"""Fail-closed evidence, vessel strata, and selection primitives for D2.

All component thresholds are derived from development-train labels.  Callers
must never pass validation/test rows to ``derive_vessel_strata``.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import binary_dilation, label as connected_components

from sabids.config import load_config
from sabids.experiments.dose_response import (
    asset_inventory,
    audit_selection_history,
    effective_split_sha,
    resolve,
    stable_sha,
    write_strict_json_exclusive,
)
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file


STRATA_VERSION = "vessel-strata-model-grid-v1"
BEST_BINDING_VERSION = "training-checkpoint-binding-v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _filtered_manifest(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for role in ("train", "val"):
        split = str(config["data"].get(f"{role}_split", role))
        part = manifest[manifest["split"].astype(str).eq(split)].copy()
        datasets = config["data"].get(f"{role}_datasets")
        groups = config["data"].get(f"{role}_groups")
        if datasets:
            part = part[part["dataset"].astype(str).isin([str(v) for v in datasets])]
        if groups:
            part = part[part["group_id"].astype(str).isin([str(v) for v in groups])]
        part["split"] = role
        parts.append(part)
    filtered = pd.concat(parts, ignore_index=True)
    _require(not filtered.empty and set(filtered["split"]) == {"train", "val"},
             "Effective checkpoint cohort must contain train and validation")
    return filtered


def bind_best_checkpoint_evidence(
    project_root: Path,
    initial_inventory: Path,
    checkpoint: Path,
    history: Path,
    resolved_config: Path,
    run_metadata: Path,
    protocol_lock: Path,
    split_contract: Path,
    output: Path,
) -> dict:
    """Derive a best-checkpoint proof without modifying any source evidence."""
    root = project_root.resolve()
    sources = {
        "initial_inventory": initial_inventory.resolve(),
        "checkpoint": checkpoint.resolve(),
        "history": history.resolve(),
        "resolved_config": resolved_config.resolve(),
        "run_metadata": run_metadata.resolve(),
        "protocol_lock": protocol_lock.resolve(),
        "split_contract": split_contract.resolve(),
    }
    for name, path in sources.items():
        _require(path.is_file(), f"Missing binding source {name}: {path}")
    _require(checkpoint.name == "best.pth", "Best binding requires best.pth")
    _require(output.resolve() not in sources.values(), "Binding output cannot replace source evidence")

    initial = json.loads(initial_inventory.read_text(encoding="utf-8-sig"))
    _require(initial.get("recorded_at_training") is True
             and initial.get("recorded_before_optimizer_step") is True,
             "Initial inventory is not immutable training-start evidence")
    config = load_config(resolved_config)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(isinstance(raw, dict) and raw.get("config") and raw.get("model"),
             "Checkpoint lacks embedded config/model")
    _require(isinstance(raw.get("optimizer"), dict)
             and isinstance(raw.get("scheduler"), dict),
             "Best checkpoint lacks optimizer/scheduler training state")
    optimizer_steps = []
    for state in raw["optimizer"].get("state", {}).values():
        step = state.get("step") if isinstance(state, dict) else None
        if step is not None:
            optimizer_steps.append(int(step.item() if torch.is_tensor(step) else step))
    _require(bool(optimizer_steps) and max(optimizer_steps) > 0,
             "Best checkpoint lacks a positive optimizer global step")
    checkpoint_global_step = max(optimizer_steps)
    for section in ("model", "loss", "data", "train", "seed"):
        _require(raw["config"].get(section) == config.get(section),
                 f"Checkpoint/resolved config mismatch: {section}")
    # A state-dict-shaped dictionary is not sufficient provenance. Instantiate
    # the embedded architecture and require a strict load without perturbing
    # the caller's model RNG stream.
    from sabids.engine.trainer import build_model
    with torch.random.fork_rng(devices=[]):
        compatibility_model = build_model(config)
    try:
        compatibility_model.load_state_dict(raw["model"], strict=True)
    except RuntimeError as error:
        raise ValueError("Best checkpoint state dict is incompatible with resolved model") from error
    _require(config["train"].get("monitor") == "psnr", "Best D1 monitor is not psnr")
    history_audit = audit_selection_history(
        history, int(config["train"]["epochs"]), "best_validation_psnr"
    )
    checkpoint_epoch = int(raw.get("epoch", -1)) + 1
    _require(checkpoint_epoch == int(history_audit["best_epoch"]),
             "Best checkpoint epoch differs from history best PSNR epoch")
    metadata = json.loads(run_metadata.read_text(encoding="utf-8-sig"))
    _require(np.isfinite(float(raw.get("best_metric", float("nan"))))
             and abs(float(raw["best_metric"]) - float(history_audit["best_val_psnr"])) <= 1e-8,
             "Checkpoint best_metric differs from history best PSNR")
    checkpoint_sha = sha256_file(checkpoint)
    _require(metadata.get("best_checkpoint_sha256") == checkpoint_sha,
             "run_metadata best checkpoint SHA mismatch")
    _require(int(metadata.get("best_epoch", -1)) == checkpoint_epoch,
             "run_metadata best epoch mismatch")
    _require(metadata.get("monitor") == "psnr", "run_metadata monitor mismatch")
    _require(abs(float(metadata.get("best_metric", float("nan")))
                 - float(history_audit["best_val_psnr"])) <= 1e-8,
             "run_metadata best metric mismatch")
    metadata_checkpoint = Path(str(metadata.get("best_checkpoint", ""))).expanduser()
    metadata_checkpoint = (
        metadata_checkpoint.resolve()
        if metadata_checkpoint.is_absolute()
        else (root / metadata_checkpoint).resolve()
    )
    _require(metadata_checkpoint == checkpoint.resolve(),
             "run_metadata best checkpoint path mismatch")
    metadata_run_id = str(metadata.get("run_id") or metadata_checkpoint.parent.name)
    _require(metadata_run_id == checkpoint.parent.name, "run_metadata run ID mismatch")
    if metadata.get("selection_rule") is not None:
        _require(metadata.get("selection_rule") == "best_validation_psnr",
                 "run_metadata selection rule mismatch")

    lock = load_protocol_lock(protocol_lock)
    _require(config.get("protocol_id") == lock["protocol_id"], "Protocol ID mismatch")
    embedded_lock = config.get("runtime", {}).get("active_protocol_lock", {})
    for key in ("data_plan_sha256", "label_inventory_sha256", "dataset_inventory_sha256",
                "split_contract_sha256"):
        _require(embedded_lock.get(key, config.get(key)) == lock[key],
                 f"Protocol lock mismatch: {key}")
    _require(sha256_file(split_contract) == lock["split_contract_sha256"],
             "Split contract SHA mismatch")
    manifest_path = resolve(root, config["data"]["manifest"])
    _require(sha256_file(manifest_path) == initial.get("manifest_sha256")
             == config.get("runtime", {}).get("manifest_sha256"),
             "Manifest SHA differs from training-start evidence")
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    filtered = _filtered_manifest(config, manifest)
    split_sha = effective_split_sha(filtered)
    _require(split_sha == initial.get("effective_split_sha256")
             == config.get("runtime", {}).get("effective_split_sha256"),
             "Effective split SHA mismatch")
    data_root = resolve(root, config["data"].get("root") or root)
    current_records = asset_inventory(data_root, filtered, include_labels=False)
    records_sha = stable_sha(current_records)
    _require(initial.get("records") == current_records
             and initial.get("records_sha256") == records_sha
             and initial.get("train_val_noisy_clean_asset_sha256") == records_sha,
             "Training pixel assets differ from initial inventory")
    run_id = checkpoint.parent.name
    configured_run_id = Path(str(config["train"]["output_dir"])).name
    _require(run_id == configured_run_id, "Run ID differs from resolved output directory")
    initial_git = initial.get("git_commit")
    config_git = config.get("runtime", {}).get("git_commit")
    _require(bool(initial_git) and initial_git == config_git, "Training Git commit mismatch")
    initialization_checkpoint = config.get("runtime", {}).get("initialization_checkpoint")
    initialization_sha = config.get("runtime", {}).get("initialization_checkpoint_sha256")
    _require(bool(initialization_checkpoint) == bool(initialization_sha),
             "Initialization checkpoint path/SHA evidence is incomplete")

    source_sha = {name: sha256_file(path) for name, path in sources.items()}
    result = {
        "schema_version": BEST_BINDING_VERSION,
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "protocol_id": lock["protocol_id"],
        "effective_split_sha256": split_sha,
        "train_val_noisy_clean_asset_sha256": records_sha,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_global_optimizer_step": checkpoint_global_step,
        "selection_rule": "best_validation_psnr",
        "monitor": "psnr",
        "run_metadata_selection_rule_source": (
            "explicit" if metadata.get("selection_rule") is not None
            else "best_checkpoint_path+monitor+best_epoch"
        ),
        "history_best_epoch": int(history_audit["best_epoch"]),
        "history_best_value": float(history_audit["best_val_psnr"]),
        "git_commit": initial_git,
        "initialization_checkpoint": initialization_checkpoint,
        "initialization_checkpoint_sha256": initialization_sha,
        "initialization_mode": "checkpoint" if initialization_checkpoint else "seeded_random",
        "source_files": {name: str(path) for name, path in sources.items()},
        "source_sha256": source_sha,
        "history_audit": history_audit,
        "unverifiable_fields": ["external_timestamp_or_cryptographic_signature"],
        "warnings": [
            "The local hash chain detects changes after binding but historical files are not externally signed."
        ],
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output, result)
    return result


def audit_best_checkpoint_binding(
    binding_path: Path,
    checkpoint: Path,
    initial_inventory: Path,
    history: Path,
    resolved_config: Path,
    run_metadata: Path,
    protocol_lock: Path,
    split_contract: Path,
) -> dict:
    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    _require(binding.get("schema_version") == BEST_BINDING_VERSION
             and binding.get("status") == "passed", "Invalid best binding schema/status")
    expected = {
        "initial_inventory": initial_inventory.resolve(),
        "checkpoint": checkpoint.resolve(),
        "history": history.resolve(),
        "resolved_config": resolved_config.resolve(),
        "run_metadata": run_metadata.resolve(),
        "protocol_lock": protocol_lock.resolve(),
        "split_contract": split_contract.resolve(),
    }
    for name, path in expected.items():
        _require(path.is_file(), f"Best binding source missing: {name}")
        _require(binding.get("source_sha256", {}).get(name) == sha256_file(path),
                 f"Best binding source changed: {name}")
    _require(binding.get("checkpoint_sha256") == sha256_file(checkpoint),
             "Best binding checkpoint SHA mismatch")
    _require(binding.get("selection_rule") == "best_validation_psnr"
             and binding.get("monitor") == "psnr", "Best binding selection mismatch")
    return binding


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    return connected_components(mask.astype(bool), structure=structure)


def local_component_contrast(
    image: np.ndarray,
    component: np.ndarray,
    all_vessels: np.ndarray,
    layer: np.ndarray,
    valid: np.ndarray,
    ring_width: int = 3,
    eps: float = 1e-6,
) -> tuple[float, int]:
    ring = binary_dilation(component, iterations=max(1, int(ring_width)))
    ring &= ~all_vessels.astype(bool)
    ring &= layer.astype(bool) & valid.astype(bool)
    vessel = component.astype(bool) & valid.astype(bool)
    if not vessel.any() or not ring.any():
        return float("nan"), int(ring.sum())
    ring_values = image[ring].astype(np.float64)
    q25, q75 = np.percentile(ring_values, (25, 75))
    # OCT vessel lumina are expected to be darker than the surrounding stroma.
    # Keep the signed, preregistered definition: a negative value is meaningful
    # evidence that the local appearance is not a dark-lumen contrast.
    contrast = float(np.median(ring_values)) - float(np.median(image[vessel]))
    return contrast / max(float(q75 - q25), eps), int(ring.sum())


def component_measurements(
    vessel: np.ndarray,
    layer: np.ndarray,
    valid: np.ndarray,
    noisy: np.ndarray,
    clean: np.ndarray | None = None,
    ring_width: int = 3,
) -> list[dict]:
    valid_bool = valid.astype(bool)
    target = vessel.astype(bool) & valid_bool
    labels, count = _components(target)
    rows = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        noisy_contrast, ring_pixels = local_component_contrast(
            noisy, component, target, layer, valid_bool, ring_width
        )
        clean_contrast = float("nan")
        if clean is not None:
            clean_contrast, _ = local_component_contrast(
                clean, component, target, layer, valid_bool, ring_width
            )
        rows.append({
            "component_id": int(component_id),
            "area_pixels": int(component.sum()),
            "noisy_local_contrast": noisy_contrast,
            "clean_local_contrast": clean_contrast,
            "ring_pixels": ring_pixels,
            "noisy_contrast_valid": bool(np.isfinite(noisy_contrast)),
            "noisy_contrast_invalid_reason": (
                None if np.isfinite(noisy_contrast) else "empty_local_stroma_ring"
            ),
            "clean_contrast_valid": bool(np.isfinite(clean_contrast)),
            "clean_contrast_invalid_reason": (
                None if np.isfinite(clean_contrast) else
                "clean_not_available" if clean is None else "empty_local_stroma_ring"
            ),
        })
    return rows


def derive_vessel_strata(
    training_samples: Iterable[dict],
    fixed_area_thresholds: tuple[int, int] | None = None,
    ring_width: int = 3,
) -> dict:
    measurements = []
    sample_count = 0
    for sample in training_samples:
        _require(str(sample.get("split", "train")) == "train",
                 "Vessel strata thresholds may only use train samples")
        sample_count += 1
        measurements.extend(component_measurements(
            sample["vessel"], sample["layer"], sample["valid"], sample["noisy"],
            sample.get("clean"), ring_width,
        ))
    _require(bool(measurements), "No valid train vessel components for strata")
    areas = np.asarray([row["area_pixels"] for row in measurements], dtype=np.float64)
    noisy_contrasts = np.asarray(
        [row["noisy_local_contrast"] for row in measurements], dtype=np.float64
    )
    finite_contrasts = noisy_contrasts[np.isfinite(noisy_contrasts)]
    clean_contrasts = np.asarray(
        [row["clean_local_contrast"] for row in measurements], dtype=np.float64
    )
    finite_clean_contrasts = clean_contrasts[np.isfinite(clean_contrasts)]
    _require(finite_contrasts.size > 0, "No train component has a valid local contrast ring")
    q33, q67 = np.quantile(areas, (1 / 3, 2 / 3), method="linear")
    q25 = float(np.quantile(finite_contrasts, 0.25, method="linear"))
    definition = {
        "version": STRATA_VERSION,
        "threshold_source": "development_train_gt_only",
        "coordinate_system": "model_grid_pixels",
        "resize_note": "areas are measured after the configured resize/pad transform",
        "training_sample_count": sample_count,
        "training_component_count": int(len(measurements)),
        "training_contrast_component_count": int(finite_contrasts.size),
        "area_q33": float(q33),
        "area_q67": float(q67),
        "fixed_area_thresholds": (
            [int(fixed_area_thresholds[0]), int(fixed_area_thresholds[1])]
            if fixed_area_thresholds is not None else None
        ),
        "low_contrast_q25_noisy": q25,
        "low_contrast_q25_clean": (
            float(np.quantile(finite_clean_contrasts, 0.25, method="linear"))
            if finite_clean_contrasts.size else None
        ),
        "primary_contrast_source": "noisy",
        "clean_contrast_role": "sensitivity_only",
        "ring_width_pixels": int(ring_width),
        "component_connectivity": 8,
        "coverage_primary_threshold": 0.25,
        "test_assets_opened": 0,
    }
    definition["definition_sha256"] = stable_sha(definition)
    return definition


def evaluate_vessel_components(
    prediction: np.ndarray,
    vessel: np.ndarray,
    layer: np.ndarray,
    valid: np.ndarray,
    noisy: np.ndarray,
    definition: dict,
    clean: np.ndarray | None = None,
) -> list[dict]:
    expected = dict(definition)
    claimed = expected.pop("definition_sha256", None)
    _require(claimed == stable_sha(expected), "Vessel strata definition SHA mismatch")
    pred = prediction.astype(bool) & valid.astype(bool)
    rows = component_measurements(
        vessel, layer, valid, noisy, clean, int(definition["ring_width_pixels"])
    )
    fixed = definition.get("fixed_area_thresholds")
    for row in rows:
        area = row["area_pixels"]
        row["area_bin_quantile"] = (
            "small" if area <= definition["area_q33"] else
            "medium" if area <= definition["area_q67"] else "large"
        )
        row["area_bin_fixed"] = (
            None if fixed is None else
            "small" if area <= fixed[0] else "medium" if area <= fixed[1] else "large"
        )
        component = connected_components(
            (vessel.astype(bool) & valid.astype(bool)),
            structure=np.ones((3, 3), dtype=np.uint8),
        )[0] == row["component_id"]
        coverage = float(pred[component].sum() / max(int(component.sum()), 1))
        row.update({
            "coverage": coverage,
            "any_overlap": float(coverage > 0.0),
            "recall_at_025": float(coverage >= 0.25),
            "recall_at_050": float(coverage >= 0.5),
            "completely_missed": float(coverage == 0.0),
            "contrast_bin": (
                "unknown" if not np.isfinite(row["noisy_local_contrast"])
                else "low" if row["noisy_local_contrast"] <= definition["low_contrast_q25_noisy"]
                else "normal_high"
            ),
            "clean_contrast_bin": (
                "unknown"
                if definition.get("low_contrast_q25_clean") is None
                or not np.isfinite(row["clean_local_contrast"])
                else "low"
                if row["clean_local_contrast"] <= definition["low_contrast_q25_clean"]
                else "normal_high"
            ),
        })
    return rows


def component_strata_masks(
    vessel: np.ndarray,
    layer: np.ndarray,
    valid: np.ndarray,
    noisy: np.ndarray,
    definition: dict,
) -> dict[str, np.ndarray]:
    """Return frozen train-defined GT masks for D2 residual diagnostics."""
    expected = dict(definition)
    claimed = expected.pop("definition_sha256", None)
    _require(claimed == stable_sha(expected), "Vessel strata definition SHA mismatch")
    target = vessel.astype(bool) & valid.astype(bool)
    labels, count = _components(target)
    masks = {
        "small": np.zeros_like(target, dtype=bool),
        "low_contrast": np.zeros_like(target, dtype=bool),
        "small_low_contrast": np.zeros_like(target, dtype=bool),
    }
    ring_width = int(definition["ring_width_pixels"])
    for component_id in range(1, count + 1):
        component = labels == component_id
        is_small = int(component.sum()) <= float(definition["area_q33"])
        contrast, _ = local_component_contrast(
            noisy, component, target, layer, valid, ring_width
        )
        is_low = bool(
            np.isfinite(contrast)
            and contrast <= float(definition["low_contrast_q25_noisy"])
        )
        if is_small:
            masks["small"] |= component
        if is_low:
            masks["low_contrast"] |= component
        if is_small and is_low:
            masks["small_low_contrast"] |= component
    return masks


def aggregate_component_rows(
    rows: list[dict], area_key: str = "area_bin_quantile",
    contrast_key: str = "contrast_bin",
) -> dict:
    result: dict[str, Any] = {"component_count": float(len(rows))}
    if not rows:
        for key in ("any_overlap_recall", "recall_at_025", "recall_at_050",
                    "mean_coverage", "completely_missed_count"):
            result[key] = float("nan")
        result["component_metric_reason"] = "no_valid_gt_components"
        return result
    def add(prefix: str, part: list[dict]) -> None:
        for key in ("any_overlap", "recall_at_025", "recall_at_050", "coverage"):
            result[f"{prefix}{'mean_coverage' if key == 'coverage' else key + '_recall' if key == 'any_overlap' else key}"] = (
                float(np.mean([row[key] for row in part])) if part else float("nan")
            )
        result[f"{prefix}completely_missed_count"] = (
            float(sum(row["completely_missed"] for row in part)) if part else float("nan")
        )
        result[f"{prefix}component_count"] = float(len(part))
    add("", rows)
    for size in ("small", "medium", "large"):
        add(f"{size}_", [row for row in rows if row.get(area_key) == size])
    for contrast in ("low", "normal_high"):
        add(f"{contrast}_contrast_", [row for row in rows if row.get(contrast_key) == contrast])
    add("small_low_contrast_", [
        row for row in rows
        if row.get(area_key) == "small" and row.get(contrast_key) == "low"
    ])
    result["component_metric_reason"] = "ok"
    return result


def select_d2_checkpoints(
    history: pd.DataFrame,
    psnr_column: str = "val_psnr",
    task_column: str = "val_teacher_task_preservation",
    psnr_noninferiority_db: float = 0.2,
) -> dict:
    required = {"epoch", psnr_column, task_column}
    _require(required.issubset(history.columns), f"Selection history missing {sorted(required - set(history))}")
    table = history[list(required)].copy()
    for column in required:
        table[column] = pd.to_numeric(table[column], errors="raise")
    _require(np.isfinite(table.to_numpy(dtype=float)).all(), "Selection history contains NaN/Inf")
    _require(not table["epoch"].duplicated().any(), "Selection history has duplicate epoch")
    table = table.sort_values("epoch", kind="stable")
    best_psnr = float(table[psnr_column].max())
    best_pixel = table[table[psnr_column].eq(best_psnr)].iloc[0]
    eligible = table[table[psnr_column].ge(best_psnr - float(psnr_noninferiority_db))]
    best_task_value = float(eligible[task_column].max())
    best_task = eligible[eligible[task_column].eq(best_task_value)].iloc[0]
    return {
        "selection_rule_version": "d2-hierarchical-v1",
        "psnr_noninferiority_db": float(psnr_noninferiority_db),
        "best_pixel_epoch": int(best_pixel["epoch"]),
        "best_pixel_psnr": best_psnr,
        "best_task_preserving_epoch": int(best_task["epoch"]),
        "best_task_preserving_psnr": float(best_task[psnr_column]),
        "best_task_preserving_value": best_task_value,
        "task_metric": task_column,
        "tie_break": "earliest_epoch",
    }


def write_d2_checkpoint_binding(
    initial_inventory: Path,
    checkpoint: Path,
    output: Path,
    checkpoint_kind: str,
    selection: dict,
    selection_history: Path,
    teacher_audit: Path | None = None,
    parameter_audit: Path | None = None,
) -> dict:
    _require(initial_inventory.is_file(), "D2 initial training inventory missing")
    _require(checkpoint.is_file(), f"D2 checkpoint missing: {checkpoint}")
    _require(selection_history.is_file(), "D2 selection history audit missing")
    initial = json.loads(initial_inventory.read_text(encoding="utf-8-sig"))
    _require(initial.get("recorded_at_training") is True
             and initial.get("recorded_before_optimizer_step") is True,
             "D2 initial inventory lacks pre-optimizer evidence")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_epoch = {
        "best_pixel": selection.get("best_pixel_epoch"),
        "best_task_preserving": selection.get("best_task_preserving_epoch"),
        "last": selection.get("completed_epochs"),
    }.get(checkpoint_kind)
    _require(expected_epoch is not None and int(raw.get("epoch", -1)) + 1 == int(expected_epoch),
             f"D2 {checkpoint_kind} checkpoint epoch mismatch")
    checkpoint_details = selection.get("checkpoint_details", {}).get(checkpoint_kind)
    _require(isinstance(checkpoint_details, dict),
             f"D2 {checkpoint_kind} selection lacks checkpoint details")
    _require(int(raw.get("global_optimizer_step", -1))
             == int(checkpoint_details.get("global_optimizer_step", -2)),
             f"D2 {checkpoint_kind} global optimizer step mismatch")
    _require(teacher_audit is not None and teacher_audit.is_file(),
             "D2 teacher audit missing")
    teacher = json.loads(teacher_audit.read_text(encoding="utf-8-sig"))
    _require(teacher.get("status") in {"passed", "not_applicable"}
             and int(teacher.get("changed_parameter_count", -1)) == 0
             and int(teacher.get("requires_grad_parameter_count", -1)) == 0,
             "D2 teacher freeze audit failed")
    _require(parameter_audit is not None and parameter_audit.is_file(),
             "D2 parameter audit missing")
    parameters = json.loads(parameter_audit.read_text(encoding="utf-8-sig"))
    _require(parameters.get("status") == "passed"
             and int(parameters.get("changed_trainable_parameter_count", 0)) > 0
             and int(parameters.get("changed_frozen_parameter_count", -1)) == 0,
             "D2 parameter update audit failed")
    result = {
        "schema_version": "d2-checkpoint-binding-v1",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(expected_epoch),
        "global_optimizer_step": int(checkpoint_details["global_optimizer_step"]),
        "monitor_values": {
            "val_psnr": float(checkpoint_details["val_psnr"]),
            "val_teacher_task_preservation": float(
                checkpoint_details["val_teacher_task_preservation"]
            ),
        },
        "run_id": checkpoint.parent.name,
        "protocol_id": raw.get("config", {}).get("protocol_id"),
        "effective_split_sha256": initial.get("effective_split_sha256"),
        "train_val_noisy_clean_asset_sha256": initial.get(
            "train_val_noisy_clean_asset_sha256"
        ),
        "git_commit": initial.get("git_commit"),
        "initialization_checkpoint": raw.get("config", {}).get("runtime", {}).get(
            "initialization_checkpoint"
        ),
        "initialization_checkpoint_sha256": raw.get("config", {}).get("runtime", {}).get(
            "initialization_checkpoint_sha256"
        ),
        "selection": selection,
        "initial_inventory_path": str(initial_inventory.resolve()),
        "initial_inventory_sha256": sha256_file(initial_inventory),
        "selection_history_path": str(selection_history.resolve()),
        "selection_history_sha256": sha256_file(selection_history),
        "teacher_audit_path": str(teacher_audit.resolve()) if teacher_audit and teacher_audit.is_file() else None,
        "teacher_audit_sha256": sha256_file(teacher_audit) if teacher_audit and teacher_audit.is_file() else None,
        "parameter_audit_path": str(parameter_audit.resolve()),
        "parameter_audit_sha256": sha256_file(parameter_audit),
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output, result)
    return result


def audit_d2_checkpoint_binding(
    binding_path: Path,
    checkpoint: Path,
    expected_kind: str,
) -> dict:
    binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
    _require(binding.get("schema_version") == "d2-checkpoint-binding-v1"
             and binding.get("status") == "passed", "Invalid D2 checkpoint binding")
    _require(binding.get("checkpoint_kind") == expected_kind,
             "D2 checkpoint kind differs from binding")
    _require(binding.get("checkpoint_sha256") == sha256_file(checkpoint),
             "D2 checkpoint SHA differs from binding")
    sources = {
        "initial_inventory": (binding.get("initial_inventory_path"), binding.get("initial_inventory_sha256")),
        "selection_history": (binding.get("selection_history_path"), binding.get("selection_history_sha256")),
    }
    if binding.get("teacher_audit_path"):
        sources["teacher_audit"] = (binding["teacher_audit_path"], binding.get("teacher_audit_sha256"))
    sources["parameter_audit"] = (
        binding.get("parameter_audit_path"), binding.get("parameter_audit_sha256")
    )
    for name, (value, expected_sha) in sources.items():
        path = Path(str(value)).expanduser().resolve()
        _require(path.is_file() and sha256_file(path) == expected_sha,
                 f"D2 binding source changed: {name}")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(int(raw.get("epoch", -1)) + 1 == int(binding.get("checkpoint_epoch", -2)),
             "D2 checkpoint epoch differs from binding")
    _require(int(raw.get("global_optimizer_step", -1))
             == int(binding.get("global_optimizer_step", -2)),
             "D2 checkpoint global optimizer step differs from binding")
    teacher = json.loads(Path(binding["teacher_audit_path"]).read_text(encoding="utf-8-sig"))
    parameters = json.loads(Path(binding["parameter_audit_path"]).read_text(encoding="utf-8-sig"))
    _require(teacher.get("status") in {"passed", "not_applicable"}
             and int(teacher.get("changed_parameter_count", -1)) == 0
             and int(teacher.get("requires_grad_parameter_count", -1)) == 0,
             "D2 bound teacher audit is not clean")
    _require(parameters.get("status") == "passed"
             and int(parameters.get("changed_trainable_parameter_count", 0)) > 0
             and int(parameters.get("changed_frozen_parameter_count", -1)) == 0,
             "D2 bound parameter audit is not clean")
    return binding


def formal_d2_preflight(
    root: Path,
    checkpoint: Path,
    binding_path: Path,
    checkpoint_kind: str,
    protocol_lock: Path,
    split_contract: Path,
) -> dict:
    """Validate one completed D2 checkpoint without opening sealed test assets."""
    report = {"mode": "formal", "status": "blocked", "issues": [],
              "test_assets_opened": 0, "test_evaluation_performed": False}
    try:
        root = root.resolve(); checkpoint = checkpoint.resolve()
        _require(checkpoint.is_file(), f"Missing D2 checkpoint: {checkpoint}")
        expected_name = {"d2_pixel": "best_pixel.pth", "d2_task": "best_task_preserving.pth",
                         "d2_last": "last.pth"}.get(checkpoint_kind)
        _require(expected_name is not None and checkpoint.name == expected_name,
                 "D2 checkpoint filename/kind mismatch")
        binding_kind = {"d2_pixel": "best_pixel", "d2_task": "best_task_preserving",
                        "d2_last": "last"}[checkpoint_kind]
        lock = load_protocol_lock(protocol_lock)
        _require(sha256_file(split_contract) == lock["split_contract_sha256"],
                 "D2 split contract SHA mismatch")
        contract = yaml.safe_load(split_contract.read_text(encoding="utf-8-sig"))
        _require(contract.get("protocol_id") == lock["protocol_id"], "D2 split protocol mismatch")
        binding = audit_d2_checkpoint_binding(binding_path, checkpoint, binding_kind)
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = raw.get("config", {})
        _require(config.get("d2", {}).get("enabled") is True
                 and config.get("loss", {}).get("restoration_mode") == "structure_d2",
                 "Checkpoint is not registered structure_d2")
        _require(config.get("d2", {}).get("run_mode") not in {"smoke", "overfit"},
                 "Smoke/overfit D2 checkpoint is not eligible for scientific dose evaluation")
        _require(config.get("train", {}).get("stage") == "denoise", "D2 stage is not denoise")
        _require(config.get("protocol_id") == lock["protocol_id"], "D2 protocol mismatch")
        for key in ("data_plan_sha256", "label_inventory_sha256"):
            found = config.get("runtime", {}).get("active_protocol_lock", {}).get(key, config.get(key))
            _require(found == lock[key], f"D2 protocol lock mismatch: {key}")
        resolved_config = checkpoint.parent / "resolved_config.yaml"
        _require(resolved_config.is_file(), "D2 resolved_config.yaml missing")
        resolved = load_config(resolved_config)
        for section in ("model", "loss", "data", "train", "d2", "seed"):
            _require(config.get(section) == resolved.get(section),
                     f"D2 checkpoint/resolved config mismatch: {section}")
        initial_path = Path(binding["initial_inventory_path"]).resolve()
        initial = json.loads(initial_path.read_text(encoding="utf-8-sig"))
        _require(initial.get("recorded_before_optimizer_step") is True,
                 "D2 initial inventory was not recorded before optimisation")
        manifest = resolve(root, config["data"]["manifest"])
        _require(sha256_file(manifest) == initial.get("manifest_sha256"),
                 "D2 training manifest changed")
        table = pd.read_csv(manifest, dtype=str).fillna("")
        filtered = _filtered_manifest(config, table)
        data_root = resolve(root, config["data"].get("root") or root)
        records = asset_inventory(data_root, filtered, include_labels=False)
        records_sha = stable_sha(records)
        _require(records == initial.get("records")
                 and records_sha == initial.get("records_sha256")
                 and effective_split_sha(filtered) == initial.get("effective_split_sha256"),
                 "D2 current train/val assets differ from training-start evidence")
        protocol_root = resolve(root, lock["manifest_root"])
        segmentation_manifest = protocol_root / "train_segment.csv"
        _require(segmentation_manifest.is_file(), "D2 segmentation manifest missing")
        segmentation = pd.read_csv(segmentation_manifest, dtype=str).fillna("")
        _require(segmentation["split"].isin(["train", "val"]).all(),
                 "D2 downstream manifest contains non-development rows")
        report.update({
            "status": "passed", "protocol": lock,
            "protocol_lock": str(protocol_lock.resolve()),
            "protocol_lock_sha256": sha256_file(protocol_lock),
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_binding": str(binding_path.resolve()),
            "checkpoint_binding_sha256": sha256_file(binding_path),
            "checkpoint_kind": checkpoint_kind,
            "d1_resolved_config": str(resolved_config),
            "resolved_config_sha256": sha256_file(resolved_config),
            "restoration_mode": "structure_d2",
            "input_resolution": list(config["data"]["target_size"]),
            "segmentation_manifest": str(segmentation_manifest),
            "segmentation_manifest_sha256": sha256_file(segmentation_manifest),
            "segmentation_asset_inventory": asset_inventory(root, segmentation, include_labels=True),
            "train_val_noisy_clean_asset_sha256": records_sha,
        })
    except Exception as exc:
        report["issues"].append(str(exc))
    return report
