from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.data.io import write_gray, write_rgb
from sabids.engine.evaluator import evaluate_model
from sabids.engine.trainer import build_diagnostic_loader, build_model
from sabids.utils import get_device, load_checkpoint, write_json
from tools.input_oracle_cv_common import ARMS, resolve, sha256_file
from tools.run_input_oracle_cv import _clean_once_loader


ARM_LABELS = {"noisy": "I-NOISY", "denoised": "I-DENOISED", "clean": "I-CLEAN"}
INPUT_COLUMNS = {"noisy": "noisy_cache_path", "denoised": "denoised_cache_path", "clean": "clean_cache_path"}
REQUIRED_EXPORTS = (
    "input.png", "layer_prob.png", "layer_mask.png", "vessel_prob.png", "vessel_mask.png",
    "layer_overlay.png", "vessel_overlay.png", "combined_overlay.png", "gt_layer_mask.png",
    "gt_vessel_mask.png", "gt_layer_overlay.png", "gt_vessel_overlay.png",
    "gt_combined_overlay.png", "layer_prob.npy", "vessel_prob.npy", "metadata.json",
)
METRIC_MAP = {
    "layer_dice": ("layer_dice", "p0_layer_dice"),
    "upper_boundary_mae": ("upper_boundary_mae", "p0_upper_boundary_mae"),
    "lower_boundary_mae": ("lower_boundary_mae", "p0_lower_boundary_mae"),
    "thickness_mae": ("thickness_mae", "p0_thickness_mae"),
    "vessel_dice": ("p0_vessel_dice", "vessel_dice"),
    "vessel_precision": ("p0_vessel_precision", "vessel_precision"),
    "vessel_recall": ("p0_vessel_recall", "vessel_recall"),
    "vessel_roi_dice": ("vessel_roi_dice",),
    "vessel_roi_fp": ("vessel_roi_fp_pixels",),
    "vessel_roi_fn": ("vessel_roi_fn_pixels",),
    "vessel_outside_gt_layer_fraction": ("vessel_outside_gt_layer_fraction",),
    "vessel_area_fraction": ("vessel_area_fraction_pred",),
    "vessel_area_fraction_mae": ("vessel_area_fraction_mae",),
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _load_resolved(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return value if isinstance(value, dict) else {}


def _path(root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    candidate = Path(text).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _state_hashes(state: dict[str, torch.Tensor]) -> tuple[str, str, dict[str, str]]:
    aggregate, common = hashlib.sha256(), hashlib.sha256()
    tensors: dict[str, str] = {}
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
        tensors[name] = digest.hexdigest()
        payload = f"{name}|{tensors[name]}\n".encode("utf-8")
        aggregate.update(payload)
        if not name.startswith("interactions."):
            common.update(payload)
    return aggregate.hexdigest(), common.hexdigest(), tensors


def _module_names(names: Iterable[str]) -> str:
    modules: set[str] = set()
    for name in names:
        parts = str(name).split(".")
        if parts[0] in {"adapters", "decoders", "interactions"} and len(parts) > 1:
            modules.add(".".join(parts[:2]))
        else:
            modules.add(parts[0])
    return ";".join(sorted(modules))


def _changed_names(initial: dict[str, torch.Tensor], final: dict[str, torch.Tensor], names: Iterable[str]) -> list[str]:
    changed = []
    for name in names:
        if name not in initial or name not in final:
            changed.append(str(name))
            continue
        if not torch.equal(initial[name].detach().cpu(), final[name].detach().cpu()):
            changed.append(str(name))
    return changed


def _archive_output(output: Path, archive_existing: bool) -> Path | None:
    if not output.exists() or not any(output.iterdir()):
        output.mkdir(parents=True, exist_ok=True)
        return None
    if not archive_existing:
        raise FileExistsError(f"Refusing to overwrite visualization output: {output}; use --archive-existing")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = output.with_name(f"{output.name}_archive_{stamp}")
    counter = 1
    while archive.exists():
        archive = output.with_name(f"{output.name}_archive_{stamp}_{counter:02d}")
        counter += 1
    output.rename(archive)
    output.mkdir(parents=True)
    return archive


def _arm_enabled(config: dict[str, Any], short: str, legacy: str) -> bool:
    model = config.get("model", {})
    return bool(model.get(short, model.get(legacy, False)))


def _optimizer_objects(payload: dict[str, Any]) -> int:
    optimizer = payload.get("optimizer") or {}
    return sum(len(group.get("params", [])) for group in optimizer.get("param_groups", []))


def _requires_grad_names(config: dict[str, Any]) -> list[str]:
    if not config:
        return []
    model = build_model(config)
    model.set_train_stage(
        str(config.get("train", {}).get("stage", "input_segment")),
        private_train_encoder_levels=config.get("model", {}).get("private_train_encoder_levels", []),
        freeze_shared_encoder=bool(config.get("model", {}).get("freeze_shared_encoder", config.get("model", {}).get("stage2_freeze_shared_encoder", False))),
        train_denoise_to_seg=bool(config.get("model", {}).get("stage2_train_denoise_to_seg", False)),
    )
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    del model
    return names


def audit_runs(
    root: Path,
    runs: Path,
    folds: list[int],
    seed: int,
    checkpoint_name: str,
    epoch: int,
    threshold: float,
    sealed: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[tuple[int, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    parameters: list[dict[str, Any]] = []
    contexts: dict[tuple[int, str], dict[str, Any]] = {}
    for fold in folds:
        for arm in ARMS:
            run = runs / f"fold{fold}" / f"{arm}_seed{seed}"
            config_path = run / "resolved_config.yaml"
            checkpoint = run / checkpoint_name
            config = _load_resolved(config_path)
            initialization_audit = _read_json(run / "initialization_audit.json")
            parameter_audit = _read_json(run / "parameter_audit.json")
            notes: list[str] = []
            payload: dict[str, Any] = {}
            checkpoint_sha = ""
            if checkpoint.is_file():
                checkpoint_sha = sha256_file(checkpoint)
                try:
                    payload = torch.load(checkpoint, map_location="cpu")
                except Exception as exc:
                    notes.append(f"checkpoint_load_failed:{exc}")
            else:
                notes.append("MISSING CHECKPOINT")
            if not config:
                notes.append("missing_resolved_config")

            initialization = _path(root, config.get("train", {}).get("pretrained")) if config else None
            if initialization is None:
                initialization = runs / f"fold{fold}" / "preseg_initialization.pth"
            initialization_sha = sha256_file(initialization) if initialization.is_file() else ""
            if not initialization.is_file():
                notes.append("missing_preseg_initialization")
            initial_payload = torch.load(initialization, map_location="cpu") if initialization.is_file() else {}
            initial_state = initial_payload.get("model", initial_payload if initial_payload else {})
            final_state = payload.get("model", payload if payload else {})
            initial_model_hash, initial_common_hash, _ = _state_hashes(initial_state) if initial_state else ("", "", {})
            final_model_hash, _, _ = _state_hashes(final_state) if final_state else ("", "", {})

            trainable = list(map(str, parameter_audit.get("trainable", [])))
            frozen = list(map(str, parameter_audit.get("frozen", [])))
            requires_grad = _requires_grad_names(config)
            requires_grad_match = sorted(trainable) == sorted(requires_grad)
            if not requires_grad_match:
                notes.append("parameter_audit_requires_grad_mismatch")
            changed_trainable = _changed_names(initial_state, final_state, trainable) if initial_state and final_state else []
            changed_frozen = _changed_names(initial_state, final_state, frozen) if initial_state and final_state else []
            trainable_elements = sum(int(final_state[name].numel()) for name in trainable if name in final_state)
            frozen_elements = sum(int(final_state[name].numel()) for name in frozen if name in final_state)
            optimizer_objects = _optimizer_objects(payload)
            encoder_prefixes = ("stem.", "encoder_blocks.", "downsamples.")
            encoder_trainable = any(name.startswith(encoder_prefixes) for name in trainable)
            denoise_trainable = any(name.startswith(("adapters.denoise.", "decoders.denoise.", "residual_head.")) for name in trainable)
            interaction_trainable = any(name.startswith("interactions.") for name in trainable)
            declared_freeze = bool(config.get("model", {}).get("freeze_shared_encoder", config.get("model", {}).get("stage2_freeze_shared_encoder", False))) if config else False
            manifest = _resolve_manifest(root, config) if config else None
            manifest_table = pd.read_csv(manifest, dtype=str).fillna("") if manifest else pd.DataFrame()
            train_groups = sorted(manifest_table.loc[manifest_table["split"] == str(config.get("data", {}).get("train_split", "train")), "group_id"].astype(str).unique()) if len(manifest_table) else []
            val_groups = sorted(manifest_table.loc[manifest_table["split"] == str(config.get("data", {}).get("val_split", "val")), "group_id"].astype(str).unique()) if len(manifest_table) else []
            augmentation_sha = hashlib.sha256(json.dumps(config.get("data", {}).get("augmentation", {}), sort_keys=True).encode("utf-8")).hexdigest() if config else ""
            completed_epoch = int(payload.get("epoch", -1)) + 1 if payload else -1
            d2s = _arm_enabled(config, "d2s_enabled", "enable_denoise_to_seg") if config else False
            s2d = _arm_enabled(config, "s2d_enabled", "enable_seg_to_denoise") if config else False
            optimizer_count_audit = int(parameter_audit.get("optimizer_parameter_count", -1))
            if declared_freeze and encoder_trainable:
                notes.append("CONFIG_FREEZE_MISMATCH: shared encoder was actually trainable")
            if denoise_trainable:
                notes.append("denoising_path_trainable")
            if interaction_trainable:
                notes.append("interaction_parameters_trainable")
            if d2s or s2d:
                notes.append("interaction_forward_enabled")
            if completed_epoch != epoch:
                notes.append(f"completed_epoch={completed_epoch}, expected={epoch}")
            if initialization_audit.get("model_state_sha256") and initialization_audit.get("model_state_sha256") != initial_model_hash:
                notes.append("initialization_model_state_hash_mismatch")
            if initialization_audit.get("common_state_sha256") and initialization_audit.get("common_state_sha256") != initial_common_hash:
                notes.append("initialization_common_state_hash_mismatch")
            if optimizer_count_audit >= 0 and optimizer_count_audit != trainable_elements:
                notes.append("optimizer_element_count_mismatch")
            if optimizer_objects and optimizer_objects != len(trainable):
                notes.append("optimizer_object_count_mismatch")
            if changed_frozen:
                notes.append(f"frozen_parameter_changes={len(changed_frozen)}")

            row = {
                "fold": fold, "seed": seed, "arm": ARM_LABELS[arm], "run_dir": str(run),
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                "initialization_path": str(initialization), "initialization_sha256": initialization_sha,
                "model_state_sha256": initialization_audit.get("model_state_sha256", initial_model_hash),
                "common_state_sha256": initialization_audit.get("common_state_sha256", initial_common_hash),
                "final_model_state_sha256": final_model_hash,
                "data_plan_sha256": initialization_audit.get("data_plan_sha256", ""),
                "manifest_sha256": sha256_file(manifest) if manifest else "",
                "train_groups": ";".join(train_groups), "validation_groups": ";".join(val_groups),
                "augmentation_sha256": augmentation_sha,
                "configured_epochs": int(config.get("train", {}).get("epochs", -1)) if config else -1,
                "completed_epoch": completed_epoch, "trainable_parameter_count": trainable_elements,
                "frozen_parameter_count": frozen_elements, "trainable_modules": _module_names(trainable),
                "d2s_enabled": d2s, "s2d_enabled": s2d,
                "declared_freeze_shared_encoder": declared_freeze,
                "actual_shared_encoder_trainable": encoder_trainable,
                "denoising_path_frozen": not denoise_trainable,
                "interaction_modules_frozen": not interaction_trainable,
                "optimizer_parameter_objects": optimizer_objects,
                "optimizer_parameter_elements": optimizer_count_audit,
                "requires_grad_matches_parameter_audit": requires_grad_match,
                "changed_trainable_parameter_tensors": len(changed_trainable),
                "changed_frozen_parameter_tensors": len(changed_frozen),
                "threshold": threshold, "postprocess": "P0",
                "audit_pass": not notes, "audit_notes": "; ".join(notes),
            }
            rows.append(row)
            parameters.append({
                "fold": fold, "seed": seed, "arm": ARM_LABELS[arm],
                "trainable_modules": _module_names(trainable), "frozen_modules": _module_names(frozen),
                "trainable_parameter_names": ";".join(trainable),
                "changed_trainable_parameter_names": ";".join(changed_trainable),
                "unexpected_changed_frozen_parameter_names": ";".join(changed_frozen),
                "declared_freeze_shared_encoder": declared_freeze,
                "actual_shared_encoder_trainable": encoder_trainable,
                "denoising_path_frozen": not denoise_trainable,
                "interaction_modules_frozen": not interaction_trainable,
                "optimizer_parameter_objects": optimizer_objects,
                "optimizer_parameter_elements": optimizer_count_audit,
                "requires_grad_parameter_names": ";".join(requires_grad),
                "requires_grad_matches_parameter_audit": requires_grad_match,
            })
            contexts[(fold, arm)] = {
                "run": run, "config": config, "checkpoint": checkpoint,
                "checkpoint_sha256": checkpoint_sha,
            }

    table = pd.DataFrame(rows)
    parameter_table = pd.DataFrame(parameters)
    pairing: dict[str, Any] = {}
    for fold, part in table.groupby("fold", sort=True):
        checks = {
            "all_three_arms_present": set(part["arm"]) == set(ARM_LABELS.values()),
            "same_initialization_sha256": part["initialization_sha256"].nunique() == 1 and bool(part["initialization_sha256"].iloc[0]),
            "same_model_state_sha256": part["model_state_sha256"].nunique() == 1 and bool(part["model_state_sha256"].iloc[0]),
            "same_common_state_sha256": part["common_state_sha256"].nunique() == 1 and bool(part["common_state_sha256"].iloc[0]),
            "same_data_plan_sha256": part["data_plan_sha256"].nunique() == 1 and bool(part["data_plan_sha256"].iloc[0]),
            "same_manifest_sha256": part["manifest_sha256"].nunique() == 1 and bool(part["manifest_sha256"].iloc[0]),
            "same_training_positions": part["train_groups"].nunique() == 1,
            "same_validation_positions": part["validation_groups"].nunique() == 1,
            "same_augmentation_plan": part["augmentation_sha256"].nunique() == 1,
            "same_configured_epoch": part["configured_epochs"].nunique() == 1 and int(part["configured_epochs"].iloc[0]) == epoch,
            "all_epoch_60": bool((part["completed_epoch"] == epoch).all()),
            "all_d2s_off": bool((~part["d2s_enabled"]).all()),
            "all_s2d_off": bool((~part["s2d_enabled"]).all()),
            "all_denoising_frozen": bool(part["denoising_path_frozen"].all()),
            "all_interactions_frozen": bool(part["interaction_modules_frozen"].all()),
        }
        pairing[str(fold)] = {"checks": checks, "paired_protocol_pass": all(checks.values())}
    return table, parameter_table, pairing, contexts


def _write_audit_summary(path: Path, table: pd.DataFrame, pairing: dict[str, Any]) -> None:
    freeze_mismatch = table[table["declared_freeze_shared_encoder"] & table["actual_shared_encoder_trainable"]]
    text = [
        "# I-NOISY / I-DENOISED / I-CLEAN audit", "",
        "This report audits fixed-final checkpoints and does not select a checkpoint or threshold.", "",
        "## Interpretation", "",
    ]
    if len(freeze_mismatch):
        text.append("> 当前实验是“相同 Stage 1 预训练初始化后，针对三种输入分别进行匹配微调”，不是整个网络从随机权重训练，也不是完全冻结编码器的输入探针。")
        text.append("")
        text.append(f"`stage2_freeze_shared_encoder: true` conflicts with the actual trainable `stem/encoder_blocks/downsamples` in {len(freeze_mismatch)} run(s).")
    else:
        text.append("No declared-versus-actual shared-encoder freeze mismatch was found in the available run audits.")
    text.extend(["", "## Fold pairing", "", "```json", json.dumps(pairing, ensure_ascii=False, indent=2), "```", ""])
    path.write_text("\n".join(text), encoding="utf-8")


def _resolve_manifest(root: Path, config: dict[str, Any]) -> Path:
    value = config.get("data", {}).get("manifest")
    path = _path(root, value)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Missing configured input-oracle manifest: {path}")
    return path


def _expected_rows(root: Path, config: dict[str, Any], arm: str, val_groups: set[str], sealed: set[str]) -> pd.DataFrame:
    manifest = pd.read_csv(_resolve_manifest(root, config), dtype=str).fillna("")
    val = manifest[(manifest["split"] == str(config.get("data", {}).get("val_split", "val"))) & manifest["group_id"].isin(val_groups)].copy()
    overlap = set(val["group_id"]) & sealed
    if overlap:
        raise RuntimeError(f"Sealed test groups entered visualization selection: {sorted(overlap)}")
    val = val.sort_values(["group_id", "sample_id"]).reset_index(drop=True)
    if arm == "clean":
        val = val.drop_duplicates("group_id", keep="first").reset_index(drop=True)
    return val


def _evaluation_complete(directory: Path, rows: pd.DataFrame) -> bool:
    frame_path = directory / "frame_metrics.csv"
    if not frame_path.is_file():
        return False
    try:
        metrics = pd.read_csv(frame_path, dtype={"sample_id": str})
    except Exception:
        return False
    expected = set(rows["sample_id"].astype(str))
    if not expected.issubset(set(metrics["sample_id"].astype(str))):
        return False
    for row in rows.itertuples(index=False):
        base = directory / "predictions" / str(row.dataset)
        for suffix in ("noisy.png", "layer_prob.png", "vessel_prob.png", "layer_prob_float32.npy", "vessel_prob_float32.npy", "layer_mask.png", "vessel_mask.png", "layer_gt.png", "vessel_gt.png"):
            if not (base / f"{row.sample_id}_{suffix}").is_file():
                return False
    return True


def _run_evaluation(config: dict[str, Any], checkpoint: Path, destination: Path, arm: str, device: str, workers: int, val_groups: set[str]) -> None:
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Refusing to merge a partial inference cache: {destination}")
    eval_config = json.loads(json.dumps(config))
    eval_config.setdefault("evaluation", {})["num_workers"] = workers
    eval_config["evaluation"]["batch_size"] = 1
    eval_config.setdefault("data", {})["val_groups"] = sorted(val_groups)
    model = build_model(eval_config)
    target = get_device(device)
    model.to(target)
    load_checkpoint(checkpoint, model, strict=True, map_location=target)
    loader = build_diagnostic_loader(eval_config, split=str(eval_config.get("data", {}).get("val_split", "val")))
    if arm == "clean":
        loader = _clean_once_loader(loader)
    evaluate_model(
        model, loader, target, output_dir=destination, threshold=0.5,
        layer_threshold=0.5, vessel_threshold=0.5, save_predictions=True,
        stage="input_segment", tasks=("layer", "vessel"), postprocess_modes=("p0",),
        restore_original_geometry=True,
        input_normalization=str(eval_config.get("data", {}).get("normalization", "fixed")),
    )


def _read_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False).astype(np.float32)
    else:
        raw = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"DECODE FAIL: {path}")
        if raw.ndim == 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        maximum = float(np.iinfo(raw.dtype).max) if np.issubdtype(raw.dtype, np.integer) else 1.0
        value = raw.astype(np.float32) / max(maximum, 1.0)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise RuntimeError(f"Invalid 2-D finite image: {path}")
    return np.clip(value, 0.0, 1.0)


def overlay_mask(background: np.ndarray, layer: np.ndarray | None = None, vessel: np.ndarray | None = None, alpha_layer: float = 0.23, alpha_vessel: float = 0.40) -> np.ndarray:
    if background.ndim != 2:
        raise ValueError("Overlay background must be 2-D grayscale")
    output = np.repeat(np.clip(background, 0.0, 1.0)[..., None], 3, axis=2).astype(np.float32)
    if layer is not None:
        mask = layer.astype(bool)
        output[mask] = (1.0 - alpha_layer) * output[mask] + alpha_layer * np.array((0.0, 1.0, 0.0), np.float32)
    if vessel is not None:
        mask = vessel.astype(bool)
        output[mask] = (1.0 - alpha_vessel) * output[mask] + alpha_vessel * np.array((1.0, 0.0, 0.0), np.float32)
    return np.clip(output, 0.0, 1.0)


def _pick_metric(row: pd.Series, candidates: tuple[str, ...]) -> float:
    for name in candidates:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    return float("nan")


def _copy_position_assets(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_EXPORTS:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)


def _materialize_sample(
    source: Path,
    destination: Path,
    metadata: dict[str, Any],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    sample_id = str(metadata["sample_id"])
    required = {
        "input": source / f"{sample_id}_noisy.png",
        "layer_prob_png": source / f"{sample_id}_layer_prob.png",
        "vessel_prob_png": source / f"{sample_id}_vessel_prob.png",
        "layer_prob": source / f"{sample_id}_layer_prob_float32.npy",
        "vessel_prob": source / f"{sample_id}_vessel_prob_float32.npy",
        "layer_mask": source / f"{sample_id}_layer_mask.png",
        "vessel_mask": source / f"{sample_id}_vessel_mask.png",
        "gt_layer": source / f"{sample_id}_layer_gt.png",
        "gt_vessel": source / f"{sample_id}_vessel_gt.png",
    }
    missing = [{"status": "MISSING GT" if key.startswith("gt_") else "MISSING INPUT" if key == "input" else "MISSING PREDICTION", "asset": key, "path": str(path), **metadata} for key, path in required.items() if not path.is_file()]
    if missing:
        return {}, missing
    try:
        image = _read_gray(required["input"])
        layer_prob = _read_gray(required["layer_prob"])
        vessel_prob = _read_gray(required["vessel_prob"])
        layer_mask = _read_gray(required["layer_mask"]) > 0.5
        vessel_mask = _read_gray(required["vessel_mask"]) > 0.5
        gt_layer = _read_gray(required["gt_layer"]) > 0.5
        gt_vessel = _read_gray(required["gt_vessel"]) > 0.5
    except Exception as exc:
        return {}, [{"status": "DECODE FAIL", "asset": "sample", "path": str(source), "notes": str(exc), **metadata}]
    shapes = {tuple(x.shape) for x in (image, layer_prob, vessel_prob, layer_mask, vessel_mask, gt_layer, gt_vessel)}
    if len(shapes) != 1:
        return {}, [{"status": "CROP OOB", "asset": "shape_alignment", "path": str(source), "notes": str(sorted(shapes)), **metadata}]
    expected_shape = (int(metadata.get("expected_original_height", image.shape[0])), int(metadata.get("expected_original_width", image.shape[1])))
    if image.shape != expected_shape:
        return {}, [{"status": "CROP OOB", "asset": "original_geometry", "path": str(source), "notes": f"got={image.shape}, expected={expected_shape}", **metadata}]
    destination.mkdir(parents=True, exist_ok=True)
    write_gray(destination / "input.png", image)
    write_gray(destination / "layer_prob.png", layer_prob)
    write_gray(destination / "vessel_prob.png", vessel_prob)
    np.save(destination / "layer_prob.npy", layer_prob.astype(np.float32), allow_pickle=False)
    np.save(destination / "vessel_prob.npy", vessel_prob.astype(np.float32), allow_pickle=False)
    write_gray(destination / "layer_mask.png", layer_mask.astype(np.float32))
    write_gray(destination / "vessel_mask.png", vessel_mask.astype(np.float32))
    write_gray(destination / "gt_layer_mask.png", gt_layer.astype(np.float32))
    write_gray(destination / "gt_vessel_mask.png", gt_vessel.astype(np.float32))
    write_rgb(destination / "layer_overlay.png", overlay_mask(image, layer=layer_mask))
    write_rgb(destination / "vessel_overlay.png", overlay_mask(image, vessel=vessel_mask))
    write_rgb(destination / "combined_overlay.png", overlay_mask(image, layer_mask, vessel_mask))
    write_rgb(destination / "gt_layer_overlay.png", overlay_mask(image, layer=gt_layer))
    write_rgb(destination / "gt_vessel_overlay.png", overlay_mask(image, vessel=gt_vessel))
    write_rgb(destination / "gt_combined_overlay.png", overlay_mask(image, gt_layer, gt_vessel))
    metadata = {**metadata, "original_height": image.shape[0], "original_width": image.shape[1], "probability_dtype": "float32", "display_mapping": "fixed_[0,1]_to_[0,255]", "layer_color_rgb": [0, 255, 0], "vessel_color_rgb": [255, 0, 0], "alpha_layer": 0.23, "alpha_vessel": 0.40}
    write_json(metadata, destination / "metadata.json")
    paths: dict[str, str] = {}
    for name in REQUIRED_EXPORTS:
        if name == "metadata.json":
            continue
        stem = name.removesuffix(".png").removesuffix(".npy")
        if name.endswith(".npy"):
            stem += "_float"
        paths[f"{stem}_path"] = str((destination / name).resolve())
    paths["metadata_path"] = str((destination / "metadata.json").resolve())
    return paths, []


def _asset_inventory(output: Path, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for record in registry.to_dict("records"):
        for column, value in record.items():
            if not column.endswith("_path") or not str(value).strip():
                continue
            path = Path(str(value))
            item = {"fold": record["fold"], "group_id": record["group_id"], "sample_id": record["sample_id"], "arm": record["arm"], "asset": column, "path": str(path), "status": "OK" if path.is_file() else "MISSING"}
            if path.is_file():
                item.update({"size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
            rows.append(item)
    return pd.DataFrame(rows)


def _load_rgb(path: Path) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    if value is None:
        raise RuntimeError(f"DECODE FAIL: {path}")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _panel(group_id: str, fold: int, seed: int, per_position: Path, target: Path, epoch: int, threshold: float) -> None:
    arms = ("I-NOISY", "I-DENOISED", "I-CLEAN")
    assets = ("input.png", "layer_overlay.png", "vessel_overlay.png", "combined_overlay.png", "gt_combined_overlay.png")
    row_labels = ("Input", "Layer prediction", "Vessel prediction", "Combined prediction", "GT reference")
    tile_w, tile_h, left, top, gap = 420, 320, 230, 120, 8
    canvas = np.full((top + len(assets) * (tile_h + gap) + 80, left + len(arms) * (tile_w + gap) + 20, 3), 255, np.uint8)
    cv2.putText(canvas, f"{group_id} | fold={fold} | seed={seed} | last.pth epoch={epoch} | threshold={threshold:g} | P0", (24, 42), cv2.FONT_HERSHEY_SIMPLEX, .72, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Clean is a position-level reference, not a repeated noisy frame.", (24, 78), cv2.FONT_HERSHEY_SIMPLEX, .57, (70, 70, 70), 1, cv2.LINE_AA)
    for col, arm in enumerate(arms):
        cv2.putText(canvas, arm, (left + col * (tile_w + gap) + 10, 110), cv2.FONT_HERSHEY_SIMPLEX, .68, (20, 20, 20), 2, cv2.LINE_AA)
    for row, (asset, label) in enumerate(zip(assets, row_labels)):
        y = top + row * (tile_h + gap)
        cv2.putText(canvas, label, (18, y + tile_h // 2), cv2.FONT_HERSHEY_SIMPLEX, .57, (25, 25, 25), 1, cv2.LINE_AA)
        for col, arm in enumerate(arms):
            path = per_position / group_id / arm / asset
            image = _load_rgb(path) if path.is_file() else np.full((tile_h, tile_w, 3), 235, np.uint8)
            if not path.is_file():
                cv2.putText(image, "MISSING", (60, image.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            x = left + col * (tile_w + gap)
            canvas[y:y + tile_h, x:x + tile_w] = _fit(image, tile_w, tile_h)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Panel encode failed: {target}")
    encoded.tofile(str(target))


def _contact_sheet(panels: list[Path], target: Path) -> None:
    tile_w, tile_h, columns = 420, 480, 4
    rows = max(1, int(np.ceil(len(panels) / columns)))
    canvas = np.full((rows * tile_h, columns * tile_w, 3), 255, np.uint8)
    for index, path in enumerate(panels):
        image = _load_rgb(path)
        row, col = divmod(index, columns)
        canvas[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = _fit(image, tile_w, tile_h)
    target.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Contact sheet encode failed")
    encoded.tofile(str(target))


def _copy_tree_files(files: Iterable[Path], base: Path, destination: Path) -> None:
    for source in files:
        if not source.is_file():
            continue
        target = destination / source.relative_to(base)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _make_bundle(output: Path, sealed: set[str]) -> tuple[Path, dict[str, Any]]:
    bundle = output / "gpt_analysis_bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    _copy_tree_files((output / "audit").rglob("*"), output, bundle)
    table_names = {
        "fixed_final_metrics_by_position.csv", "fixed_final_metrics_summary.csv",
        "missing_assets.csv",
    }
    _copy_tree_files(
        (path for path in (output / "tables").rglob("*") if path.name in table_names),
        output,
        bundle,
    )
    atlas = output / "atlas"
    _copy_tree_files(atlas.rglob("*"), output, bundle)
    for path in (output / "README.md", output / "build_summary.json"):
        if path.is_file():
            shutil.copy2(path, bundle / path.name)
    per_position = output / "per_position"
    allowed = {"input.png", "layer_overlay.png", "vessel_overlay.png", "combined_overlay.png", "metadata.json"}
    _copy_tree_files((path for path in per_position.rglob("*") if path.name in allowed), output, bundle)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = output / f"SABIDS_I_NOISY_I_DENOISED_I_CLEAN_visualization_seed42_{stamp}.zip"
    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in files:
            handle.write(path, path.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        bad = handle.testzip()
    forbidden = [name for name in names if Path(name).suffix.lower() in {".pth", ".pt", ".npy", ".npz"} or any(token in name.lower().split("/") for token in ("test", "test_results", "predictions", "cache"))]
    leaked = [group for group in sealed if any(group in name for name in names)]
    if bad or forbidden or leaked:
        raise RuntimeError(f"GPT package validation failed: crc={bad}, forbidden={forbidden[:5]}, sealed={leaked}")
    result = {"path": str(archive), "sha256": sha256_file(archive), "files": len(names), "crc_check": "passed", "forbidden_assets": 0, "sealed_test_groups": 0}
    return archive, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fixed-final I-NOISY/I-DENOISED/I-CLEAN segmentations")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--runs-root", default="runs/input_oracle_cv")
    parser.add_argument("--output-root", default="runs/input_oracle_visualization_seed42_fixed_final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", default="last.pth")
    parser.add_argument("--epoch", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--postprocess", choices=("p0",), default="p0")
    parser.add_argument("--folds", type=int, nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--all-outer-val-frames", action="store_true")
    parser.add_argument("--make-atlas", action="store_true")
    parser.add_argument("--make-gpt-bundle", action="store_true")
    parser.add_argument("--archive-existing", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Export only the lexicographically first validation position")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.seed != 42 or args.checkpoint != "last.pth" or args.epoch != 60 or args.threshold != 0.5 or args.postprocess != "p0":
        raise ValueError("The primary visualization protocol is fixed to seed=42, last.pth, epoch=60, threshold=0.5, P0")
    root = Path(args.project_root).expanduser().resolve()
    runs = resolve(root, args.runs_root)
    output = resolve(root, args.output_root)
    archived = _archive_output(output, args.archive_existing)
    for name in ("audit", "tables", "per_frame", "per_position", "atlas/panels"):
        (output / name).mkdir(parents=True, exist_ok=True)

    phase0 = _read_json(runs / "audit/audit_summary.json")
    split = _read_json(runs / "splits/split_audit.json")
    if phase0.get("status") != "passed" or not phase0.get("training_authorized") or int(phase0.get("test_assets_opened", -1)) != 0:
        raise RuntimeError(f"Missing or blocked Phase 0 audit: {runs / 'audit/audit_summary.json'}")
    if not split or split.get("status") != "passed":
        raise RuntimeError(f"Missing or blocked outer-fold audit: {runs / 'splits/split_audit.json'}")
    folds = args.folds or sorted(map(int, split.get("folds", {}).keys()))
    protocol_path = root / "configs/current/input_oracle_cv/protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    sealed = set(map(str, protocol.get("sealed_test_groups", [])))
    sealed.update(map(str, phase0.get("sealed_test_groups", [])))
    sealed.update(map(str, split.get("sealed_test_groups_metadata_only", [])))
    if not args.audit_only and not args.smoke_test and not args.all_outer_val_frames:
        raise ValueError("Full export requires --all-outer-val-frames")
    declared_val = [str(group) for fold in folds for group in split["folds"][str(fold)].get("val_groups", [])]
    duplicate_val = sorted({group for group in declared_val if declared_val.count(group) > 1})
    if duplicate_val:
        raise RuntimeError(f"Outer-validation groups are duplicated across folds: {duplicate_val}")
    if set(declared_val) & sealed:
        raise RuntimeError(f"Outer-validation groups overlap the sealed test denylist: {sorted(set(declared_val) & sealed)}")
    if not args.smoke_test and not args.audit_only:
        development = set(map(str, phase0.get("development_groups", [])))
        expected_count = int(protocol.get("expected_development_labelled_groups", 16))
        if set(declared_val) != development or len(declared_val) != expected_count:
            raise RuntimeError(
                f"Full external-validation coverage must equal the {expected_count} audited development positions; "
                f"got {len(declared_val)}, missing={sorted(development - set(declared_val))}, extra={sorted(set(declared_val) - development)}"
            )

    initialization, parameters, pairing, contexts = audit_runs(root, runs, folds, args.seed, args.checkpoint, args.epoch, args.threshold, sealed)
    initialization.to_csv(output / "audit/initialization_audit.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(output / "audit/parameter_audit.csv", index=False, encoding="utf-8-sig")
    _write_audit_summary(output / "audit/AUDIT_SUMMARY.md", initialization, pairing)
    protocol_audit = {
        "phase0_status": phase0.get("status", "missing"), "split_status": split.get("status"),
        "folds": folds, "seed": args.seed, "checkpoint": args.checkpoint, "epoch": args.epoch,
        "threshold": args.threshold, "postprocess": "P0", "sealed_test_denylist": sorted(sealed),
        "pairing": pairing, "audit_failures": int((~initialization["audit_pass"]).sum()),
        "test_assets_opened": 0,
        "interpretation": "matched fine-tuning from the same Stage 1 initialization; not random initialization and not a frozen-encoder probe" if bool(initialization["actual_shared_encoder_trainable"].any()) else "frozen-encoder input probe",
    }
    write_json(protocol_audit, output / "audit/protocol_audit.json")
    if args.audit_only:
        write_json({"status": "audit_only", "archived_previous_output": str(archived or ""), "test_assets_opened": 0}, output / "build_summary.json")
        print(json.dumps(protocol_audit, ensure_ascii=False, indent=2))
        return

    hard_errors = initialization[~initialization["checkpoint_path"].map(lambda value: Path(value).is_file())]
    if len(hard_errors):
        raise FileNotFoundError("Missing fixed-final checkpoints; inspect audit/initialization_audit.csv")

    metric_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    selections: dict[str, dict[str, Any]] = {}
    seen_outer_groups: set[str] = set()
    for fold in folds:
        val_groups = set(map(str, split["folds"][str(fold)]["val_groups"]))
        overlap = val_groups & sealed
        if overlap:
            raise RuntimeError(f"Fold {fold} external validation overlaps sealed test denylist: {sorted(overlap)}")
        if seen_outer_groups & val_groups:
            raise RuntimeError(f"External validation position appears in multiple folds: {sorted(seen_outer_groups & val_groups)}")
        seen_outer_groups.update(val_groups)
        if args.smoke_test:
            val_groups = {sorted(val_groups)[0]}
        for arm in ARMS:
            context = contexts[(fold, arm)]
            config = context["config"]
            expected = _expected_rows(root, config, arm, val_groups, sealed)
            existing = context["run"] / "final_validation"
            evaluation = existing if _evaluation_complete(existing, expected) else output / "_evaluation_cache" / f"fold{fold}" / f"{arm}_seed{args.seed}"
            if not _evaluation_complete(evaluation, expected):
                _run_evaluation(config, context["checkpoint"], evaluation, arm, args.device, args.num_workers, val_groups)
            metrics = pd.read_csv(evaluation / "frame_metrics.csv", dtype={"sample_id": str}).fillna(np.nan)
            for manifest_row in expected.to_dict("records"):
                sample_id, group_id, dataset = str(manifest_row["sample_id"]), str(manifest_row["group_id"]), str(manifest_row["dataset"])
                arm_label = ARM_LABELS[arm]
                destination = (output / "per_position" / group_id / arm_label) if arm == "clean" else (output / "per_frame" / f"fold{fold}" / group_id / arm_label / sample_id)
                row_match = metrics[(metrics["sample_id"].astype(str) == sample_id) & (metrics["group_id"].astype(str) == group_id)]
                source_metrics = row_match.iloc[0] if not row_match.empty else pd.Series(dtype=object)
                metadata = {
                    "fold": fold, "seed": args.seed, "arm": arm_label, "group_id": group_id,
                    "sample_id": sample_id, "dataset": dataset, "checkpoint": str(context["checkpoint"]),
                    "checkpoint_sha256": context["checkpoint_sha256"], "epoch": args.epoch,
                    "threshold": args.threshold, "postprocess": "P0",
                    "input_source": str(_path(root, manifest_row.get(INPUT_COLUMNS[arm], "")) or ""),
                    "clean_is_position_level_reference": arm == "clean",
                    "expected_original_height": int(source_metrics.get("original_height", source_metrics.get("evaluation_height", 0)) or 0),
                    "expected_original_width": int(source_metrics.get("original_width", source_metrics.get("evaluation_width", 0)) or 0),
                }
                if metadata["expected_original_height"] <= 0 or metadata["expected_original_width"] <= 0:
                    metadata.pop("expected_original_height")
                    metadata.pop("expected_original_width")
                paths, failures = _materialize_sample(evaluation / "predictions" / dataset, destination, metadata)
                missing.extend(failures)
                status = "OK" if paths else (failures[0]["status"] if failures else "INFERENCE FAIL")
                registry = {**metadata, **paths, "status": status, "notes": "Clean is a position-level reference, not a repeated noisy frame." if arm == "clean" else ""}
                registry_rows.append(registry)
                if row_match.empty:
                    missing.append({**metadata, "status": "INFERENCE FAIL", "asset": "frame_metrics row", "path": str(evaluation / "frame_metrics.csv")})
                    continue
                metric = {**metadata, **{name: _pick_metric(source_metrics, candidates) for name, candidates in METRIC_MAP.items()}, "image_directory": str(destination.resolve()), "input_image_path": paths.get("input_path", ""), "layer_overlay_path": paths.get("layer_overlay_path", ""), "vessel_overlay_path": paths.get("vessel_overlay_path", ""), "combined_overlay_path": paths.get("combined_overlay_path", ""), "original_height": source_metrics.get("evaluation_height", source_metrics.get("original_height", "")), "original_width": source_metrics.get("evaluation_width", source_metrics.get("original_width", ""))}
                metric_rows.append(metric)
                choice = selections.setdefault(group_id, {"fold": fold, "group_id": group_id, "seed": args.seed, "selection_rule": "lexicographically_first_valid_sample_id"})
                choice.setdefault(f"{arm}_sample_id", sample_id)
                if arm != "clean" and sample_id == expected[expected["group_id"] == group_id]["sample_id"].astype(str).min() and paths:
                    _copy_position_assets(destination, output / "per_position" / group_id / arm_label)

    registry = pd.DataFrame(registry_rows)
    metrics = pd.DataFrame(metric_rows)
    shape_failures = []
    for group_id in sorted(selections):
        shapes: dict[str, tuple[int, int]] = {}
        for arm_label in ARM_LABELS.values():
            path = output / "per_position" / group_id / arm_label / "input.png"
            if path.is_file():
                image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
                if image is not None:
                    shapes[arm_label] = tuple(image.shape[:2])
        if len(set(shapes.values())) > 1:
            shape_failures.append({"fold": selections[group_id]["fold"], "seed": args.seed, "arm": "ALL", "group_id": group_id, "sample_id": "", "dataset": "PKU37", "status": "CROP OOB", "asset": "cross_arm_original_geometry", "path": str(output / "per_position" / group_id), "notes": json.dumps(shapes)})
    missing.extend(shape_failures)
    registry.to_csv(output / "tables/visualization_registry.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(output / "tables/fixed_final_metrics_by_sample.csv", index=False, encoding="utf-8-sig")
    numeric = list(METRIC_MAP)
    position = metrics.groupby(["fold", "group_id", "arm", "seed"], as_index=False)[numeric].mean(numeric_only=True)
    counts = metrics.groupby(["fold", "group_id", "arm", "seed"]).size().rename("n_samples").reset_index()
    position = position.merge(counts, on=["fold", "group_id", "arm", "seed"])
    position.to_csv(output / "tables/fixed_final_metrics_by_position.csv", index=False, encoding="utf-8-sig")
    summary = position.groupby("arm", as_index=False)[numeric].agg(["mean", "std", "count"])
    summary.columns = ["_".join(str(value) for value in column if value) for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output / "tables/fixed_final_metrics_summary.csv", index=False, encoding="utf-8-sig")
    assets = _asset_inventory(output, registry)
    assets.to_csv(output / "tables/asset_inventory.csv", index=False, encoding="utf-8-sig")
    missing_columns = ["fold", "seed", "arm", "group_id", "sample_id", "dataset", "status", "asset", "path", "notes"]
    pd.DataFrame(missing, columns=missing_columns).to_csv(output / "tables/missing_assets.csv", index=False, encoding="utf-8-sig")

    selections_table = pd.DataFrame(selections.values()).sort_values(["fold", "group_id"])
    selections_table["clean_mapping_note"] = "I-CLEAN uses one position-level reference; it is not duplicated as repeat frames."
    selections_table.to_csv(output / "atlas/atlas_selection.csv", index=False, encoding="utf-8-sig")
    panels: list[Path] = []
    if args.make_atlas:
        page_rows = []
        for selected in selections_table.to_dict("records"):
            group_id, fold = str(selected["group_id"]), int(selected["fold"])
            target = output / "atlas/panels" / f"{group_id}_three_input_comparison.png"
            _panel(group_id, fold, args.seed, output / "per_position", target, args.epoch, args.threshold)
            panels.append(target)
            page_rows.append(f"<h2>{html.escape(group_id)} | fold {fold}</h2><a href='panels/{html.escape(target.name)}'><img loading='lazy' style='max-width:100%' src='panels/{html.escape(target.name)}'></a>")
        (output / "atlas/index.html").write_text("<html><body><h1>Fixed-final three-input comparison</h1>" + "\n".join(page_rows) + "</body></html>", encoding="utf-8")
        _contact_sheet(panels, output / "atlas/contact_sheet.png")

    readme = f"""# I-NOISY / I-DENOISED / I-CLEAN fixed-final visualization

- Seed: {args.seed}
- Checkpoint: `{args.checkpoint}`; required completed epoch: {args.epoch}
- Threshold: {args.threshold}; postprocess: P0
- Selection: lexicographically first valid sample ID per outer-validation position
- CLEAN is evaluated once per anatomical position and is not duplicated into repeat frames.
- Layer overlay: RGB(0,255,0), alpha=0.23. Vessel overlay: RGB(255,0,0), alpha=0.40.
- Probabilities are restored to original geometry with bilinear interpolation; masks/GT use nearest-neighbor restoration in the evaluator.
- Test denylist was applied before any image read. Test assets opened: 0.
- Inspect `audit/AUDIT_SUMMARY.md`: existing input-segment training fine-tuned the shared encoder despite the inherited freeze flag.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    build_summary = {
        "status": "passed_with_missing_assets" if missing else "passed_with_audit_warnings" if bool((~initialization["audit_pass"]).any()) else "passed",
        "smoke_test": args.smoke_test, "folds": folds, "positions": int(position["group_id"].nunique()),
        "sample_counts": metrics.groupby("arm").size().to_dict(),
        "position_counts": position.groupby("arm")["group_id"].nunique().to_dict(),
        "missing_assets": len(missing), "panels": len(panels), "test_denylist": sorted(sealed),
        "cross_arm_geometry_check": "passed" if not shape_failures else "failed",
        "test_assets_opened": 0, "archived_previous_output": str(archived or ""),
    }
    write_json(build_summary, output / "build_summary.json")
    package = None
    if args.make_gpt_bundle:
        package, package_info = _make_bundle(output, sealed)
        build_summary["gpt_package"] = package_info
        write_json(build_summary, output / "build_summary.json")
    print(json.dumps({**build_summary, "output": str(output), "package": str(package or "")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
