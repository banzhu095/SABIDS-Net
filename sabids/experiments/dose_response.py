"""Opt-in, fail-closed primitives for the model-grid dose experiment.

Oracle uses clean references and is NOT deployable inference. No test assets
are opened here. A passed preflight is an engineering/provenance result, not
evidence for the dose-response hypothesis.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import csv
from collections import Counter
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
import cv2

from sabids.config import load_config
from sabids.data.transforms import _resize_pad
from sabids.experiments.protocol_lock import CONSISTENCY_KEYS, load_protocol_lock, sha256_file

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)
DOSE_CURVES = ("oracle", "d1", "d2_pixel", "d2_task", "d2_last")
VERSION = "dose-model-grid-v1"
SMOKE_NOTICE = "NOT FOR SCIENTIFIC EVALUATION"
REQUIRED_METADATA = (
    "protocol_id", "sample_id", "group_id", "curve_type", "alpha",
    "source_noisy_path", "source_noisy_sha256", "source_clean_path",
    "source_clean_sha256", "checkpoint_path", "checkpoint_sha256",
    "resolved_config_sha256", "restoration_mode", "geometry_sha256",
    "code_version", "split",
)


@contextmanager
def dose_deterministic_algorithms():
    """Do not leak dose-only global settings into another experiment."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_strict_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2,
                               sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_strict_json_exclusive(path: Path, value: Any) -> None:
    """Create provenance once; never replace an earlier training-time record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        json_safe(value), ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def code_fingerprint(root: Path) -> str:
    # Include dirty source contents: HEAD alone cannot distinguish uncommitted fixes.
    paths = ["sabids/experiments/dose_response.py", "tools/prepare_dose_response_inputs.py",
             "sabids/experiments/d2.py", "sabids/losses/d2.py",
             "sabids/data/dataset.py", "sabids/data/transforms.py",
             "sabids/engine/trainer.py", "sabids/losses/total.py", "configs/base.yaml",
             "configs/adaptive_denoising/dose_response_common.yaml",
             "configs/adaptive_denoising/dose_oracle.yaml", "configs/adaptive_denoising/dose_d1.yaml",
             "sabids/models/sabids_net.py", "sabids/models/ugbi.py", "sabids/models/blocks.py",
             "sabids/losses/common.py", "sabids/data/io.py", "evaluate.py", "sabids/engine/evaluator.py"]
    return stable_sha({p: sha256_file(root / p) for p in paths})


def git_commit(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                          text=True, check=False).stdout.strip() or "unknown"


def alpha_code(alpha: float) -> str:
    if isinstance(alpha, bool) or float(alpha) not in ALPHAS:
        raise ValueError(f"Unsupported alpha {alpha!r}; frozen grid is {ALPHAS}")
    return f"a{round(float(alpha) * 100):03d}"


def _image(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError("Dose inputs must be nonempty, finite 2-D images")
    if value.min() < 0 or value.max() > 1:
        raise ValueError("Dose inputs must already be normalized to [0,1]")
    return value


def read_binary_mask(path: Path) -> np.ndarray:
    """Dose-only decoder: uint8 0/1 is not divided by 255 into background.

    Binary 255 means foreground, NOT multiclass unknown. Unknown pixels require
    an explicit annotation-valid mask. Multiclass/nonbinary data are rejected.
    """
    if path.suffix.lower() == ".npy":
        value = np.load(path, allow_pickle=False)
        allowed = {0, 1}
    else:
        value = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        allowed = {0, 1, 255}
    if value is None or value.ndim != 2 or not value.size or not np.isfinite(value).all() or not set(np.unique(value)) <= allowed:
        raise ValueError(f"Expected binary mask/explicit annotation validity, not multiclass: {path}")
    return (value > 0).astype(np.float32)


def dose_input(noisy: np.ndarray, reference: np.ndarray, alpha: float,
               curve_type: str, valid: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """reference is clean for oracle, CLIPPED forward_denoise_only for d1."""
    alpha_code(alpha)
    if curve_type not in DOSE_CURVES:
        raise ValueError(f"curve_type must be one of {DOSE_CURVES}")
    x, target = _image(noisy), _image(reference)
    if x.shape != target.shape:
        raise ValueError("All inputs must use the same model grid")
    # Explicit endpoints avoid floating cancellation and give exact identities.
    raw = x.copy() if alpha == 0 else target.copy() if alpha == 1 else (
        x + np.float32(alpha) * (target - x) if curve_type == "oracle"
        else x - np.float32(alpha) * (x - target)
    )
    region = np.ones(x.shape, dtype=bool) if valid is None else np.asarray(valid) > 0.5
    if region.shape != x.shape or not region.any():
        raise ValueError("Clipping statistics need a nonempty valid region")
    stats = {"below_zero_fraction": float((raw[region] < 0).mean()),
             "above_one_fraction": float((raw[region] > 1).mean()),
             "total_clip_fraction": float(((raw[region] < 0) | (raw[region] > 1)).mean()),
             "clip_statistics_region": "spatial_valid_model_grid",
             "alpha_description": "extrapolation/residual amplification" if alpha > 1 else "interpolation",
             "preclip_min": float(raw[region].min()), "preclip_max": float(raw[region].max())}
    return np.clip(raw, 0, 1).astype(np.float32), stats


def cache_key(metadata: dict) -> str:
    missing = [k for k in REQUIRED_METADATA if metadata.get(k) in (None, "")]
    if missing:
        raise ValueError(f"Missing cache identity metadata: {missing}")
    alpha_code(metadata["alpha"])
    if metadata["curve_type"] not in DOSE_CURVES:
        raise ValueError("Unknown curve type")
    keys = list(REQUIRED_METADATA) + [k for k in ("source_label_assets_sha256", "cache_content_role") if k in metadata]
    return stable_sha({k: metadata[k] for k in keys})


def save_cache(path: Path, array: np.ndarray, metadata: dict) -> str:
    """Reuse only exact identities AND exact array bytes; never overwrite."""
    key = cache_key(metadata)
    value = _image(array)
    digest = hashlib.sha256(value.tobytes()).hexdigest()
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        if not path.is_file() or not sidecar.is_file():
            raise FileExistsError(f"Incomplete cache (do not overwrite): {path}")
        old = json.loads(sidecar.read_text(encoding="utf-8"))
        cached = np.load(path, allow_pickle=False)
        if (old.get("cache_key") != key or old.get("array_sha256") != digest
                or cached.dtype != np.float32 or not np.array_equal(cached, value)):
            raise FileExistsError(f"Cache identity/content mismatch: {path}")
        return key
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create also prevents another preparer from overwriting a cache.
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
    write_strict_json(sidecar, {**metadata, "cache_key": key, "array_sha256": digest})
    return key


def geometry_plan(shape: tuple[int, int], target: tuple[int, int]) -> dict:
    h, w = shape
    th, tw = target
    scale = min(th / h, tw / w)
    rh, rw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = (th - rh) // 2, (tw - rw) // 2
    return {"original_noisy_shape": list(shape), "resize_shape": [rh, rw],
            "pad_top_bottom_left_right": [top, th - rh - top, left, tw - rw - left],
            "crop_yxhw": [top, left, rh, rw], "model_input_shape": list(target),
            "prediction_shape": list(target), "metric_evaluation_shape": [rh, rw],
            "image_interpolation": "linear" if scale > 1 else "area",
            "label_interpolation": "nearest", "normalization": "fixed",
            "metric_coordinate_system": "cropped model grid (px)",
            "original_resolution_boundary_metrics": "NOT IMPLEMENTED"}


def to_model_grid(arrays: dict[str, np.ndarray], target: tuple[int, int]) -> tuple[dict, dict]:
    shape = arrays["noisy"].shape
    if any(value.shape != shape for value in arrays.values()):
        raise ValueError("Raw noisy/clean/GT/validity geometry differs; no implicit GT upsampling")
    plan = geometry_plan(shape, target)
    plan["original_clean_shape"] = list(arrays["clean"].shape)
    plan["original_gt_shapes"] = {k: list(v.shape) for k, v in arrays.items()
                                  if k not in {"noisy", "clean"}}
    values = {k: _resize_pad(v, target, is_mask=k not in {"noisy", "clean"})
              for k, v in arrays.items()}
    values["spatial_valid"] = _resize_pad(np.ones(shape, np.float32), target, is_mask=True)
    plan["geometry_sha256"] = stable_sha(plan)
    return values, plan


def deterministic_flip(seed: int, epoch: int, sample_id: str, probability: float) -> bool:
    if not 0 <= probability <= 1:
        raise ValueError("Invalid flip probability")
    digest = hashlib.sha256(f"dose-flip-v1|{seed}|{epoch}|{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < probability


def augmentation_plan_sha(seed: int, epochs: int, ids: list[str], probability: float) -> str:
    digest = hashlib.sha256()
    for epoch in range(epochs):
        for sample_id in ids:
            digest.update(f"{epoch}|{sample_id}|{int(deterministic_flip(seed, epoch, sample_id, probability))}\n".encode())
    return digest.hexdigest()


def resolve(root: Path, value: str | Path) -> Path:
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (root / p).resolve()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


class SelectionHistoryError(ValueError):
    """Strict history failure carrying the audit produced before rejection."""

    def __init__(self, message: str, audit: dict[str, Any]):
        super().__init__(message)
        self.audit = audit


def _selection_history_epoch(value: str, audit: dict[str, Any], row_number: int) -> int:
    text = str(value).strip()
    try:
        parsed = Decimal(text)
    except (InvalidOperation, ValueError):
        raise SelectionHistoryError(
            f"Selection history epoch is not an integer at CSV row {row_number}", audit
        )
    if not parsed.is_finite() or parsed != parsed.to_integral_value():
        raise SelectionHistoryError(
            f"Selection history epoch is not an integer at CSV row {row_number}", audit
        )
    return int(parsed)


def audit_selection_history(
    path: str | Path, configured_epochs: int, selection_rule: str
) -> dict[str, Any]:
    """Audit selection history without normalizing, dropping, or rewriting rows.

    Fixed-final selection trusts only the epoch column. A best-PSNR selection
    additionally requires a rectangular schema because an old header cannot
    identify the meaning of extra values in wider legacy rows.
    """
    history_path = Path(path)
    audit: dict[str, Any] = {
        "history_path": str(history_path),
        "configured_epochs": int(configured_epochs),
        "selection_rule": selection_rule,
        "history_schema_drift_detected": False,
        "val_psnr_trusted": False,
        "history_header_width": 0,
        "history_row_count": 0,
        "history_row_width_distribution": {},
    }
    if selection_rule not in {"fixed_final", "best_validation_psnr"}:
        raise SelectionHistoryError("Unknown selection history rule", audit)
    if not history_path.is_file():
        raise SelectionHistoryError("Missing selection history", audit)
    audit["history_sha256"] = sha256_file(history_path)
    try:
        with history_path.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.reader(handle, strict=True))
    except (OSError, csv.Error, UnicodeError) as exc:
        raise SelectionHistoryError(f"Selection history CSV is unreadable: {exc}", audit)
    if not rows:
        raise SelectionHistoryError("Selection history is empty", audit)
    header, data_rows = rows[0], rows[1:]
    widths = Counter(len(row) for row in data_rows)
    audit.update(
        history_header_width=len(header),
        history_row_count=len(data_rows),
        history_row_width_distribution={str(width): count for width, count in sorted(widths.items())},
    )
    drift = len(widths) > 1 or any(width != len(header) for width in widths)
    audit["history_schema_drift_detected"] = drift
    epoch_indices = [index for index, name in enumerate(header) if name.strip() == "epoch"]
    if len(epoch_indices) != 1:
        raise SelectionHistoryError("Selection history requires exactly one epoch header", audit)
    epoch_index = epoch_indices[0]
    audit["epoch_column_index"] = epoch_index
    epochs = []
    for row_number, row in enumerate(data_rows, start=2):
        if len(row) <= epoch_index or not str(row[epoch_index]).strip():
            raise SelectionHistoryError(
                f"Selection history epoch is missing at CSV row {row_number}", audit
            )
        epochs.append(_selection_history_epoch(row[epoch_index], audit, row_number))
    expected = list(range(1, int(configured_epochs) + 1))
    if len(data_rows) != int(configured_epochs):
        raise SelectionHistoryError(
            "Formal D1 history row count does not equal configured epochs; incomplete fixed budget",
            audit,
        )
    if len(set(epochs)) != len(epochs):
        raise SelectionHistoryError("Selection history contains duplicate epoch", audit)
    if epochs != expected:
        raise SelectionHistoryError(
            "Formal D1 history epochs must be complete and strictly equal 1...configured_epochs",
            audit,
        )
    audit.update(first_epoch=epochs[0], last_epoch=epochs[-1])
    if selection_rule == "fixed_final":
        # Deliberately do not locate, parse, or validate val_psnr. Legacy rows
        # can grow without a corresponding recoverable header; fixed-final did
        # not use validation PSNR to select its checkpoint.
        return audit
    if drift:
        raise SelectionHistoryError(
            "Selection history schema drift prevents trustworthy val_psnr best-epoch selection",
            audit,
        )
    value_indices = [index for index, name in enumerate(header) if name.strip() == "val_psnr"]
    if len(value_indices) != 1:
        raise SelectionHistoryError(
            "Best-validation selection requires exactly one val_psnr header", audit
        )
    value_index = value_indices[0]
    values = []
    for row_number, row in enumerate(data_rows, start=2):
        try:
            value = float(row[value_index])
        except (IndexError, TypeError, ValueError):
            raise SelectionHistoryError(
                f"Selection history val_psnr is invalid at CSV row {row_number}", audit
            )
        if not math.isfinite(value):
            raise SelectionHistoryError(
                f"Selection history val_psnr is nonfinite at CSV row {row_number}", audit
            )
        values.append(value)
    audit.update(
        val_psnr_trusted=True,
        val_psnr_column_index=value_index,
        best_epoch=epochs[max(range(len(values)), key=values.__getitem__)],
        best_val_psnr=max(values),
    )
    return audit


def split_audit(table: pd.DataFrame) -> dict:
    _require({"sample_id", "group_id", "split", "image_path"}.issubset(table), "Invalid manifest schema")
    _require(not table.sample_id.duplicated().any(), "Duplicate sample_id")
    _require(table.split.isin(["train", "val", "test"]).all(), "Unrecognized split")
    for key in ("group_id", "patient_id"):
        if key in table:
            known = table[table[key].astype(str).str.strip().ne("")]
            _require(not (known.groupby(key).split.nunique() > 1).any(), f"Split {key} leakage")
    return {s: {"groups": sorted(t.group_id.unique().tolist()), "frames": len(t)}
            for s, t in table.groupby("split")}


def asset_inventory(root: Path, table: pd.DataFrame, include_labels: bool) -> list[dict]:
    # Enforce split selection BEFORE touching any asset, including hashes.
    allowed = table[table.split.isin(["train", "val"])]
    cols = ["image_path", "clean_path"] + (["layer_mask_path", "vessel_mask_path",
            "label_valid_mask_path", "vessel_valid_mask_path"] if include_labels else [])
    result = []
    for row in allowed.to_dict("records"):
        for column in cols:
            if row.get(column):
                path = resolve(root, row[column])
                _require(path.is_file(), f"Missing train/val asset: {path}")
                result.append({"sample_id": row["sample_id"], "split": row["split"],
                               "column": column, "sha256": sha256_file(path)})
    return sorted(result, key=lambda r: (r["sample_id"], r["column"]))


def effective_split_sha(table: pd.DataFrame) -> str:
    """Hash the effective train/validation group contract used by Trainer."""
    payload = "\n".join(
        f"{role}:{group_id}"
        for role in ("train", "val")
        for group_id in sorted(
            table.loc[table["split"].astype(str).eq(role), "group_id"]
            .astype(str)
            .unique()
            .tolist()
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_reproduction_config(value: dict, root: Path) -> dict:
    """Remove runtime-only state and canonicalise reusable path spellings."""
    normalised = json.loads(json.dumps(json_safe(value)))
    normalised.pop("runtime", None)
    data = normalised.get("data", {})
    for key in ("manifest", "root"):
        if data.get(key):
            data[key] = str(resolve(root, data[key]))
    train = normalised.get("train", {})
    if train.get("output_dir"):
        train["output_dir"] = str(resolve(root, train["output_dir"]))
    # A historical run may record the checkpoint used to continue that same
    # run.  A fresh reproduction must not resume it.  Compare this separately
    # as run control, not as model/data/optimisation semantics.
    train["resume"] = None
    return normalised


def _config_differences(reference: Any, candidate: Any, prefix: str = "") -> list[dict]:
    if isinstance(reference, dict) and isinstance(candidate, dict):
        differences = []
        for key in sorted(set(reference) | set(candidate)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in reference or key not in candidate:
                differences.append({
                    "path": path,
                    "reference": reference.get(key),
                    "candidate": candidate.get(key),
                })
            else:
                differences.extend(_config_differences(reference[key], candidate[key], path))
        return differences
    if reference != candidate:
        return [{"path": prefix, "reference": reference, "candidate": candidate}]
    return []


def audit_d1_reproduction_config(root: Path, cfg: dict) -> dict:
    """Fail closed unless only output and opt-in evidence differ from old D1."""
    evidence_cfg = cfg.get("training_asset_evidence", {})
    reference_value = evidence_cfg.get("reference_resolved_config")
    _require(bool(reference_value), "D1 reproduction requires reference_resolved_config")
    reference_path = resolve(root, reference_value)
    _require(reference_path.is_file(), f"Missing reference D1 resolved config: {reference_path}")
    raw_reference = load_config(reference_path)
    reference_resume = raw_reference.get("train", {}).get("resume")
    candidate_resume = cfg.get("train", {}).get("resume")
    operational_differences = []
    if reference_resume != candidate_resume:
        operational_differences.append({
            "path": "train.resume",
            "reference": reference_resume,
            "candidate": candidate_resume,
            "classification": "run_control_not_training_semantics",
            "reason": "fresh reproduction must not resume the historical run",
        })
    reference = _normalise_reproduction_config(raw_reference, root)
    candidate = _normalise_reproduction_config(cfg, root)
    differences = _config_differences(reference, candidate)
    allowed = set(evidence_cfg.get("allowed_semantic_differences", []))
    actual = {item["path"] for item in differences}
    _require(
        actual == allowed,
        f"D1 reproduction semantic differences are {sorted(actual)}, expected exactly {sorted(allowed)}",
    )
    return {
        "status": "passed",
        "reference_resolved_config": str(reference_path),
        "reference_resolved_config_sha256": sha256_file(reference_path),
        "allowed_semantic_differences": sorted(allowed),
        "semantic_differences": differences,
        "operational_differences": operational_differences,
        "protocol_lock_bindings": cfg.get("runtime", {}).get(
            "training_asset_protocol_bindings", []
        ),
        "test_assets_opened": 0,
    }


def create_training_asset_evidence(
    root: Path,
    cfg: dict,
    filtered: pd.DataFrame,
    output_path: Path,
) -> dict:
    """Record effective train/val noisy+clean pixels before optimisation."""
    evidence_cfg = cfg.get("training_asset_evidence", {})
    _require(evidence_cfg.get("enabled") is True, "Training asset evidence is not enabled")
    _require(cfg.get("train", {}).get("stage") == "denoise", "Asset evidence is denoise-only")
    _require(filtered["split"].astype(str).isin(["train", "val"]).all(),
             "Filtered training evidence contains a non-development split")
    _require(set(filtered["split"].astype(str)) == {"train", "val"},
             "Training evidence requires both train and validation rows")
    manifest = resolve(root, cfg["data"]["manifest"])
    records = asset_inventory(resolve(root, cfg["data"].get("root") or root), filtered, False)
    records_sha = stable_sha(records)
    payload = {
        "schema_version": "denoiser-training-assets-v1",
        "recorded_at_training": True,
        "recorded_before_optimizer_step": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(root),
        "protocol_id": cfg.get("protocol_id"),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "effective_split_sha256": effective_split_sha(filtered),
        "records": records,
        "records_sha256": records_sha,
        "train_val_noisy_clean_asset_sha256": records_sha,
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output_path, payload)
    return payload


def bind_training_asset_evidence(
    initial_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
    current_records: list[dict],
    completed_epochs: int,
    configured_epochs: int,
    selection_rule: str = "fixed_final",
) -> dict:
    """Bind a completed checkpoint to the immutable pre-optimisation record."""
    _require(initial_path.is_file(), "Missing training-start asset evidence; retroactive binding forbidden")
    initial = json.loads(initial_path.read_text(encoding="utf-8-sig"))
    _require(initial.get("recorded_at_training") is True
             and initial.get("recorded_before_optimizer_step") is True,
             "Initial evidence was not recorded before optimisation")
    _require(selection_rule == "fixed_final", "This binding supports fixed_final only")
    _require(checkpoint_path.name == "last.pth" and checkpoint_path.is_file(),
             "fixed_final evidence must bind an existing last.pth")
    _require(int(completed_epochs) == int(configured_epochs),
             "Cannot bind an incomplete fixed-final training run")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(int(raw.get("epoch", -1)) + 1 == int(configured_epochs),
             "last.pth epoch differs from configured fixed budget")
    manifest_sha = sha256_file(manifest_path)
    _require(initial.get("manifest_sha256") == manifest_sha,
             "Manifest changed after training-start evidence")
    records_sha = stable_sha(current_records)
    _require(initial.get("records") == current_records
             and initial.get("records_sha256") == records_sha
             and initial.get("train_val_noisy_clean_asset_sha256") == records_sha,
             "Train/validation noisy-clean assets changed after training-start evidence")
    source_sha = sha256_file(initial_path)
    payload = {
        **initial,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest_sha256": manifest_sha,
        "records": current_records,
        "records_sha256": records_sha,
        "train_val_noisy_clean_asset_sha256": records_sha,
        "source_initial_evidence_file": initial_path.name,
        "source_initial_evidence_path": str(initial_path.resolve()),
        "source_initial_evidence_sha256": source_sha,
        "completed_epochs": int(completed_epochs),
        "selection_rule": selection_rule,
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output_path, payload)
    return payload


def audit_bound_training_asset_evidence(
    evidence_path: Path,
    checkpoint_sha256: str,
    manifest_sha256: str,
    current_records: list[dict],
    selection_rule: str,
    completed_epochs: int,
) -> dict:
    """Verify the bound file and its immutable training-start evidence chain."""
    evidence = json.loads(evidence_path.read_text(encoding="utf-8-sig"))
    _require(evidence.get("recorded_at_training") is True,
             "Training asset inventory is not marked recorded_at_training")
    _require(evidence.get("checkpoint_sha256") == checkpoint_sha256,
             "Training asset inventory checkpoint SHA mismatch")
    _require(evidence.get("manifest_sha256") == manifest_sha256,
             "Training asset inventory manifest SHA mismatch")
    _require(evidence.get("selection_rule") == selection_rule
             and int(evidence.get("completed_epochs", -1)) == int(completed_epochs),
             "Training asset inventory selection/budget mismatch")
    source_name = evidence.get("source_initial_evidence_file")
    _require(bool(source_name), "Training asset inventory lacks initial evidence chain")
    source = evidence_path.parent / str(source_name)
    _require(source.is_file(), "Training asset inventory initial evidence is missing")
    _require(sha256_file(source) == evidence.get("source_initial_evidence_sha256"),
             "Training-start evidence SHA mismatch")
    initial = json.loads(source.read_text(encoding="utf-8-sig"))
    _require(initial.get("recorded_at_training") is True
             and initial.get("recorded_before_optimizer_step") is True,
             "Initial evidence lacks pre-optimisation provenance")
    records_sha = stable_sha(current_records)
    for document, label in ((initial, "initial"), (evidence, "bound")):
        _require(document.get("records") == current_records
                 and document.get("records_sha256") == records_sha
                 and document.get("train_val_noisy_clean_asset_sha256") == records_sha,
                 f"{label} training asset records differ from current pixels")
        _require(document.get("manifest_sha256") == manifest_sha256,
                 f"{label} training asset manifest SHA mismatch")
        _require(document.get("test_assets_opened") == 0,
                 f"{label} training evidence does not prove sealed-test exclusion")
    return evidence


def formal_preflight(root: Path, checkpoint_path: str | None, lock_path: str | None,
                     split_contract: str | None, selection_rule: str | None,
                     training_asset_inventory: str | None = None,
                     checkpoint_binding: str | None = None) -> dict:
    """Validate protocol metadata first, then train/val pixels. Trusted local pth only."""
    report = {"mode": "formal", "status": "blocked", "issues": [],
              "test_assets_opened": 0, "test_evaluation_performed": False,
              "code_commit": git_commit(root), "selection_rule": selection_rule,
              "historical_protocol_note": "13/3 and 8/2/3 are NOT interchangeable; no fallback"}
    try:
        _require(bool(checkpoint_path) and checkpoint_path != "auto", "An explicit D1 checkpoint is required; auto forbidden")
        cp = resolve(root, checkpoint_path)
        _require(cp.is_file(), f"Missing checkpoint: {cp}")
        report.update(checkpoint_path=str(cp), checkpoint_sha256=sha256_file(cp))
        _require(not re.search(r"smoke|pilot", str(cp), flags=re.I), "Smoke/pilot checkpoint is not formal D1")
        _require(bool(lock_path), "Missing explicit active protocol lock")
        lp = resolve(root, lock_path)
        locks = sorted((root / "Manifests").rglob("active_protocol_lock.json"))
        locks = [p.resolve() for p in locks if not any("archive" in part.lower() for part in p.parts)]
        _require(len(locks) == 1 and locks[0] == lp, f"Active protocol missing/nonunique or explicit lock not active: {locks}")
        lock = load_protocol_lock(lp)
        report.update(protocol_lock=str(lp), protocol_lock_sha256=sha256_file(lp), protocol=lock)
        _require(selection_rule in {"best_validation_psnr", "fixed_final"}, "Freeze selection rule explicitly")
        _require(bool(split_contract), "Explicit --split-contract required (no guessed contract)")
        contract_path = resolve(root, split_contract)
        _require(contract_path.is_file(), "Missing split contract")
        _require(sha256_file(contract_path) == lock["split_contract_sha256"], "Split contract SHA mismatch")
        contract = yaml.safe_load(contract_path.read_text(encoding="utf-8-sig"))
        _require(contract.get("protocol_id") == lock["protocol_id"], "Contract protocol mismatch")
        def positions(values):
            return {"pku_%04d" % int("".join(c for c in str(v) if c.isdigit())) for v in values}
        _require(positions(contract["validation_positions"]) == set(lock["validation_positions"])
                 and positions(contract["test_positions"]) == set(lock["sealed_test_positions"]), "Contract/lock held-out groups mismatch")
        raw = torch.load(cp, map_location="cpu", weights_only=False)
        _require(isinstance(raw, dict) and raw.get("model") and raw.get("config"), "Checkpoint source/config/model untraceable")
        cfg = raw["config"]
        rp = cp.parent / "resolved_config.yaml"
        _require(rp.is_file(), "Missing checkpoint resolved_config.yaml")
        resolved = load_config(rp)
        for section in ("model", "loss", "data", "train", "seed"):
            _require(cfg.get(section) == resolved.get(section), f"Embedded/resolved config {section} mismatch")
        _require(cfg["train"].get("stage") == "denoise" and cfg["loss"].get("restoration_mode") == "structure_d1",
                 "Checkpoint is not traced structure_d1 denoising")
        _require(int(cfg["train"].get("epochs", 0)) >= 60 and not cfg.get("runtime", {}).get("smoke_test"),
                 "Smoke/short denoiser budget is not formal")
        _require(cfg["data"].get("normalization") == "fixed", "Only audited fixed normalization supported")
        _require(list(cfg["data"]["target_size"]) == list(lock["input_resolution"]), "D1/protocol geometry mismatch")
        for key in CONSISTENCY_KEYS:
            found = cfg.get("runtime", {}).get("active_protocol_lock", {}).get(key, cfg.get(key))
            _require(found == lock[key], f"Checkpoint {key} mismatch")
        provenance_commit = cfg.get("runtime", {}).get("git_commit") or lock.get("git_commit_at_d1_start")
        _require(bool(provenance_commit), "Checkpoint code source untraceable")
        root_data = resolve(root, cfg["data"].get("root") or root)
        manifest = resolve(root, cfg["data"]["manifest"])
        _require(manifest.is_file(), "Missing checkpoint manifest")
        runtime = cfg.get("runtime", {})
        _require(runtime.get("manifest_sha256") == sha256_file(manifest), "Checkpoint manifest SHA missing/mismatch")
        table = pd.read_csv(manifest, dtype=str).fillna("")
        report["checkpoint_splits"] = split_audit(table)
        filtered = table.copy()
        for split in ("train", "val"):
            part = filtered[filtered.split.eq(split)]
            for field, column in ((f"{split}_datasets", "dataset"), (f"{split}_groups", "group_id")):
                values = cfg["data"].get(field)
                if values:
                    part = part[part[column].isin([str(v) for v in values])]
            filtered = pd.concat([filtered[~filtered.split.eq(split)], part], ignore_index=True)
        trains = sorted(filtered[filtered.split.eq("train")].group_id.unique())
        vals = sorted(filtered[filtered.split.eq("val")].group_id.unique())
        held = set(lock["validation_positions"]) | set(lock["sealed_test_positions"])
        _require(bool(trains) and not set(trains) & held, "D1 train overlaps validation/sealed test or empty")
        _require(set(trains) <= set(lock["train_positions"]) and set(vals) == set(lock["validation_positions"]),
                 "D1 effective cohort differs from active protocol")
        effective = hashlib.sha256("\n".join(f"{s}:{g}" for s, gs in (("train", trains), ("val", vals)) for g in gs).encode()).hexdigest()
        _require(runtime.get("effective_split_sha256") == effective, "D1 effective split SHA missing/mismatch")
        protocol_root = resolve(root, lock["manifest_root"])
        audit = json.loads((protocol_root / "protocol_audit.json").read_text(encoding="utf-8-sig"))
        _require(audit.get("status") == "passed", "Current protocol audit is not passed")
        for key in CONSISTENCY_KEYS:
            _require(audit.get(key) == lock[key], f"Protocol audit {key} mismatch")
        for name, key in (("label_inventory.csv", "label_inventory_sha256"),
                          ("dataset_inventory.csv", "dataset_inventory_sha256")):
            inv = pd.read_csv(protocol_root / name, dtype=str).fillna("")
            _require(hashlib.sha256(inv.to_csv(index=False).encode()).hexdigest() == lock[key], f"{name} SHA mismatch")
        joint = pd.read_csv(protocol_root / "train_joint.csv", dtype=str).fillna("")
        sealed = pd.read_csv(protocol_root / "test_sealed.csv", dtype=str).fillna("")  # metadata ONLY
        allrows = pd.concat([joint, sealed], ignore_index=True).sort_values("image_path", kind="stable").reset_index(drop=True)
        report["protocol_splits"] = split_audit(allrows)
        _require(hashlib.sha256(allrows.to_csv(index=False).encode()).hexdigest() == lock["data_plan_sha256"],
                 "Data-plan SHA mismatch/unreconstructable; no fallback protocol")
        for split, key in (("train", "train_positions"), ("val", "validation_positions"), ("test", "sealed_test_positions")):
            _require(set(allrows[allrows.split.eq(split)].group_id) == set(lock[key]), f"Protocol {split} groups mismatch")
        # Reject any shared train/val -> test asset path before opening assets.
        cols = [c for c in allrows if c.endswith("_path")]
        forbidden = {str(resolve(root, v)) for c in cols for v in sealed[c] if v}
        development = allrows[allrows.split.isin(["train", "val"])]
        _require(not any(str(resolve(root, v)) in forbidden for c in cols for v in development[c] if v),
                 "Development asset aliases sealed test asset")
        by_id = allrows.set_index("sample_id")
        for row in filtered[filtered.split.isin(["train", "val"])].to_dict("records"):
            _require(row["sample_id"] in by_id.index, "D1 sample not in locked data plan")
            reference = by_id.loc[row["sample_id"]]
            _require(all(str(row[c]) == str(reference[c]) for c in ("group_id", "split", "dataset"))
                     and all(resolve(root_data, row[c]) == resolve(root, reference[c]) for c in ("image_path", "clean_path")),
                     "D1 manifest row differs from locked data plan (possible held-out asset leakage)")
        for column in ("image_path", "clean_path", "layer_mask_path", "vessel_mask_path"):
            paths = allrows[allrows[column].ne("")][["split", column]].copy()
            paths["resolved_path"] = paths[column].map(lambda p: str(resolve(root, p)))
            _require(not (paths.groupby("resolved_path").split.nunique() > 1).any(), "Asset path crosses train/val/test splits")
        seg_path = protocol_root / "train_segment.csv"
        _require(seg_path.is_file(), "Missing segmentation manifest")
        seg = pd.read_csv(seg_path, dtype=str).fillna("")
        split_audit(seg)
        _require(seg.split.isin(["train", "val"]).all(), "Segmentation manifest contains non-development rows")
        _require(set(seg[seg.split.eq("val")].group_id) == set(lock["validation_positions"]), "Incomplete validation cohort")
        for row in seg.to_dict("records"):
            _require(row["sample_id"] in by_id.index, "Segmentation row not in locked data plan")
            _require(all(str(by_id.loc[row["sample_id"], c]) == str(v) for c, v in row.items() if c in by_id),
                     "Segmentation row differs from locked data plan")
            _require(all(row.get(c) for c in ("clean_path", "layer_mask_path", "vessel_mask_path")),
                     "Common paired/labelled cohort incomplete; do not silently discard frames")
        checkpoint_epoch = raw.get("epoch")
        try:
            checkpoint_epoch_value = Decimal(str(checkpoint_epoch))
        except (InvalidOperation, ValueError):
            checkpoint_epoch_value = Decimal("NaN")
        _require(checkpoint_epoch_value.is_finite()
                 and checkpoint_epoch_value == checkpoint_epoch_value.to_integral_value()
                 and checkpoint_epoch_value >= 0, "Checkpoint epoch is not a valid integer index")
        report.update(segmentation_manifest=str(seg_path), segmentation_manifest_sha256=sha256_file(seg_path),
                      d1_resolved_config=str(rp), resolved_config_sha256=sha256_file(rp),
                      restoration_mode="structure_d1", normalization="fixed", input_resolution=cfg["data"]["target_size"],
                      d1_epoch=int(checkpoint_epoch_value) + 1, provenance_commit=provenance_commit)
        history_path = cp.parent / "history.csv"
        try:
            history_audit = audit_selection_history(
                history_path, int(cfg["train"]["epochs"]), selection_rule
            )
        except SelectionHistoryError as exc:
            history_audit = exc.audit
            report.update(
                selection_history_audit=history_audit,
                history_schema_drift_detected=history_audit["history_schema_drift_detected"],
                history_row_width_distribution=history_audit["history_row_width_distribution"],
                val_psnr_trusted=history_audit["val_psnr_trusted"],
            )
            raise ValueError(str(exc))
        report.update(
            selection_history_audit=history_audit,
            history_schema_drift_detected=history_audit["history_schema_drift_detected"],
            history_row_width_distribution=history_audit["history_row_width_distribution"],
            val_psnr_trusted=history_audit["val_psnr_trusted"],
        )
        if selection_rule == "best_validation_psnr":
            _require(cp.name == "best.pth" and cfg["train"].get("monitor") == "psnr", "Not predeclared best PSNR checkpoint")
            _require(bool(training_asset_inventory) and bool(checkpoint_binding),
                     "Best checkpoint requires training-start inventory and derived binding")
            best_epoch = int(history_audit["best_epoch"])
            _require(report["d1_epoch"] == best_epoch, "Checkpoint epoch is not validation PSNR best (first tie)")
            metadata = json.loads((cp.parent / "run_metadata.json").read_text(encoding="utf-8-sig"))
            _require(metadata.get("best_checkpoint_sha256") == report["checkpoint_sha256"], "Best checkpoint provenance SHA missing/mismatch")
        else:
            _require(cp.name == "last.pth" and report["d1_epoch"] == int(cfg["train"]["epochs"])
                     and int(history_audit["last_epoch"]) == report["d1_epoch"], "Incomplete fixed-final checkpoint")
        # The protocol inventory hashes are table hashes, not historical noisy
        # pixel hashes. Do not retroactively invent the latter from today's data.
        pixel_records = asset_inventory(root_data, filtered, include_labels=False)
        pixel_sha = stable_sha(pixel_records)
        expected_pixel_sha = runtime.get("train_val_noisy_clean_asset_sha256")
        if training_asset_inventory:
            inventory_path = resolve(root, training_asset_inventory)
            if selection_rule == "best_validation_psnr":
                _require(bool(checkpoint_binding), "Best checkpoint requires explicit derived binding")
                initial = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
                _require(initial.get("recorded_at_training") is True
                         and initial.get("recorded_before_optimizer_step") is True,
                         "Best checkpoint requires original training-start inventory")
                _require(initial.get("manifest_sha256") == sha256_file(manifest)
                         and initial.get("records") == pixel_records
                         and initial.get("records_sha256") == pixel_sha
                         and initial.get("train_val_noisy_clean_asset_sha256") == pixel_sha,
                         "Best checkpoint initial inventory differs from current train/val pixels")
                from sabids.experiments.d2 import audit_best_checkpoint_binding
                binding = audit_best_checkpoint_binding(
                    resolve(root, checkpoint_binding), cp, inventory_path, history_path,
                    rp, cp.parent / "run_metadata.json", lp, contract_path,
                )
                _require(binding.get("checkpoint_epoch") == report["d1_epoch"]
                         and binding.get("protocol_id") == lock["protocol_id"]
                         and binding.get("effective_split_sha256") == effective,
                         "Best checkpoint binding identity mismatch")
                expected_pixel_sha = initial.get("train_val_noisy_clean_asset_sha256")
                report["checkpoint_binding"] = str(resolve(root, checkpoint_binding))
                report["checkpoint_binding_sha256"] = sha256_file(resolve(root, checkpoint_binding))
            else:
                evidence = audit_bound_training_asset_evidence(
                    inventory_path,
                    checkpoint_sha256=report["checkpoint_sha256"],
                    manifest_sha256=sha256_file(manifest),
                    current_records=pixel_records,
                    selection_rule=str(selection_rule),
                    completed_epochs=report["d1_epoch"],
                )
                _require(resolve(root, evidence.get("checkpoint_path", "")) == cp,
                         "Training asset inventory checkpoint path mismatch")
                expected_pixel_sha = evidence.get("train_val_noisy_clean_asset_sha256")
        _require(expected_pixel_sha == pixel_sha, "Historical D1 noisy/clean pixel fingerprint missing/mismatch; require immutable training evidence")
        labels = pd.read_csv(protocol_root / "label_inventory.csv", dtype=str).fillna("")
        for row in labels[labels.group_id.isin(set(development.group_id))].to_dict("records"):
            for task in ("layer", "vessel"):
                if row.get(f"{task}_path"):
                    _require(sha256_file(resolve(root, row[f"{task}_path"])) == row[f"{task}_sha256"], "Current label pixel SHA mismatch")
                    referenced = development[development.group_id.eq(row["group_id"])][f"{task}_mask_path"]
                    _require(all(str(resolve(root, p)) == str(resolve(root, row[f"{task}_path"])) for p in referenced if p),
                             "Manifest labels do not match locked label inventory paths")
        report.update(train_val_noisy_clean_asset_sha256=pixel_sha,
                      segmentation_asset_inventory=asset_inventory(root, seg, include_labels=True),
                      status="passed")
    except Exception as exc:
        report["issues"].append(str(exc))
    return report


def validate_dose_config(cfg: dict, require_fresh: bool = True) -> None:
    if not cfg.get("dose_response", {}).get("enabled", False):
        return
    d, m, t, loss = cfg["data"], cfg["model"], cfg["train"], cfg["loss"]
    _require(t["stage"] == "input_segment" and not t.get("pretrained") and not t.get("resume"), "Dose requires independent from-scratch input_segment")
    _require(cfg.get("deterministic") is True and t.get("num_workers") == 0, "Minimal strict pairing requires deterministic=True,num_workers=0")
    _require(d.get("pretransformed_model_grid") is True and d.get("deterministic_augmentation") is True
             and d.get("input_column") == "dose_path" and d.get("normalization") == "fixed", "Dose data contract missing")
    _require(d.get("augmentation", {}).get("strong_private_only") is True, "Only deterministic paired horizontal flip is supported")
    _require(not any(m.get(k, True) for k in ("d2s_enabled", "s2d_enabled", "enable_denoise_to_seg", "enable_seg_to_denoise")), "Dose interactions must be explicitly off")
    _require(not m.get("causal_interaction_experiment", False) and float(m.get("dropout", 0)) == 0, "No extra stochastic/causal paths")
    _require(float(loss.get("auxiliary_weight", -1)) == 0, "Auxiliary supervision must be explicitly zero")
    _require(loss.get("zero_source") == "final_segmentation", "Inactive losses must not attach to restoration graph")
    for key in ("reconstruction", "residual", "rmac", "pseudo", "identity", "vessel_stroma", "vessel_area"):
        _require(float(loss.get("weights", {}).get(key, -1)) == 0, f"Unexpected dose loss {key}")
    _require(loss.get("vessel_supervision_mode") == "roi_bce_dice_outside"
             and loss.get("exclude_annotation_invalid_containment") is True
             and loss.get("exclude_annotation_invalid_boundary") is True, "Dose requires uniform annotation-valid E3b supervision")
    _require(loss["weights"] == {"reconstruction": 0., "residual": 0., "identity": 0., "rmac": 0., "pseudo": 0.,
                                "vessel_stroma": 0., "vessel_area": 0., "layer": 1., "vessel": 1.,
                                "vessel_outside": .5, "containment": .1}, "Dose weights differ from common E3b contract")
    _require(int(t["early_stopping_patience"]) > int(t["epochs"]) and not t.get("use_ema"), "Fixed budget/final selection required")
    _require(all(float(cfg["evaluation"].get(k, -1)) == .5 for k in ("threshold", "layer_threshold", "vessel_threshold")), "P0 thresholds must be 0.5")
    root = Path(cfg["dose_response"]["project_root"]).resolve()
    output = resolve(root, t["output_dir"])
    expected = root / "runs" / "adaptive_denoising" / cfg["dose_response"]["protocol_id"] / "dose_v1"
    _require(expected.resolve() in output.parents, "Dose run must be a child of new adaptive_denoising/dose_v1 directory")
    if require_fresh:
        _require(not output.exists() or not any(output.iterdir()), f"Refusing existing dose run: {output}")
    identity_path = resolve(root, cfg["dose_response"]["preparation_registry"])
    registry = json.loads(identity_path.read_text(encoding="utf-8"))
    config_path = resolve(root, cfg["dose_response"]["generated_config_path"])
    _require(sha256_file(config_path) == registry["config_sha256"][str(config_path)], "Generated dose config fingerprint mismatch")
    expected_cfg = load_config(config_path)
    _require({k: v for k, v in cfg.items() if k != "runtime"} ==
             {k: v for k, v in expected_cfg.items() if k != "runtime"}, "Dose config differs from registered generated config")
    _require(registry["code_version"] == code_fingerprint(root), "Prepared dose inputs use different code; do not silently reuse")
    manifest = resolve(root, d["manifest"])
    _require(sha256_file(manifest) == cfg["dose_response"]["manifest_sha256"], "Prepared dose manifest SHA mismatch")
    table = pd.read_csv(manifest, dtype=str).fillna("")
    split_audit(table)
    _require(table.split.isin(["train", "val"]).all(), "Dose training never reads test")
    for row in table.to_dict("records"):
        p = resolve(root, row["dose_path"])
        meta = json.loads(p.with_suffix(".json").read_text(encoding="utf-8"))
        _require(meta["cache_key"] == cache_key(meta), "Dose cache metadata identity mismatch")
        arr = np.load(p, allow_pickle=False)
        _require(hashlib.sha256(arr.tobytes()).hexdigest() == meta["array_sha256"], "Dose cache content mismatch")
        _require(meta["alpha"] == cfg["dose_response"]["alpha"] and meta["curve_type"] == cfg["dose_response"]["curve_type"], "Dose arm metadata mismatch")
        _require(meta["protocol_id"] == cfg["dose_response"]["protocol_id"] and meta["sample_id"] == row["sample_id"]
                 and meta["group_id"] == row["group_id"] and meta["split"] == row["split"]
                 and meta["checkpoint_sha256"] == registry["identity"]["checkpoint_sha256"]
                 and meta["resolved_config_sha256"] == registry["identity"]["resolved_config_sha256"]
                 and meta["code_version"] == registry["code_version"]
                 and meta["geometry"] == registry["samples"][row["sample_id"]]["geometry"], "Dose cache lineage mismatch")
        for role, column in (("clean", "clean_path"), ("layer", "layer_mask_path"),
                             ("vessel", "vessel_mask_path"), ("label_valid", "label_valid_mask_path"),
                             ("vessel_valid", "vessel_valid_mask_path"), ("spatial_valid", "spatial_valid_mask_path")):
            asset = resolve(root, row[column])
            common_meta = json.loads(asset.with_suffix(".json").read_text(encoding="utf-8"))
            common_arr = np.load(asset, allow_pickle=False)
            _require(common_meta["cache_content_role"] == role and common_meta["cache_key"] == cache_key(common_meta)
                     and hashlib.sha256(common_arr.tobytes()).hexdigest() == common_meta["array_sha256"], "Prepared GT/validity cache drift")
    if cfg["dose_response"]["mode"] == "formal":
        _require(registry["preflight"]["status"] == "passed", "Dose registry preflight blocked")
        pf = registry["preflight"]
        if pf.get("restoration_mode") == "structure_d2":
            from sabids.experiments.d2 import formal_d2_preflight
            live = formal_d2_preflight(
                root, resolve(root, pf["checkpoint_path"]),
                resolve(root, registry["checkpoint_binding"]), pf["checkpoint_kind"],
                resolve(root, pf["protocol_lock"]), resolve(root, registry["split_contract"]),
            )
        else:
            live = formal_preflight(root, pf["checkpoint_path"], pf["protocol_lock"],
                                    registry["split_contract"], pf["selection_rule"],
                                    registry.get("training_asset_inventory"),
                                    registry.get("checkpoint_binding"))
        _require(live["status"] == "passed", f"Live formal gate blocked: {live['issues']}")
        for key in ("checkpoint_sha256", "protocol_lock_sha256", "segmentation_manifest_sha256", "segmentation_asset_inventory"):
            _require(live[key] == pf[key], f"Live fingerprint drift: {key}")
