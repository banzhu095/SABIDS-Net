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
from contextlib import contextmanager
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
    if curve_type not in {"oracle", "d1"}:
        raise ValueError("curve_type must be oracle or d1")
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
    if metadata["curve_type"] not in {"oracle", "d1"}:
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


def formal_preflight(root: Path, checkpoint_path: str | None, lock_path: str | None,
                     split_contract: str | None, selection_rule: str | None,
                     training_asset_inventory: str | None = None) -> dict:
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
        report.update(segmentation_manifest=str(seg_path), segmentation_manifest_sha256=sha256_file(seg_path),
                      d1_resolved_config=str(rp), resolved_config_sha256=sha256_file(rp),
                      restoration_mode="structure_d1", normalization="fixed", input_resolution=cfg["data"]["target_size"],
                      d1_epoch=int(raw["epoch"]) + 1, provenance_commit=provenance_commit)
        history_path = cp.parent / "history.csv"
        _require(history_path.is_file(), "Missing selection history")
        history = pd.read_csv(history_path)
        _require({"epoch", "val_psnr"}.issubset(history), "Selection history lacks epoch/val_psnr")
        _require(np.isfinite(history.val_psnr.to_numpy(float)).all(), "Nonfinite selection history")
        _require(history.epoch.astype(int).tolist() == list(range(1, int(cfg["train"]["epochs"]) + 1)),
                 "Formal D1 history does not prove complete fixed budget; partial best is not formal")
        if selection_rule == "best_validation_psnr":
            _require(cp.name == "best.pth" and cfg["train"].get("monitor") == "psnr", "Not predeclared best PSNR checkpoint")
            best_epoch = int(history.loc[history.val_psnr.idxmax(), "epoch"])
            _require(report["d1_epoch"] == best_epoch, "Checkpoint epoch is not validation PSNR best (first tie)")
            metadata = json.loads((cp.parent / "run_metadata.json").read_text(encoding="utf-8-sig"))
            _require(metadata.get("best_checkpoint_sha256") == report["checkpoint_sha256"], "Best checkpoint provenance SHA missing/mismatch")
        else:
            _require(cp.name == "last.pth" and report["d1_epoch"] == int(cfg["train"]["epochs"])
                     and int(history.epoch.max()) == report["d1_epoch"], "Incomplete fixed-final checkpoint")
        # The protocol inventory hashes are table hashes, not historical noisy
        # pixel hashes. Do not retroactively invent the latter from today's data.
        pixel_records = asset_inventory(root_data, filtered, include_labels=False)
        pixel_sha = stable_sha(pixel_records)
        expected_pixel_sha = runtime.get("train_val_noisy_clean_asset_sha256")
        if training_asset_inventory:
            evidence = json.loads(resolve(root, training_asset_inventory).read_text(encoding="utf-8-sig"))
            _require(evidence.get("checkpoint_sha256") == report["checkpoint_sha256"]
                     and evidence.get("manifest_sha256") == sha256_file(manifest)
                     and evidence.get("recorded_at_training") is True, "Unbound/unhistorical pixel inventory")
            _require(evidence.get("records") == pixel_records, "Training pixel inventory differs from current pixels")
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
        live = formal_preflight(root, pf["checkpoint_path"], pf["protocol_lock"],
                                registry["split_contract"], pf["selection_rule"], registry.get("training_asset_inventory"))
        _require(live["status"] == "passed", f"Live formal gate blocked: {live['issues']}")
        for key in ("checkpoint_sha256", "protocol_lock_sha256", "segmentation_manifest_sha256", "segmentation_asset_inventory"):
            _require(live[key] == pf[key], f"Live fingerprint drift: {key}")
