from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.data.io import read_gray, read_mask
from sabids.engine.trainer import _make_transform, build_model
from sabids.metrics import (
    edge_preservation_index, psnr, reference_edge_mae, region_cnr, rmse, ssim,
)
from sabids.utils import get_device, load_checkpoint, save_checkpoint, write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array, dtype=np.float32), allow_pickle=False)
    os.replace(temporary, path)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def audit_d0(root: Path, checkpoint: Path, output: Path) -> Dict:
    segmentation_manifest = root / "Manifests/segmentation_folds/manifest_seg_fold0.csv"
    stage1_manifest = root / "Manifests/joint_folds/manifest_joint_fold0.csv"
    missing = [str(path) for path in (checkpoint, segmentation_manifest, stage1_manifest) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing D0 audit input: " + ", ".join(missing))
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("D0 checkpoint has no resolved config; leakage cannot be proved")
    runtime = config.get("runtime", {})
    effective_groups = runtime.get("effective_groups", {})
    d0_train_groups = {str(value) for value in effective_groups.get("train", [])}
    required = {
        "manifest_sha256": runtime.get("manifest_sha256"),
        "effective_split_sha256": runtime.get("effective_split_sha256"),
        "d0_train_groups": sorted(d0_train_groups),
    }
    unknown = [key for key, value in required.items() if value in (None, "", [])]
    stage1_manifest_sha = sha256_file(stage1_manifest)
    manifest_match = runtime.get("manifest_sha256") == stage1_manifest_sha
    segmentation = pd.read_csv(segmentation_manifest, dtype=str).fillna("")
    held_out = segmentation[segmentation["split"].isin(["val", "test"])]
    held_out_groups = set(held_out["group_id"].astype(str))
    overlap = sorted(d0_train_groups & held_out_groups)

    stage1 = pd.read_csv(stage1_manifest, dtype=str).fillna("")
    asset_rows = stage1[stage1["split"].isin(["train", "val"])].copy()
    asset_hash = hashlib.sha256()
    asset_count = 0
    missing_assets = []
    for _, row in asset_rows.iterrows():
        for column in ("image_path", "clean_path"):
            value = str(row.get(column, "")).strip()
            if not value:
                continue
            path = resolve_path(root, value)
            if not path.is_file():
                missing_assets.append(str(path))
                continue
            asset_hash.update(f"{row['sample_id']}|{column}|{sha256_file(path)}\n".encode("utf-8"))
            asset_count += 1
    passed = not unknown and manifest_match and not overlap and not missing_assets
    report = {
        "status": "passed" if passed else "blocked",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)) + 1,
        "checkpoint_declared_manifest": config.get("data", {}).get("manifest"),
        "checkpoint_declared_normalization": config.get("data", {}).get("normalization"),
        "checkpoint_declared_target_size": config.get("data", {}).get("target_size"),
        "checkpoint_manifest_sha256": runtime.get("manifest_sha256"),
        "current_stage1_manifest": str(stage1_manifest),
        "current_stage1_manifest_sha256": stage1_manifest_sha,
        "manifest_fingerprint_match": manifest_match,
        "effective_split_sha256": runtime.get("effective_split_sha256"),
        "d0_train_groups": sorted(d0_train_groups),
        "segmentation_validation_groups": sorted(set(segmentation.loc[segmentation["split"] == "val", "group_id"])),
        "segmentation_test_groups_metadata_only": sorted(set(segmentation.loc[segmentation["split"] == "test", "group_id"])),
        "held_out_overlap_with_d0_train": overlap,
        "unknown_identity_fields": unknown,
        "current_train_val_noisy_clean_asset_sha256": asset_hash.hexdigest(),
        "current_train_val_asset_count": asset_count,
        "missing_current_train_val_assets": missing_assets,
        "test_assets_opened": 0,
        "test_evaluation_performed": False,
        "required_remediation_if_blocked": (
            "python train.py --config configs/current/stage1_denoise_fold0.yaml"
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, output)
    return report


def build_cache(root: Path, checkpoint: Path, output_root: Path, device_name: str) -> None:
    audit_path = output_root / "d0_leakage_audit.json"
    audit = audit_d0(root, checkpoint, audit_path)
    if audit["status"] != "passed":
        raise RuntimeError(f"D0 leakage audit is blocked; inspect {audit_path}")
    payload = torch.load(checkpoint, map_location="cpu")
    d0_config = payload["config"]
    # Resolve the local data root while retaining the checkpoint's exact model
    # and Stage-1 preprocessing definition.
    d0_config["data"]["root"] = str(root)
    device = get_device(device_name)
    model = build_model(d0_config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    if any(module.training for module in model.modules()):
        raise RuntimeError("D0 must be fully in eval mode")

    source_manifest = root / "Manifests/segmentation_folds/manifest_seg_fold0.csv"
    source_table = pd.read_csv(source_manifest, dtype=str).fillna("")
    transform_config = json.loads(json.dumps(d0_config))
    transform_config["data"]["manifest"] = str(source_manifest)
    transform_config["data"]["root"] = str(root)
    transform_config["data"]["normalization"] = d0_config["data"].get("normalization", "fixed")
    cache_rows, quality_rows = [], []
    derived = source_table.copy()
    derived["noisy_cache_path"] = ""
    derived["denoised_cache_path"] = ""
    checkpoint_sha = sha256_file(checkpoint)
    for split in ("train", "val"):
        dataset = OCTManifestDataset(
            source_manifest, split=split, transform=_make_transform(transform_config, False),
            sample_repeat=False, root=root,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            image = batch["image"].to(device)
            with torch.no_grad():
                first = model(image, return_features=False, return_auxiliary=False)["denoised"][0, 0].cpu().numpy()
                second = model(image, return_features=False, return_auxiliary=False)["denoised"][0, 0].cpu().numpy()
            deterministic_diff = float(np.max(np.abs(first - second)))
            if deterministic_diff > 1e-7:
                raise RuntimeError(f"D0 is non-deterministic for {batch['sample_id'][0]}: {deterministic_diff}")
            valid = batch["valid_mask"][0, 0].numpy() > 0.5
            coordinates = np.argwhere(valid)
            y0, x0 = coordinates.min(axis=0)
            y1, x1 = coordinates.max(axis=0) + 1
            cropped = first[y0:y1, x0:x1]
            original_size = (int(batch["original_width"][0]), int(batch["original_height"][0]))
            denoised_original = cv2.resize(cropped.astype(np.float32), original_size, interpolation=cv2.INTER_LINEAR)
            noisy_original = read_gray(batch["original_path"][0])
            sample_id, group_id = batch["sample_id"][0], batch["group_id"][0]
            noisy_path = output_root / "noisy_float32" / split / f"{sample_id}.npy"
            denoised_path = output_root / "denoised_float32" / split / f"{sample_id}.npy"
            for path, array in ((noisy_path, noisy_original), (denoised_path, denoised_original)):
                if path.exists():
                    existing = np.load(path, allow_pickle=False)
                    if existing.shape != array.shape or not np.array_equal(existing.astype(np.float32), array.astype(np.float32)):
                        raise FileExistsError(f"Existing cache differs; refusing overwrite: {path}")
                else:
                    atomic_save_npy(path, array)
            identity = np.load(noisy_path, allow_pickle=False)
            identity_diff = float(np.max(np.abs(identity.astype(np.float32) - noisy_original.astype(np.float32))))
            if identity_diff != 0.0:
                raise RuntimeError(f"Identity cache round-trip changed {sample_id}: {identity_diff}")
            row_index = derived.index[(derived["split"] == split) & (derived["sample_id"] == sample_id)]
            if len(row_index) != 1:
                raise RuntimeError(f"sample_id must be unique within split: {sample_id}")
            derived.loc[row_index, "noisy_cache_path"] = str(noisy_path.relative_to(root)).replace("\\", "/")
            derived.loc[row_index, "denoised_cache_path"] = str(denoised_path.relative_to(root)).replace("\\", "/")
            cache_rows.append({
                "split": split, "sample_id": sample_id, "group_id": group_id,
                "input_path": batch["original_path"][0], "noisy_cache_path": str(noisy_path),
                "denoised_cache_path": str(denoised_path), "d0_checkpoint_sha256": checkpoint_sha,
                "shape": str(tuple(denoised_original.shape)), "dtype": "float32",
                "minimum": float(denoised_original.min()), "maximum": float(denoised_original.max()),
                "noisy_cache_sha256": sha256_file(noisy_path), "denoised_cache_sha256": sha256_file(denoised_path),
                "identity_roundtrip_max_abs_diff": identity_diff,
                "repeat_d0_max_abs_diff": deterministic_diff,
            })
            clean_path = str(batch["clean_path"][0])
            if clean_path:
                clean = read_gray(clean_path)
                quality = {
                    "split": split, "sample_id": sample_id, "group_id": group_id,
                    "psnr_noisy": psnr(noisy_original, clean), "psnr_d0": psnr(denoised_original, clean),
                    "ssim_noisy": ssim(noisy_original, clean), "ssim_d0": ssim(denoised_original, clean),
                    "rmse_noisy": rmse(noisy_original, clean), "rmse_d0": rmse(denoised_original, clean),
                    "epi_noisy": edge_preservation_index(noisy_original, clean),
                    "epi_d0": edge_preservation_index(denoised_original, clean),
                    "reference_edge_mae_noisy": reference_edge_mae(noisy_original, clean),
                    "reference_edge_mae_d0": reference_edge_mae(denoised_original, clean),
                }
                layer_path, vessel_path = str(batch["layer_mask_path"][0]), str(batch["vessel_mask_path"][0])
                if layer_path:
                    layer = read_mask(layer_path).astype(bool)
                    quality["layer_roi_psnr_noisy"] = psnr(noisy_original[layer], clean[layer])
                    quality["layer_roi_psnr_d0"] = psnr(denoised_original[layer], clean[layer])
                    quality["layer_roi_mse_noisy"] = float(np.mean((noisy_original[layer] - clean[layer]) ** 2))
                    quality["layer_roi_mse_d0"] = float(np.mean((denoised_original[layer] - clean[layer]) ** 2))
                    if vessel_path:
                        vessel = read_mask(vessel_path).astype(bool)
                        stroma = layer & ~vessel
                        for name, value in (("noisy", noisy_original), ("d0", denoised_original), ("clean", clean)):
                            quality[f"vessel_stroma_cnr_{name}"] = region_cnr(value, vessel, stroma)
                        quality["vessel_stroma_cnr_abs_error_noisy"] = abs(quality["vessel_stroma_cnr_noisy"] - quality["vessel_stroma_cnr_clean"])
                        quality["vessel_stroma_cnr_abs_error_d0"] = abs(quality["vessel_stroma_cnr_d0"] - quality["vessel_stroma_cnr_clean"])
                quality_rows.append(quality)
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cache_rows).to_csv(output_root / "denoised_cache_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality_rows).to_csv(output_root / "input_quality_metrics.csv", index=False, encoding="utf-8-sig")
    repeat_rows = []
    cache_table = pd.DataFrame(cache_rows)
    for (split, group_id), group in cache_table.groupby(["split", "group_id"], sort=True):
        noisy_mae, denoised_mae = [], []
        for (_, left), (_, right) in itertools.combinations(group.iterrows(), 2):
            left_noisy = np.load(left["noisy_cache_path"], allow_pickle=False)
            right_noisy = np.load(right["noisy_cache_path"], allow_pickle=False)
            left_d0 = np.load(left["denoised_cache_path"], allow_pickle=False)
            right_d0 = np.load(right["denoised_cache_path"], allow_pickle=False)
            if left_noisy.shape != right_noisy.shape or left_d0.shape != right_d0.shape:
                continue
            noisy_mae.append(float(np.mean(np.abs(left_noisy - right_noisy))))
            denoised_mae.append(float(np.mean(np.abs(left_d0 - right_d0))))
        repeat_rows.append({
            "split": split, "group_id": group_id, "n_frames": int(len(group)),
            "n_shape_compatible_pairs": int(len(noisy_mae)),
            "repeat_noisy_mae": float(np.mean(noisy_mae)) if noisy_mae else np.nan,
            "repeat_denoised_mae": float(np.mean(denoised_mae)) if denoised_mae else np.nan,
            "repeat_denoised_mae_improvement": (
                float(np.mean(noisy_mae) - np.mean(denoised_mae)) if noisy_mae else np.nan
            ),
        })
    pd.DataFrame(repeat_rows).to_csv(
        output_root / "input_repeat_stability_by_position.csv",
        index=False, encoding="utf-8-sig",
    )
    write_json({
        "cache_files": int(2 * len(cache_rows)),
        "cache_bytes": int(sum(Path(row[key]).stat().st_size for row in cache_rows for key in ("noisy_cache_path", "denoised_cache_path"))),
        "dtype": "float32", "test_rows_cached": 0, "test_assets_opened": 0,
        "stage1_normalization": d0_config["data"].get("normalization", "fixed"),
        "stage1_target_size": d0_config["data"].get("target_size"),
    }, output_root / "cache_summary.json")
    derived_path = root / "Manifests/input_factorial/manifest_input_fold0.csv"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    if derived_path.exists():
        existing = pd.read_csv(derived_path, dtype=str).fillna("")
        if not existing.equals(derived.astype(str)):
            raise FileExistsError(f"Derived manifest differs; refusing overwrite: {derived_path}")
    else:
        derived.to_csv(derived_path, index=False, encoding="utf-8-sig")


def build_initialization(root: Path, checkpoint: Path, output: Path) -> None:
    audit = audit_d0(root, checkpoint, output.parent / "d0_leakage_audit.json")
    if audit["status"] != "passed":
        raise RuntimeError("Cannot create common initialization before D0 leakage audit passes")
    if output.exists():
        raise FileExistsError(f"Initialization snapshot already exists: {output}")
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    config = load_config(root / "configs/current/input_noisy_fold0.yaml")
    config.setdefault("runtime", {})["stage1_checkpoint"] = str(checkpoint.resolve())
    config["runtime"]["stage1_checkpoint_sha256"] = sha256_file(checkpoint)
    save_checkpoint(output, model, None, None, -1, float("-inf"), config)
    state_hash = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        state_hash.update(name.encode("utf-8"))
        state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    write_json({
        "snapshot": str(output.resolve()), "snapshot_sha256": sha256_file(output),
        "stage1_checkpoint_sha256": sha256_file(checkpoint),
        "model_state_sha256": state_hash.hexdigest(),
        "segmentation_initialization_source": "frozen/untrained segmentation modules stored in the fold-specific Stage-1 checkpoint",
    }, output.parent / "initialization_audit.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the attributable I_NOISY/I_DENOISED experiment")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--stage1-checkpoint", default="runs/current/stage1_denoise_fold0/best.pth")
    parser.add_argument("--mode", choices=("audit", "cache", "initialize"), required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    checkpoint = resolve_path(root, args.stage1_checkpoint)
    common = root / "runs/current/input_factorial_common_fold0"
    if args.mode == "audit":
        report = audit_d0(root, checkpoint, common / "d0_leakage_audit.json")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.mode == "cache":
        build_cache(root, checkpoint, common / "cache", args.device)
    else:
        build_initialization(root, checkpoint, common / "preseg_initialization.pth")


if __name__ == "__main__":
    main()
