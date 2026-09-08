"""Create D0/D1 float inputs for the paired downstream segmentation probe."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.data.io import read_gray
from sabids.data.transforms import JointOCTTransform
from sabids.engine.trainer import build_model
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file, validate_checkpoint_config


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _resolve_d1_checkpoint(root: Path, value: str, prefix: str, seed: int, lock: dict) -> Path:
    if value != "auto": return _resolve(root, value)
    matches = []
    for run in (root / "runs" / "current").glob(f"{prefix}*seed{seed}"):
        candidate = run / "last.pth"
        if not candidate.is_file(): continue
        try:
            raw = torch.load(candidate, map_location="cpu", weights_only=False)
            validate_checkpoint_config(raw, lock, str(candidate))
            if int(raw.get("config", {}).get("train", {}).get("epochs", 0)) < 60:
                continue
            matches.append(candidate)
        except RuntimeError:
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Expected one active-protocol {prefix} seed {seed} checkpoint, found {matches}")
    return matches[0]


def _load_denoiser(path: Path, lock: dict, device: torch.device):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_config(raw, lock, str(path))
    cfg = raw.get("config", {})
    model = build_model(cfg).to(device)
    model.load_state_dict(raw.get("model", raw), strict=True)
    model.eval()
    return model, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--expected-protocol-lock", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--d0-checkpoint", required=True)
    parser.add_argument("--d1-checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    lock_path = _resolve(root, args.expected_protocol_lock)
    lock = load_protocol_lock(lock_path)
    manifest_root = _resolve(root, str(lock["manifest_root"]))
    source = manifest_root / "train_segment.csv"
    table = pd.read_csv(source, dtype=str).fillna("")
    # Filter using metadata before resolving or opening any asset.
    table = table[table["split"].isin(["train", "val"])].copy()
    table = table[(table["clean_path"].str.strip() != "") & (table["layer_mask_path"].str.strip() != "") & (table["vessel_mask_path"].str.strip() != "")].copy()
    if table.empty:
        raise RuntimeError("No aligned clean+layer+vessel train/validation rows are available")
    missing_assets = [
        str(_resolve(root, getattr(row, column)))
        for row in table.itertuples(index=False)
        for column in ("image_path", "clean_path", "layer_mask_path", "vessel_mask_path")
        if not _resolve(root, getattr(row, column)).is_file()
    ]
    if missing_assets:
        raise FileNotFoundError(f"Input probe cohort has missing assets: {missing_assets[:10]}")
    device = torch.device(args.device)
    d0_path = _resolve_d1_checkpoint(root, args.d0_checkpoint, "d1_denoise_d0", args.seed, lock)
    d1_path = _resolve_d1_checkpoint(root, args.d1_checkpoint, "d1_denoise_struct", args.seed, lock)
    d0, d0_cfg = _load_denoiser(d0_path, lock, device)
    d1, d1_cfg = _load_denoiser(d1_path, lock, device)
    target_size = tuple(int(value) for value in lock.get("input_resolution") or d0_cfg["data"]["target_size"])
    if target_size != tuple(int(value) for value in (lock.get("input_resolution") or d1_cfg["data"]["target_size"])):
        raise RuntimeError("D0 and D1 input resolution differs")
    transform = JointOCTTransform(target_size=target_size, training=False, normalization=str(lock.get("normalization") or "fixed"))
    cache = root / "runs" / "input_probe_cache" / str(lock["protocol_id"]) / f"seed{args.seed}"
    output_manifest = manifest_root / f"input_probe_manifest_seed{args.seed}.csv"
    if output_manifest.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite input probe manifest: {output_manifest}")
    d0_paths, d1_paths = [], []
    for row in table.itertuples(index=False):
        image = read_gray(_resolve(root, row.image_path))
        tensor = transform({"image": image}, {}, allow_strong=False)["image"].unsqueeze(0).to(device)
        destinations = []
        with torch.no_grad():
            for name, model in (("d0", d0), ("d1", d1)):
                destination = cache / name / f"{row.sample_id}.npy"
                if not destination.is_file() or not args.resume:
                    prediction = model.forward_denoise_only(tensor)["denoised"][0, 0].detach().cpu().numpy().astype(np.float32)
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError(f"Non-finite {name} output: {row.sample_id}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    np.save(destination, np.clip(prediction, 0.0, 1.0), allow_pickle=False)
                destinations.append(str(destination.relative_to(root)).replace("\\", "/"))
        d0_paths.append(destinations[0]); d1_paths.append(destinations[1])
    table["d0_path"], table["d1_path"] = d0_paths, d1_paths
    table.to_csv(output_manifest, index=False, encoding="utf-8-sig")
    audit = {"status": "passed", "protocol_id": lock["protocol_id"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "seed": args.seed, "row_count": len(table), "splits": table["split"].value_counts().to_dict(), "d0_checkpoint": str(d0_path), "d0_checkpoint_sha256": sha256_file(d0_path), "d1_checkpoint": str(d1_path), "d1_checkpoint_sha256": sha256_file(d1_path), "manifest": str(output_manifest), "test_assets_opened": 0}
    audit_path = output_manifest.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
