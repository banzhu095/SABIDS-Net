from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .data import load_protocol_manifest
from .io import sha256_file
from .registry import stable_sha256
from .table_store import atomic_write_csv


METHOD_NAMES = {
    "noisy_identity": "noisy", "bm3d_standard": "bm3d", "tv_chambolle": "tv", "nlm": "nlm",
    "ksvd_self": "ksvd", "dncnn_paired": "dncnn", "nafnet_paired": "nafnet",
    "sabids_current": "sabids-current", "tcfl_dncnn": "tcfl-dncnn",
}


def build(project_root: Path, run_dir: Path, expected_per_method: int | None = None) -> pd.DataFrame:
    root, run = project_root.resolve(), run_dir.resolve()
    lock_path = run / "audit" / "extension_config_lock.json"
    if not lock_path.is_file(): lock_path = run / "audit" / "config_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "locked":
        raise RuntimeError("downstream manifest requires the formal locked run; sealed references remain closed")
    protocol = load_protocol_manifest(root)
    evaluated_path = run / "manifests" / "denoised_dataset_manifest.csv"
    evaluated = pd.read_csv(evaluated_path)
    evaluated = evaluated[evaluated.status == "success"].copy()
    evaluated = evaluated[evaluated.method_id.isin(METHOD_NAMES)].copy()
    rows = []
    for row in evaluated.itertuples():
        rows.append({"input_group": METHOD_NAMES[row.method_id], "method_id": row.method_id, "dataset": row.dataset,
                     "split": row.split, "position_id": row.position_id, "frame_id": row.frame_id,
                     "sample_id": getattr(row, "sample_id", f"{row.position_id}_{row.frame_id}"), "path": row.denoised_path,
                     "config_sha256": row.config_sha256, "checkpoint_sha256": getattr(row, "checkpoint_sha256", ""),
                     "image_path": row.denoised_path, "image_sha256": getattr(row, "output_sha256", ""), "denoiser_seed": int(getattr(row, "seed", 0)), "is_primary": bool(getattr(row, "is_primary_seed", True))})
    clean_config_hash = stable_sha256({"method_id": "clean_oracle", "role": "downstream_oracle_not_deployable"})
    for row in protocol.itertuples():
        clean = Path(row.clean_path)
        rows.append({"input_group": "clean oracle", "method_id": "clean_oracle", "dataset": row.dataset, "split": row.split,
                     "position_id": row.position_id, "frame_id": row.frame_id, "sample_id": row.sample_id, "image_path": str(clean),
                     "config_sha256": clean_config_hash, "checkpoint_sha256": "", "image_sha256": sha256_file(clean), "denoiser_seed": 0, "is_primary": True})
    result = pd.DataFrame(rows)
    labels = protocol[["sample_id", "layer_mask_path", "vessel_mask_path"]].copy()
    result = result.merge(labels, on="sample_id", how="left")
    result["has_layer_label"] = result["layer_mask_path"].fillna("").astype(str).str.len().gt(0)
    result["has_vessel_label"] = result["vessel_mask_path"].fillna("").astype(str).str.len().gt(0)
    if expected_per_method is None: expected_per_method = int(protocol.sample_id.nunique())
    primary_rows = result[result.is_primary.astype(str).str.lower().isin({"true", "1"})]
    counts = primary_rows.groupby("method_id").sample_id.nunique().to_dict()
    expected = set(METHOD_NAMES) | {"clean_oracle"}
    if set(counts) != expected or any(counts[name] != expected_per_method for name in expected):
        raise RuntimeError(f"downstream coverage mismatch; expected {expected_per_method} per method, got {counts}")
    if result.duplicated(["method_id", "dataset", "sample_id", "denoiser_seed"]).any():
        raise RuntimeError("duplicate downstream logical keys")
    path = run / "manifests" / "downstream_segmentation_inputs.csv"
    atomic_write_csv(result.sort_values(["method_id", "dataset", "position_id", "frame_id"]), path, run)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(".")); parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-per-method", type=int)
    args = parser.parse_args(argv)
    result = build(args.project_root, args.run_dir, args.expected_per_method)
    print(result.groupby("method_id").size().to_dict())


if __name__ == "__main__":
    main()
