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
}


def build(project_root: Path, run_dir: Path, expected_per_method: int = 1779) -> pd.DataFrame:
    root, run = project_root.resolve(), run_dir.resolve()
    lock = json.loads((run / "audit" / "config_lock.json").read_text(encoding="utf-8"))
    if lock.get("status") != "locked":
        raise RuntimeError("downstream manifest requires the formal locked run; sealed references remain closed")
    protocol = load_protocol_manifest(root)
    evaluated_path = run / "manifests" / "denoised_dataset_manifest.csv"
    evaluated = pd.read_csv(evaluated_path)
    if "is_primary_seed" in evaluated:
        evaluated = evaluated[evaluated.is_primary_seed.astype(str).str.lower().isin({"true", "1"})]
    evaluated = evaluated[evaluated.status == "success"].copy()
    rows = []
    for row in evaluated.itertuples():
        rows.append({"method": METHOD_NAMES[row.method_id], "method_id": row.method_id, "dataset": row.dataset,
                     "split": row.split, "position_id": row.position_id, "frame_id": row.frame_id,
                     "sample_id": getattr(row, "sample_id", f"{row.position_id}_{row.frame_id}"), "path": row.denoised_path,
                     "config_sha256": row.config_sha256, "checkpoint_sha256": getattr(row, "checkpoint_sha256", ""),
                     "source_sha256": getattr(row, "output_sha256", ""), "primary_seed": 42 if row.method_id in {"dncnn_paired", "nafnet_paired"} else 0})
    clean_config_hash = stable_sha256({"method_id": "clean_oracle", "role": "downstream_oracle_not_deployable"})
    for row in protocol.itertuples():
        clean = Path(row.clean_path)
        rows.append({"method": "clean oracle", "method_id": "clean_oracle", "dataset": row.dataset, "split": row.split,
                     "position_id": row.position_id, "frame_id": row.frame_id, "sample_id": row.sample_id, "path": str(clean),
                     "config_sha256": clean_config_hash, "checkpoint_sha256": "", "source_sha256": sha256_file(clean), "primary_seed": 0})
    result = pd.DataFrame(rows)
    counts = result.groupby("method_id").sample_id.nunique().to_dict()
    expected = set(METHOD_NAMES) | {"clean_oracle"}
    if set(counts) != expected or any(counts[name] != expected_per_method for name in expected):
        raise RuntimeError(f"downstream coverage mismatch; expected {expected_per_method} per method, got {counts}")
    if result.duplicated(["method_id", "dataset", "sample_id"]).any():
        raise RuntimeError("duplicate downstream logical keys")
    path = run / "manifests" / "downstream_segmentation_inputs.csv"
    atomic_write_csv(result.sort_values(["method_id", "dataset", "position_id", "frame_id"]), path, run)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(".")); parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-per-method", type=int, default=1779)
    args = parser.parse_args(argv)
    result = build(args.project_root, args.run_dir, args.expected_per_method)
    print(result.groupby("method_id").size().to_dict())


if __name__ == "__main__":
    main()
