from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .data import load_protocol_manifest
from .io import read_image, save_image, sha256_file
from .metrics import compute_metrics
from .methods import AdapterContext, denoise
from .registry import load_yaml, stable_sha256
from .statistics import aggregate, bootstrap_confidence_intervals, paired_differences


DEEP = {"dncnn_paired", "nafnet_paired"}


def _selected(table: pd.DataFrame, requested: list[str]) -> pd.DataFrame:
    masks = []
    for split in requested:
        if split == "external_test":
            masks.append(table.split == "external_test")
        else:
            masks.append((table.dataset == "PKU37") & (table.split == split))
    if not masks:
        raise ValueError("no splits requested")
    mask = masks[0]
    for extra in masks[1:]: mask |= extra
    return table[mask].copy()


def evaluate(args: argparse.Namespace) -> None:
    root, run_dir = args.project_root.resolve(), args.run_dir.resolve()
    table = _selected(load_protocol_manifest(root, args.manifest), args.splits)
    if any(split in {"test", "external_test"} for split in args.splits):
        lock = run_dir / "audit" / "config_lock.json"
        if not lock.is_file():
            raise RuntimeError("sealed test requires audit/config_lock.json created before test launch")
    registry = load_yaml(args.registry or run_dir / "configs" / "inference_registry.yaml")
    methods = list(registry["methods"]) if args.methods == ["all"] else args.methods
    metric_path, failure_path = run_dir / "metrics" / "per_image_metrics.csv", run_dir / "failures.csv"
    existing = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    rows, failures, assets, runtimes = [], [], [], []
    for method in methods:
        entry = dict(registry["methods"][method]); config = dict(entry.get("config", entry)); config["method_id"] = method
        seed = int(entry.get("seed", 0)); config_hash = stable_sha256(config)
        checkpoint = entry.get("checkpoint")
        checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
        checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else ""
        if method in DEEP and not checkpoint_hash:
            failures.append({"method_id": method, "stage": "evaluate", "error": "locked checkpoint missing"}); continue
        for index, row in enumerate(table.itertuples(), 1):
            key_ok = False
            if not existing.empty:
                match = existing[(existing.sample_id.astype(str) == str(row.sample_id)) & (existing.method_id == method) & (existing.config_sha256 == config_hash) & (existing.checkpoint_sha256.fillna("") == checkpoint_hash)]
                key_ok = not match.empty and Path(match.iloc[-1].denoised_path).is_file()
            if key_ok and not args.overwrite:
                continue
            split_name = "external_test" if row.dataset in {"Duke17", "Duke28"} else row.split
            seed_part = Path(f"seed_{seed}") if method in DEEP else Path()
            source = Path(row.image_path)
            destination = run_dir / "images" / row.dataset / split_name / method / seed_part / source.name
            try:
                io_started = time.perf_counter(); noisy, metadata = read_image(source); reference, _ = read_image(Path(row.clean_path)); io_seconds = time.perf_counter() - io_started
                context = AdapterContext(device=args.device, checkpoint=checkpoint_path, seed=seed, tile_size=args.tile_size, tile_overlap=args.tile_overlap)
                started = time.perf_counter(); output = denoise(noisy, config, context); algorithm_seconds = time.perf_counter() - started
                save_started = time.perf_counter(); destination = save_image(destination, output, metadata, True); io_seconds += time.perf_counter() - save_started
                output_hash = sha256_file(destination)
                values = compute_metrics(noisy, reference, output, algorithm_seconds, io_seconds)
                rows.append({"dataset": row.dataset, "split": split_name, "position_id": row.position_id, "frame_id": row.frame_id, "sample_id": row.sample_id, "method_id": method, "seed": seed, "noisy_path": str(source), "reference_path": row.clean_path, "denoised_path": str(destination), "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash, "output_sha256": output_hash, "width": metadata["width"], "height": metadata["height"], "bit_depth": metadata["bit_depth"], "status": "success", **values})
                assets.append({"path": str(destination), "sha256": output_hash, "bytes": destination.stat().st_size, "kind": "denoised_image"})
                runtimes.append({"method_id": method, "dataset": row.dataset, "sample_id": row.sample_id, "algorithm_seconds": algorithm_seconds, "io_seconds": io_seconds, "width": metadata["width"], "height": metadata["height"]})
            except Exception as exc:
                failures.append({"dataset": row.dataset, "split": split_name, "sample_id": row.sample_id, "method_id": method, "source": str(source), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=4)})
            if index % 20 == 0:
                print(f"{method} {row.dataset}/{split_name}: {index}/{len(table)}", flush=True)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
    if not combined.empty:
        combined = combined.drop_duplicates(["sample_id", "method_id", "seed", "config_sha256", "checkpoint_sha256"], keep="last")
        metric_path.parent.mkdir(parents=True, exist_ok=True); combined.to_csv(metric_path, index=False)
        summaries = aggregate(combined)
        for name, frame in summaries.items(): frame.to_csv(run_dir / "metrics" / f"{name}.csv", index=False)
        paired_differences(summaries["per_position_metrics"]).to_csv(run_dir / "metrics" / "paired_method_differences.csv", index=False)
        bootstrap_confidence_intervals(summaries["per_position_metrics"], args.bootstrap_iterations, 42).to_csv(run_dir / "metrics" / "bootstrap_confidence_intervals.csv", index=False)
        manifest_columns = ["dataset", "split", "position_id", "frame_id", "noisy_path", "reference_path", "denoised_path", "method_id", "seed", "config_sha256", "checkpoint_sha256", "width", "height", "bit_depth", "output_sha256", "status"]
        (run_dir / "manifests").mkdir(exist_ok=True); combined[manifest_columns].to_csv(run_dir / "manifests" / "denoised_dataset_manifest.csv", index=False)
    old_failures = pd.read_csv(failure_path) if failure_path.exists() and failure_path.stat().st_size else pd.DataFrame()
    pd.concat([old_failures, pd.DataFrame(failures)], ignore_index=True).to_csv(failure_path, index=False)
    if assets: pd.DataFrame(assets).to_csv(run_dir / "metrics" / "asset_inventory.csv", index=False)
    if runtimes: pd.DataFrame(runtimes).groupby(["method_id", "dataset"], as_index=False).agg(images=("sample_id", "size"), algorithm_seconds_mean=("algorithm_seconds", "mean"), algorithm_seconds_median=("algorithm_seconds", "median"), io_seconds_mean=("io_seconds", "mean")).to_csv(run_dir / "metrics" / "runtime_summary.csv", index=False)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, default=Path(".")); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--manifest", type=Path); p.add_argument("--registry", type=Path)
    p.add_argument("--methods", nargs="+", default=["all"]); p.add_argument("--splits", nargs="+", required=True, choices=["train", "val", "test", "external_test"]); p.add_argument("--device", default="cpu"); p.add_argument("--tile-size", type=int); p.add_argument("--tile-overlap", type=int, default=32); p.add_argument("--bootstrap-iterations", type=int, default=10000); p.add_argument("--overwrite", action="store_true"); return p


def main(argv: Sequence[str] | None = None) -> None: evaluate(parser().parse_args(argv))


if __name__ == "__main__": main()
