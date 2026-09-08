from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from .data import load_protocol_manifest
from .io import read_image, save_image, sha256_file
from .metrics import compute_metrics
from .methods import AdapterContext, denoise
from .registry import load_yaml, stable_sha256
from .statistics import aggregate, bootstrap_confidence_intervals, paired_differences


DEEP = {"dncnn_paired", "nafnet_paired"}


def _cpu_peak_memory_mb() -> float:
    try:
        import resource
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value / (1024.0 if value > 10_000 else 1.0)
    except (ImportError, AttributeError):
        try:
            import psutil
            return float(psutil.Process().memory_info().peak_wset) / (1024**2)
        except (ImportError, AttributeError):
            return float("nan")


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
    lock_data: dict[str, Any] | None = None
    if any(split in {"test", "external_test"} for split in args.splits):
        lock = run_dir / "audit" / "config_lock.json"
        if not lock.is_file():
            raise RuntimeError("sealed test requires audit/config_lock.json created before test launch")
        lock_data = json.loads(lock.read_text(encoding="utf-8"))
        if lock_data.get("status") != "locked":
            raise RuntimeError("sealed test remains closed because config_lock status is not locked")
        for name, expected in lock_data.get("config_sha256", {}).items():
            current = run_dir / "configs" / name
            if not current.is_file() or sha256_file(current) != expected:
                raise RuntimeError(f"locked configuration changed after lock: {name}")
        if lock_data.get("test_started_at_utc") is None:
            lock_data["test_started_at_utc"] = datetime.now(timezone.utc).isoformat()
            lock.write_text(json.dumps(lock_data, indent=2, ensure_ascii=False), encoding="utf-8")
    registry = load_yaml(args.registry or run_dir / "configs" / "inference_registry.yaml")
    methods = list(registry["methods"]) if args.methods == ["all"] else args.methods
    metric_path, failure_path = run_dir / "metrics" / "per_image_metrics.csv", run_dir / "failures.csv"
    existing = pd.read_csv(metric_path) if metric_path.exists() else pd.DataFrame()
    rows, failures, assets, runtimes = [], [], [], []
    planned: list[tuple[str, dict[str, Any]]] = []
    for method in methods:
        entry = dict(registry["methods"][method])
        variants = entry.get("evaluation_checkpoints", []) if args.all_deep_seeds and method in DEEP else []
        if variants:
            for variant in variants:
                planned.append((method, {**entry, "seed": int(variant["seed"]), "checkpoint": variant["checkpoint"]}))
        else:
            planned.append((method, entry))
    for method, entry in planned:
        config = dict(entry.get("config", entry)); config["method_id"] = method
        seed = int(entry.get("seed", 0)); config_hash = stable_sha256(config)
        checkpoint = entry.get("checkpoint")
        checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
        checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else ""
        if method in DEEP and not checkpoint_hash:
            failures.append({"method_id": method, "stage": "evaluate", "error": "locked checkpoint missing"}); continue
        if lock_data is not None and method in DEEP:
            allowed = {item.get("sha256") for item in lock_data.get("checkpoint_sha256", {}).get(method, [])}
            if checkpoint_hash not in allowed:
                raise RuntimeError(f"checkpoint hash for {method} seed {seed} is not in config_lock.json")
        first_success = True
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        context = AdapterContext(device=args.device, checkpoint=checkpoint_path, seed=seed, tile_size=args.tile_size, tile_overlap=args.tile_overlap)
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
                if args.device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.synchronize(torch.device(args.device))
                started = time.perf_counter(); output = denoise(noisy, config, context)
                if args.device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.synchronize(torch.device(args.device))
                algorithm_seconds = time.perf_counter() - started
                save_started = time.perf_counter(); destination = save_image(destination, output, metadata, True); io_seconds += time.perf_counter() - save_started
                output_hash = sha256_file(destination)
                values = compute_metrics(noisy, reference, output, algorithm_seconds, io_seconds)
                rows.append({"dataset": row.dataset, "split": split_name, "position_id": row.position_id, "frame_id": row.frame_id, "sample_id": row.sample_id, "method_id": method, "seed": seed, "noisy_path": str(source), "reference_path": row.clean_path, "denoised_path": str(destination), "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash, "output_sha256": output_hash, "width": metadata["width"], "height": metadata["height"], "bit_depth": metadata["bit_depth"], "status": "success", **values})
                assets.append({"path": str(destination), "sha256": output_hash, "bytes": destination.stat().st_size, "kind": "denoised_image", "dataset": row.dataset, "split": split_name, "method_id": method, "seed": seed})
                gpu_peak = torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024**2) if args.device.startswith("cuda") and torch.cuda.is_available() else float("nan")
                runtimes.append({"method_id": method, "seed": seed, "dataset": row.dataset, "sample_id": row.sample_id, "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash, "is_startup_image": first_success, "algorithm_seconds": algorithm_seconds, "io_seconds": io_seconds, "width": metadata["width"], "height": metadata["height"], "cpu_peak_memory_mb": _cpu_peak_memory_mb(), "gpu_peak_memory_mb": gpu_peak, "cpu_model": platform.processor(), "gpu_model": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") and torch.cuda.is_available() else ""})
                first_success = False
            except Exception as exc:
                failures.append({"dataset": row.dataset, "split": split_name, "sample_id": row.sample_id, "method_id": method, "source": str(source), "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(limit=4)})
            if index % 20 == 0:
                print(f"{method} {row.dataset}/{split_name}: {index}/{len(table)}", flush=True)
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if not existing.empty else pd.DataFrame(rows)
    if not combined.empty:
        combined["checkpoint_sha256"] = combined["checkpoint_sha256"].fillna("").astype(str)
        combined["config_sha256"] = combined["config_sha256"].fillna("").astype(str)
        combined = combined.drop_duplicates(["sample_id", "method_id", "seed", "config_sha256", "checkpoint_sha256"], keep="last")
        metric_path.parent.mkdir(parents=True, exist_ok=True); combined.to_csv(metric_path, index=False)
        summaries = aggregate(combined)
        for name, frame in summaries.items(): frame.to_csv(run_dir / "metrics" / f"{name}.csv", index=False)
        paired_differences(summaries["per_position_metrics"]).to_csv(run_dir / "metrics" / "paired_method_differences.csv", index=False)
        bootstrap_confidence_intervals(summaries["per_position_metrics"], args.bootstrap_iterations, 42).to_csv(run_dir / "metrics" / "bootstrap_confidence_intervals.csv", index=False)
        primary_seeds = {method: int(entry.get("seed", 0)) for method, entry in registry["methods"].items()}
        combined["is_primary_seed"] = [int(seed) == primary_seeds.get(method, 0) for method, seed in zip(combined.method_id, combined.seed)]
        manifest_columns = ["dataset", "split", "position_id", "frame_id", "noisy_path", "reference_path", "denoised_path", "method_id", "seed", "is_primary_seed", "config_sha256", "checkpoint_sha256", "width", "height", "bit_depth", "output_sha256", "status"]
        (run_dir / "manifests").mkdir(exist_ok=True)
        combined[manifest_columns].to_csv(run_dir / "manifests" / "denoised_dataset_manifest.csv", index=False)
        combined.loc[combined.is_primary_seed, manifest_columns].to_csv(run_dir / "manifests" / "denoised_dataset_manifest_primary.csv", index=False)
    old_failures = pd.read_csv(failure_path) if failure_path.exists() and failure_path.stat().st_size else pd.DataFrame()
    pd.concat([old_failures, pd.DataFrame(failures)], ignore_index=True).to_csv(failure_path, index=False)
    if assets:
        asset_path = run_dir / "metrics" / "asset_inventory.csv"
        old_assets = pd.read_csv(asset_path) if asset_path.exists() and asset_path.stat().st_size else pd.DataFrame()
        pd.concat([old_assets, pd.DataFrame(assets)], ignore_index=True).drop_duplicates("path", keep="last").to_csv(asset_path, index=False)
    if runtimes:
        runtime_path = run_dir / "metrics" / "runtime_records.csv"
        old_runtime = pd.read_csv(runtime_path) if runtime_path.exists() and runtime_path.stat().st_size else pd.DataFrame()
        runtime = pd.concat([old_runtime, pd.DataFrame(runtimes)], ignore_index=True)
        runtime = runtime.drop_duplicates(["method_id", "seed", "dataset", "sample_id", "config_sha256", "checkpoint_sha256"], keep="last")
        runtime.to_csv(runtime_path, index=False)
        summary_rows = []
        for (method, seed, dataset), group in runtime.groupby(["method_id", "seed", "dataset"], sort=True):
            steady = group.loc[~group["is_startup_image"], "algorithm_seconds"]
            startup = group.loc[group["is_startup_image"], "algorithm_seconds"]
            summary_rows.append({"method_id": method, "seed": seed, "dataset": dataset, "images": len(group),
                                 "image_heights": ",".join(map(str, sorted(group.height.unique()))), "image_widths": ",".join(map(str, sorted(group.width.unique()))),
                                 "startup_algorithm_seconds": float(startup.iloc[0]) if len(startup) else float("nan"),
                                 "steady_algorithm_seconds_mean": float(steady.mean()) if len(steady) else float("nan"), "algorithm_seconds_mean": float(group.algorithm_seconds.mean()), "algorithm_seconds_median": float(group.algorithm_seconds.median()), "io_seconds_mean": float(group.io_seconds.mean()), "cpu_peak_memory_mb": float(group.cpu_peak_memory_mb.max()), "gpu_peak_memory_mb": float(group.gpu_peak_memory_mb.max()), "cpu_model": group.iloc[0].cpu_model, "gpu_model": group.iloc[0].gpu_model})
        pd.DataFrame(summary_rows).to_csv(run_dir / "metrics" / "runtime_summary.csv", index=False)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, default=Path(".")); p.add_argument("--run-dir", type=Path, required=True); p.add_argument("--manifest", type=Path); p.add_argument("--registry", type=Path)
    p.add_argument("--methods", nargs="+", default=["all"]); p.add_argument("--splits", nargs="+", required=True, choices=["train", "val", "test", "external_test"]); p.add_argument("--device", default="cpu"); p.add_argument("--tile-size", type=int); p.add_argument("--tile-overlap", type=int, default=32); p.add_argument("--bootstrap-iterations", type=int, default=10000); p.add_argument("--all-deep-seeds", action="store_true"); p.add_argument("--overwrite", action="store_true"); return p


def main(argv: Sequence[str] | None = None) -> None: evaluate(parser().parse_args(argv))


if __name__ == "__main__": main()
