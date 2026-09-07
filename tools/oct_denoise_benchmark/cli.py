from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

from .data import audit_protocol, development_rows, load_protocol_manifest
from .io import read_image, save_image, sha256_file
from .metrics import compute_metrics
from .methods import AdapterContext, denoise
from .registry import lock_run, save_yaml, stable_sha256


MAIN_METHODS = ["noisy_identity", "bm3d_standard", "tv_chambolle", "nlm", "ksvd_self", "dncnn_paired", "nafnet_paired"]


def create_run(project_root: Path, run_dir: Path | None = None) -> Path:
    root = project_root.resolve()
    run = run_dir.resolve() if run_dir else root / "runs" / f"denoise_benchmark_pku_protocol_{datetime.now():%Y%m%d_%H%M%S}"
    for subdir in ("audit", "configs", "checkpoints", "logs", "metrics", "reports", "previews", "images", "manifests", "gpt_light", "scripts"):
        (run / subdir).mkdir(parents=True, exist_ok=True)
    template = root / "configs" / "protocol.yaml"
    if not template.is_file(): raise FileNotFoundError(template)
    protocol = yaml.safe_load(template.read_text(encoding="utf-8"))
    save_yaml(run / "configs" / "protocol.yaml", protocol)
    save_yaml(run / "configs" / "locked_classical_configs.yaml", {"status": "unlocked", "methods": {}})
    save_yaml(run / "configs" / "locked_deep_configs.yaml", {"status": "unlocked", "methods": {}})
    save_yaml(run / "configs" / "inference_registry.yaml", {"status": "unlocked", "methods": {}})
    return run


def audit(project_root: Path, run_dir: Path, manifest: Path | None = None) -> dict[str, Any]:
    table = load_protocol_manifest(project_root, manifest)
    result = audit_protocol(table)
    versions = {}
    for package in ("numpy", "pandas", "scikit-image", "scipy", "torch", "bm3d", "Pillow", "PyYAML"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = "missing"
    result.update({"platform": platform.platform(), "python": sys.version, "packages": versions, "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True).stdout.strip() or "unavailable"})
    path = run_dir / "audit" / "data_split_audit.json"; path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(result["rows_by_dataset_split"]).to_csv(run_dir / "audit" / "dataset_inventory.csv", index=False)
    if not result["passed"]: raise RuntimeError(f"protocol audit failed; see {path}")
    return result


def _smoke_configs() -> dict[str, dict[str, Any]]:
    return {
        "noisy_identity": {"method_id": "noisy_identity"},
        "bm3d_standard": {"method_id": "bm3d_standard", "sigma_psd": 0.08, "profile": "standard", "stage": "all"},
        "tv_chambolle": {"method_id": "tv_chambolle", "weight": 0.05, "eps": 0.0002, "max_num_iter": 20},
        "nlm": {"method_id": "nlm", "h_sigma_multiplier": 0.8, "patch_size": 3, "patch_distance": 3, "fast_mode": True, "provide_sigma": True},
        "ksvd_self": {"method_id": "ksvd_self", "patch_size": 4, "dictionary_atoms": 16, "iterations": 1, "omp_max_nonzero": 2, "stride": 4, "max_training_patches": 64, "aggregation_weight": 1.0},
    }


def smoke(project_root: Path, run_dir: Path, methods: list[str]) -> pd.DataFrame:
    table = load_protocol_manifest(project_root)
    sample = development_rows(table, "val").sort_values(["position_id", "frame_id"]).iloc[0]
    noisy, metadata = read_image(Path(sample.image_path)); reference, _ = read_image(Path(sample.clean_path))
    # Bound local smoke cost without changing geometry contract: adapters see a fixed source crop and must return it unchanged.
    crop = noisy[:64, :64].copy(); crop_reference = reference[:64, :64].copy()
    rows = []
    for method in methods:
        config = _smoke_configs()[method]
        started = time.perf_counter()
        try:
            output = denoise(crop, config, AdapterContext(device="cpu", seed=42))
            elapsed = time.perf_counter() - started
            destination = run_dir / "previews" / "smoke" / f"{method}.png"
            save_image(destination, output, {**metadata, "source_dtype": np.dtype("uint16")}, True)
            rows.append({"method_id": method, "status": "passed", "shape": str(output.shape), "finite": bool(np.isfinite(output).all()), "min": float(output.min()), "max": float(output.max()), "seconds": elapsed, **compute_metrics(crop, crop_reference, output, elapsed)})
        except Exception as exc:
            rows.append({"method_id": method, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    result = pd.DataFrame(rows); result.to_csv(run_dir / "metrics" / "adapter_smoke_test.csv", index=False)
    return result


def _candidate_grid(method: str) -> list[dict[str, Any]]:
    if method == "bm3d_standard": return [{"method_id": method, "sigma_psd": value, "profile": "standard", "stage": "all"} for value in (0.02, 0.04, 0.06, 0.08, 0.12)]
    if method == "tv_chambolle": return [{"method_id": method, "weight": weight, "eps": eps, "max_num_iter": iterations} for weight in (0.01, 0.03, 0.06, 0.12, 0.24) for eps in (0.0001, 0.0002) for iterations in (100, 300)]
    if method == "nlm": return [{"method_id": method, "h_sigma_multiplier": h, "patch_size": patch, "patch_distance": distance, "fast_mode": True, "provide_sigma": provide} for h in (0.5, 0.8, 1.1, 1.5) for patch in (3, 5, 7) for distance in (3, 6, 10) for provide in (True, False)]
    if method == "ksvd_self": return [{"method_id": method, "patch_size": patch, "dictionary_atoms": atoms, "iterations": iterations, "omp_max_nonzero": sparsity, "stride": stride, "noise_weight": 1.0, "aggregation_weight": 1.0, "max_training_patches": 2000} for patch in (5, 7) for atoms in (32, 64) for iterations in (3, 5) for sparsity in (3, 5) for stride in (2, 3)]
    raise ValueError(method)


def calibrate(project_root: Path, run_dir: Path, methods: list[str], limit_frames_per_position: int | None = None) -> None:
    val = development_rows(load_protocol_manifest(project_root), "val").sort_values(["position_id", "frame_id"])
    if limit_frames_per_position: val = val.groupby("position_id", as_index=False, group_keys=False).head(limit_frames_per_position)
    all_rows, selected = [], {}
    for method in methods:
        candidates = _candidate_grid(method)
        round_index = 0
        while True:
            round_rows = []
            for candidate_index, config in enumerate(candidates):
                for row in val.itertuples():
                    noisy, _ = read_image(Path(row.image_path)); reference, _ = read_image(Path(row.clean_path))
                    started = time.perf_counter(); output = denoise(noisy, config, AdapterContext(seed=42)); elapsed = time.perf_counter() - started
                    metrics = compute_metrics(noisy, reference, output, elapsed)
                    round_rows.append({"method_id": method, "search_round": round_index, "candidate_index": candidate_index, "candidate_json": json.dumps(config, sort_keys=True), "position_id": row.position_id, "sample_id": row.sample_id, **metrics})
            all_rows.extend(round_rows)
            summary = pd.DataFrame(round_rows).groupby(["candidate_index", "candidate_json"], as_index=False)[["psnr", "ssim"]].mean()
            summary = summary.sort_values(["psnr", "ssim"], ascending=False); best = summary.iloc[0]; best_config = json.loads(best.candidate_json)
            boundary = False; stop_reason = "best_candidate_interior_or_nonadaptive_grid"
            if method == "bm3d_standard":
                sigmas = sorted(config["sigma_psd"] for config in candidates); best_sigma = best_config["sigma_psd"]
                boundary = best_sigma in {sigmas[0], sigmas[-1]}
                if boundary and round_index < 3:
                    if best_sigma == sigmas[-1] and best_sigma < 0.4: new_values = [best_sigma, min(best_sigma * 1.5, 0.4), min(best_sigma * 2, 0.4)]
                    elif best_sigma == sigmas[0] and best_sigma > 0.002: new_values = [max(best_sigma / 2, 0.002), max(best_sigma * 0.75, 0.002), best_sigma]
                    else: new_values = []
                    new_values = sorted(set(new_values) - set(sigmas))
                    if new_values:
                        candidates = [{"method_id": method, "sigma_psd": value, "profile": "standard", "stage": "all"} for value in new_values]
                        round_index += 1; continue
                    stop_reason = "scientific_sigma_limit_reached"
                elif boundary: stop_reason = "maximum_adaptive_rounds_reached"
                else: stop_reason = "best_sigma_interior"
            best_config["selection_rule"] = "PKU37 validation position-macro PSNR; SSIM tie-break"
            best_config["search_stop_reason"] = stop_reason
            selected[method] = best_config; break
        result_name = "parameter_search_smoke_results.csv" if limit_frames_per_position else "parameter_search_results.csv"
        pd.DataFrame(all_rows).to_csv(run_dir / "metrics" / result_name, index=False)
    if limit_frames_per_position:
        save_yaml(run_dir / "configs" / "calibration_smoke_selected.yaml", {
            "status": "smoke_only_not_locked",
            "reason": "subset calibration cannot satisfy the complete-PKU37-validation protocol",
            "methods": selected,
        })
        return
    save_yaml(run_dir / "configs" / "locked_classical_configs.yaml", {"status": "locked_on_pku37_validation", "methods": selected})
    pd.DataFrame([{"method_id": key, **value} for key, value in selected.items()]).to_csv(run_dir / "metrics" / "selected_parameters.csv", index=False)
    save_yaml(run_dir / "configs" / "inference_registry.yaml", {"status": "classical_locked", "methods": {key: {"config": value, "seed": 0} for key, value in selected.items()}})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "audit", "smoke", "calibrate", "lock"):
        p = sub.add_parser(name); p.add_argument("--project-root", type=Path, default=Path(".")); p.add_argument("--run-dir", type=Path)
        if name == "smoke": p.add_argument("--methods", nargs="+", default=list(_smoke_configs()))
        if name == "calibrate": p.add_argument("--methods", nargs="+", default=["bm3d_standard", "tv_chambolle", "nlm", "ksvd_self"]); p.add_argument("--limit-frames-per-position", type=int)
    args = parser.parse_args(argv); run = args.run_dir
    if args.command == "init": print(create_run(args.project_root, run)); return
    if run is None: parser.error("--run-dir is required")
    if args.command == "audit": print(json.dumps(audit(args.project_root, run), ensure_ascii=False)); return
    if args.command == "smoke": print(smoke(args.project_root, run, args.methods).to_string(index=False)); return
    if args.command == "calibrate": calibrate(args.project_root, run, args.methods, args.limit_frames_per_position); return
    if args.command == "lock": print(json.dumps(lock_run(args.project_root, run), indent=2)); return


if __name__ == "__main__": main()
