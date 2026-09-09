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
from .table_store import atomic_write_csv


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
    if method == "ksvd_self":
        base = {"method_id": method, "patch_size": 7, "dictionary_atoms": 64, "iterations": 5, "omp_max_nonzero": 4, "omp_residual_threshold": 0.0, "stride": 6, "noise_weight": 1.0, "aggregation_weight": 1.0, "max_training_patches": 2000}
        # A deterministic fractional grid varies every required dimension while
        # avoiding the prohibitive 2^7 full Cartesian product.
        variants = [
            {}, {"patch_size": 5}, {"dictionary_atoms": 32}, {"dictionary_atoms": 96},
            {"iterations": 3}, {"iterations": 7}, {"omp_max_nonzero": 3},
            {"omp_max_nonzero": 5}, {"noise_weight": 0.6}, {"noise_weight": 0.8},
            {"stride": 4}, {"stride": 8}, {"aggregation_weight": 0.0},
            {"aggregation_weight": 3.0},
        ]
        return [{**base, **variant} for variant in variants]
    raise ValueError(method)


def _position_macro_candidates(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    positions = frame.groupby(["candidate_uid", "candidate_json", "position_id"], as_index=False).agg(
        psnr=("psnr", "mean"), ssim=("ssim", "mean"), samples=("sample_id", "nunique")
    )
    return positions.groupby(["candidate_uid", "candidate_json"], as_index=False).agg(
        position_macro_psnr=("psnr", "mean"), position_macro_ssim=("ssim", "mean"),
        positions=("position_id", "nunique"), samples=("samples", "sum")
    )


def _complete_summary(rows: list[dict[str, Any]], expected_rows: pd.DataFrame) -> pd.DataFrame:
    summary = _position_macro_candidates(rows)
    complete = summary[(summary.positions == expected_rows.position_id.nunique()) & (summary.samples == expected_rows.sample_id.nunique())]
    if complete.empty:
        raise RuntimeError("no calibration candidate completed every registered validation sample")
    return complete


def _select_candidate(summary: pd.DataFrame, tolerance: float = 1e-4) -> pd.Series:
    best_psnr = float(summary["position_macro_psnr"].max())
    tied = summary[summary["position_macro_psnr"] >= best_psnr - tolerance]
    return tied.sort_values(["position_macro_ssim", "position_macro_psnr", "candidate_uid"], ascending=[False, False, True]).iloc[0]


def _numeric_boundaries(best: dict[str, Any], evaluated: list[dict[str, Any]], keys: list[str]) -> list[str]:
    boundaries = []
    for key in keys:
        values = sorted({config[key] for config in evaluated if key in config})
        if values and best.get(key) in {values[0], values[-1]}:
            boundaries.append(key)
    return boundaries


def _expand_non_bm3d(method: str, best: dict[str, Any], evaluated: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    keys = ["weight", "eps", "max_num_iter"] if method == "tv_chambolle" else ["h_sigma_multiplier", "patch_size", "patch_distance"]
    boundaries = _numeric_boundaries(best, evaluated, keys)
    proposals: dict[str, list[Any]] = {}
    if method == "tv_chambolle":
        ranges = {key: sorted({config[key] for config in evaluated}) for key in keys}
        if "weight" in boundaries:
            value = float(best["weight"]); proposals["weight"] = ([max(value / 2, 0.001)] if value == ranges["weight"][0] else [min(value * 1.5, 1.0), min(value * 2, 1.0)])
        if "eps" in boundaries:
            value = float(best["eps"]); proposals["eps"] = ([max(value / 2, 1e-6), max(value / 5, 1e-6)] if value == ranges["eps"][0] else [min(value * 2, 0.01)])
        if "max_num_iter" in boundaries:
            value = int(best["max_num_iter"]); proposals["max_num_iter"] = ([max(value // 2, 20)] if value == ranges["max_num_iter"][0] else [min(value * 2, 1000)])
    elif method == "nlm":
        ranges = {key: sorted({config[key] for config in evaluated}) for key in keys}
        if "h_sigma_multiplier" in boundaries:
            value = float(best["h_sigma_multiplier"]); proposals["h_sigma_multiplier"] = ([max(value / 2, 0.1)] if value == ranges["h_sigma_multiplier"][0] else [min(value * 1.5, 5.0), min(value * 2, 5.0)])
        if "patch_size" in boundaries:
            value = int(best["patch_size"]); proposals["patch_size"] = ([max(value - 2, 1)] if value == ranges["patch_size"][0] else [min(value + 2, 15)])
        if "patch_distance" in boundaries:
            value = int(best["patch_distance"]); proposals["patch_distance"] = ([max(value // 2, 1)] if value == ranges["patch_distance"][0] else [min(round(value * 1.5), 30), min(value * 2, 30)])
    seen = {stable_sha256(config) for config in evaluated}
    expanded = []
    for key, values in proposals.items():
        for value in values:
            candidate = dict(best); candidate[key] = value
            if stable_sha256(candidate) not in seen:
                expanded.append(candidate); seen.add(stable_sha256(candidate))
    return expanded, boundaries


def _evaluate_candidates(method: str, candidates: list[dict[str, Any]], rows: pd.DataFrame, search_round: int, phase: str, partial_path: Path | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    total = len(candidates) * len(rows)
    completed = 0; resumed = 0; failed = 0; batch_started = time.perf_counter()
    existing = pd.read_csv(partial_path) if partial_path and partial_path.is_file() else pd.DataFrame()

    def flush() -> None:
        if partial_path is None or not results:
            return
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.concat([existing, pd.DataFrame(results)], ignore_index=True) if not existing.empty else pd.DataFrame(results)
        combined = combined.drop_duplicates(["method_id", "search_phase", "search_round", "candidate_uid", "sample_id"], keep="last")
        temporary = partial_path.with_suffix(partial_path.suffix + ".tmp")
        combined.to_csv(temporary, index=False)
        temporary.replace(partial_path)

    for candidate_index, config in enumerate(candidates):
        candidate_json = json.dumps(config, sort_keys=True)
        candidate_uid = stable_sha256(config)[:16]
        for row in rows.itertuples():
            if not existing.empty:
                match = existing[
                    (existing.method_id == method) & (existing.search_phase == phase)
                    & (existing.search_round == search_round) & (existing.candidate_uid == candidate_uid)
                    & (existing.sample_id.astype(str) == str(row.sample_id)) & (existing.status == "success")
                ]
                if not match.empty:
                    results.append(match.iloc[-1].to_dict()); completed += 1; resumed += 1
                    continue
            record = {"method_id": method, "search_phase": phase, "search_round": search_round,
                      "candidate_index": candidate_index, "candidate_uid": candidate_uid,
                      "candidate_json": candidate_json, "position_id": row.position_id,
                      "sample_id": row.sample_id}
            try:
                noisy, _ = read_image(Path(row.image_path)); reference, _ = read_image(Path(row.clean_path))
                started = time.perf_counter(); output = denoise(noisy, config, AdapterContext(seed=42)); elapsed = time.perf_counter() - started
                metrics = compute_metrics(noisy, reference, output, elapsed)
                results.append({**record, "status": "success", "error": "", **metrics})
            except Exception as exc:
                results.append({**record, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
                failed += 1
            completed += 1
            if completed % 20 == 0 or completed == total:
                flush()
                elapsed = time.perf_counter() - batch_started
                eta = elapsed / max(completed - resumed, 1) * max(total - completed, 0)
                print(f"calibrate {method}/{phase}: candidate={candidate_index + 1}/{len(candidates)} sample={row.sample_id} position={row.position_id} frame={getattr(row, 'frame_id', '')} completed={completed}/{total} resumed={resumed} failed={failed} eta_seconds={eta:.1f}", flush=True)
    if resumed:
        print(f"calibrate {method}/{phase}: resume reused {resumed}/{total} hash-matched successful rows", flush=True)
    flush()
    return [record for record in results if record.get("status") == "success"]


def _calibrate_method(method: str, full_val: pd.DataFrame, val: pd.DataFrame, partial_path: Path, smoke: bool, ksvd_coarse_frames_per_position: int, ksvd_top_candidates: int, psnr_tolerance: float) -> dict[str, Any]:
    candidates = _candidate_grid(method)
    if method == "ksvd_self" and not smoke:
        coarse = pd.concat([group.iloc[np.linspace(0, len(group) - 1, min(ksvd_coarse_frames_per_position, len(group)), dtype=int)] for _, group in full_val.groupby("position_id", sort=True)], ignore_index=True)
        coarse_rows = _evaluate_candidates(method, candidates, coarse, 0, "coarse_equal_frames_per_position", partial_path)
        coarse_summary = _complete_summary(coarse_rows, coarse).sort_values(["position_macro_psnr", "position_macro_ssim"], ascending=False)
        finalists = [json.loads(value) for value in coarse_summary.head(ksvd_top_candidates)["candidate_json"]]
        final_rows = _evaluate_candidates(method, finalists, full_val, 1, "full_validation_reevaluation", partial_path)
        best = _select_candidate(_complete_summary(final_rows, full_val), psnr_tolerance)
        best_config = json.loads(best.candidate_json)
        stop_reason = f"two_stage_search_top_{len(finalists)}_fully_reevaluated"
        selection_evaluated = candidates
    else:
        method_rows: list[dict[str, Any]] = []; round_index = 0; evaluated_sigmas: set[float] = set()
        while True:
            phase = "smoke_subset" if smoke else "full_validation"
            round_rows = _evaluate_candidates(method, candidates, val, round_index, phase, partial_path)
            method_rows.extend(round_rows)
            summary = _complete_summary(method_rows, val)
            best = _select_candidate(summary, psnr_tolerance); best_config = json.loads(best.candidate_json)
            stop_reason = "nonadaptive_grid_complete"
            evaluated_configs = [json.loads(value) for value in summary["candidate_json"]]
            if method in {"tv_chambolle", "nlm"}:
                expanded, boundaries = _expand_non_bm3d(method, best_config, evaluated_configs)
                if not boundaries: stop_reason = "best_numeric_parameters_interior"; break
                if expanded and round_index < 3: candidates = expanded; round_index += 1; continue
                stop_reason = "maximum_adaptive_rounds_or_scientific_limit_reached"; break
            if method != "bm3d_standard": break
            evaluated_sigmas.update(float(json.loads(value)["sigma_psd"]) for value in summary["candidate_json"])
            best_sigma = float(best_config["sigma_psd"]); low, high = min(evaluated_sigmas), max(evaluated_sigmas)
            if best_sigma not in {low, high}: stop_reason = "best_sigma_interior"; break
            if round_index >= 3: stop_reason = "maximum_adaptive_rounds_reached"; break
            if best_sigma == high and high < 0.4: new_values = [min(high * 1.5, 0.4), min(high * 2.0, 0.4)]
            elif best_sigma == low and low > 0.002: new_values = [max(low / 2.0, 0.002), max(low * 0.75, 0.002)]
            else: new_values = []
            new_values = sorted(set(new_values) - evaluated_sigmas)
            if not new_values: stop_reason = "scientific_sigma_limit_reached"; break
            candidates = [{"method_id": method, "sigma_psd": value, "profile": "standard", "stage": "all"} for value in new_values]
            round_index += 1
        selection_evaluated = [json.loads(value) for value in summary["candidate_json"]]
    parameter_values = {key: sorted({config.get(key) for config in selection_evaluated if key in config}) for key in selection_evaluated[0] if key != "method_id"}
    boundary = {key: best_config.get(key) in {values[0], values[-1]} for key, values in parameter_values.items() if values and isinstance(values[0], (int, float)) and not isinstance(values[0], bool)}
    best_config.update({"selection_rule": f"PKU37 validation frame-to-position macro PSNR; SSIM tie-break within {psnr_tolerance} dB", "search_stop_reason": stop_reason, "boundary_parameters": [key for key, value in boundary.items() if value]})
    return best_config


def calibrate(project_root: Path, run_dir: Path, methods: list[str], limit_frames_per_position: int | None = None, ksvd_coarse_frames_per_position: int = 2, ksvd_top_candidates: int = 4, psnr_tolerance: float = 1e-4) -> None:
    full_val = development_rows(load_protocol_manifest(project_root), "val").sort_values(["position_id", "frame_index"])
    val = full_val.groupby("position_id", as_index=False, group_keys=False).head(limit_frames_per_position) if limit_frames_per_position else full_val
    smoke = limit_frames_per_position is not None
    partial_path = run_dir / "metrics" / ("parameter_search_smoke_partial.csv" if smoke else "parameter_search_partial.csv")
    config_path = run_dir / "configs" / ("calibration_smoke_selected.yaml" if smoke else "locked_classical_configs.yaml")
    prior = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    selected: dict[str, dict[str, Any]] = dict(prior.get("methods", {}))
    failures = []
    for method in methods:
        try:
            selected[method] = _calibrate_method(method, full_val, val, partial_path, smoke, ksvd_coarse_frames_per_position, ksvd_top_candidates, psnr_tolerance)
        except Exception as exc:
            failures.append({"method_id": method, "stage": "calibration", "error": f"{type(exc).__name__}: {exc}", "severity": "blocking_for_method", "status": "incomplete", "recorded_at": datetime.now().isoformat()})
        if partial_path.is_file():
            atomic_write_csv(pd.read_csv(partial_path), run_dir / "metrics" / ("parameter_search_smoke_results.csv" if smoke else "parameter_search_results.csv"), run_dir)
        if smoke:
            save_yaml(config_path, {"status": "smoke_only_not_locked", "reason": "subset calibration cannot satisfy the complete-PKU37-validation protocol", "methods": selected})
        else:
            pending = sorted(set(("bm3d_standard", "tv_chambolle", "nlm", "ksvd_self")) - set(selected))
            status = "locked_on_pku37_validation" if not pending else "partially_locked_on_pku37_validation"
            save_yaml(config_path, {"status": status, "completed_methods": sorted(selected), "pending_methods": pending, "methods": selected})
            save_yaml(run_dir / "configs" / "inference_registry.yaml", {"status": "classical_locked" if not pending else "classical_partially_locked", "methods": {key: {"config": value, "seed": 0} for key, value in selected.items()}})
    if failures:
        failure_path = run_dir / "failures.csv"; existing = pd.read_csv(failure_path) if failure_path.is_file() and failure_path.stat().st_size else pd.DataFrame()
        atomic_write_csv(pd.concat([existing, pd.DataFrame(failures)], ignore_index=True), failure_path, run_dir)
    if not smoke:
        selected_path = run_dir / "metrics" / "selected_parameters.csv"; existing = pd.read_csv(selected_path) if selected_path.is_file() else pd.DataFrame()
        if not existing.empty: existing = existing[~existing.method_id.isin(selected)]
        atomic_write_csv(pd.concat([existing, pd.DataFrame([{"method_id": key, "status": "selected_on_complete_PKU37_validation", **value} for key, value in selected.items()])], ignore_index=True), selected_path, run_dir)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "audit", "smoke", "calibrate", "lock"):
        p = sub.add_parser(name); p.add_argument("--project-root", type=Path, default=Path(".")); p.add_argument("--run-dir", type=Path)
        if name == "smoke": p.add_argument("--methods", nargs="+", default=list(_smoke_configs()))
        if name == "calibrate":
            p.add_argument("--methods", nargs="+", default=["bm3d_standard", "tv_chambolle", "nlm", "ksvd_self"]); p.add_argument("--limit-frames-per-position", type=int)
            p.add_argument("--ksvd-coarse-frames-per-position", type=int, default=2); p.add_argument("--ksvd-top-candidates", type=int, default=4); p.add_argument("--psnr-tolerance", type=float, default=1e-4)
    args = parser.parse_args(argv); run = args.run_dir
    if args.command == "init": print(create_run(args.project_root, run)); return
    if run is None: parser.error("--run-dir is required")
    if args.command == "audit": print(json.dumps(audit(args.project_root, run), ensure_ascii=False)); return
    if args.command == "smoke": print(smoke(args.project_root, run, args.methods).to_string(index=False)); return
    if args.command == "calibrate": calibrate(args.project_root, run, args.methods, args.limit_frames_per_position, args.ksvd_coarse_frames_per_position, args.ksvd_top_candidates, args.psnr_tolerance); return
    if args.command == "lock": print(json.dumps(lock_run(args.project_root, run), indent=2)); return


if __name__ == "__main__": main()
