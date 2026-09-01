"""Create a server-side, validation-only SABIDS presentation archive.

The tool has three deliberately separated phases:

``audit``
    Inventories run/checkpoint/prediction assets without opening test files.
``evaluate``
    Runs only the repository's existing fixed-threshold validation entry points
    when their outputs are absent.  It never invokes a training mode.
``assemble``
    Copies/derives tables and fixed visual assets from completed validation
    outputs, records missing evidence, and creates full/GPT packages.

Use ``all`` to execute the phases in order.  ``--run-missing-validation`` is an
explicit authorization gate; without it, missing inference products remain
MISSING and the tool only assembles available evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.data.io import read_gray, write_gray, write_rgb
from sabids.metrics import (
    binary_metrics,
    edge_preservation_index,
    layer_boundary_mae,
    psnr,
    reference_edge_mae,
    rmse,
    ssim,
)
from sabids.postprocessing import clean_layer_mask, hard_contain_vessel, regularize_lower_boundary


VERSION = "presentation-archive-v1"
FORBIDDEN_PARTS = {"test", "test_results", "test-results", "sealed_test", "sealed-test"}
SEEDS = (42, 43, 44)
J_VARIANTS = ("j00", "j10", "j01", "j11")
STAGE2_RUNS = {
    "E1-current": "stage2_segment_safe_current_fold0",
    "E3-current": "stage2_segment_roi_current_fold0",
    "E3b": "stage2_segment_roi_outside_fold0",
    "E3b-noD2S": "stage2_segment_roi_outside_no_d2s_fold0",
}
EXPECTED_RUNS = {
    "S1-Denoise": "stage1_denoise_fold0",
    "E0": "stage2_overfit_safe_fold0",
    **STAGE2_RUNS,
    **{f"J{variant[1:]}-s{seed}": f"interaction_{variant}_fold0_seed{seed}" for seed in SEEDS for variant in J_VARIANTS},
    **{f"I-{variant.upper()}-s{seed}": f"input_{variant}_fold0_seed{seed}" for seed in SEEDS for variant in ("noisy", "denoised")},
}
AUDIT_FILES = (
    "history.csv", "resolved_config.yaml", "run_metadata.json", "initialization_audit.json",
    "parameter_audit.json", "label_asset_inventory.json", "evaluation_registry.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=("audit", "evaluate", "assemble", "package", "all"), default="audit")
    parser.add_argument("--output", help="Existing/new presentation_archive directory; generated when omitted")
    parser.add_argument("--stage12-report", help="Completed stage12_validation report")
    parser.add_argument("--interaction-report", help="Completed interaction factorial summary")
    parser.add_argument("--input-report", help="Completed input factorial summary")
    parser.add_argument("--run-missing-validation", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--skip-workbooks", action="store_true", help="Keep CSVs if @oai/artifact-tool is unavailable")
    return parser.parse_args()


def forbidden(path: Path) -> bool:
    return any(part.lower() in FORBIDDEN_PARTS for part in path.parts)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_table(path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> pd.DataFrame:
    rows = list(rows)
    frame = pd.DataFrame(rows, columns=columns or (list(rows[0]) if rows else ["status", "reason"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def safe_resolve(root: Path, value: Any) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_yaml_text(path: Path) -> dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def nested(mapping: dict[str, Any], dotted: str, default: Any = "") -> Any:
    current: Any = mapping
    for key in dotted.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def checkpoint_epoch(path: Path) -> tuple[Any, str]:
    try:
        payload = torch.load(path, map_location="cpu")
        epoch = payload.get("epoch") if isinstance(payload, dict) else None
        return (int(epoch) + 1 if epoch is not None else "", "checkpoint_payload")
    except Exception as exc:
        return "", f"unreadable:{type(exc).__name__}"


def run_dirs(root: Path) -> list[tuple[str, Path]]:
    current = root / "runs" / "current"
    found: dict[Path, str] = {}
    for alias, name in EXPECTED_RUNS.items():
        found[current / name] = alias
    if current.is_dir():
        for directory, names, files in os.walk(current):
            names[:] = [name for name in names if name.lower() not in FORBIDDEN_PARTS]
            path = Path(directory)
            if any(name in files for name in AUDIT_FILES) or any(name in files for name in ("best.pth", "last.pth")):
                found.setdefault(path, path.name)
    return sorted(((alias, path) for path, alias in found.items()), key=lambda item: (item[0], str(item[1])))


def audit(root: Path, output: Path) -> dict[str, Any]:
    audit_dir = output / "audit"; audit_dir.mkdir(parents=True, exist_ok=True)
    run_rows, checkpoint_rows, prediction_rows, missing = [], [], [], []
    for alias, path in run_dirs(root):
        config_path = path / "resolved_config.yaml"
        metadata_path = path / "run_metadata.json"
        config = read_yaml_text(config_path) if config_path.is_file() else {}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        stage = nested(config, "train.stage", metadata.get("stage", ""))
        history = path / "history.csv"
        history_rows = 0
        last_history_epoch: Any = ""
        if history.is_file():
            table = pd.read_csv(history)
            history_rows = len(table)
            last_history_epoch = int(table["epoch"].max()) if "epoch" in table and len(table) else ""
        run_rows.append({
            "alias": alias, "run_path": path.relative_to(root).as_posix(), "exists": path.is_dir(),
            "stage": stage, "seed": nested(config, "train.seed", metadata.get("seed", "")),
            "fold": nested(config, "data.fold", metadata.get("fold", 0)),
            "input_type": nested(config, "data.input_column", "image_path"),
            "d2s_enabled": nested(config, "model.d2s_enabled", nested(config, "model.enable_denoise_to_seg", "unknown")),
            "s2d_enabled": nested(config, "model.s2d_enabled", nested(config, "model.enable_seg_to_denoise", "unknown")),
            "encoder_frozen": nested(config, "train.freeze_shared_encoder", "unknown"),
            "denoiser_frozen": nested(config, "train.freeze_denoiser", "unknown"),
            "outside_bce_weight": nested(config, "loss.weights.vessel_outside", "unknown"),
            "configured_epochs": nested(config, "train.epochs", ""), "last_history_epoch": last_history_epoch,
            "monitor": nested(config, "train.monitor", metadata.get("monitor", "")),
            "history_rows": history_rows, "history_present": history.is_file(),
            "config_present": config_path.is_file(), "metadata_present": metadata_path.is_file(),
            "initialization_audit_present": (path / "initialization_audit.json").is_file(),
            "parameter_audit_present": (path / "parameter_audit.json").is_file(),
            "label_inventory_present": (path / "label_asset_inventory.json").is_file(),
            "final_validation_present": (path / "final_validation" / "frame_metrics.csv").is_file(),
            "test_opened": False,
        })
        if not path.is_dir():
            missing.append({"family": "run", "alias": alias, "asset": "run_directory", "path": path.relative_to(root).as_posix(), "status": "MISSING", "impact": "experiment unavailable"})
            continue
        for name in ("best.pth", "last.pth"):
            checkpoint = path / name
            if checkpoint.is_file():
                epoch, epoch_source = checkpoint_epoch(checkpoint)
                checkpoint_rows.append({
                    "alias": alias, "run_path": path.relative_to(root).as_posix(),
                    "checkpoint_type": checkpoint.stem, "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                    "checkpoint_sha256": sha256(checkpoint), "epoch": epoch, "epoch_source": epoch_source,
                    "configured_epochs": nested(config, "train.epochs", ""),
                    "selection_monitor": nested(config, "train.monitor", metadata.get("monitor", "")),
                    "config_sha256": sha256(config_path) if config_path.is_file() else "",
                })
            elif alias in EXPECTED_RUNS:
                missing.append({"family": "checkpoint", "alias": alias, "asset": name, "path": checkpoint.relative_to(root).as_posix(), "status": "MISSING", "impact": "cannot run corresponding inference"})
        for validation_name in ("validation", "validity_v0", "final_validation"):
            validation = path / validation_name
            if not validation.is_dir():
                continue
            predictions = validation / "predictions"
            png_count = 0; npy_count = 0
            if predictions.is_dir():
                for directory, names, files in os.walk(predictions):
                    names[:] = [n for n in names if n.lower() not in FORBIDDEN_PARTS]
                    png_count += sum(Path(name).suffix.lower() == ".png" for name in files)
                    npy_count += sum(Path(name).suffix.lower() == ".npy" for name in files)
            prediction_rows.append({
                "alias": alias, "evaluation": validation_name,
                "path": validation.relative_to(root).as_posix(),
                "frame_metrics": (validation / "frame_metrics.csv").is_file(),
                "group_metrics": (validation / "group_metrics.csv").is_file(),
                "summary": (validation / "summary.json").is_file(),
                "evaluation_registry": (validation / "evaluation_registry.json").is_file(),
                "prediction_png_count": png_count, "float_probability_count": npy_count,
                "contains_test": False,
            })
    run_table = write_table(audit_dir / "run_path_audit.csv", run_rows)
    checkpoint_table = write_table(audit_dir / "checkpoint_audit.csv", checkpoint_rows)
    prediction_table = write_table(audit_dir / "prediction_asset_audit.csv", prediction_rows)
    missing_table = write_table(audit_dir / "missing_assets.csv", missing)
    summary = {
        "tool_version": VERSION, "generated_at": datetime.now().isoformat(),
        "runs_inventoried": len(run_table), "checkpoints_hashed": len(checkpoint_table),
        "prediction_directories": len(prediction_table), "missing_assets": len(missing_table),
        "test_assets_opened": 0, "test_evaluation_performed": False,
    }
    write_json(audit_dir / "audit_summary.json", summary)
    return summary


def run_command(command: list[str], root: Path, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}); see {log}: {' '.join(command)}")


def latest_dir(parent: Path, pattern: str) -> Path | None:
    candidates = sorted(path for path in parent.glob(pattern) if path.is_dir())
    return candidates[-1] if candidates else None


def evaluate_missing(root: Path, output: Path, args: argparse.Namespace) -> dict[str, str]:
    if not args.run_missing_validation:
        raise PermissionError("Evaluation phase requires explicit --run-missing-validation")
    staging = output / "_staging"; logs = output / "audit" / "logs"
    stage12 = safe_resolve(root, args.stage12_report) if args.stage12_report else staging / "stage12_validation"
    if not (stage12 / "segmentation_per_frame.csv").is_file() or not (stage12 / "predictions_index.csv").is_file():
        run_command([
            sys.executable, "tools/export_stage12_results.py", "--project-root", str(root),
            "--output", str(stage12), "--device", args.device, "--batch-size", "1",
            "--num-workers", str(args.num_workers), "--full-non-test", "--save-all-float", "--resume",
        ], root, logs / "stage12_validation.log")

    b0 = root / "runs/current/interaction_b0_fold0_seed42/validation/frame_metrics.csv"
    if not b0.is_file():
        run_command([sys.executable, "tools/run_interaction_factorial.py", "--project-root", str(root), "--mode", "b0", "--seeds", "42", "--device", args.device, "--save-predictions"], root, logs / "interaction_b0.log")
    for seed in SEEDS:
        missing_variants = [variant for variant in J_VARIANTS if not (root / f"runs/current/interaction_{variant}_fold0_seed{seed}/final_validation/frame_metrics.csv").is_file()]
        if missing_variants:
            run_command([sys.executable, "tools/run_interaction_factorial.py", "--project-root", str(root), "--mode", "evaluate", "--seeds", str(seed), "--variants", *missing_variants, "--epochs", "20", "--device", args.device, "--save-predictions"], root, logs / f"interaction_seed{seed}.log")
    interaction = safe_resolve(root, args.interaction_report) if args.interaction_report else staging / "interaction_factorial_report"
    if not (interaction / "paired_gains_summary.csv").is_file():
        run_command([sys.executable, "tools/summarize_interaction_factorial.py", "--project-root", str(root), "--seeds", *map(str, SEEDS), "--output", str(interaction)], root, logs / "interaction_summary.log")
    interaction_atlas = staging / "interaction_atlas_seed42"
    if not (interaction_atlas / "atlas_selection.csv").is_file():
        run_command([sys.executable, "tools/build_interaction_atlas.py", "--project-root", str(root), "--seed", "42", "--output", str(interaction_atlas)], root, logs / "interaction_atlas.log")

    for seed in SEEDS:
        evals = [root / f"runs/current/input_{variant}_fold0_seed{seed}/final_validation/frame_metrics.csv" for variant in ("noisy", "denoised")]
        present = [path.is_file() for path in evals]
        if any(present) and not all(present):
            raise RuntimeError(f"Partial input evaluation for seed {seed}; preserve existing output and resolve explicitly: {evals}")
        if not all(present):
            run_command([sys.executable, "tools/run_input_factorial.py", "--project-root", str(root), "--mode", "evaluate", "--seeds", str(seed), "--epochs", "60", "--device", args.device, "--save-predictions"], root, logs / f"input_seed{seed}.log")
    input_report = safe_resolve(root, args.input_report) if args.input_report else staging / "input_factorial_report"
    if not (input_report / "paired_input_gains_summary.csv").is_file():
        run_command([sys.executable, "tools/summarize_input_factorial.py", "--project-root", str(root), "--seeds", *map(str, SEEDS), "--output", str(input_report)], root, logs / "input_summary.log")
    input_atlas = staging / "input_atlas_seed42"
    if not (input_atlas / "atlas_selection.csv").is_file():
        run_command([sys.executable, "tools/build_input_factorial_atlas.py", "--project-root", str(root), "--seed", "42", "--output", str(input_atlas)], root, logs / "input_atlas.log")
    state = {"stage12_report": str(stage12), "interaction_report": str(interaction), "interaction_atlas": str(interaction_atlas), "input_report": str(input_report), "input_atlas": str(input_atlas)}
    write_json(output / "evaluation_sources.json", state)
    return state


def load_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise RuntimeError(f"Image decode failed: {path}")
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    maximum = float(np.iinfo(array.dtype).max) if np.issubdtype(array.dtype, np.integer) else max(float(np.nanmax(array)), 1.0)
    return np.clip(array.astype(np.float32) / maximum, 0.0, 1.0)


def resolve_column(row: pd.Series, root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value and value.lower() != "nan":
            return safe_resolve(root, value)
    return None


def labelled_canvas(images: list[tuple[str, np.ndarray]], title: str, metrics: str = "") -> np.ndarray:
    if not images:
        return np.full((540, 960, 3), 255, np.uint8)
    height = max(image.shape[0] for _, image in images)
    resized = []
    for label, image in images:
        width = max(1, round(image.shape[1] * height / image.shape[0]))
        tile = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        tile = np.repeat(tile[..., None], 3, axis=2) if tile.ndim == 2 else tile
        resized.append((label, tile))
    header, footer = 70, 58
    canvas = np.full((height + header + footer, sum(tile.shape[1] for _, tile in resized), 3), 255, np.uint8)
    x = 0
    for label, tile in resized:
        canvas[header:header+height, x:x+tile.shape[1]] = np.round(np.clip(tile, 0, 1) * 255).astype(np.uint8)
        cv2.putText(canvas, label, (x+12, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        x += tile.shape[1]
    cv2.putText(canvas, title[:130], (12, height+header+25), cv2.FONT_HERSHEY_SIMPLEX, .55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, metrics[:180], (12, height+header+49), cv2.FONT_HERSHEY_SIMPLEX, .45, (50, 50, 50), 1, cv2.LINE_AA)
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def denoise_metrics(noisy: np.ndarray, denoised: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    common_h = min(noisy.shape[0], denoised.shape[0], clean.shape[0])
    common_w = min(noisy.shape[1], denoised.shape[1], clean.shape[1])
    noisy, denoised, clean = noisy[:common_h, :common_w], denoised[:common_h, :common_w], clean[:common_h, :common_w]
    return {
        "psnr_noisy": psnr(noisy, clean), "psnr": psnr(denoised, clean),
        "psnr_gain_db": psnr(denoised, clean) - psnr(noisy, clean),
        "ssim_noisy": ssim(noisy, clean), "ssim": ssim(denoised, clean),
        "rmse_noisy": rmse(noisy, clean), "rmse": rmse(denoised, clean),
        "epi_noisy": edge_preservation_index(noisy, clean), "epi": edge_preservation_index(denoised, clean),
        "reference_edge_mae_noisy": reference_edge_mae(noisy, clean),
        "reference_edge_mae": reference_edge_mae(denoised, clean),
    }


def choose_frames(table: pd.DataFrame, dataset: str, group_id: str) -> list[tuple[str, str]]:
    part = table[(table["dataset"].astype(str) == dataset) & (table["group_id"].astype(str) == group_id)].sort_values("sample_id")
    ids = part["sample_id"].astype(str).tolist()
    if not ids:
        return []
    chosen = [(ids[0], "first")]
    if dataset.lower() == "pku37" and len(ids) > 2:
        midpoint = next((sample for sample in ids if re.search(r"(?:f|_)?25$", sample, re.I)), ids[len(ids)//2])
        chosen.extend([(midpoint, "middle_or_f25"), (ids[-1], "last")])
    return list(dict.fromkeys(chosen))


def materialize_stage1(root: Path, report: Path, output: Path, missing: list[dict[str, Any]]) -> pd.DataFrame:
    index_path = report / "predictions_index.csv"
    manifest_path = report / "non_test_inference_manifest.csv"
    if not index_path.is_file() or not manifest_path.is_file():
        missing.append({"family": "stage1", "asset": "full non-test prediction index/manifest", "status": "MISSING", "impact": "all-position triptychs unavailable"})
        return pd.DataFrame()
    index = pd.read_csv(index_path, dtype=str).fillna("")
    index = index[index["experiment"] == "S1-Denoise"].copy()
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    keys = [key for key in ("sample_id", "group_id", "dataset") if key in index and key in manifest]
    merged = index.merge(manifest, on=keys, how="left", suffixes=("", "_manifest"))
    metric_rows, selection_rows = [], []
    for (dataset, group_id), _ in merged.groupby(["dataset", "group_id"], sort=True):
        for sample_id, role in choose_frames(merged, str(dataset), str(group_id)):
            selection_rows.append({"dataset": dataset, "group_id": group_id, "sample_id": sample_id, "selection_role": role, "selection_rule": "first for every non-test position; PKU37 also f25/middle and last"})
    selection = pd.DataFrame(selection_rows).drop_duplicates(["dataset", "group_id", "sample_id"])
    selection.to_csv(output / "stage1_denoising" / "stage1_triptych_selection.csv", index=False, encoding="utf-8-sig")
    selected_ids = set(selection["sample_id"].astype(str))
    for _, row in merged.iterrows():
        sample_id, dataset, group_id = str(row["sample_id"]), str(row["dataset"]), str(row["group_id"])
        prediction = safe_resolve(report, row.get("denoised", ""))
        sample_root = prediction.parent if prediction else None
        float_path = sample_root / "raw_outputs_float32.npz" if sample_root else None
        noisy_path = resolve_column(row, root, ("image_path", "original_path", "noisy_path", "source_image"))
        clean_path = resolve_column(row, root, ("clean_path", "target_path", "clean_image_path"))
        if not noisy_path or not noisy_path.is_file() or not clean_path or not clean_path.is_file() or not float_path or not float_path.is_file():
            missing.append({"family": "stage1", "dataset": dataset, "group_id": group_id, "sample_id": sample_id, "asset": "noisy/clean/float prediction", "status": "MISSING", "impact": "metrics or triptych unavailable"})
            continue
        noisy, clean = read_gray(noisy_path), read_gray(clean_path)
        with np.load(float_path, allow_pickle=False) as cache:
            key = "denoised_clipped" if "denoised_clipped" in cache.files else "denoised"
            denoised = np.asarray(cache[key], dtype=np.float32)
        metrics = {"dataset": dataset, "group_id": group_id, "sample_id": sample_id, **denoise_metrics(noisy, denoised, clean)}
        metric_rows.append(metrics)
        if sample_id not in selected_ids:
            continue
        target = output / "stage1_denoising" / dataset / "all_positions" / group_id / sample_id
        target.mkdir(parents=True, exist_ok=True)
        write_gray(target / "noisy.png", noisy); write_gray(target / "clean.png", clean); write_gray(target / "denoised.png", denoised)
        write_gray(target / "noisy_clean_residual.png", np.abs(noisy-clean)); write_gray(target / "denoised_clean_residual.png", np.abs(denoised-clean))
        line = f"PSNR {metrics['psnr']:.2f} dB | SSIM {metrics['ssim']:.3f} | RMSE {metrics['rmse']:.4f} | EPI {metrics['epi']:.3f} | edge MAE {metrics['reference_edge_mae']:.4f}"
        contact = labelled_canvas([("Noisy", noisy), ("Clean", clean), ("Denoised", denoised)], f"{dataset} | {group_id} | {sample_id}", line)
        write_rgb(target / "noisy_clean_denoised_triptych.png", contact.astype(np.float32)/255.0)
        write_json(target / "metrics.json", metrics)
    frames = pd.DataFrame(metric_rows)
    if not frames.empty:
        frames.to_csv(output / "metrics" / "stage1_metrics_frame.csv", index=False, encoding="utf-8-sig")
        numeric = frames.select_dtypes(include=[np.number]).columns.tolist()
        positions = frames.groupby(["dataset", "group_id"], as_index=False)[numeric].mean()
        positions["n_frames"] = frames.groupby(["dataset", "group_id"]).size().values
        positions.to_csv(output / "metrics" / "stage1_metrics_position.csv", index=False, encoding="utf-8-sig")
        datasets = positions.groupby("dataset", as_index=False)[numeric].mean()
        datasets["n_positions"] = positions.groupby("dataset").size().values
        datasets.to_csv(output / "metrics" / "stage1_metrics_dataset.csv", index=False, encoding="utf-8-sig")
    return frames


def rgb_overlay(noisy: np.ndarray, layer: np.ndarray, vessel: np.ndarray) -> np.ndarray:
    base = np.repeat(noisy[..., None], 3, axis=2).astype(np.float32)
    layer = layer > .5; vessel = vessel > .5
    base[layer] = .72 * base[layer] + .28 * np.array((0.0, 1.0, 0.0), np.float32)
    base[vessel] = .40 * base[vessel] + .60 * np.array((1.0, 0.0, 0.0), np.float32)
    return np.clip(base, 0, 1)


def error_map(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    pred, true = pred > .5, true > .5
    image = np.zeros((*pred.shape, 3), np.float32)
    image[pred & true] = (1, 1, 1)
    image[pred & ~true] = (0, 0, 1)  # blue FP
    image[~pred & true] = (1, .55, 0)  # orange FN
    return image


def materialize_stage2(report: Path, output: Path, missing: list[dict[str, Any]]) -> pd.DataFrame:
    frame_path = report / "segmentation_per_frame.csv"
    if not frame_path.is_file():
        missing.append({"family": "stage2", "asset": "segmentation_per_frame.csv", "status": "MISSING", "impact": "ablation table/atlas unavailable"})
        return pd.DataFrame()
    frames = pd.read_csv(frame_path)
    frames.to_csv(output / "metrics" / "stage2_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    group_path = report / "segmentation_per_group.csv"
    if group_path.is_file(): shutil.copy2(group_path, output / "metrics" / "stage2_ablation_by_position.csv")
    selected = []
    e3b = frames[frames["experiment"] == "E3b"]
    for (dataset, group_id), _ in e3b.groupby(["dataset", "group_id"]):
        selected.extend((dataset, group_id, sample, role) for sample, role in choose_frames(e3b, str(dataset), str(group_id)))
    selection = pd.DataFrame(selected, columns=["dataset", "group_id", "sample_id", "selection_role"])
    selection.to_csv(output / "stage2_ablation" / "atlas_selection.csv", index=False, encoding="utf-8-sig")
    contacts = output / "stage2_ablation" / "stage2_ablation_contact"; contacts.mkdir(parents=True, exist_ok=True)
    for row in selection.itertuples(index=False):
        method_tiles = []
        for alias in STAGE2_RUNS:
            source = report / "validation" / alias / "predictions" / str(row.dataset)
            required = {
                "input.png": source / f"{row.sample_id}_noisy.png",
                "layer_gt.png": source / f"{row.sample_id}_layer_gt.png",
                "vessel_gt.png": source / f"{row.sample_id}_vessel_gt.png",
                "layer_pred.png": source / f"{row.sample_id}_layer_mask.png",
                "vessel_pred.png": source / f"{row.sample_id}_vessel_mask.png",
            }
            target = output / "stage2_ablation" / str(row.group_id) / str(row.sample_id) / alias
            target.mkdir(parents=True, exist_ok=True)
            if any(not path.is_file() for path in required.values()):
                missing.append({"family": "stage2", "alias": alias, "sample_id": row.sample_id, "asset": "matched prediction set", "status": "MISSING", "impact": "contact incomplete"})
                continue
            for name, path in required.items(): shutil.copy2(path, target / name)
            noisy = load_image(required["input.png"]); layer_gt = load_image(required["layer_gt.png"]); vessel_gt = load_image(required["vessel_gt.png"])
            layer = load_image(required["layer_pred.png"]); vessel = load_image(required["vessel_pred.png"])
            write_rgb(target / "gt_overlay.png", rgb_overlay(noisy, layer_gt, vessel_gt))
            overlay = rgb_overlay(noisy, layer, vessel); write_rgb(target / "pred_overlay.png", overlay)
            write_rgb(target / "vessel_error_tp_fp_fn.png", error_map(vessel, vessel_gt))
            method_tiles.append((alias, overlay))
            metric_row = frames[(frames["experiment"] == alias) & (frames["sample_id"].astype(str) == str(row.sample_id))]
            write_json(target / "metrics.json", metric_row.iloc[0].dropna().to_dict() if len(metric_row) else {"status": "MISSING"})
        if method_tiles:
            contact = labelled_canvas(method_tiles, f"Stage2 matched comparison | {row.group_id} | {row.sample_id}", "Threshold 0.5, P0, same sample and display window")
            write_rgb(contacts / f"{row.group_id}_{row.sample_id}.png", contact.astype(np.float32)/255.0)
    paired = report / "paired_comparisons.csv"
    if paired.is_file(): shutil.copy2(paired, output / "metrics" / "stage2_paired_comparisons.csv")
    return frames


def postprocess_assets(report: Path, output: Path, missing: list[dict[str, Any]]) -> None:
    selection_path = output / "stage2_ablation" / "atlas_selection.csv"
    if not selection_path.is_file():
        return
    selection = pd.read_csv(selection_path)
    rows = []
    for row in selection.itertuples(index=False):
        validation = report / "validation" / "E3b" / "predictions" / str(row.dataset)
        full = report / "predictions" / "E3b" / str(row.dataset) / "val" / str(row.group_id) / str(row.sample_id)
        noisy_path = validation / f"{row.sample_id}_noisy.png"
        layer_gt_path = validation / f"{row.sample_id}_layer_gt.png"
        vessel_gt_path = validation / f"{row.sample_id}_vessel_gt.png"
        p0_layer_path = validation / f"{row.sample_id}_layer_mask.png"
        p0_vessel_path = validation / f"{row.sample_id}_vessel_mask.png"
        required = (noisy_path, layer_gt_path, vessel_gt_path, p0_layer_path, p0_vessel_path)
        if any(not path.is_file() for path in required):
            missing.append({"family": "postprocessing", "sample_id": row.sample_id, "asset": "E3b base prediction/GT", "status": "MISSING", "impact": "P0-P3 page unavailable"})
            continue
        noisy, layer_gt, vessel_gt, p0_layer, p0_vessel = map(load_image, required)
        valid = np.ones_like(p0_layer, dtype=bool)
        p1, _ = clean_layer_mask(p0_layer > .5, valid)
        p2a, _ = regularize_lower_boundary(p1, valid, smoothness=0.0)
        p2b, _ = regularize_lower_boundary(p2a, valid, smoothness=2.0, max_displacement=8)
        p3, _ = hard_contain_vessel(p0_vessel > .5, p2b, valid)
        variants = {
            "P0": (p0_layer > .5, p0_vessel > .5), "P1": (p1, p0_vessel > .5),
            "P2a": (p2a, p0_vessel > .5), "P2b": (p2b, p0_vessel > .5),
            "P3": (p2b, p3), "GT-layer-oracle": (layer_gt > .5, (p0_vessel > .5) & (layer_gt > .5)),
        }
        sample_root = output / "postprocessing" / str(row.group_id) / str(row.sample_id)
        contacts = []
        previous_layer, previous_vessel = None, None
        for name, (layer, vessel) in variants.items():
            target = sample_root / name; target.mkdir(parents=True, exist_ok=True)
            write_gray(target / "layer_mask.png", layer.astype(np.float32)); write_gray(target / "vessel_mask.png", vessel.astype(np.float32))
            overlay = rgb_overlay(noisy, layer, vessel); write_rgb(target / "pred_overlay.png", overlay); contacts.append((name, overlay))
            change = np.zeros((*layer.shape, 3), np.float32)
            if previous_layer is not None:
                change[layer & ~previous_layer] = (0, 1, 0); change[previous_layer & ~layer] = (1, 0, 1)
                change[vessel & ~previous_vessel] = (1, 1, 0); change[previous_vessel & ~vessel] = (0, 0, 1)
            write_rgb(target / "change_from_previous.png", change)
            metrics = {f"layer_{key}": value for key, value in binary_metrics(layer[valid], (layer_gt > .5)[valid]).items()}
            metrics.update({f"vessel_{key}": value for key, value in binary_metrics(vessel[valid], (vessel_gt > .5)[valid]).items()})
            upper, lower, thickness = layer_boundary_mae(layer, layer_gt > .5, 1.0)
            metrics.update({"upper_boundary_mae_px": upper, "lower_boundary_mae_px": lower, "thickness_mae_px": thickness, "label": "POSTPROCESSING — NOT NETWORK TRAINING GAIN"})
            write_json(target / "metrics.json", metrics)
            rows.append({"group_id": row.group_id, "sample_id": row.sample_id, "variant": name, **metrics})
            previous_layer, previous_vessel = layer, vessel
        contact = labelled_canvas(contacts, f"POSTPROCESSING — NOT NETWORK TRAINING GAIN | {row.group_id} | {row.sample_id}")
        write_rgb(sample_root / "postprocessing_contact.png", contact.astype(np.float32)/255.0)
    write_table(output / "metrics" / "postprocessing_metrics.csv", rows)


def text_from_office(path: Path) -> str:
    if path.suffix.lower() not in {".xlsx", ".docx"}:
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return ""
    try:
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.endswith(".xml")]
            return "\n".join(archive.read(name).decode("utf-8", errors="ignore") for name in members)
    except (OSError, zipfile.BadZipFile):
        return ""


def literature_table(root: Path, output: Path) -> pd.DataFrame:
    methods = ["TCFL-DnCNN", "PSCAT", "PSB-DSN", "OCTNet", "N2N-DnCNN", "BM3D", "NLM"]
    search_roots = [root / name for name in ("reports", "docs", "references") if (root / name).exists()]
    candidates = []
    for search_root in search_roots:
        for directory, names, files in os.walk(search_root):
            names[:] = [name for name in names if name.lower() not in FORBIDDEN_PARTS]
            for name in files:
                path = Path(directory) / name
                if path.suffix.lower() in {".md", ".csv", ".txt", ".xlsx", ".docx"} and not forbidden(path):
                    candidates.append(path)
    rows = []
    for method in methods:
        sources = []
        for path in candidates:
            if method.lower() in text_from_office(path).lower():
                sources.append(path.relative_to(root).as_posix())
        rows.append({
            "paper": "SOURCE REQUIRED", "year": "", "method": method, "journal": "",
            "training_type": "", "task_type": "denoising+super-resolution" if method == "PSCAT" else "denoising",
            "train_dataset": "", "test_dataset": "", "split": "", "cross_domain": "",
            "resolution": "", "normalization": "", "PSNR": "", "SSIM": "", "RMSE": "",
            "EPI": "", "CNR": "", "ENL": "", "source": ";".join(sources) or "SOURCE REQUIRED",
            "protocol_match": "unverified", "notes": "Do not rank until the cited protocol and value are independently verified" + ("; PSCAT kept separate from pure denoising" if method == "PSCAT" else ""),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metrics" / "literature_denoising_comparison.csv", index=False, encoding="utf-8-sig")
    return frame


def debug_archive(root: Path, output: Path, missing: list[dict[str, Any]]) -> None:
    timeline = [
        ["NaN/full-layer collapse", "historical", "Observed non-finite loss and high-recall/low-precision vessel expansion", "diagnostic only"],
        ["softplus/FP32 repair", "code repair", "Stable stroma-negative softplus and FP32 reductions", "implementation confirmed; result separate"],
        ["freeze protection", "code repair", "Stage1 denoising isolation and drift audit", "implementation confirmed"],
        ["E0", "overfit check", "Training-position memorization check", "NOT GENERALIZATION"],
        ["ROI supervision", "E3-current", "Layer-ROI segmentation supervision", "current protocol history"],
        ["outside BCE", "E3b", "Layer-exterior vessel negative supervision", "current matched contrast"],
        ["same-protocol retraining", "current controls", "E1/E3/E3b/noD2S share current data identity", "requires protocol hashes"],
        ["D→S diagnosis", "dependency/ablation", "same-checkpoint disable vs independently retrained noD2S", "do not conflate"],
    ]
    write_table(output / "stage2_debug" / "debug_timeline.csv", [dict(zip(("event", "category", "finding", "evidence_scope"), row)) for row in timeline])
    (output / "stage2_debug" / "debug_summary.md").write_text("# Stage 2 debugging timeline\n\nHistorical failures explain engineering decisions and are excluded from the current matched-protocol ranking. E0 is an overfit check, not generalization. Same-checkpoint D→S disabling measures dependency; E3b-noD2S is the retraining contrast.\n", encoding="utf-8")
    e0 = root / "runs/current/stage2_overfit_safe_fold0"
    if (e0 / "history.csv").is_file():
        shutil.copy2(e0 / "history.csv", output / "stage2_debug" / "e0_overfit" / "history.csv")
        (output / "stage2_debug" / "e0_overfit" / "README.md").write_text("# OVERFIT CHECK — NOT GENERALIZATION\n", encoding="utf-8")
    else:
        missing.append({"family": "E0", "asset": "checkpoint/history/predictions", "status": "MISSING", "impact": "E0 images cannot be reconstructed; no retraining performed"})


def copy_factorial(root: Path, output: Path, sources: dict[str, Path], missing: list[dict[str, Any]]) -> None:
    mappings = {
        "joint": (sources.get("interaction_report"), {
            "position_metrics_long.csv": "joint_absolute_metrics.csv", "paired_gains_by_position.csv": "joint_by_position.csv",
            "paired_gains_by_seed.csv": "joint_by_seed.csv", "paired_gains_summary.csv": "joint_paired_effects.csv",
        }),
        "input_experiment": (sources.get("input_report"), {
            "position_metrics_long.csv": "input_absolute_metrics.csv", "paired_input_gains_by_position.csv": "input_paired_gains_by_position.csv",
            "paired_input_gains_by_seed.csv": "input_paired_gains_by_seed.csv", "paired_input_gains_summary.csv": "input_paired_gains_summary.csv",
        }),
    }
    for family, (source, files) in mappings.items():
        target = output / family; target.mkdir(parents=True, exist_ok=True)
        if source is None or not source.is_dir():
            missing.append({"family": family, "asset": "formal summary directory", "status": "MISSING", "impact": "no absolute or paired result"})
            continue
        for source_name, target_name in files.items():
            path = source / source_name
            if path.is_file(): shutil.copy2(path, output / "metrics" / target_name)
            else: missing.append({"family": family, "asset": source_name, "status": "MISSING", "impact": f"{target_name} unavailable"})
    for family, source_key in (("joint", "interaction_atlas"), ("input_experiment", "input_atlas")):
        source = sources.get(source_key)
        if source and source.is_dir():
            shutil.copytree(source, output / family / "atlas_seed42", dirs_exist_ok=True)
        else:
            missing.append({"family": family, "asset": "seed42 fixed atlas", "status": "MISSING", "impact": "qualitative comparison unavailable"})


def decode_rgb(path: Path) -> np.ndarray | None:
    if not path.is_file(): return None
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None


def presentation_figure(path: Path, title: str, candidates: list[Path], note: str) -> None:
    width, height = 1920, 1080
    canvas = np.full((height, width, 3), 255, np.uint8)
    canvas[:125] = (23, 54, 93)
    cv2.putText(canvas, title, (65, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.45, (255, 255, 255), 3, cv2.LINE_AA)
    images = [(candidate, decode_rgb(candidate)) for candidate in candidates]
    images = [(p, image) for p, image in images if image is not None][:6]
    if not images:
        cv2.putText(canvas, "MISSING", (760, 515), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (192, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(canvas, note[:150], (170, 650), cv2.FONT_HERSHEY_SIMPLEX, .65, (55, 55, 55), 2, cv2.LINE_AA)
    else:
        columns = 3 if len(images) > 2 else len(images)
        rows = int(np.ceil(len(images) / columns))
        tile_w, tile_h = 560, 360
        start_x, start_y = 80, 165
        for index, (source, image) in enumerate(images):
            row, col = divmod(index, columns)
            scale = min(tile_w / image.shape[1], tile_h / image.shape[0])
            resized = cv2.resize(image, (max(1, int(image.shape[1]*scale)), max(1, int(image.shape[0]*scale))), interpolation=cv2.INTER_AREA)
            x, y = start_x + col*610, start_y + row*405
            canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
            cv2.putText(canvas, source.stem[:48], (x, min(y+resized.shape[0]+28, 1025)), cv2.FONT_HERSHEY_SIMPLEX, .48, (30,30,30), 1, cv2.LINE_AA)
        cv2.putText(canvas, note[:170], (70, 1040), cv2.FONT_HERSHEY_SIMPLEX, .55, (60,60,60), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", bgr)
    if not success: raise RuntimeError(f"Cannot encode figure: {path}")
    encoded.tofile(str(path))
    try:
        from PIL import Image
        Image.fromarray(canvas).save(path.with_suffix(".pdf"), "PDF", resolution=150.0)
    except Exception as exc:
        (path.with_suffix(".pdf.missing.txt")).write_text(f"PDF export unavailable: {exc}\n", encoding="utf-8")


def make_figures(output: Path) -> None:
    stage1 = list((output / "stage1_denoising").rglob("noisy_clean_denoised_triptych.png"))
    by_dataset = {dataset: [p for p in stage1 if dataset.lower() in str(p).lower()][:3] for dataset in ("PKU37", "Duke17", "Duke28")}
    collapse = list((output / "stage2_debug" / "collapse").rglob("*contact*.png"))
    e0 = list((output / "stage2_debug" / "e0_overfit").rglob("*.png"))
    stage2_contacts = list((output / "stage2_ablation" / "stage2_ablation_contact").glob("*.png"))
    post = list((output / "postprocessing").rglob("postprocessing_contact.png"))
    joint = list((output / "joint" / "atlas_seed42").glob("*.png"))
    input_images = list((output / "input_experiment" / "atlas_seed42").glob("*.png"))
    specs = [
        ("01_stage1_pku37_triptychs.png", "Stage 1 — PKU37 denoising", by_dataset["PKU37"], "Same display window; fixed non-test position selection."),
        ("02_stage1_duke17_triptychs.png", "Stage 1 — Duke17 denoising", by_dataset["Duke17"], "Reported separately from PKU37 and Duke28."),
        ("03_stage1_duke28_triptychs.png", "Stage 1 — Duke28 denoising", by_dataset["Duke28"], "Reported separately from PKU37 and Duke17."),
        ("04_stage1_metrics_literature.png", "Denoising metrics and literature protocol", by_dataset["PKU37"][:1], "Literature values remain SOURCE REQUIRED unless verified."),
        ("05_collapse_failure.png", "Historical whole-layer vessel collapse", collapse, "Historical debugging evidence; excluded from current ranking."),
        ("06_e0_overfit.png", "E0 overfit check", e0, "OVERFIT CHECK — NOT GENERALIZATION"),
        ("07_stage2_design.png", "Stage 2 ablation design", stage2_contacts[:1], "E3b−E3-current isolates outside BCE; E3b−E1-current is multifactor."),
        ("08_stage2_ablation_metrics.png", "Stage 2 matched-protocol metrics", stage2_contacts[:3], "Best checkpoint per run; fixed threshold 0.5; position-first interpretation."),
        ("09_stage2_ablation_visual.png", "Stage 2 matched qualitative comparison", stage2_contacts[:3], "Same sample IDs and display parameters across methods."),
        ("10_postprocessing.png", "P0–P3 anatomical postprocessing", post[:3], "POSTPROCESSING — NOT NETWORK TRAINING GAIN"),
        ("11_joint_design_and_metrics.png", "Joint 2×2 factorial", joint[:2], "Fixed-final last.pth, threshold 0.5, P0; three seeds."),
        ("12_joint_visual.png", "Joint qualitative comparison", joint[:3], "Seed 42 fixed atlas; no method-specific sample selection."),
        ("13_input_experiment_metrics.png", "I-NOISY vs I-DENOISED", input_images[:2], "Epoch-60 last.pth, threshold 0.5, P0; three seeds."),
        ("14_input_experiment_visual.png", "Input experiment qualitative comparison", input_images[:3], "Seed 42 fixed atlas; inspect FP/FN and vessel boundary changes."),
        ("15_current_evidence_and_next_step.png", "Current evidence and next step", stage2_contacts[:1] + joint[:1] + input_images[:1], "Only claims supported by present validation assets are retained; test remains sealed."),
    ]
    for name, title, candidates, note in specs:
        presentation_figure(output / "figures" / name, title, candidates, note)


def write_documents(output: Path, sources: dict[str, Path], missing: list[dict[str, Any]]) -> None:
    content = """# SABIDS-Net 阶段性汇报内容

本归档以 validation 和非封存位置的推理展示为范围，不读取封存 test、不重新训练、不重新选阈值。

汇报顺序建议：Stage 1 三数据集降噪 → 历史整层血管退化与修复 → 当前 Stage 2 同协议消融 → P0–P3 后处理 → Joint 2×2 → I-NOISY/I-DENOISED → 证据边界与下一步。

后处理、oracle 和同 checkpoint 关闭模块只能作为推理或依赖诊断，不能表述为网络训练增益。Joint 与输入实验仅在固定终轮、阈值 0.5、P0、位置优先归约且三个 seed 完整时进入主要结论。
"""
    technical = f"""# Technical notes

- Tool: `{VERSION}`
- Stage12 source: `{sources.get('stage12_report', 'MISSING')}`
- Joint source: `{sources.get('interaction_report', 'MISSING')}`
- Input source: `{sources.get('input_report', 'MISSING')}`
- Layer overlay RGB=(0,255,0), alpha=0.28; vessel RGB=(255,0,0), alpha=0.60, applied second.
- Fixed thresholds: layer=0.5, vessel=0.5. Stage2 uses metadata best; J uses fixed-final last; input uses epoch-60 last.
- Repeated frames are reduced within group_id before seed summaries.
"""
    limitations = "# Limitations\n\n" + "\n".join(f"- {row.get('family','')} / {row.get('asset','')}: {row.get('impact','')}" for row in missing) + "\n- Test assets were not opened; this is not a final test report.\n"
    guide = """# Analysis guide

1. Verify `audit/run_path_audit.csv` and `audit/checkpoint_audit.csv` before comparing metrics.
2. Keep historical debug, current Stage2 ablation, dependency diagnostics, oracle and postprocessing separate.
3. Use group_id as the primary unit; do not treat repeat frames as independent cases.
4. Check Dice together with Precision/Recall, ROI FP/FN, outside fraction, area fraction and boundary metrics.
5. Joint claims require all four arms and three seeds; input claims require both arms and three seeds.
6. Inspect fixed images for oversmoothing, whole-layer vessel FP, FN and structural hallucination.
7. Do not use test, threshold calibration, P3, oracle or selective samples to manufacture a gain.
"""
    (output / "PRESENTATION_CONTENT.md").write_text(content, encoding="utf-8")
    (output / "TECHNICAL_NOTES.md").write_text(technical, encoding="utf-8")
    (output / "LIMITATIONS.md").write_text(limitations, encoding="utf-8")
    (output / "ANALYSIS_GUIDE.md").write_text(guide, encoding="utf-8")
    first = [
        "PRESENTATION_CONTENT.md", "figures/01_stage1_pku37_triptychs.png", "figures/05_collapse_failure.png",
        "figures/08_stage2_ablation_metrics.png", "figures/09_stage2_ablation_visual.png", "figures/10_postprocessing.png",
        "figures/11_joint_design_and_metrics.png", "figures/12_joint_visual.png", "figures/13_input_experiment_metrics.png",
        "figures/14_input_experiment_visual.png", "SABIDS_presentation_results.xlsx", "audit/run_path_audit.csv",
        "audit/checkpoint_audit.csv", "audit/missing_assets.csv", "LIMITATIONS.md",
    ]
    (output / "README_FIRST.md").write_text("# Read first\n\n" + "\n".join(f"{index}. `{name}`" for index, name in enumerate(first, 1)) + "\n", encoding="utf-8")


def workbook_inputs(output: Path) -> None:
    target = output / "workbook_inputs"; target.mkdir(exist_ok=True)
    mappings = {
        "dataset.csv": output / "metrics" / "stage1_metrics_dataset.csv",
        "denoising_metrics.csv": output / "metrics" / "stage1_metrics_frame.csv",
        "literature_comparison.csv": output / "metrics" / "literature_denoising_comparison.csv",
        "debug.csv": output / "stage2_debug" / "debug_timeline.csv",
        "e0.csv": output / "stage2_debug" / "e0_overfit" / "history.csv",
        "stage2_ablation.csv": output / "metrics" / "stage2_ablation_metrics.csv",
        "postprocessing.csv": output / "metrics" / "postprocessing_metrics.csv",
        "joint.csv": output / "metrics" / "joint_paired_effects.csv",
        "input_experiment.csv": output / "metrics" / "input_paired_gains_summary.csv",
        "position_results.csv": output / "metrics" / "stage1_metrics_position.csv",
        "seed_results.csv": output / "metrics" / "joint_by_seed.csv",
        "image_index.csv": output / "audit" / "prediction_asset_audit.csv",
        "missing_assets.csv": output / "audit" / "missing_assets.csv",
    }
    placeholder = pd.DataFrame([{"status": "MISSING", "reason": "source asset unavailable; see audit/missing_assets.csv"}])
    for name, source in mappings.items():
        if source.is_file(): shutil.copy2(source, target / name)
        else: placeholder.to_csv(target / name, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"conclusion": "Use only validation-supported claims; test remains sealed."}]).to_csv(target / "conclusions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"limitation": line[2:]} for line in (output / "LIMITATIONS.md").read_text(encoding="utf-8").splitlines() if line.startswith("- ")]).to_csv(target / "limitations.csv", index=False, encoding="utf-8-sig")


def validate_archive(output: Path) -> dict[str, Any]:
    failures, csvs, images = [], [], []
    for path in output.rglob("*.csv"):
        if forbidden(path): failures.append(f"forbidden CSV path: {path}"); continue
        try:
            frame = pd.read_csv(path)
            infinite = int(np.isinf(frame.select_dtypes(include=[np.number]).to_numpy(dtype=float, na_value=np.nan)).sum()) if not frame.empty else 0
            csvs.append({"path": path.relative_to(output).as_posix(), "rows": len(frame), "columns": len(frame.columns), "infinite_values": infinite})
            if infinite: failures.append(f"infinite values: {path}")
        except Exception as exc: failures.append(f"CSV failure {path}: {exc}")
    for path in output.rglob("*.png"):
        image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None: failures.append(f"decode failure: {path}")
        else: images.append({"path": path.relative_to(output).as_posix(), "height": image.shape[0], "width": image.shape[1]})
    return {"status": "passed" if not failures else "failed", "failures": failures, "csv": csvs, "images": images, "test_assets_opened": 0}


def make_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or "packages" in path.parts: continue
        rows.append({"relative_path": path.relative_to(output).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path), "contains_image": path.suffix.lower() in {".png", ".pdf"}, "test_content": False})
    write_json(output / "MANIFEST.json", {"tool_version": VERSION, "scope": "validation and non-sealed non-test inference only", "files": rows, "test_assets_opened": 0})


def package_archive(output: Path) -> dict[str, Any]:
    packages = output / "packages"; packages.mkdir(exist_ok=True)
    stamp = output.name.removeprefix("presentation_archive_")
    full = packages / f"SABIDS_presentation_full_{stamp}.tar.gz"
    gpt = packages / f"SABIDS_presentation_for_GPT_{stamp}.zip"
    existing = [path for path in (full, gpt) if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite presentation packages: "
            + ", ".join(str(path) for path in existing)
        )
    allowed = [path for path in output.rglob("*") if path.is_file() and "packages" not in path.parts and not forbidden(path) and path.suffix.lower() not in {".pth", ".npy", ".npz"}]
    with tarfile.open(full, "w:gz") as archive:
        for path in allowed: archive.add(path, arcname=path.relative_to(output).as_posix(), recursive=False)
    with tarfile.open(full, "r:gz") as archive: archive.getmembers()
    gpt_files = [path for path in allowed if path.suffix.lower() in {".csv", ".xlsx", ".md", ".json"} or "figures" in path.parts or (path.suffix.lower() == ".png" and any(part in {"joint", "input_experiment", "stage2_ablation"} for part in path.parts))]
    with zipfile.ZipFile(gpt, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in gpt_files: archive.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(gpt) as archive:
        bad = archive.testzip()
        if bad: raise RuntimeError(f"GPT ZIP CRC failed: {bad}")
        gpt_count = len(archive.infolist())
    result = {
        "full": {"path": str(full.resolve()), "size_bytes": full.stat().st_size, "sha256": sha256(full), "file_count": len(allowed), "extract_check": "passed"},
        "gpt": {"path": str(gpt.resolve()), "size_bytes": gpt.stat().st_size, "sha256": sha256(gpt), "file_count": gpt_count, "crc_check": "passed"},
    }
    write_json(output / "package_summary.json", result)
    return result


def resolve_sources(root: Path, output: Path, args: argparse.Namespace) -> dict[str, Path]:
    state_path = output / "evaluation_sources.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    defaults = {
        "stage12_report": latest_dir(root / "runs" / "reports", "stage12_validation_*") if (root / "runs" / "reports").is_dir() else None,
        "interaction_report": root / "runs/interaction_factorial_report",
        "interaction_atlas": root / "runs/interaction_factorial_report/atlas",
        "input_report": root / "runs/input_factorial_report_v1",
        "input_atlas": root / "runs/input_factorial_report_v1/atlas_seed42",
    }
    explicit = {"stage12_report": args.stage12_report, "interaction_report": args.interaction_report, "input_report": args.input_report}
    resolved = {}
    for key in defaults:
        value = state.get(key) or explicit.get(key) or defaults[key]
        if value: resolved[key] = safe_resolve(root, value) if not isinstance(value, Path) else value.resolve()
    return resolved


def assemble(root: Path, output: Path, args: argparse.Namespace) -> dict[str, Any]:
    sources = resolve_sources(root, output, args)
    missing_path = output / "audit" / "missing_assets.csv"
    missing = pd.read_csv(missing_path).fillna("").to_dict("records") if missing_path.is_file() else []
    report = sources.get("stage12_report")
    if report and report.is_dir():
        materialize_stage1(root, report, output, missing)
        materialize_stage2(report, output, missing)
        postprocess_assets(report, output, missing)
    else:
        missing.append({"family": "stage1/stage2", "asset": "stage12 report", "status": "MISSING", "impact": "denoising and Stage2 outputs unavailable"})
    literature_table(root, output)
    debug_archive(root, output, missing)
    copy_factorial(root, output, sources, missing)
    if args.skip_workbooks:
        missing.append({"family": "workbook", "asset": "SABIDS_presentation_results.xlsx and topic workbooks", "status": "MISSING", "impact": "artifact runtime unavailable or workbook generation intentionally deferred; CSV sources retained"})
    write_table(missing_path, missing)
    make_figures(output)
    write_documents(output, sources, missing)
    workbook_inputs(output)
    validation = validate_archive(output)
    write_json(output / "missing_and_failure_checklist.json", validation)
    make_manifest(output)
    return {"sources": {key: str(value) for key, value in sources.items()}, "validation": validation["status"], "workbook_required": not args.skip_workbooks}


def prepare_output(root: Path, value: str | None, mode: str) -> Path:
    if value:
        output = safe_resolve(root, value)
        assert output is not None
    else:
        output = root / "runs" / f"presentation_archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if mode in {"audit", "all"}:
        if output.exists(): raise FileExistsError(f"Refusing to overwrite presentation archive: {output}")
        for name in ("audit", "metrics", "tables", "figures", "stage1_denoising", "stage2_debug", "stage2_ablation", "postprocessing", "joint", "input_experiment", "gpt_bundle"):
            (output / name).mkdir(parents=True, exist_ok=True)
    elif not output.is_dir():
        raise FileNotFoundError(f"Existing --output is required for mode={mode}: {output}")
    return output


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = prepare_output(root, args.output, args.mode)
    result: dict[str, Any] = {"tool_version": VERSION, "output": str(output), "mode": args.mode}
    if args.mode in {"audit", "all"}:
        result["audit"] = audit(root, output)
    if args.mode in {"evaluate", "all"}:
        result["evaluation"] = evaluate_missing(root, output, args)
    if args.mode in {"assemble", "all"}:
        result["assembly"] = assemble(root, output, args)
    if args.mode == "package":
        if not args.skip_workbooks and not (output / "SABIDS_presentation_results.xlsx").is_file():
            raise FileNotFoundError(
                "Workbook is required before packaging. Run tools/build_presentation_workbooks.mjs "
                "with the @oai/artifact-tool runtime, or explicitly use --skip-workbooks."
            )
        result["packages"] = package_archive(output)
    write_json(output / "presentation_archive_state.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
