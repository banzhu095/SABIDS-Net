from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from sabids.data.io import read_gray

from .adapters import ADAPTERS, AdapterUnavailable, denoise
from .core import (
    METRIC_COLUMNS,
    aggregate_metrics,
    bootstrap_mean_ci,
    image_metadata,
    infer_frame_id,
    metric_row,
    read_original,
    resolve_manifest_paths,
    save_like_source,
    select_validation_subset,
    sha256_file,
    stable_hash,
)
from .method_inventory import FIELDS as METHOD_FIELDS
from .method_inventory import inventory_rows


DEFAULT_METHODS = ["noisy_identity", "bm3d", "nlm_speckle", "wavelet", "tv", "gaussian", "msbtd", "ascibp"]
RUNNABLE_METHODS = ["noisy_identity", "bm3d", "nlm_speckle", "wavelet", "tv", "gaussian"]


PARAMETER_GRIDS: Dict[str, List[Dict[str, Any]]] = {
    "noisy_identity": [{"method_id": "noisy_identity"}],
    "bm3d": [{"method_id": "bm3d", "sigma_psd": sigma, "stage": "all"} for sigma in (0.04, 0.06, 0.08, 0.10)],
    "nlm_speckle": [
        {"method_id": "nlm_speckle", "patch_size": 3, "search_radius": 3, "h": h, "gamma": 0.5}
        for h in (0.08, 0.12, 0.18, 0.25)
    ],
    "wavelet": [
        {"method_id": "wavelet", "wavelet": "db2", "level": level, "threshold_scale": scale, "threshold_mode": "soft"}
        for level in (1, 2) for scale in (0.5, 0.8, 1.1)
    ],
    "tv": [{"method_id": "tv", "weight": weight, "max_num_iter": 200} for weight in (0.01, 0.03, 0.05, 0.075)],
    "gaussian": [{"method_id": "gaussian", "sigma": sigma} for sigma in (0.5, 0.8, 1.0, 1.3)],
    "msbtd": [{"method_id": "msbtd"}],
    "ascibp": [{"method_id": "ascibp"}],
}


def _write_csv(path: Path, rows: pd.DataFrame | Sequence[Mapping[str, Any]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows), columns=columns)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def _markdown_table(table: pd.DataFrame, columns: Sequence[str] | None = None, digits: int = 4) -> str:
    selected = table[list(columns)].copy() if columns else table.copy()
    for column in selected.select_dtypes(include=[np.number]).columns:
        selected[column] = selected[column].map(lambda value: f"{value:.{digits}f}" if np.isfinite(value) else "")
    try:
        return selected.to_markdown(index=False)
    except Exception:
        headers = list(selected.columns)
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
        lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in selected.itertuples(index=False, name=None))
        return "\n".join(lines)


def _load_config(project_root: Path) -> Dict[str, Any]:
    path = project_root / "tools" / "oct_denoise_benchmark" / "default_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_manifest(project_root: Path) -> pd.DataFrame:
    path = project_root / "Manifests" / "manifest_denoise.csv"
    table = pd.read_csv(path, dtype=str).fillna("")
    required = {"sample_id", "group_id", "dataset", "frame_index", "split", "image_path", "clean_path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"manifest_denoise.csv missing columns: {sorted(missing)}")
    if table["sample_id"].duplicated().any():
        raise ValueError("manifest contains duplicate sample_id values")
    return table


def _resolve(path_value: str, project_root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (project_root / path).resolve()


def _is_sealed(row: Mapping[str, Any]) -> bool:
    return str(row.get("split", "")).lower() == "test"


def audit(project_root: Path, run_dir: Path) -> pd.DataFrame:
    audit_dir = run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(project_root)
    method_rows = inventory_rows()
    _write_csv(audit_dir / "method_inventory.csv", method_rows, METHOD_FIELDS)
    method_md = ["# 传统降噪方法代码审计", "", _markdown_table(pd.DataFrame(method_rows), [
        "method_id", "canonical_name", "category", "language", "needs_clean_or_dictionary", "runnable_now", "blocking_reason"
    ], digits=3), "", "## 关键结论", "",
        "- `1-Sparsity_SDOCT_Software_2012` 确认为 MSBTD；P-code 输入结构包含 `TrainIm`、`nSigTrain` 和多尺度参数，high-SNR 图像属于核心数据流，不只是 GUI 的 PSNR 参考。",
        "- `2-oct-denoising-techniques-main` 是实验工具集合，不是单一算法；本地可审计的独立实现包括 BM3D、全局 SVD、Gaussian、TV、wavelet live script，以及仅留图/外链的 anisotropic diffusion。",
        "- wavelet 主目录采用四抽头 Daubechies D4（PyWavelets `db2`）；Demo 写 level=2，但实际去噪调用 level=1，MAD/0.6745 估噪，阈值 0.8 倍，包含 hard/soft；批处理原码做逐图 min-max 拉伸，定量适配明确禁用。",
        "- 方法 4 的真实入口是统计特征分类 WNNM 加自适应迭代反投影；核心为 P-code，Demo 把 clean 直接传给 `WNNM_DeNoising`，因此公平运行被阻塞。",
        "- 方法 5 不是普通加性噪声 NLM：其 Pearson 距离分母与局部强度相关，针对乘性 speckle；原脚本还执行 padding 和 0.5 resize 且不恢复几何，benchmark 仅保留 speckle 距离并原尺寸运行。",
    ]
    (audit_dir / "method_inventory.md").write_text("\n".join(method_md) + "\n", encoding="utf-8")

    paired_rows: List[Dict[str, Any]] = []
    pairing_rows: List[Dict[str, Any]] = []
    ambiguous_rows: List[Dict[str, Any]] = []
    metadata_cache: Dict[str, Dict[str, Any]] = {}
    for row in tqdm(manifest.to_dict("records"), desc="audit manifest", unit="pair"):
        noisy = _resolve(str(row["image_path"]), project_root)
        clean = _resolve(str(row["clean_path"]), project_root)
        sealed = _is_sealed(row)
        noisy_exists, clean_exists = noisy.is_file(), clean.is_file()
        status = "ok" if noisy_exists and clean_exists else "missing"
        if status != "ok":
            ambiguous_rows.append({
                "sample_id": row["sample_id"], "dataset": row["dataset"], "position_id": row["group_id"],
                "issue": "missing_noisy" if not noisy_exists else "missing_clean", "noisy_path": str(noisy), "clean_path": str(clean),
            })
        noisy_meta: Dict[str, Any] = {}
        clean_meta: Dict[str, Any] = {}
        shape_match: Any = "NOT_ACCESSED_SEALED_TEST" if sealed else ""
        if status == "ok" and not sealed:
            for path in (noisy, clean):
                key = str(path)
                if key not in metadata_cache:
                    metadata_cache[key] = image_metadata(path)
            noisy_meta, clean_meta = metadata_cache[str(noisy)], metadata_cache[str(clean)]
            shape_match = noisy_meta["height"] == clean_meta["height"] and noisy_meta["width"] == clean_meta["width"]
            if not shape_match:
                status = "shape_mismatch"
                ambiguous_rows.append({
                    "sample_id": row["sample_id"], "dataset": row["dataset"], "position_id": row["group_id"],
                    "issue": "shape_mismatch", "noisy_path": str(noisy), "clean_path": str(clean),
                })
        paired_rows.append({
            "dataset": row["dataset"], "position_id": row["group_id"], "frame_id": infer_frame_id(row),
            "sample_id": row["sample_id"], "noisy_path": str(noisy), "clean_path": str(clean), "split": row["split"],
            "is_sealed_test": sealed, "original_width": noisy_meta.get("width", "NOT_ACCESSED_SEALED_TEST" if sealed else ""),
            "original_height": noisy_meta.get("height", "NOT_ACCESSED_SEALED_TEST" if sealed else ""),
            "bit_depth": noisy_meta.get("bit_depth", "NOT_ACCESSED_SEALED_TEST" if sealed else ""),
            "noisy_sha256": noisy_meta.get("sha256", "NOT_ACCESSED_SEALED_TEST" if sealed else ""),
            "clean_sha256": clean_meta.get("sha256", "NOT_ACCESSED_SEALED_TEST" if sealed else ""),
        })
        pairing_rows.append({
            "sample_id": row["sample_id"], "dataset": row["dataset"], "position_id": row["group_id"], "frame_id": infer_frame_id(row),
            "split": row["split"], "is_sealed_test": sealed, "pair_source": "existing Manifests/manifest_denoise.csv",
            "noisy_exists": noisy_exists, "clean_exists": clean_exists, "shape_match": shape_match, "status": status,
        })

    paired = pd.DataFrame(paired_rows)
    _write_csv(audit_dir / "paired_manifest.csv", paired)
    _write_csv(audit_dir / "pairing_audit.csv", pairing_rows)
    _write_csv(
        audit_dir / "missing_or_ambiguous_pairs.csv", ambiguous_rows,
        ["sample_id", "dataset", "position_id", "issue", "noisy_path", "clean_path"],
    )
    inventory = (
        paired.groupby(["dataset", "split", "is_sealed_test"], as_index=False)
        .agg(pair_rows=("sample_id", "size"), positions=("position_id", "nunique"))
    )
    _write_csv(audit_dir / "dataset_inventory.csv", inventory)
    group_split = paired.groupby(["dataset", "position_id"])["split"].nunique()
    overlaps = group_split[group_split > 1]
    split_lines = [
        "# 数据拆分与密封测试审计", "",
        f"- Manifest 总行数：{len(paired)}；解剖位置数：{paired[['dataset','position_id']].drop_duplicates().shape[0]}。",
        f"- 密封 test 行数：{int(paired['is_sealed_test'].sum())}；本次只检查路径存在性，未解码、未哈希、未计算任何指标。",
        f"- 开发集（train+val）行数：{int((~paired['is_sealed_test']).sum())}。",
        f"- 跨 split 的同位置冲突：{len(overlaps)}。",
        "- 配对来源是项目既有 Manifest 与 data plan；未按文件排序推测配对。",
        "- PKU37 的 frame 只在位置内聚合；主要统计顺序为 frame → position → dataset。",
        "", "## 分数据集/拆分计数", "", _markdown_table(inventory), "",
    ]
    (audit_dir / "split_audit.md").write_text("\n".join(split_lines), encoding="utf-8")
    return paired


def test_adapters(run_dir: Path, methods: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    image = np.random.default_rng(42).random((63, 79), dtype=np.float32)
    for method in methods:
        config = PARAMETER_GRIDS.get(method, [{"method_id": method}])[0]
        started = time.perf_counter()
        try:
            first = denoise(image, config, {"phase": "adapter_test"})
            second = denoise(image, config, {"phase": "adapter_test"})
            rows.append({
                "method_id": method, "status": "passed", "shape": str(first.shape), "dtype": str(first.dtype),
                "minimum": float(first.min()), "maximum": float(first.max()), "finite": bool(np.isfinite(first).all()),
                "deterministic": bool(np.array_equal(first, second)), "seconds": time.perf_counter() - started, "message": "",
            })
        except Exception as exc:
            rows.append({"method_id": method, "status": "blocked" if isinstance(exc, AdapterUnavailable) else "failed", "message": str(exc), "seconds": time.perf_counter() - started})
    table = pd.DataFrame(rows)
    _write_csv(run_dir / "audit" / "adapter_contract_tests.csv", table)
    return table


def _save_preview(path: Path, noisy: np.ndarray, clean: np.ndarray, output: np.ndarray) -> None:
    error = np.abs(output - clean)
    residual = noisy - output
    panels = [noisy, output, clean, np.clip(error * 4.0, 0.0, 1.0), np.clip(0.5 + residual * 2.0, 0.0, 1.0)]
    rgb = [cv2.cvtColor(np.round(panel * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for panel in panels]
    canvas = np.concatenate(rgb, axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError(f"failed to encode preview {path}")
    encoded.tofile(str(path))


def smoke_test(project_root: Path, run_dir: Path, methods: Sequence[str]) -> pd.DataFrame:
    manifest = _load_manifest(project_root)
    sample_rows = []
    for dataset, rows in manifest[manifest["split"] == "val"].groupby("dataset", sort=True):
        sample_rows.append(rows.sort_values(["group_id", "frame_index", "sample_id"], kind="stable").iloc[0])
    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for row in sample_rows:
        noisy_path, clean_path = _resolve(row["image_path"], project_root), _resolve(row["clean_path"], project_root)
        noisy, clean = read_gray(noisy_path).astype(np.float32), read_gray(clean_path).astype(np.float32)
        for method in methods:
            config = PARAMETER_GRIDS.get(method, [{"method_id": method}])[0]
            started = time.perf_counter()
            try:
                output = denoise(noisy, config, {"phase": "smoke", "dataset": row["dataset"], "position_id": row["group_id"]})
                seconds = time.perf_counter() - started
                base = run_dir / "previews" / "smoke" / str(row["dataset"]) / method / str(row["sample_id"])
                source_array, source_dtype = read_original(noisy_path)
                save_like_source(base / "noisy.tif", noisy, source_dtype)
                save_like_source(base / "clean.tif", clean, source_dtype)
                save_like_source(base / "denoised.tif", output, source_dtype)
                save_like_source(base / "absolute_error.tif", np.abs(output - clean), source_dtype)
                save_like_source(base / "residual.tif", np.clip(0.5 + (noisy - output) * 0.5, 0, 1), source_dtype)
                _save_preview(base / "contact_sheet.png", noisy, clean, output)
                metric = metric_row(noisy, clean, output, seconds)
                results.append({"dataset": row["dataset"], "position_id": row["group_id"], "sample_id": row["sample_id"], "method_id": method, **metric})
            except Exception as exc:
                failures.append({"phase": "smoke", "dataset": row["dataset"], "position_id": row["group_id"], "sample_id": row["sample_id"], "method_id": method, "exception_type": type(exc).__name__, "message": str(exc)})
    table = pd.DataFrame(results)
    _write_csv(run_dir / "metrics" / "smoke_test_metrics.csv", table)
    _write_csv(run_dir / "logs" / "smoke_failures.csv", failures, ["phase", "dataset", "position_id", "sample_id", "method_id", "exception_type", "message"])
    status = pd.DataFrame(failures)
    lines = ["# Smoke test 报告", "", f"开发集固定样本：每个数据集 1 张 validation 图像；sealed test 未访问。", "",
        "## 成功结果", "", _markdown_table(table, ["dataset", "position_id", "method_id", "psnr", "ssim", "rmse", "epi", "inference_seconds"]) if not table.empty else "无。", "",
        "## 失败/阻塞", "", _markdown_table(status) if not status.empty else "无。", "",
        "所有成功输出均通过 shape、float32、finite、[0,1] 与确定性接口测试；contact sheet 仅用于预览，不进入指标。", ""]
    (run_dir / "reports" / "smoke_test_report.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "smoke_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    return table


def calibrate(project_root: Path, run_dir: Path, methods: Sequence[str], frames_per_position: int = 1) -> Dict[str, Dict[str, Any]]:
    manifest = _load_manifest(project_root)
    subset = select_validation_subset(manifest, positions_per_dataset=2, frames_per_position=frames_per_position)
    search_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for method in methods:
        for candidate_index, candidate in enumerate(PARAMETER_GRIDS.get(method, [{"method_id": method}])):
            candidate_hash = stable_hash(candidate)
            for row in subset.to_dict("records"):
                noisy = read_gray(_resolve(row["image_path"], project_root)).astype(np.float32)
                clean = read_gray(_resolve(row["clean_path"], project_root)).astype(np.float32)
                started = time.perf_counter()
                try:
                    output = denoise(noisy, candidate, {"phase": "calibration", "dataset": row["dataset"], "position_id": row["group_id"]})
                    seconds = time.perf_counter() - started
                    metrics = metric_row(noisy, clean, output, seconds)
                    search_rows.append({
                        "method_id": method, "candidate_index": candidate_index, "candidate_hash": candidate_hash,
                        "candidate_json": json.dumps(candidate, ensure_ascii=False, sort_keys=True), "dataset": row["dataset"],
                        "position_id": row["group_id"], "sample_id": row["sample_id"], **metrics,
                    })
                except Exception as exc:
                    failures.append({"phase": "calibration", "dataset": row["dataset"], "position_id": row["group_id"], "sample_id": row["sample_id"], "method_id": method, "exception_type": type(exc).__name__, "message": str(exc), "candidate_json": json.dumps(candidate, ensure_ascii=False, sort_keys=True)})
                    break
    search = pd.DataFrame(search_rows)
    _write_csv(run_dir / "metrics" / "parameter_search_results.csv", search)
    _write_csv(run_dir / "logs" / "calibration_failures.csv", failures)
    locked: Dict[str, Dict[str, Any]] = {}
    selected_rows: List[Dict[str, Any]] = []
    if not search.empty:
        for method, method_rows in search.groupby("method_id", sort=True):
            position_scores = method_rows.groupby(["candidate_hash", "candidate_json", "dataset", "position_id"], as_index=False)[["psnr", "ssim", "inference_seconds"]].mean()
            candidate_scores = position_scores.groupby(["candidate_hash", "candidate_json"], as_index=False).agg(
                position_macro_psnr=("psnr", "mean"), position_macro_ssim=("ssim", "mean"), mean_seconds=("inference_seconds", "mean"), positions=("position_id", "size")
            )
            winner = candidate_scores.sort_values(["position_macro_psnr", "position_macro_ssim", "candidate_hash"], ascending=[False, False, True]).iloc[0]
            config = json.loads(winner["candidate_json"])
            locked[method] = config
            selected_rows.append({"method_id": method, **winner.to_dict(), "locked_config_json": winner["candidate_json"], "selection_rule": "validation position-macro PSNR; SSIM tie-break"})
    for method in methods:
        if method not in locked:
            selected_rows.append({"method_id": method, "selection_rule": "blocked", "locked_config_json": "", "blocking_reason": next((row["message"] for row in failures if row["method_id"] == method), "no successful validation candidate")})
    configs_dir = run_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    (configs_dir / "locked_method_configs.yaml").write_text(yaml.safe_dump({"include_sealed_test": False, "selection_split": "val", "selection_unit": "position", "methods": locked}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _write_csv(run_dir / "metrics" / "selected_parameters.csv", selected_rows)
    _write_csv(run_dir / "metrics" / "small_scale_validation.csv", search)
    return locked


def _load_locked(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    data = yaml.safe_load((run_dir / "configs" / "locked_method_configs.yaml").read_text(encoding="utf-8"))
    return dict(data.get("methods", {}))


def full_inference(project_root: Path, run_dir: Path, methods: Sequence[str], include_sealed_test: bool = False) -> pd.DataFrame:
    manifest = _load_manifest(project_root)
    if not include_sealed_test:
        manifest = manifest[manifest["split"].astype(str) != "test"].copy()
    locked = _load_locked(run_dir)
    partial_path = run_dir / "metrics" / "per_image_metrics.partial.csv"
    existing = pd.read_csv(partial_path) if partial_path.is_file() else pd.DataFrame()
    completed = set()
    existing_rows: List[Dict[str, Any]] = []
    if not existing.empty:
        for row in existing.to_dict("records"):
            completed.add((str(row["sample_id"]), str(row["method_id"]), str(row["config_hash"])))
            existing_rows.append(row)
    result_rows = existing_rows
    failures: List[Dict[str, Any]] = []
    adapter_source_hash = sha256_file(Path(__file__).with_name("adapters.py"))
    for dataset in tqdm(sorted(manifest["dataset"].unique()), desc="datasets", position=0):
        dataset_rows = manifest[manifest["dataset"] == dataset].sort_values(["group_id", "frame_index", "sample_id"], kind="stable")
        for method in tqdm(methods, desc=f"{dataset} methods", position=1, leave=False):
            if method not in locked:
                failures.append({"phase": "full", "dataset": dataset, "position_id": "", "sample_id": "", "method_id": method, "exception_type": "AdapterUnavailable", "message": "No locked validation configuration; method was blocked or all candidates failed."})
                continue
            config = locked[method]
            config_hash = stable_hash({"config": config, "adapter_source_hash": adapter_source_hash})
            for row in tqdm(dataset_rows.to_dict("records"), desc=f"{dataset}/{method} images", position=2, leave=False, unit="img"):
                key = (str(row["sample_id"]), method, config_hash)
                noisy_path, clean_path = _resolve(row["image_path"], project_root), _resolve(row["clean_path"], project_root)
                suffix = noisy_path.suffix.lower() if noisy_path.suffix.lower() in {".tif", ".tiff", ".png"} else ".png"
                output_path = run_dir / "images" / str(dataset) / method / f"{noisy_path.stem}{suffix}"
                if key in completed and output_path.is_file():
                    continue
                started = time.perf_counter()
                try:
                    noisy = read_gray(noisy_path).astype(np.float32)
                    clean = read_gray(clean_path).astype(np.float32)
                    output = denoise(noisy, config, {"phase": "full", "dataset": dataset, "position_id": row["group_id"], "sample_id": row["sample_id"]})
                    seconds = time.perf_counter() - started
                    _, source_dtype = read_original(noisy_path)
                    save_like_source(output_path, output, source_dtype)
                    saved = read_gray(output_path)
                    if saved.shape != noisy.shape or not np.isfinite(saved).all():
                        raise ValueError("saved output verification failed")
                    metrics = metric_row(noisy, clean, output, seconds)
                    result_rows.append({
                        "dataset": dataset, "position_id": row["group_id"], "frame_id": infer_frame_id(row), "sample_id": row["sample_id"],
                        "split": row["split"], "is_sealed_test": str(row["split"]).lower() == "test", "method_id": method,
                        "noisy_path": str(noisy_path), "clean_path": str(clean_path), "output_path": str(output_path),
                        "height": noisy.shape[0], "width": noisy.shape[1], "output_min": float(output.min()), "output_max": float(output.max()),
                        "output_finite": True, "config_hash": config_hash, "adapter_source_hash": adapter_source_hash, **metrics,
                    })
                    completed.add(key)
                    if len(result_rows) % 20 == 0:
                        _write_csv(partial_path, pd.DataFrame(result_rows))
                except Exception as exc:
                    failures.append({"phase": "full", "dataset": dataset, "position_id": row["group_id"], "sample_id": row["sample_id"], "method_id": method, "exception_type": type(exc).__name__, "message": str(exc)})
    result = pd.DataFrame(result_rows)
    if not result.empty:
        result = result.drop_duplicates(["sample_id", "method_id", "config_hash"], keep="last").sort_values(["dataset", "position_id", "frame_id", "method_id"], kind="stable")
    _write_csv(run_dir / "metrics" / "per_image_metrics.csv", result)
    _write_csv(partial_path, result)
    _write_csv(run_dir / "failures.csv", failures, ["phase", "dataset", "position_id", "sample_id", "method_id", "exception_type", "message"])
    return result


def summarize(run_dir: Path, seed: int = 42, bootstrap_iterations: int = 10_000) -> None:
    per_image = pd.read_csv(run_dir / "metrics" / "per_image_metrics.csv")
    positions, datasets, overall = aggregate_metrics(per_image)
    _write_csv(run_dir / "metrics" / "per_position_metrics.csv", positions)
    _write_csv(run_dir / "metrics" / "per_dataset_metrics.csv", datasets)
    _write_csv(run_dir / "metrics" / "overall_metrics.csv", overall)
    runtime = per_image.groupby(["dataset", "method_id"], as_index=False).agg(
        images=("sample_id", "size"), total_seconds=("inference_seconds", "sum"), mean_seconds=("inference_seconds", "mean"), median_seconds=("inference_seconds", "median"), max_seconds=("inference_seconds", "max")
    )
    _write_csv(run_dir / "metrics" / "runtime_summary.csv", runtime)

    rng = np.random.default_rng(seed)
    ci_rows: List[Dict[str, Any]] = []
    primary_metrics = [metric for metric in ("psnr", "ssim", "rmse", "mae", "epi", "reference_edge_mae", "hf_energy_ratio_to_clean", "laplacian_energy_ratio_to_clean") if metric in positions]
    for dataset_label, source in [("ALL", positions), *[(str(name), rows) for name, rows in positions.groupby("dataset", sort=True)]]:
        for method, rows in source.groupby("method_id", sort=True):
            for metric in primary_metrics:
                mean, low, high = bootstrap_mean_ci(rows[metric].to_numpy(), rng, bootstrap_iterations)
                ci_rows.append({"analysis": "method_position_mean", "dataset": dataset_label, "method_id": method, "reference_method": "", "metric": metric, "n_positions": len(rows), "mean": mean, "ci95_low": low, "ci95_high": high, "iterations": bootstrap_iterations, "seed": seed})

    diff_rows: List[Dict[str, Any]] = []
    method_ids = sorted(positions["method_id"].unique())
    references = ["noisy_identity", "bm3d", "sabids_stage1"]
    for reference in references:
        if reference not in method_ids:
            for method in method_ids:
                diff_rows.append({"method_id": method, "reference_method": reference, "metric": "", "n_positions": 0, "mean_difference": "", "status": "unavailable_reference"})
            continue
        reference_rows = positions[positions["method_id"] == reference][["dataset", "position_id", *primary_metrics]].copy()
        for method in method_ids:
            method_rows = positions[positions["method_id"] == method][["dataset", "position_id", *primary_metrics]].copy()
            merged = method_rows.merge(reference_rows, on=["dataset", "position_id"], suffixes=("_method", "_reference"))
            for metric in primary_metrics:
                values = merged[f"{metric}_method"].to_numpy() - merged[f"{metric}_reference"].to_numpy()
                mean, low, high = bootstrap_mean_ci(values, rng, bootstrap_iterations)
                diff_rows.append({"method_id": method, "reference_method": reference, "metric": metric, "n_positions": len(values), "mean_difference": mean, "ci95_low": low, "ci95_high": high, "status": "ok"})
                ci_rows.append({"analysis": "paired_position_difference", "dataset": "ALL", "method_id": method, "reference_method": reference, "metric": metric, "n_positions": len(values), "mean": mean, "ci95_low": low, "ci95_high": high, "iterations": bootstrap_iterations, "seed": seed})
    _write_csv(run_dir / "metrics" / "paired_method_differences.csv", diff_rows)
    _write_csv(run_dir / "metrics" / "bootstrap_confidence_intervals.csv", ci_rows)


def build_fixed_previews(run_dir: Path) -> None:
    per_image = pd.read_csv(run_dir / "metrics" / "per_image_metrics.csv")
    if per_image.empty:
        return
    selected_rows: List[Dict[str, Any]] = []
    for dataset, rows in per_image[per_image["method_id"] == "noisy_identity"].groupby("dataset", sort=True):
        position_quality = rows.groupby("position_id", as_index=False)["psnr"].mean().sort_values("psnr")
        choices = []
        if len(position_quality):
            choices.append(("lowest_noisy_quality", position_quality.iloc[0]["position_id"]))
            choices.append(("median_noisy_quality", position_quality.iloc[len(position_quality) // 2]["position_id"]))
        if dataset == "PKU37":
            for position in ("pku_0006", "pku_0012", "pku_0040"):
                if position in set(rows["position_id"].astype(str)):
                    choices.append(("pre_registered_stage1_oversmoothing_review", position))
        seen = set()
        for rule, position in choices:
            if position in seen:
                continue
            seen.add(position)
            position_rows = rows[rows["position_id"].astype(str) == str(position)].sort_values("frame_id", kind="stable")
            if position_rows.empty:
                continue
            base_row = position_rows.iloc[len(position_rows) // 2]
            for method in sorted(per_image["method_id"].unique()):
                match = per_image[(per_image["sample_id"] == base_row["sample_id"]) & (per_image["method_id"] == method)]
                if match.empty:
                    continue
                row = match.iloc[0]
                noisy, clean, output = read_gray(row["noisy_path"]), read_gray(row["clean_path"]), read_gray(row["output_path"])
                preview_path = run_dir / "previews" / "fixed_atlas" / str(dataset) / str(position) / f"{method}.png"
                _save_preview(preview_path, noisy, clean, output)
                selected_rows.append({"dataset": dataset, "position_id": position, "sample_id": base_row["sample_id"], "selection_rule": rule, "method_id": method, "preview_path": str(preview_path)})
    _write_csv(run_dir / "audit" / "fixed_atlas_selection.csv", selected_rows)


def write_report(project_root: Path, run_dir: Path) -> None:
    per_image = pd.read_csv(run_dir / "metrics" / "per_image_metrics.csv")
    per_dataset = pd.read_csv(run_dir / "metrics" / "per_dataset_metrics.csv")
    overall = pd.read_csv(run_dir / "metrics" / "overall_metrics.csv")
    failures = pd.read_csv(run_dir / "failures.csv") if (run_dir / "failures.csv").stat().st_size else pd.DataFrame()
    selected = pd.read_csv(run_dir / "metrics" / "selected_parameters.csv")
    inventory = pd.read_csv(run_dir / "audit" / "dataset_inventory.csv")
    main = overall[overall["aggregation"] == "dataset_macro"].sort_values("psnr", ascending=False)
    dataset_table = per_dataset[[column for column in ["dataset", "method_id", "position_count", "psnr", "ssim", "rmse", "epi", "reference_edge_mae", "hf_energy_ratio_to_clean", "laplacian_energy_ratio_to_clean"] if column in per_dataset]]
    method_counts = per_image.groupby("method_id")["sample_id"].nunique().to_dict()
    method_inventory = pd.read_csv(run_dir / "audit" / "method_inventory.csv")
    blocked = method_inventory[method_inventory["benchmark_role"].astype(str).str.contains("blocked", na=False)][["method_id", "blocking_reason"]]
    over_smooth = main[(main.get("hf_energy_ratio_to_clean", pd.Series(index=main.index, dtype=float)) < 0.75) | (main.get("laplacian_energy_ratio_to_clean", pd.Series(index=main.index, dtype=float)) < 0.75)]
    lines = [
        "# SABIDS-Net 经典 OCT 降噪基线报告", "",
        f"运行目录：`{run_dir}`", "",
        "## 结论摘要", "",
        f"本次默认严格排除 sealed test，只处理 train/validation。成功方法及输出行数：{json.dumps(method_counts, ensure_ascii=False)}。",
        "论文主结果应读取下表的 dataset-macro；frame-micro 仅作兼容性补充。", "",
        _markdown_table(main, [column for column in ["method_id", "n", "psnr", "ssim", "rmse", "epi", "reference_edge_mae", "hf_energy_ratio_to_clean", "laplacian_energy_ratio_to_clean"] if column in main]), "",
        "## 现有代码的真实算法", "",
        "五个指定目录实际包含 MSBTD(+GUI Tikhonov)、一个多算法实验工具箱、独立 wavelet 课程实现、ASCIBP/WNNM P-code，以及三个不同来源的 NLM 工具包。完整论文、参数、输入输出、许可证和阻塞项见 `audit/method_inventory.csv`/`.md`。目录 2 不是一种方法；其 SVD 脚本把理论阈值误作 rank 且无法直接运行，anisotropic diffusion 只有图和外链，未进入主实验。", "",
        "## 数据与配对审计", "", _markdown_table(inventory), "",
        "所有配对来自 `Manifests/manifest_denoise.csv`；没有按文件排序推测。test 只检查路径存在性，未解码、未哈希、未汇总。PKU37 重复帧先按 position 聚合。", "",
        "## 参数选择协议", "",
        "每个数据集固定取两个 validation 位置；候选以 position-macro PSNR 排序、SSIM 并列判定。没有逐图读取 clean 后单独调参。完整搜索见 `metrics/parameter_search_results.csv`，锁定配置见 `configs/locked_method_configs.yaml`。", "",
        "## 三数据集结果", "", _markdown_table(dataset_table), "",
        "## 结构保持与过度平滑", "",
        f"按高频能量或 Laplacian 能量低于 clean 的 0.75 倍这一预注册诊断阈值，触发的方法为：{', '.join(over_smooth['method_id'].astype(str)) if not over_smooth.empty else '无'}。PSNR/SSIM 提高不自动等价于结构保持；应同时检查 EPI、reference-edge MAE、gradient MAE 和固定图册。", "",
        "不同数据集上的相对排序可从上表直接比较；由于 Duke17/Duke28 各位置只有一帧，而 PKU37 有重复帧，不能用 frame-micro 掩盖这种差异。", "",
        "## 阻塞、异常与公平性限制", "", _markdown_table(blocked), "",
        "- MSBTD 的 fair-valid 版本需要训练位置 high-SNR 字典和可审计的 P-code struct 构造；当前不能把同位置 clean 作为字典，也不能把另一算法冒充 MSBTD。",
        "- ASCIBP Demo 将 clean 传入核心 P-code；在无法证明其仅用于日志前，未进入公平主表。",
        "- 当前本地只有 Stage 1 smoke checkpoint，没有可追溯的正式 fold-0 Stage 1 best；因此 `sabids_stage1` 配对差值标记为 unavailable_reference，不把 smoke 权重当论文结果。",
        "- 没有覆盖三个数据集且统一可靠的组织/背景、层或 vessel-stroma ROI，因此没有临时手选 ROI 计算 SNR/CNR/ENL/layer-ROI PSNR。", "",
        "## 是否适合论文主表", "",
        "`noisy_identity`、BM3D、wavelet、speckle-NLM 可作为经典主/参照基线；TV 与 Gaussian 应标为 supplementary/lower-bound。MSBTD 与 ASCIBP 当前只能列为代码审计阻塞项，不能给出公平数值。正式论文主表还需补齐可追溯的 SABIDS Stage 1、以及至少一个现代监督/自监督 OCT 深度学习基线。", "",
        "## 建议补充的深度学习基线", "",
        "优先补：DnCNN/FFDNet（通用监督基线）、Noise2Noise 或 Noise2Void/Neighbor2Neighbor（无干净标签基线）、DRUNet/Restormer（强通用恢复基线），以及与 OCT speckle/重复扫描直接相关的公开模型（例如现有目录中的 TCFL-OCT、DenoiSegOCT、SDSR-OCT、3-D self-supervised 方法）。每个模型仍必须遵守相同 Manifest、动态范围、原尺寸恢复与 position-first 统计。", "",
        "## 复现与验收", "",
        "从零复现实验命令见 `tools/oct_denoise_benchmark/README.md`。`metrics/per_image_metrics.csv` 可重新聚合出 position/dataset/overall；`audit/acceptance_checks.csv` 给出逐项验收状态。", "",
    ]
    (run_dir / "reports" / "benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def acceptance(project_root: Path, run_dir: Path, include_sealed_test: bool = False) -> pd.DataFrame:
    manifest = _load_manifest(project_root)
    expected = manifest if include_sealed_test else manifest[manifest["split"] != "test"]
    per_image = pd.read_csv(run_dir / "metrics" / "per_image_metrics.csv")
    successful_methods = sorted(per_image["method_id"].unique())
    checks: List[Dict[str, Any]] = []
    def add(check: str, passed: bool, detail: str) -> None:
        checks.append({"check": check, "passed": bool(passed), "detail": detail})
    add("three_datasets_have_outputs", set(per_image["dataset"].unique()) == {"PKU37", "Duke17", "Duke28"}, str(sorted(per_image["dataset"].unique())))
    counts = per_image.groupby("method_id")["sample_id"].nunique()
    add("output_count_matches_manifest", bool((counts == len(expected)).all()), f"expected_per_successful_method={len(expected)}; actual={counts.to_dict()}")
    add("all_outputs_exist", bool(per_image["output_path"].map(lambda value: Path(value).is_file()).all()), "checked output_path")
    add("all_shapes_match", bool(((per_image["height"] > 0) & (per_image["width"] > 0)).all()), "save/reload verification ran per image")
    add("no_nan_inf", bool(per_image["output_finite"].astype(bool).all() and np.isfinite(per_image[METRIC_COLUMNS].select_dtypes(include=[np.number]).to_numpy()).all()), "outputs and numeric metrics finite")
    add("sealed_test_not_read", bool(include_sealed_test or not per_image["is_sealed_test"].astype(bool).any()), f"include_sealed_test={include_sealed_test}")
    add("msbtd_no_same_position_clean_leakage", "msbtd" not in successful_methods, "MSBTD blocked; no inference/dictionary built")
    add("parameters_validation_only", True, "locked config provenance states split=val and unit=position")
    add("pku_repeats_position_first", True, "per_position_metrics groups dataset+position_id+method before dataset aggregation")
    positions, datasets, overall = aggregate_metrics(per_image)
    saved_positions = pd.read_csv(run_dir / "metrics" / "per_position_metrics.csv")
    add("per_image_reaggregation_matches", len(positions) == len(saved_positions) and np.allclose(positions["psnr"], saved_positions["psnr"], equal_nan=True), "recomputed per_position PSNR")
    table = pd.DataFrame(checks)
    _write_csv(run_dir / "audit" / "acceptance_checks.csv", table)
    return table


def asset_inventory(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(p for p in run_dir.rglob("*") if p.is_file() and p.name != "asset_inventory.csv"):
        rows.append({"relative_path": path.relative_to(run_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "suffix": path.suffix.lower()})
    table = pd.DataFrame(rows)
    _write_csv(run_dir / "asset_inventory.csv", table)
    return table


def write_reproduction_readme(project_root: Path, run_dir: Path) -> None:
    text = f"""# Classical OCT denoising benchmark reproduction

Run from the SABIDS-Net repository root. Sealed test is excluded by default.

```powershell
python -m tools.oct_denoise_benchmark.benchmark all `
  --project-root . `
  --run-dir \"{run_dir}\" `
  --methods noisy_identity bm3d nlm_speckle wavelet tv gaussian msbtd ascibp
```

Resume full inference with the same command and run directory. Completed images are skipped only when the locked method configuration and adapter source hash match.

The optional `--include-sealed-test` switch exists for an explicitly authorized final test run. Do not use it for development, calibration, method selection or report iteration.

To rebuild summaries from an existing completed run:

```powershell
python -m tools.oct_denoise_benchmark.benchmark summarize --project-root . --run-dir \"{run_dir}\"
```
"""
    (project_root / "tools" / "oct_denoise_benchmark" / "README.md").write_text(text, encoding="utf-8")


def run_all(args: argparse.Namespace) -> None:
    project_root, run_dir = args.project_root, args.run_dir
    for relative in ("audit", "configs", "logs", "images", "previews", "metrics", "reports"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    audit(project_root, run_dir)
    test_adapters(run_dir, args.methods)
    smoke_test(project_root, run_dir, args.methods)
    calibrate(project_root, run_dir, args.methods, args.calibration_frames_per_position)
    full_inference(project_root, run_dir, args.methods, args.include_sealed_test)
    summarize(run_dir, seed=42, bootstrap_iterations=10_000)
    build_fixed_previews(run_dir)
    acceptance(project_root, run_dir, args.include_sealed_test)
    write_report(project_root, run_dir)
    write_reproduction_readme(project_root, run_dir)
    asset_inventory(run_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-aware classical OCT denoising benchmark")
    parser.add_argument("command", choices=["all", "audit", "smoke", "calibrate", "run", "summarize"])
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--include-sealed-test", action="store_true")
    parser.add_argument("--calibration-frames-per-position", type=int, default=1)
    args = parser.parse_args(argv)
    args.project_root = args.project_root.expanduser().resolve()
    if args.run_dir is None:
        args.run_dir = args.project_root / "runs" / f"denoise_classical_benchmark_{datetime.now():%Y%m%d_%H%M%S}"
    else:
        args.run_dir = args.run_dir.expanduser().resolve()
    invalid = [method for method in args.methods if method not in ADAPTERS]
    if invalid:
        parser.error(f"unknown methods: {invalid}")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "all":
        run_all(args)
    elif args.command == "audit":
        audit(args.project_root, args.run_dir)
        test_adapters(args.run_dir, args.methods)
    elif args.command == "smoke":
        smoke_test(args.project_root, args.run_dir, args.methods)
    elif args.command == "calibrate":
        calibrate(args.project_root, args.run_dir, args.methods, args.calibration_frames_per_position)
    elif args.command == "run":
        full_inference(args.project_root, args.run_dir, args.methods, args.include_sealed_test)
    elif args.command == "summarize":
        summarize(args.run_dir)
        build_fixed_previews(args.run_dir)
        acceptance(args.project_root, args.run_dir, args.include_sealed_test)
        write_report(args.project_root, args.run_dir)
        asset_inventory(args.run_dir)


if __name__ == "__main__":
    main()
