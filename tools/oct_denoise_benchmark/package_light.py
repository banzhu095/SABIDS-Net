from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from .data import load_protocol_manifest
from .io import read_image, sha256_file
from .registry import load_yaml


REQUIRED_CSV = {
    "per_image_metrics.csv": ["dataset", "split", "position_id", "frame_id", "method_id", "seed", "psnr", "ssim"],
    "per_position_metrics.csv": ["dataset", "split", "position_id", "method_id", "seed", "psnr", "ssim"],
    "per_seed_metrics.csv": ["dataset", "split", "method_id", "seed", "psnr", "ssim"],
    "per_dataset_metrics.csv": ["dataset", "split", "method_id", "psnr", "ssim"],
    "paired_method_differences.csv": ["dataset", "baseline_method", "method_id", "metric", "mean_paired_difference"],
    "bootstrap_confidence_intervals.csv": ["dataset", "method_id", "metric", "mean", "ci95_low", "ci95_high"],
    "parameter_search_results.csv": ["method_id", "search_round", "candidate_index", "candidate_json", "position_id", "psnr", "ssim"],
    "training_curves.csv": ["method_id", "seed", "optimizer_update", "train_loss", "val_position_macro_psnr", "val_position_macro_ssim"],
    "selected_parameters.csv": ["method_id", "status", "selection_rule"],
    "checkpoint_inventory.csv": ["method_id", "seed", "checkpoint", "sha256", "bytes", "status"],
    "runtime_summary.csv": ["method_id", "dataset", "images", "algorithm_seconds_mean", "io_seconds_mean"],
    "model_complexity.csv": ["method_id", "configuration", "parameters", "flops", "status"],
    "failures.csv": ["method_id", "stage", "error"],
    "asset_inventory.csv": ["path", "sha256", "bytes", "kind"],
}


def _ensure_result_tables(run_dir: Path) -> None:
    metrics = run_dir / "metrics"; metrics.mkdir(exist_ok=True)
    for name, columns in REQUIRED_CSV.items():
        destination = run_dir / name if name == "failures.csv" else metrics / name
        if not destination.exists(): pd.DataFrame(columns=columns).to_csv(destination, index=False)
    curves = []; checkpoints = []
    for path in (run_dir / "checkpoints").glob("**/training_curves.csv"):
        frame = pd.read_csv(path)
        frame["source_run"] = path.parent.name
        frame["status"] = "invalid_smoke_loss_sign" if path.parent.name == "smoke_nafnet_seed42" else ("smoke_only" if "smoke_" in path.as_posix() else "formal_training")
        curves.append(frame)
    for path in (run_dir / "checkpoints").glob("**/checkpoint_inventory.csv"):
        frame = pd.read_csv(path); frame["status"] = "smoke_only_not_formal" if "smoke_" in path.as_posix() else "formal_checkpoint"; checkpoints.append(frame)
    if curves: pd.concat(curves, ignore_index=True).to_csv(metrics / "training_curves.csv", index=False)
    if checkpoints: pd.concat(checkpoints, ignore_index=True).to_csv(metrics / "checkpoint_inventory.csv", index=False)


def select_fixed_atlas(project_root: Path, run_dir: Path) -> pd.DataFrame:
    table = load_protocol_manifest(project_root)
    selected = []
    pku = table[(table.dataset == "PKU37") & (table.split == "test")]
    for position, rows in pku.groupby("position_id", sort=True):
        row = rows.sort_values(["frame_index", "sample_id"]).iloc[len(rows) // 2]
        selected.append(row)
    for dataset in ("Duke17", "Duke28"):
        rows = table[table.dataset == dataset].sort_values("sample_id").reset_index(drop=True)
        for index in np.linspace(0, len(rows) - 1, 6, dtype=int): selected.append(rows.iloc[index])
    records = []
    selected_rows = []
    for row in selected:
        image, metadata = read_image(Path(row.image_path)); h, w = image.shape
        residual = image - cv2.GaussianBlur(image, (0, 0), 1.0)
        contrast = float(np.percentile(image, 95) - np.percentile(image, 5))
        quality_proxy = contrast / (float(np.median(np.abs(residual))) + 1e-8)
        record = {"dataset": row.dataset, "split": "test" if row.dataset == "PKU37" else "external_test", "position_id": row.position_id, "sample_id": row.sample_id,
                  "atlas_role": "pku_test_position" if row.dataset == "PKU37" else "external_fixed_case",
                  "selection_rule": "middle_frame_per_PKU_test_position" if row.dataset == "PKU37" else "six_evenly_spaced_sorted_cases",
                  "criterion_score": np.nan, "selected_without_method_metrics": True,
                  "noisy_sha256": sha256_file(Path(row.image_path)),
                  "detail_crop_x": w // 4, "detail_crop_y": h // 3, "detail_crop_width": w // 4, "detail_crop_height": h // 4,
                  "lower_crop_x": w // 4, "lower_crop_y": h // 2, "lower_crop_width": w // 2, "lower_crop_height": h // 4,
                  "limitation": "Deterministic preregistration; benchmark outputs and clean/reference metrics were not inspected."}
        records.append(record); selected_rows.append((row, image, quality_proxy, record))

    # Low-quality selection uses a noisy-only contrast-to-high-frequency proxy.
    row, _, score, base = min(selected_rows, key=lambda item: item[2])
    records.append({**base, "atlas_role": "low_quality_noisy", "selection_rule": "minimum_noisy_only_robust_contrast_to_high_frequency_proxy", "criterion_score": score})

    labelled = pku[pku.has_manual_label.astype(bool)].groupby("position_id", sort=True).first().reset_index()
    anatomy = []
    for row in labelled.itertuples():
        label_path = Path(row.multiclass_label_path)
        if not label_path.is_absolute(): label_path = project_root / label_path
        raw = np.asarray(Image.open(label_path))
        noisy, _ = read_image(Path(row.image_path))
        if raw is None or raw.shape != noisy.shape: continue
        layer, vessel = np.isin(raw, (1, 2)), raw == 2
        lower = []
        for x in range(raw.shape[1]):
            ys = np.flatnonzero(layer[:, x])
            if ys.size and ys[-1] > 0: lower.append((ys[-1], x))
        boundary_score = float(np.mean([abs(float(noisy[y, x]) - float(noisy[y - 1, x])) for y, x in lower])) if lower else np.inf
        anatomy.append((row, noisy, vessel, boundary_score, float(vessel.mean())))
    if anatomy:
        weak = min(anatomy, key=lambda item: item[3])
        rich = max(anatomy, key=lambda item: item[4])
        for role, item, rule, score in (
            ("weak_layer_boundary", weak, "minimum_noisy_intensity_step_along_manual_lower_layer_boundary", weak[3]),
            ("small_vessel_rich", rich, "maximum_manual_vessel_pixel_fraction_among_labelled_PKU_test_positions", rich[4]),
        ):
            row, noisy, vessel, _, _ = item; h, w = noisy.shape
            ys, xs = np.nonzero(vessel)
            cx = int(np.median(xs)) if len(xs) else w // 2; cy = int(np.median(ys)) if len(ys) else h // 2
            records.append({"dataset": row.dataset, "split": "test", "position_id": row.position_id, "sample_id": row.sample_id,
                            "atlas_role": role, "selection_rule": rule, "criterion_score": score,
                            "selected_without_method_metrics": True, "noisy_sha256": sha256_file(Path(row.image_path)),
                            "detail_crop_x": max(0, min(cx - w // 8, 3 * w // 4)), "detail_crop_y": max(0, min(cy - h // 8, 3 * h // 4)),
                            "detail_crop_width": w // 4, "detail_crop_height": h // 4,
                            "lower_crop_x": w // 4, "lower_crop_y": h // 2, "lower_crop_width": w // 2, "lower_crop_height": h // 4,
                            "limitation": "Manual anatomy mask was used only to preregister the display case/crop; no benchmark output or clean/reference metric was inspected."})

    # A traceable pre-existing SABIDS denoising asset supplies the requested D0
    # case; this is not selected from any method in the new benchmark.
    d0_asset = project_root / "runs" / "stage_summary_20260901_121853" / "atlas" / "stage2_e3b_postprocess" / "pku_0006_f01_denoised.png"
    d0_row = pku[pku.sample_id == "pku_0006_f01"]
    if d0_asset.is_file() and not d0_row.empty:
        row = d0_row.iloc[0]; noisy, _ = read_image(Path(row.image_path)); h, w = noisy.shape
        records.append({"dataset": "PKU37", "split": "test", "position_id": row.position_id, "sample_id": row.sample_id,
                        "atlas_role": "sabids_d0_oversmoothing_candidate", "selection_rule": "traceable_preexisting_SABIDS_D0_atlas_asset",
                        "criterion_score": np.nan, "selected_without_method_metrics": True, "noisy_sha256": sha256_file(Path(row.image_path)),
                        "detail_crop_x": w // 4, "detail_crop_y": h // 3, "detail_crop_width": w // 4, "detail_crop_height": h // 4,
                        "lower_crop_x": w // 4, "lower_crop_y": h // 2, "lower_crop_width": w // 2, "lower_crop_height": h // 4,
                        "limitation": f"Candidate anchored to prior asset {d0_asset}; over-smoothing label requires human confirmation and did not use new benchmark metrics."})
    result = pd.DataFrame(records); result.to_csv(run_dir / "audit" / "fixed_atlas_selection.csv", index=False); return result


def _write_montage(path: Path, rows: list[list[tuple[str, np.ndarray]]], cell_width: int = 240) -> None:
    rendered = []
    for row in rows:
        cells = []
        for label, image in row:
            values = np.clip(image, 0, 1)
            h, w = values.shape
            resized = cv2.resize(values, (cell_width, max(1, round(h * cell_width / w))), interpolation=cv2.INTER_AREA)
            canvas = np.round(resized * 255).astype(np.uint8)
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
            cv2.putText(canvas, label, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)
            cells.append(canvas)
        target_h = max(cell.shape[0] for cell in cells)
        cells = [cv2.copyMakeBorder(cell, 0, target_h - cell.shape[0], 0, 0, cv2.BORDER_CONSTANT) for cell in cells]
        rendered.append(np.hstack(cells))
    target_w = max(row.shape[1] for row in rendered)
    rendered = [cv2.copyMakeBorder(row, 0, 0, 0, target_w - row.shape[1], cv2.BORDER_CONSTANT) for row in rendered]
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", np.vstack(rendered), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok: raise RuntimeError(f"cannot encode atlas montage {path}")
    encoded.tofile(str(path))


def materialize_fixed_atlas(project_root: Path, run_dir: Path, selection: pd.DataFrame) -> dict[str, object]:
    """Create display-only atlas sheets, but never open sealed references before lock."""
    lock_path = run_dir / "audit" / "config_lock.json"
    status_path = run_dir / "audit" / "fixed_atlas_materialization.json"
    if not lock_path.is_file() or json.loads(lock_path.read_text(encoding="utf-8")).get("status") != "locked":
        status = {"status": "not_materialized_test_sealed", "sheets": 0}
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8"); return status
    metrics_path = run_dir / "metrics" / "per_image_metrics.csv"
    if not metrics_path.is_file():
        status = {"status": "not_materialized_metrics_missing", "sheets": 0}
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8"); return status
    metrics = pd.read_csv(metrics_path)
    registry = load_yaml(run_dir / "configs" / "inference_registry.yaml")
    primary_seeds = {method: int(entry.get("seed", 0)) for method, entry in registry.get("methods", {}).items()}
    methods = [method for method in ("noisy_identity", "bm3d_standard", "tv_chambolle", "nlm", "ksvd_self", "dncnn_paired", "nafnet_paired") if method in primary_seeds]
    manifest = load_protocol_manifest(project_root).set_index("sample_id")
    sheets = []
    for item in selection.itertuples():
        if item.sample_id not in manifest.index: continue
        source = manifest.loc[item.sample_id]
        noisy, _ = read_image(Path(source.image_path)); reference, _ = read_image(Path(source.clean_path))
        images: list[tuple[str, np.ndarray]] = [("noisy", noisy), ("reference", reference)]
        complete = True
        for method in methods:
            subset = metrics[(metrics.sample_id.astype(str) == str(item.sample_id)) & (metrics.method_id == method) & (metrics.seed == primary_seeds[method]) & (metrics.status == "success")]
            if subset.empty or not Path(subset.iloc[-1].denoised_path).is_file(): complete = False; break
            output, _ = read_image(Path(subset.iloc[-1].denoised_path)); images.append((method, output))
        if not complete: continue
        errors = [("", np.zeros_like(noisy)), ("absolute error", np.zeros_like(noisy))] + [(name, np.abs(image - reference)) for name, image in images[2:]]
        residuals = [("", np.zeros_like(noisy)), ("residual x2", np.zeros_like(noisy))] + [(name, np.clip(0.5 + 2.0 * (noisy - image), 0, 1)) for name, image in images[2:]]
        role = str(item.atlas_role); stem = f"{role}__{item.sample_id}"
        full_path = run_dir / "previews" / "fixed_atlas" / f"{stem}__full.jpg"
        _write_montage(full_path, [images, errors, residuals])
        x, y, w, h = int(item.detail_crop_x), int(item.detail_crop_y), int(item.detail_crop_width), int(item.detail_crop_height)
        lx, ly, lw, lh = int(item.lower_crop_x), int(item.lower_crop_y), int(item.lower_crop_width), int(item.lower_crop_height)
        crop_path = run_dir / "previews" / "fixed_atlas" / f"{stem}__crops.jpg"
        _write_montage(crop_path, [[(name, image[y:y+h, x:x+w]) for name, image in images], [(name, image[ly:ly+lh, lx:lx+lw]) for name, image in images]])
        sheets.extend([str(full_path), str(crop_path)])
    status = {"status": "materialized" if sheets else "not_materialized_outputs_incomplete", "sheets": len(sheets),
              "display_note": "JPEG sheets are qualitative previews. Absolute error uses [0,1]; signed noisy-output residual is centered at 0.5 and amplified 2x. Metrics use original float outputs."}
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"); return status


def write_report(project_root: Path, run_dir: Path) -> None:
    audit = json.loads((run_dir / "audit" / "data_split_audit.json").read_text(encoding="utf-8"))
    dataset_path = run_dir / "metrics" / "per_dataset_metrics.csv"
    dataset = pd.read_csv(dataset_path) if dataset_path.exists() else pd.DataFrame()
    table = dataset.to_markdown(index=False) if not dataset.empty else "当前仅完成工程 smoke，尚无可报告的正式比较结果。"
    lines = [
        "# SABIDS-Net OCT 降噪基准报告", "",
        "## 当前完成状态", "",
        "本目录是本机可验证的工程结果包，不是完成三种子 GPU 正式实验后的论文结果。正式参数搜索、三种子训练、配置锁定与封存测试尚未执行，因此没有根据 PKU37 test、Duke17 或 Duke28 结果进行任何回调。", "",
        f"数据审计通过：PKU37 train/validation/test 为 1163/277/294 帧、25/6/6 个位置；Duke17 与 Duke28 分别为完整 17/28 例 external test。PKU37 跨 split 位置：{audit['pku_positions_crossing_splits']}。", "",
        "## 实现核验", "",
        "- BM3D：改为 `bm3d 4.0.3` 的 standard profile，并强制 hard-thresholding + Wiener 两阶段；LC 仅保留为补充。许可证仅允许非商业使用。",
        "- TV：调用 `skimage.restoration.denoise_tv_chambolle`，独立记录 weight、eps 与 max_num_iter。",
        "- NLM：调用 `skimage.restoration.denoise_nl_means` 与 noisy-only `estimate_sigma`；旧 speckle-NLM 不进入主表。",
        "- K-SVD：逐图 noisy-only 学习，OMP 编码、逐原子 `numpy.linalg.svd` 更新、重叠 patch 聚合；单测覆盖归一化、稀疏度、SVD 路径、无空洞、确定性和无 clean context。",
        "- DnCNN：单通道、经典 BN/卷积结构、预测 noisy-clean 残差，训练损失为残差 MSE。",
        "- NAFNet：单通道 I/O，复用 NAFBlock、encoder-decoder 和 padding；训练使用官方 PSNRLoss 方向。首次 smoke 发现符号错误后已修正，错误目录保留。", "",
        "## 当前数值", "", table, "",
        "上述若仅含 noisy_identity validation，它是开发基线，不是论文主测试表。", "",
        "## 尚未完成及原因", "",
        "本机只有 PyTorch CPU 2.8.0、没有 CUDA/nvidia-smi，且默认 Python 3.9.13 低于项目声明的 3.10。K-SVD、完整 BM3D 网格、三种子 DnCNN/NAFNet 正式训练和全量 7 方法推理需要 ModelWhale RTX 3090 环境。未生成正式 checkpoint 时，推理 registry 保持 unlocked，深度方法不会退回随机权重。", "",
        "因此目前不能回答 validation/test 一致性、Duke 跨域下降、正式最优方法或分割优先级。下一阶段分割实验只能在正式降噪表完成后，优先选择 noisy_identity、BM3D，以及在边缘/高频指标与 PSNR/SSIM 间表现互补的深度方法；不能仅凭 smoke PSNR 决策。", "",
        "## 公平性与结构解释", "",
        "PSNR/SSIM 与可供固定分割网络恢复的解剖信息是两个问题。正式分析必须同时看 EPI、reference-edge MAE、gradient MAE、HF/Laplacian 能量比及固定图册；能量明显低于 reference 只能提示过度平滑，不能单独证明血管信息丢失。没有统一 ROI 时不报告人工挑选 ROI 指标。",
    ]
    path = run_dir / "reports" / "benchmark_report.md"; path.write_text("\n".join(lines), encoding="utf-8")


def write_inference_commands(run_dir: Path) -> None:
    methods = ["bm3d_standard", "tv_chambolle", "nlm", "ksvd_self", "dncnn_paired", "nafnet_paired"]
    lines = ["# 单图与文件夹推理命令", "", "$RUN = \"runs\\denoise_benchmark_pku_protocol_YYYYmmdd_HHMMSS\"", "$INPUT_IMAGE = \"E:\\example\\image.png\"", "$INPUT_FOLDER = \"E:\\example\\input_folder\"", "$OUTPUT_ROOT = \"E:\\example\\denoised_outputs\"", ""]
    for method in methods:
        device = " --device cuda:0" if method in {"dncnn_paired", "nafnet_paired"} else ""
        lines.append(f"python -m tools.oct_denoise_benchmark.inference --method {method} --input $INPUT_IMAGE --output \"$OUTPUT_ROOT\\{method}\" --registry \"$RUN\\configs\\inference_registry.yaml\"{device}")
    lines.append("")
    for method in methods:
        device = " --device cuda:0" if method in {"dncnn_paired", "nafnet_paired"} else ""
        lines.append(f"python -m tools.oct_denoise_benchmark.inference --method {method} --input $INPUT_FOLDER --output \"$OUTPUT_ROOT\\{method}\" --registry \"$RUN\\configs\\inference_registry.yaml\"{device} --recursive --preserve-relative-path")
    lines.extend(["", "python -m tools.oct_denoise_benchmark.inference --method all --input $INPUT_IMAGE --output $OUTPUT_ROOT --registry \"$RUN\\configs\\inference_registry.yaml\" --device cuda:0 --save-preview"])
    (run_dir / "reports" / "inference_commands.md").write_text("\n".join(lines), encoding="utf-8")


def package(project_root: Path, run_dir: Path) -> Path:
    _ensure_result_tables(run_dir); selection = select_fixed_atlas(project_root, run_dir); materialize_fixed_atlas(project_root, run_dir, selection); write_report(project_root, run_dir); write_inference_commands(run_dir)
    stage = (run_dir / "gpt_light").resolve()
    if stage.parent != run_dir.resolve(): raise RuntimeError("unsafe GPT-light staging path")
    if stage.exists(): shutil.rmtree(stage)
    stage.mkdir()
    include = [run_dir / "metrics", run_dir / "audit", run_dir / "configs", run_dir / "reports", run_dir / "manifests", run_dir / "previews" / "smoke", run_dir / "previews" / "fixed_atlas", run_dir / "benchmark_summary.xlsx", project_root / "configs" / "dncnn_paired.yaml", project_root / "configs" / "nafnet_paired.yaml", project_root / "tools" / "oct_denoise_benchmark", project_root / "tests" / "test_denoise_protocol_v1.py", project_root / "docs" / "EXPERIMENT_LOG.md"]
    for source in include:
        if not source.exists(): continue
        if source.is_dir():
            destination = stage / ("source" if source.name == "oct_denoise_benchmark" else source.name)
            shutil.copytree(source, destination, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pth", "node_modules"))
        else: shutil.copy2(source, stage / source.name)
    guide = """# GPT analysis guide\n\n正式结果完成后重点检查：PKU37 到 Duke17/Duke28 的跨域退化；PSNR/SSIM 与 EPI、边缘 MAE、高频和 Laplacian 能量的冲突；参数搜索边界；过度平滑；以及固定分割网络应优先比较的方法。当前包若标记 incomplete，只能审查代码、协议和 smoke，不得推断正式方法排名。\n"""
    (stage / "GPT_ANALYSIS_GUIDE.md").write_text(guide, encoding="utf-8")
    inventory = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path.name != "PACKAGE_MANIFEST.csv": inventory.append({"path": path.relative_to(stage).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    pd.DataFrame(inventory).to_csv(stage / "PACKAGE_MANIFEST.csv", index=False)
    archive = run_dir.parent / f"SABIDS_PKU37_denoise_benchmark_GPT_light_{datetime.now():%Y%m%d_%H%M%S}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(stage.rglob("*")):
            if path.is_file(): bundle.write(path, path.relative_to(stage).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad: raise RuntimeError(f"zip CRC failure: {bad}")
    (run_dir / "reports" / "package_sha256.txt").write_text(f"{sha256_file(archive)}  {archive.name}\n", encoding="utf-8")
    return archive


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path(".")); parser.add_argument("--run-dir", type=Path, required=True); args = parser.parse_args(argv)
    print(package(args.project_root.resolve(), args.run_dir.resolve()))


if __name__ == "__main__": main()
