from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def orientation(metric: str) -> float:
    lower = ("mae", "mse", "rmse", "hd95", "assd", "error", "outside", "_fp", "_fn", "bias_abs")
    return -1.0 if any(token in metric for token in lower) else 1.0


def scale_unit(metric: str) -> tuple[float, str]:
    if any(token in metric for token in ("dice", "precision", "recall", "iou")):
        return 100.0, "percentage_points"
    if "psnr" in metric or metric.endswith("_db"):
        return 1.0, "dB"
    if any(token in metric for token in ("boundary", "thickness", "hd95", "assd")):
        return 1.0, "px"
    return 1.0, "native"


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-first I_DENOISED - I_NOISY report")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", default="runs/input_factorial_report")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite input-factorial report: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frames, positions, missing = [], [], []
    for seed in args.seeds:
        for variant in ("noisy", "denoised"):
            directory = root / "runs/current" / f"input_{variant}_fold0_seed{seed}" / "final_validation"
            frame_path, group_path = directory / "frame_metrics.csv", directory / "group_metrics.csv"
            if not frame_path.is_file() or not group_path.is_file():
                missing.append(str(directory))
                continue
            frame, group = pd.read_csv(frame_path), pd.read_csv(group_path)
            frame["seed"], frame["variant"] = seed, variant.upper()
            group["seed"], group["variant"] = seed, variant.upper()
            frames.append(frame)
            positions.append(group)
    checklist = output / "missing_and_failure_checklist.json"
    if missing:
        checklist.write_text(json.dumps({"missing_inputs": missing, "complete": False}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise FileNotFoundError("Missing fixed-final validation outputs:\n" + "\n".join(missing))
    frame_table, position_table = pd.concat(frames, ignore_index=True), pd.concat(positions, ignore_index=True)
    frame_table.to_csv(output / "frame_metrics_long.csv", index=False, encoding="utf-8-sig")
    position_table.to_csv(output / "position_metrics_long.csv", index=False, encoding="utf-8-sig")
    excluded = ("height", "width", "valid_pixels", "n_evaluated", "_true", "_pred", "baseline")
    metrics = [
        column for column in position_table.select_dtypes(include=[np.number]).columns
        if column != "seed" and not any(token in column for token in excluded)
    ]
    rows = []
    for metric in metrics:
        pivot = position_table.pivot_table(index=["seed", "group_id", "dataset"], columns="variant", values=metric)
        if not {"NOISY", "DENOISED"}.issubset(pivot.columns):
            continue
        factor, unit = scale_unit(metric)
        if "signed_bias" in metric:
            gain = (pivot["NOISY"].abs() - pivot["DENOISED"].abs()) * factor
        else:
            gain = orientation(metric) * (pivot["DENOISED"] - pivot["NOISY"]) * factor
        for key, value in gain.items():
            rows.append({"seed": key[0], "group_id": key[1], "dataset": key[2], "metric": metric, "improvement": value, "unit": unit})
    gains = pd.DataFrame(rows)
    gains.to_csv(output / "paired_input_gains_by_position.csv", index=False, encoding="utf-8-sig")

    quality_path = root / "runs/current/input_factorial_common_fold0/cache/input_quality_metrics.csv"
    quality_by_position = pd.DataFrame()
    associations = []
    if quality_path.is_file():
        quality = pd.read_csv(quality_path)
        quality = quality[quality["split"].astype(str) == "val"].copy()
        input_rows = []
        pairs = {
            "psnr": ("psnr_noisy", "psnr_d0", 1.0),
            "ssim": ("ssim_noisy", "ssim_d0", 1.0),
            "rmse": ("rmse_noisy", "rmse_d0", -1.0),
            "epi": ("epi_noisy", "epi_d0", 1.0),
            "reference_edge_mae": ("reference_edge_mae_noisy", "reference_edge_mae_d0", -1.0),
            "layer_roi_psnr": ("layer_roi_psnr_noisy", "layer_roi_psnr_d0", 1.0),
            "layer_roi_mse": ("layer_roi_mse_noisy", "layer_roi_mse_d0", -1.0),
            "vessel_stroma_cnr_abs_error": (
                "vessel_stroma_cnr_abs_error_noisy", "vessel_stroma_cnr_abs_error_d0", -1.0,
            ),
        }
        for group_id, part in quality.groupby("group_id", sort=True):
            for metric, (left, right, direction) in pairs.items():
                if left not in part or right not in part:
                    continue
                delta = direction * (pd.to_numeric(part[right], errors="coerce") - pd.to_numeric(part[left], errors="coerce"))
                input_rows.append({
                    "group_id": str(group_id), "metric": metric,
                    "input_quality_improvement": float(delta.mean()),
                    "n_frames": int(delta.notna().sum()),
                })
        quality_by_position = pd.DataFrame(input_rows)
        quality_by_position.to_csv(output / "input_quality_by_position.csv", index=False, encoding="utf-8-sig")
        segmentation_metrics = {
            "layer_dice", "upper_boundary_mae", "lower_boundary_mae", "thickness_mae",
            "p0_vessel_dice", "p0_vessel_precision", "p0_vessel_recall",
            "vessel_roi_dice", "vessel_gt_component_small_pixel_recall",
            "vessel_boundary_band_dice", "repeat_layer_dice", "repeat_vessel_dice",
        }
        for seed in sorted(gains["seed"].unique()):
            seed_gains = gains[gains["seed"] == seed]
            for input_metric, input_part in quality_by_position.groupby("metric"):
                for segmentation_metric in sorted(segmentation_metrics & set(seed_gains["metric"])):
                    paired = input_part.merge(
                        seed_gains[seed_gains["metric"] == segmentation_metric][["group_id", "improvement"]],
                        on="group_id", how="inner",
                    ).dropna()
                    if len(paired) < 3:
                        continue
                    associations.append({
                        "seed": int(seed), "input_metric": input_metric,
                        "segmentation_metric": segmentation_metric,
                        "n_positions": int(len(paired)),
                        "pearson_descriptive": float(paired["input_quality_improvement"].corr(paired["improvement"])),
                        "spearman_descriptive": float(
                            paired["input_quality_improvement"].rank().corr(paired["improvement"].rank())
                        ),
                        "inferential_claim_allowed": False,
                    })
    pd.DataFrame(associations).to_csv(
        output / "input_quality_segmentation_association.csv",
        index=False, encoding="utf-8-sig",
    )
    by_seed = gains.groupby(["seed", "metric", "unit"], as_index=False).agg(
        mean_improvement=("improvement", "mean"),
        positive_positions=("improvement", lambda values: int((values > 0).sum())),
        negative_positions=("improvement", lambda values: int((values < 0).sum())),
        zero_positions=("improvement", lambda values: int((values == 0).sum())),
        max_abs_position_effect=("improvement", lambda values: float(np.max(np.abs(values)))),
    )
    by_seed.to_csv(output / "paired_input_gains_by_seed.csv", index=False, encoding="utf-8-sig")
    summary = by_seed.groupby(["metric", "unit"], as_index=False).agg(
        mean=("mean_improvement", "mean"), seed_std=("mean_improvement", "std"),
        n_seeds=("seed", "nunique"), total_positive_positions=("positive_positions", "sum"),
        total_negative_positions=("negative_positions", "sum"),
        largest_single_position_effect=("max_abs_position_effect", "max"),
    )
    summary["mean_smaller_than_seed_sd"] = summary["mean"].abs() < summary["seed_std"]
    summary.to_csv(output / "paired_input_gains_summary.csv", index=False, encoding="utf-8-sig")

    def result(metric: str) -> float:
        values = summary.loc[summary["metric"] == metric, "mean"]
        return float(values.iloc[0]) if len(values) else float("nan")

    layer, vessel = result("layer_dice"), result("p0_vessel_dice")
    if np.isfinite(layer) and np.isfinite(vessel):
        stable_positive = all(
            bool((by_seed.loc[by_seed["metric"] == metric, "mean_improvement"] > 0).all())
            for metric in ("layer_dice", "p0_vessel_dice")
        )
        if layer > 0 and vessel > 0 and stable_positive:
            conclusion = "A: 降噪图像本身有利于分割，当前 J 系列阴性主要指向特征交互形式、尺度或优化问题。"
        elif abs(layer) < 0.1 and abs(vessel) < 0.1:
            conclusion = "B: 当前分割器对噪声较鲁棒，或 Stage 1 表征已经吸收降噪收益，额外图像级降噪信息边际价值有限。"
        elif layer > 0 and vessel < 0:
            conclusion = "C: 降噪保留了粗层结构，但损害小血管、暗管腔或弱边缘，后续应研究任务保持型降噪。"
        elif layer < 0 and vessel < 0:
            conclusion = "D: 当前 D0 存在输入分布偏移或任务相关结构损伤，不应继续增强 D→S；需要先修改降噪目标。"
        else:
            conclusion = "混合结果：均值或 seed 方向不支持“稳定受益”，需按位置和 seed 原样报告。"
    else:
        conclusion = "Required layer/vessel metrics are missing; no outcome conclusion is generated."
    checks = []
    def add_check(name: str, evidence: str, status: object) -> None:
        checks.append({"question": name, "evidence": evidence, "supported": status})

    precision, recall = result("p0_vessel_precision"), result("p0_vessel_recall")
    roi_dice = result("vessel_roi_dice")
    lower_boundary, thickness = result("lower_boundary_mae"), result("thickness_mae")
    stroma_fp = result("vessel_roi_fp_per_valid_pixel")
    small_recall = result("vessel_gt_component_small_pixel_recall")
    repeat_layer, repeat_vessel = result("repeat_layer_dice"), result("repeat_vessel_dice")
    add_check("layer_and_vessel_both_improve", f"layer={layer}, vessel={vessel}", bool(layer > 0 and vessel > 0))
    add_check("precision_gain_costs_recall", f"precision={precision}, recall={recall}", bool(precision > 0 and recall < 0) if np.isfinite(precision + recall) else "unknown")
    add_check("roi_gain_without_full_gain", f"roi_dice={roi_dice}, full_dice={vessel}", bool(roi_dice > 0 and vessel <= 0) if np.isfinite(roi_dice + vessel) else "unknown")
    add_check("lower_boundary_and_thickness_improve", f"lower={lower_boundary}, thickness={thickness}", bool(lower_boundary > 0 and thickness > 0) if np.isfinite(lower_boundary + thickness) else "unknown")
    add_check("stroma_fp_reduced", f"normalized_roi_fp_improvement={stroma_fp}", bool(stroma_fp > 0) if np.isfinite(stroma_fp) else "unknown")
    add_check("small_vessel_recall_declines", f"small_pixel_recall={small_recall}", bool(small_recall < 0) if np.isfinite(small_recall) else "unknown")
    add_check("repeat_segmentation_more_stable", f"layer={repeat_layer}, vessel={repeat_vessel}", bool(repeat_layer > 0 and repeat_vessel > 0) if np.isfinite(repeat_layer + repeat_vessel) else "unknown")
    if not quality_by_position.empty:
        quality_means = quality_by_position.groupby("metric")["input_quality_improvement"].mean()
        psnr_gain = float(quality_means.get("psnr", np.nan))
        cnr_gain = float(quality_means.get("vessel_stroma_cnr_abs_error", np.nan))
        add_check("clean_like_psnr_without_task_gain", f"psnr={psnr_gain}, vessel={vessel}", bool(psnr_gain > 0 and abs(vessel) < 0.1) if np.isfinite(psnr_gain + vessel) else "unknown")
        add_check("cnr_closer_to_clean_and_vessel_gain", f"cnr_error_improvement={cnr_gain}, vessel={vessel}, recall={recall}", bool(cnr_gain > 0 and vessel > 0 and recall > 0) if np.isfinite(cnr_gain + vessel + recall) else "unknown")
    pd.DataFrame(checks).to_csv(output / "interpretation_checks.csv", index=False, encoding="utf-8-sig")
    priority_names = [
        "layer_dice", "layer_surface_dice", "upper_boundary_mae", "lower_boundary_mae",
        "upper_boundary_signed_bias", "lower_boundary_signed_bias",
        "thickness_mae", "thickness_signed_bias",
        "p0_vessel_dice", "p0_vessel_precision", "p0_vessel_recall",
        "vessel_roi_dice", "vessel_roi_precision", "vessel_roi_recall",
        "vessel_roi_fp_pixels", "vessel_roi_fn_pixels",
        "vessel_roi_fp_per_valid_pixel", "vessel_roi_fn_per_valid_pixel",
        "vessel_outside_gt_layer_fraction",
        "vessel_area_fraction_mae", "vessel_boundary_band_dice",
        "vessel_boundary_band_fp_per_valid_pixel", "vessel_boundary_band_fn_per_valid_pixel",
        "vessel_gt_component_small_pixel_recall",
        "vessel_gt_component_medium_pixel_recall",
        "vessel_gt_component_large_pixel_recall",
        "whole_layer_baseline_vessel_dice",
        "repeat_layer_dice", "repeat_vessel_dice",
    ]
    priority = summary[summary["metric"].isin(priority_names)]
    (output / "SUMMARY.md").write_text("\n".join([
        "# I_NOISY / I_DENOISED paired report", "",
        conclusion, "",
        "Positive values mean improvement after error-direction reversal. Frames were averaged within anatomical position before seed summaries.",
        "No frame-level significance test is produced; three labelled validation positions do not justify treating seed×position as independent subjects.",
        "Input-quality/segmentation correlations, when available, are descriptive only because there are only three labelled positions.",
        "", "## Priority metrics", "", "```text", priority.to_string(index=False), "```",
        "", "## Constrained interpretation checks", "", "```text",
        pd.DataFrame(checks).to_string(index=False), "```",
    ]), encoding="utf-8")
    checklist.write_text(json.dumps({
        "missing_inputs": [], "complete": True, "fixed_final": True,
        "threshold": 0.5, "postprocess": "P0", "test_used": False,
        "position_first_reduction": True, "frame_level_significance_test": False,
        "input_quality_metrics_found": quality_path.is_file(),
        "descriptive_association_rows": len(associations),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
