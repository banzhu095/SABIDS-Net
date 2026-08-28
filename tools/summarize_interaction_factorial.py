from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


VARIANTS = ("j00", "j10", "j01", "j11")
IDENTIFIERS = {"group_id", "dataset", "sample_id", "patient_id", "seed", "variant"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired, group-level J00/J10/J01/J11 gain report")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", default="runs/interaction_factorial_report")
    return parser.parse_args()


def orientation(metric: str) -> float:
    lower_is_better = (
        "mae", "mse", "rmse", "hd95", "assd", "error", "outside", "_fp", "_fn",
        "pred_layer_vessel_dice", "pred_vessel_fraction_of_layer",
    )
    higher_already_is_improvement = ("gain", "reduction")
    if any(token in metric for token in higher_already_is_improvement):
        return 1.0
    return -1.0 if any(token in metric for token in lower_is_better) else 1.0


def display_scale(metric: str) -> tuple[float, str]:
    if "dice" in metric or "precision" in metric or "recall" in metric:
        return 100.0, "percentage_points"
    if "psnr" in metric or metric.endswith("_db") or "snr_" in metric:
        return 1.0, "dB"
    if "boundary_mae" in metric or "thickness_mae" in metric or "hd95" in metric or "assd" in metric:
        return 1.0, "px"
    return 1.0, "native"


def is_outcome_metric(metric: str) -> bool:
    descriptive = (
        "_noisy", "_clean", "_baseline", "_true", "_pred", "valid_pixels",
        "original_height", "original_width", "evaluation_height", "evaluation_width",
        "n_evaluated_frames", "repeat_pair_count",
    )
    return not any(token in metric for token in descriptive)


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame_parts, group_parts = [], []
    missing = []
    for seed in args.seeds:
        for variant in VARIANTS:
            run = root / "runs" / "current" / f"interaction_{variant}_fold0_seed{seed}" / "final_validation"
            frame_path, group_path = run / "frame_metrics.csv", run / "group_metrics.csv"
            if not frame_path.is_file() or not group_path.is_file():
                missing.append(str(run))
                continue
            frame = pd.read_csv(frame_path)
            group = pd.read_csv(group_path)
            frame["seed"], frame["variant"] = seed, variant.upper()
            group["seed"], group["variant"] = seed, variant.upper()
            frame_parts.append(frame)
            group_parts.append(group)
    if missing:
        (output / "missing_and_failure_checklist.json").write_text(json.dumps({
            "missing_inputs": missing,
            "checks": {"all_four_variants": False, "all_requested_seeds": False},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        raise FileNotFoundError("Missing complete fixed-final validation outputs:\n" + "\n".join(missing))
    frames = pd.concat(frame_parts, ignore_index=True)
    groups = pd.concat(group_parts, ignore_index=True)
    frames.to_csv(output / "frame_metrics_long.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(output / "position_metrics_long.csv", index=False, encoding="utf-8-sig")

    numeric = [
        column for column in groups.select_dtypes(include=[np.number]).columns
        if column != "seed" and is_outcome_metric(column)
    ]
    gain_rows = []
    formulas = {
        "d2s_on_s2d_off": ("J10", "J00"),
        "d2s_on_s2d_on": ("J11", "J01"),
        "s2d_on_d2s_off": ("J01", "J00"),
        "s2d_on_d2s_on": ("J11", "J10"),
        "total": ("J11", "J00"),
    }
    index_columns = ["seed", "group_id", "dataset"]
    for metric in numeric:
        pivot = groups.pivot_table(index=index_columns, columns="variant", values=metric, aggfunc="mean")
        if not set(VARIANTS).issubset({str(value).lower() for value in pivot.columns}):
            continue
        pivot.columns = [str(value).upper() for value in pivot.columns]
        sign, (scale, unit) = orientation(metric), display_scale(metric)
        for effect, (high, low) in formulas.items():
            values = sign * (pivot[high] - pivot[low]) * scale
            for key, value in values.items():
                gain_rows.append({
                    "seed": key[0], "group_id": key[1], "dataset": key[2],
                    "metric": metric, "effect": effect, "improvement": value, "unit": unit,
                })
        interaction = sign * (pivot["J11"] - pivot["J10"] - pivot["J01"] + pivot["J00"]) * scale
        for key, value in interaction.items():
            gain_rows.append({
                "seed": key[0], "group_id": key[1], "dataset": key[2],
                "metric": metric, "effect": "interaction", "improvement": value, "unit": unit,
            })
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(output / "paired_gains_by_position.csv", index=False, encoding="utf-8-sig")
    seed_summary = gains.groupby(["seed", "metric", "effect", "unit"], as_index=False)["improvement"].mean()
    seed_summary.to_csv(output / "paired_gains_by_seed.csv", index=False, encoding="utf-8-sig")
    overall = seed_summary.groupby(["metric", "effect", "unit"], as_index=False)["improvement"].agg(
        mean="mean", seed_std="std", n_seeds="count"
    )
    overall.to_csv(output / "paired_gains_summary.csv", index=False, encoding="utf-8-sig")

    priority = overall[overall["metric"].isin([
        "p0_vessel_dice", "p0_vessel_precision", "p0_vessel_recall", "layer_dice",
        "upper_boundary_mae", "lower_boundary_mae", "thickness_mae", "psnr", "layer_roi_psnr",
        "ssim", "rmse", "reference_edge_mae", "epi", "vessel_stroma_cnr_abs_error",
        "repeat_denoised_mae", "repeat_layer_dice", "repeat_vessel_dice",
    ])]
    markdown = [
        "# Detached 2×2 interaction paired-gain report",
        "",
        "Primary unit is anatomical position (`group_id`): frames are averaged first, positions are equally weighted within each seed, and only then are seed mean/SD calculated.",
        "Positive values always mean improvement. Dice/precision/recall are percentage points; PSNR/SNR are dB; boundary and thickness errors are pixels.",
        "No confidence interval is reported: three seeds are too few for a reliable seed-level interval, and repeated frames are not treated as independent subjects.",
        "",
        "## Priority metrics",
        "",
        ("```text\n" + priority.to_string(index=False) + "\n```") if not priority.empty else "No priority metrics were available.",
        "",
        "## Interpretation guardrails",
        "",
        "These are fixed-threshold P0, fixed-final-checkpoint comparisons. Diagnostic perturbations, oracle ROIs, calibrated thresholds and post-processing are excluded from network gain claims.",
    ]
    (output / "SUMMARY.md").write_text("\n".join(markdown), encoding="utf-8")
    (output / "missing_and_failure_checklist.json").write_text(json.dumps({
        "missing_inputs": missing,
        "checks": {
            "all_four_variants": True,
            "all_requested_seeds": True,
            "position_first_reduction": True,
            "fixed_final_checkpoint": True,
            "fixed_p0_threshold_0_5": True,
            "confidence_interval_reported": False,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
