#!/usr/bin/env python
"""Create paired position-level dual-view reports from validation-only outputs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.experiments.dose_response import write_strict_json

PRIMARY = (
    "vessel_dice", "vessel_recall", "vessel_boundary_band_dice", "vessel_roi_dice",
)
SECONDARY = (
    "vessel_precision", "layer_dice", "upper_boundary_mae", "lower_boundary_mae",
    "vessel_roi_fp_per_valid_pixel", "vessel_roi_fn_per_valid_pixel",
    "vessel_outside_gt_layer_fraction", "vessel_area_fraction_mae",
    "repeat_vessel_dice", "repeat_prediction_mae",
    "vessel_component_recall_at_025", "vessel_component_recall_at_050",
    "vessel_component_small_recall_at_025", "vessel_component_low_contrast_recall_at_025",
    "vessel_component_small_low_contrast_recall_at_025",
    "vessel_component_mean_coverage", "vessel_component_completely_missed_count",
)


def _load(run: Path, kind: str, c5: bool = False) -> pd.DataFrame:
    config = load_config(run / "resolved_config.yaml")
    arm = "C5" if c5 else str(config["dual_view"]["arm"])
    folder = run / ("validation_c5_last" if c5 else f"validation_{kind}")
    table = pd.read_csv(folder / "group_metrics.csv")
    if "source_split" in table and not table["source_split"].astype(str).eq("val").all():
        raise ValueError(f"Non-validation rows in {folder}")
    table.insert(0, "selection", kind)
    table.insert(0, "seed", int(config["seed"]))
    table.insert(0, "arm", arm)
    return table


def _paired(long: pd.DataFrame, selection: str) -> pd.DataFrame:
    table = long[long.selection.eq(selection)]
    keys = [column for column in ("seed", "group_id") if column in table]
    available = [metric for metric in (*PRIMARY, *SECONDARY) if metric in table]
    rows = []
    comparisons = {
        "dual_vs_b0": ("B3", "B0"), "dual_vs_b1": ("B3", "B1"),
        "content_vs_b6": ("B3", "B6"), "pair_vs_c1": ("B3", "C1"),
        "ablation_vs_c5": ("B3", "C5"),
    }
    for label, (left, right) in comparisons.items():
        a = table[table.arm.eq(left)].set_index(keys)
        b = table[table.arm.eq(right)].set_index(keys)
        common = a.index.intersection(b.index)
        for index in common:
            index_tuple = index if isinstance(index, tuple) else (index,)
            for metric in available:
                av, bv = a.loc[index, metric], b.loc[index, metric]
                if pd.isna(av) or pd.isna(bv):
                    continue
                improvement = float(av) - float(bv)
                if any(token in metric for token in (
                    "mae", "_fp_", "_fn_", "outside_gt_layer_fraction",
                    "completely_missed_count",
                )):
                    improvement = -improvement
                rows.append({
                    **dict(zip(keys, index_tuple)), "selection": selection,
                    "comparison": label, "metric": metric,
                    "left": float(av), "right": float(bv), "improvement": improvement,
                })
    return pd.DataFrame(rows)


def _stats(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(20260928)
    for (selection, comparison, metric), part in paired.groupby(["selection", "comparison", "metric"]):
        seed_means = part.groupby("seed")["improvement"].mean()
        values = seed_means.to_numpy(float)
        # Hierarchical cluster bootstrap: resample seeds, then positions within seed.
        boot = []
        seeds = sorted(part.seed.unique())
        for _ in range(2000):
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_values = []
            for seed in sampled_seeds:
                cluster = part[part.seed.eq(seed)].improvement.to_numpy(float)
                seed_values.append(float(rng.choice(cluster, size=len(cluster), replace=True).mean()))
            boot.append(float(np.mean(seed_values)))
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        position_values = part.improvement.to_numpy(float)
        rows.append({
            "selection": selection, "comparison": comparison, "metric": metric,
            "mean_improvement": mean, "seed_sd": sd,
            "hierarchical_bootstrap_ci_low": float(np.quantile(boot, 0.025)),
            "hierarchical_bootstrap_ci_high": float(np.quantile(boot, 0.975)),
            "paired_effect_size_dz": (
                float(position_values.mean() / position_values.std(ddof=1))
                if len(position_values) > 1 and position_values.std(ddof=1) > 0 else float("nan")
            ),
            "improved_positions": int((position_values > 0).sum()),
            "position_count": int(len(position_values)), "seed_count": int(len(values)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    output.mkdir(parents=True)
    runs = [Path(path).resolve() for path in args.run_dirs]
    initialization_rows = []
    for run in runs:
        config = load_config(run / "resolved_config.yaml")
        audit = json.loads((run / "initialization_audit.json").read_text(encoding="utf-8"))
        initialization_rows.append({
            "arm": config["dual_view"]["arm"], "seed": int(config["seed"]),
            **{key: audit.get(key) for key in (
                "model_state_sha256", "initialization_checkpoint_sha256",
                "sampler_plan_sha256", "actual_augmentation_plan_sha256",
                "paired_cohort_sha256", "label_assets_decoded_sha256",
            )},
        })
    initialization = pd.DataFrame(initialization_rows)
    pairing_fields = [column for column in initialization.columns if column not in {"arm", "seed"}]
    pairing_checks = {
        f"seed{seed}_{field}": bool(part[field].notna().all() and part[field].nunique() == 1)
        for seed, part in initialization.groupby("seed") for field in pairing_fields
    }
    if not all(pairing_checks.values()):
        raise ValueError(f"Paired initialization/data/augmentation audit failed: {pairing_checks}")
    initialization.to_csv(output / "initialization_pairing_audit.csv", index=False)
    tables = []
    for run in runs:
        tables.append(_load(run, "last"))
        if (run / "validation_best/group_metrics.csv").is_file():
            tables.append(_load(run, "best"))
        config = load_config(run / "resolved_config.yaml")
        if config["dual_view"]["arm"] == "B3" and (run / "validation_c5_last/group_metrics.csv").is_file():
            tables.append(_load(run, "last", c5=True))
    long = pd.concat(tables, ignore_index=True)
    required = {"B0", "B1", "B3", "B6", "C1", "C5"}
    if set(long[long.selection.eq("last")].arm) != required:
        raise ValueError(f"Fixed-final matrix incomplete: {sorted(set(long.arm))}")
    metric_columns = [metric for metric in (*PRIMARY, *SECONDARY) if metric in long]
    long[["arm", "seed", "selection", "group_id", *metric_columns]].to_csv(
        output / "metrics_by_seed_position.csv", index=False
    )
    paired = pd.concat(
        [_paired(long, selection) for selection in sorted(long.selection.unique())],
        ignore_index=True,
    )
    paired.to_csv(output / "paired_position_differences.csv", index=False)
    stats = _stats(paired)
    stats.to_csv(output / "paired_summary.csv", index=False)
    stats.to_csv(output / "RESULTS_TABLE.csv", index=False)
    fixed_frames = long[long.selection.eq("last")]
    failure_index = fixed_frames.pivot_table(
        index=["seed", "group_id"], columns="arm", values="vessel_dice"
    ).reset_index()
    for required_arm in ("B0", "B3"):
        if required_arm not in failure_index:
            failure_index[required_arm] = float("nan")
    failure_index["b3_minus_b0_vessel_dice"] = failure_index["B3"] - failure_index["B0"]
    failure_index = failure_index.sort_values(
        ["b3_minus_b0_vessel_dice", "B3"], kind="stable"
    )
    failure_index.to_csv(output / "failure_case_index.csv", index=False)
    fixed = stats[stats.selection.eq("last")]
    def positive(comparison: str, metric: str, allow_equal: bool = False) -> bool:
        row = fixed[(fixed.comparison == comparison) & (fixed.metric == metric)]
        if not len(row):
            return False
        value = float(row.iloc[0].mean_improvement)
        return value >= 0 if allow_equal else value > 0
    required_columns = {
        *PRIMARY,
        "vessel_component_small_recall_at_025",
        "vessel_component_low_contrast_recall_at_025",
    }
    missing_required = sorted(required_columns - set(long.columns))
    majority = all(
        bool(
            len(fixed[(fixed.comparison == comparison) & (fixed.metric == "vessel_dice")])
            and int(fixed[(fixed.comparison == comparison) & (fixed.metric == "vessel_dice")].iloc[0].improved_positions)
            > int(fixed[(fixed.comparison == comparison) & (fixed.metric == "vessel_dice")].iloc[0].position_count) / 2
        )
        for comparison in ("dual_vs_b0", "dual_vs_b1")
    )
    dual_supported = not missing_required and all(
        positive(comparison, "vessel_dice")
        for comparison in ("dual_vs_b0", "dual_vs_b1")
    ) and all(
        positive("dual_vs_b0", metric, allow_equal=True)
        for metric in (
            "vessel_recall", "vessel_boundary_band_dice",
            "vessel_component_small_recall_at_025",
            "vessel_component_low_contrast_recall_at_025",
        )
    ) and majority
    content_supported = dual_supported and all(
        positive(comparison, "vessel_dice")
        for comparison in ("content_vs_b6", "pair_vs_c1", "ablation_vs_c5")
    )
    b1_better = False
    if {"B0", "B1"}.issubset(set(long.arm)):
        pivot = long[long.selection.eq("last")].pivot_table(index=["seed", "group_id"], columns="arm", values="vessel_dice")
        b1_better = bool((pivot["B1"] - pivot["B0"]).mean() > 0)
    conclusion = (
        "mild图像本身有助于分割" if content_supported and b1_better
        else "轻度降噪视图与noisy图像具有互补信息，作为辅助视图促进分割" if content_supported
        else "未满足把收益归因于正确配对降噪信息的预注册门槛"
    )
    result = {
        "status": "passed", "dual_view_superiority_supported": dual_supported,
        "paired_denoising_content_supported": content_supported,
        "b1_mild_single_view_better_than_b0": b1_better,
        "allowed_conclusion": conclusion,
        "missing_preregistered_metrics": missing_required,
        "majority_positions_same_direction": majority,
        "statistical_unit": "anatomical_position",
        "bootstrap": "hierarchical seed then anatomical-position resampling; seeds are not independent patients",
        "limitation": "Only three validation positions; intervals are unstable and not population-level inference.",
        "test_assets_opened": 0,
        "initialization_pairing_checks": pairing_checks,
    }
    write_strict_json(output / "summary.json", result)
    (output / "SUMMARY.md").write_text(
        "# Dual-view validation summary\n\n" + conclusion + "\n\n"
        "Primary selection is fixed-final at P0/0.5. Best is sensitivity only. "
        "The unit is anatomical position; frames and seeds are not treated as independent patients.\n",
        encoding="utf-8",
    )
    (output / "EXPERIMENT_MATRIX.md").write_text(
        "# Experiment matrix\n\n"
        "| Arm | Input/control |\n|---|---|\n"
        "| B0 | noisy single view |\n| B1 | mild single view |\n"
        "| B3 | noisy + correctly paired mild |\n| B6 | noisy + noisy compute/parameter control |\n"
        "| C1 | noisy + shuffled different-position mild |\n"
        "| C5 | B3 last checkpoint with auxiliary disabled |\n",
        encoding="utf-8",
    )
    (output / "METRIC_DEFINITIONS.md").write_text(
        "# Metric definitions\n\n"
        "Primary: vessel Dice, vessel recall, vessel boundary-band Dice and GT-layer ROI vessel Dice. "
        "All use P0 threshold 0.5 on the model grid. Component recall@0.25/@0.5 is the fraction "
        "of immutable GT components with at least that predicted coverage. Size thresholds come "
        "from train GT; low contrast comes from train noisy only. Positive paired improvement means "
        "better performance; error metrics are sign-reversed.\n",
        encoding="utf-8",
    )
    failures = []
    if not dual_supported:
        failures.append("B3 did not beat both B0 and B1 on fixed-final vessel Dice.")
    if not content_supported:
        failures.append("B3 did not clear B6, C1 and C5 content-attribution controls.")
    worst = failure_index.head(min(10, len(failure_index)))
    if len(worst):
        failures.append("Lowest paired B3-B0 anatomical positions (fixed before qualitative review):")
        failures.extend(
            f"seed={int(row.seed)}, group={row.group_id}, B3-B0 Dice={row.b3_minus_b0_vessel_dice:.6f}, B3 Dice={row.B3:.6f}"
            for row in worst.itertuples()
        )
    (output / "FAILURE_CASES.md").write_text(
        "# Failure checks\n\n" + ("\n".join(f"- {item}" for item in failures) if failures else "- No aggregate gate failure; inspect per-position failure rows.") + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
