#!/usr/bin/env python
"""Apply the preregistered seed-42 dual-view pilot gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.dose_response import write_strict_json_exclusive


THRESHOLDS = {
    "vessel_dice_min_improvement": 0.0,
    "vessel_recall_max_decline": 0.01,
    "vessel_boundary_band_dice_max_decline": 0.01,
    "vessel_component_small_recall_at_025_max_decline": 0.02,
    "vessel_component_low_contrast_recall_at_025_max_decline": 0.02,
}


def evaluate_gate(table: pd.DataFrame) -> dict:
    fixed = table[table["selection"].astype(str).eq("last")].copy()
    evidence = []

    def add(comparison: str, metric: str, threshold: float, operator: str) -> None:
        row = fixed[
            fixed["comparison"].astype(str).eq(comparison)
            & fixed["metric"].astype(str).eq(metric)
        ]
        observed = None if len(row) != 1 else float(row.iloc[0]["mean_improvement"])
        passed = observed is not None and (
            observed > threshold if operator == ">" else observed >= threshold
        )
        evidence.append({
            "comparison": comparison,
            "metric": metric,
            "operator": operator,
            "threshold": threshold,
            "observed_mean_improvement": observed,
            "passed": bool(passed),
            "statistical_unit": "anatomical_position",
        })

    # Superiority and compute/content control.  These thresholds are fixed in
    # source before pilot results are observed.
    for comparison in ("dual_vs_b0", "dual_vs_b1", "content_vs_b6"):
        add(comparison, "vessel_dice", THRESHOLDS["vessel_dice_min_improvement"], ">")
    # A small Dice gain cannot hide a material loss in sensitivity, boundary,
    # or the two immutable difficult-component strata.
    for metric, key in (
        ("vessel_recall", "vessel_recall_max_decline"),
        ("vessel_boundary_band_dice", "vessel_boundary_band_dice_max_decline"),
        (
            "vessel_component_small_recall_at_025",
            "vessel_component_small_recall_at_025_max_decline",
        ),
        (
            "vessel_component_low_contrast_recall_at_025",
            "vessel_component_low_contrast_recall_at_025_max_decline",
        ),
    ):
        add("dual_vs_b0", metric, -THRESHOLDS[key], ">=")
    # Attribution controls: correct mild content and its use must both matter.
    add("pair_vs_c1", "vessel_dice", 0.0, ">")
    add("ablation_vs_c5", "vessel_dice", 0.0, ">")

    return {
        "version": "noisy-mild-dual-view-pilot-gate-v1",
        "status": "passed" if all(item["passed"] for item in evidence) else "failed",
        "primary_selection": "fixed_final_last",
        "postprocess": "P0",
        "thresholds_frozen_before_formal": THRESHOLDS,
        "checks": evidence,
        "formal_allowed": all(item["passed"] for item in evidence),
        "failure_policy": "formal is blocked; thresholds must not be changed after pilot",
        "test_assets_opened": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = evaluate_gate(pd.read_csv(args.paired_summary))
    write_strict_json_exclusive(Path(args.output).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
