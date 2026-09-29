#!/usr/bin/env python
"""Create position-primary summaries for the segmentation-guided protocol."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.dose_response import stable_sha


COMPARISONS = (("B3R", "B0"), ("B3R", "B3"), ("B3R", "B6R"))


def _parse_run(run: Path) -> tuple[int, int, str]:
    match = re.search(r"fold(\d+).*(b0|b1|b3r?|b6r?)_seed(\d+)", run.as_posix(), re.I)
    if not match:
        raise ValueError(f"Cannot parse fold/arm/seed from {run}")
    return int(match.group(1)), int(match.group(3)), match.group(2).upper()


def _bootstrap(values: pd.Series, seed: int = 20260929, draws: int = 10000) -> tuple[float, float]:
    clean = values.dropna().to_numpy(float)
    if not len(clean):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(clean, (draws, len(clean)), replace=True), axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="*", default=[])
    parser.add_argument("--intervention-report")
    parser.add_argument("--residual-report")
    parser.add_argument("--oracle-report")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing summary: {output}")
    output.mkdir(parents=True)
    tables = []
    missing = []
    for raw in args.run_dirs:
        run = Path(raw).resolve()
        fold, seed, arm = _parse_run(run)
        for endpoint, folder in (("epoch12", "validation_epoch012"),
                                 ("epoch20", "validation_last"),
                                 ("best", "validation_best")):
            path = run / folder / "group_metrics.csv"
            if not path.is_file():
                missing.append(str(path))
                continue
            part = pd.read_csv(path)
            part.insert(0, "fold", fold); part.insert(1, "seed", seed)
            part.insert(2, "arm", arm); part.insert(3, "endpoint", endpoint)
            tables.append(part)
    cv = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    cv.to_csv(output / "CV_RESULTS.csv", index=False)
    paired_rows = []
    if not cv.empty:
        numeric = cv.select_dtypes(include=[np.number]).columns.difference(["fold", "seed"])
        for endpoint in cv.endpoint.unique():
            part = cv[cv.endpoint.eq(endpoint)]
            for metric in numeric:
                pivot = part.pivot_table(index=["group_id", "seed"], columns="arm", values=metric)
                for left, right in COMPARISONS:
                    if {left, right}.issubset(pivot.columns):
                        delta = (pivot[left] - pivot[right]).rename("difference").reset_index()
                        # Seeds are repeats: aggregate within anatomical position first.
                        by_position = delta.groupby("group_id", as_index=False).difference.mean()
                        lo, hi = _bootstrap(by_position.difference)
                        paired_rows.append({"endpoint": endpoint, "comparison": f"{left}-{right}",
                                            "metric": metric, "mean": by_position.difference.mean(),
                                            "sd_across_positions": by_position.difference.std(ddof=1),
                                            "ci95_low": lo, "ci95_high": hi,
                                            "improved_positions": int((by_position.difference > 0).sum()),
                                            "position_count": len(by_position),
                                            "statistical_unit": "anatomical_position"})
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output / "PAIRED_POSITION_DIFFERENCES.csv", index=False)
    sources = {}
    for name, raw, expected in (("INTERVENTION_RESULTS.csv", args.intervention_report, "INTERVENTION_RESULTS.csv"),
                                ("RESIDUAL_ANALYSIS.csv", args.residual_report, "RESIDUAL_ANALYSIS.csv"),
                                ("ORACLE_RESULTS.csv", args.oracle_report, "ORACLE_RESULTS.csv")):
        if raw:
            source = Path(raw).resolve()
            source = source / expected if source.is_dir() else source
            if source.is_file():
                target = output / name
                target.write_bytes(source.read_bytes())
                sources[name] = str(source)
    for name in ("INTERVENTION_RESULTS.csv", "RESIDUAL_ANALYSIS.csv", "ORACLE_RESULTS.csv",
                 "GATE_OR_CONTROLLER_STATISTICS.csv"):
        if not (output / name).exists():
            pd.DataFrame().to_csv(output / name, index=False)
    (output / "EXPERIMENT_MATRIX.md").write_text(
        "# Experiment matrix\n\nB0/B1/B3/B6/B3R/B6R; primary endpoint is epoch12, P0, threshold 0.5.\n",
        encoding="utf-8")
    (output / "PROTOCOL.md").write_text(
        "# Protocol\n\nDevelopment-only grouped four-fold CV. Sealed test assets are not opened.\n",
        encoding="utf-8")
    (output / "METRIC_DEFINITIONS.md").write_text(
        "# Metrics\n\nFrame metrics are aggregated within anatomical position; seeds are training repeats.\n",
        encoding="utf-8")
    (output / "FAILURE_CASES.md").write_text("# Failure cases\n\nSee missing-items in summary.json.\n", encoding="utf-8")
    status = "passed" if not missing and tables else "incomplete"
    summary = {"status": status, "test_assets_opened": 0, "run_count": len(args.run_dirs),
               "position_count": int(cv.group_id.nunique()) if not cv.empty else 0,
               "primary_endpoint": "epoch12/P0/threshold=0.5/position-equal",
               "statistical_limitation": "16 development positions remain a small sample",
               "sources": sources, "missing": missing}
    summary["summary_sha256"] = stable_sha(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "SUMMARY.md").write_text(
        f"# Summary\n\nStatus: **{status}**. Independent unit: anatomical position. "
        "Best checkpoints are sensitivity analyses only.\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
