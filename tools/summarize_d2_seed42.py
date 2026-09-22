#!/usr/bin/env python
"""Assemble validation-only D2 tables; group_metrics.csv is position-level."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.experiments.dose_response import write_strict_json_exclusive


def _read(path: Path, run_id: str, arm: str) -> pd.DataFrame:
    table = pd.read_csv(path, low_memory=False)
    table.insert(0, "arm", arm)
    table.insert(0, "run_id", run_id)
    return table


def _arm_label(config: dict) -> str:
    d2_arm = config.get("d2", {}).get("arm")
    if d2_arm:
        return str(d2_arm)
    dose = config.get("dose_response", {})
    if dose.get("enabled") and dose.get("curve_type") is not None and dose.get("alpha") is not None:
        return f"{dose['curve_type']}:alpha={float(dose['alpha']):g}"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-sample-ids", nargs="+", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output)
    if not output.is_absolute(): output = (root / output).resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite D2 report: {output}")

    frames, positions, components, contrasts, leakages, failures, atlas = [], [], [], [], [], [], []
    for value in args.run_dirs:
        run = Path(value); run = run.resolve() if run.is_absolute() else (root / run).resolve()
        cfg_path = run / "resolved_config.yaml"
        if not cfg_path.is_file():
            failures.append({"run_id": run.name, "asset": str(cfg_path), "reason": "missing"}); continue
        cfg = load_config(cfg_path)
        if int(cfg.get("seed", -1)) != 42 or cfg.get("evaluation", {}).get("use_test") is not False:
            raise ValueError(f"Run is not seed-42 validation-only: {run}")
        arm = _arm_label(cfg)
        validation = run / "validation_results"
        common_assets = {
            "frame": validation / "frame_metrics.csv",
            # Explicit mapping: group_id is anatomical position in this protocol.
            "position": validation / "group_metrics.csv",
        }
        if cfg.get("d2", {}).get("enabled") and cfg.get("train", {}).get("stage") == "denoise":
            assets = {**common_assets, "leakage": validation / "structure_leakage_metrics.csv"}
        elif cfg.get("dose_response", {}).get("enabled"):
            assets = {
                **common_assets,
                "component": validation / "component_metrics.csv",
                "contrast": validation / "contrast_metrics.csv",
            }
        else:
            raise ValueError(f"Run is neither registered D2 nor matched dose segmentation: {run}")
        targets = {"frame": frames, "position": positions,
                   "component": components, "contrast": contrasts, "leakage": leakages}
        for kind, path in assets.items():
            if path.is_file(): targets[kind].append(_read(path, run.name, arm))
            else: failures.append({"run_id": run.name, "asset": str(path), "reason": "missing"})
        prediction_root = validation / "predictions"
        for sample_id in args.fixed_sample_ids:
            matches = sorted(prediction_root.rglob(f"{sample_id}_*.png")) if prediction_root.is_dir() else []
            if not matches:
                failures.append({"run_id": run.name, "asset": sample_id, "reason": "fixed atlas sample missing"})
            for path in matches:
                atlas.append({"run_id": run.name, "arm": arm, "sample_id": sample_id,
                              "asset": str(path), "selection_rule": "same preregistered sample IDs for every arm"})
    output.mkdir(parents=True)
    tables = {
        "metrics_by_image.csv": frames,
        "metrics_by_position.csv": positions,
        "component_metrics.csv": components,
        "contrast_metrics.csv": contrasts,
        "structure_leakage_metrics.csv": leakages,
    }
    combined = {}
    for name, pieces in tables.items():
        table = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        table.to_csv(output / name, index=False, encoding="utf-8-sig")
        combined[name] = table
    pd.DataFrame(failures, columns=["run_id", "asset", "reason"]).to_csv(
        output / "missing_assets.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(atlas, columns=["run_id", "arm", "sample_id", "asset", "selection_rule"]).to_csv(
        output / "fixed_atlas_manifest.csv", index=False, encoding="utf-8-sig"
    )
    position = combined["metrics_by_position.csv"]
    numeric = [column for column in position.select_dtypes(include="number") if column != "seed"]
    summary = position.groupby("arm", as_index=False)[numeric].mean() if not position.empty else pd.DataFrame()
    summary.to_csv(output / "RESULTS_TABLE.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "status": "passed" if not failures and not position.empty else "incomplete",
        "seed": 42, "position_level_source_filename": "group_metrics.csv",
        "position_unit": "group_id/anatomical position", "position_equal_aggregation": True,
        "run_count": len(args.run_dirs), "failure_count": len(failures),
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output / "report_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
