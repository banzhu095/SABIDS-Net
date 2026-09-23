#!/usr/bin/env python
"""Audit a completed formal-teacher overfit diagnostic without test access."""
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
from sabids.experiments.d2_teacher import audit_teacher_history
from sabids.experiments.dose_response import resolve, write_strict_json_exclusive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.01)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    run = resolve(root, args.run_dir)
    config = load_config(run / "resolved_config.yaml")
    if config.get("formal_d2_teacher", {}).get("run_mode") != "overfit":
        raise ValueError("Run is not a formal-teacher overfit diagnostic")
    history = audit_teacher_history(run / "history.csv", int(config["train"]["epochs"]))
    epoch0_path = run / "diagnostics" / "epoch000_metrics.json"
    if not epoch0_path.is_file():
        raise ValueError("Teacher overfit lacks epoch-0 validation diagnostics")
    epoch0 = json.loads(epoch0_path.read_text(encoding="utf-8-sig"))
    initial = float(epoch0["val"]["vessel_soft_dice"])
    gain = float(history["best_value"]) - initial
    parameters = json.loads(
        (run / "formal_teacher_parameter_audit.json").read_text(encoding="utf-8-sig")
    )
    if parameters.get("status") != "passed":
        raise ValueError("Teacher overfit parameter audit failed")
    if gain < float(args.minimum_gain):
        raise ValueError(
            f"Teacher overfit gain {gain:.6f} is below required {args.minimum_gain:.6f}"
        )
    report = {
        "status": "passed",
        "run_dir": str(run),
        "initial_val_vessel_soft_dice": initial,
        "best_val_vessel_soft_dice": float(history["best_value"]),
        "best_epoch": int(history["best_epoch"]),
        "gain": gain,
        "minimum_gain": float(args.minimum_gain),
        "optimizer_steps": int(parameters["optimizer_steps"]),
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(resolve(root, args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
