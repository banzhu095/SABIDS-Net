#!/usr/bin/env python
"""Fail-closed finite/update audit for a diagnostic D2 CUDA overfit run."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from sabids.experiments.dose_response import write_strict_json_exclusive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).expanduser().resolve()
    history = pd.read_csv(run / "history.csv", low_memory=False)
    required_columns = [
        column for column in history.columns
        if column.startswith("train_")
        or column in {"epoch", "lr", "val_psnr", "val_teacher_task_preservation"}
    ]
    numeric = history[required_columns].apply(pd.to_numeric, errors="coerce")
    finite = bool(not numeric.empty and np.isfinite(numeric.to_numpy()).all())
    total = pd.to_numeric(history.get("train_total"), errors="coerce")
    decreased = bool(len(total) >= 2 and np.isfinite(total).all() and total.iloc[-1] < total.iloc[0])
    teacher = json.loads((run / "teacher_audit.json").read_text(encoding="utf-8"))
    parameters = json.loads((run / "parameter_audit.json").read_text(encoding="utf-8"))
    checks = {
        "at_least_two_epochs": len(history) >= 2,
        "all_required_training_values_finite": finite,
        "training_total_decreased": decreased,
        "teacher_unchanged": int(teacher.get("changed_parameter_count", -1)) == 0,
        "frozen_parameters_unchanged": int(parameters.get("changed_frozen_parameter_count", -1)) == 0,
        "trainable_parameters_changed": int(parameters.get("changed_trainable_parameter_count", 0)) > 0,
        "checkpoint_bindings_present": all((run / name).is_file() for name in (
            "checkpoint_binding_best_pixel.json", "checkpoint_binding_best_task_preserving.json",
            "checkpoint_binding_last.json")),
    }
    result = {"status": "passed" if all(checks.values()) else "blocked",
              "checks": checks, "run_dir": str(run), "test_assets_opened": 0}
    write_strict_json_exclusive(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed": raise SystemExit(2)


if __name__ == "__main__": main()
