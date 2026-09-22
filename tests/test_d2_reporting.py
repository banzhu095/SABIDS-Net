import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from sabids.config import save_config


ROOT = Path(__file__).resolve().parents[1]


def _report_run(root: Path, name: str, config: dict, value: float) -> Path:
    run = root / name
    validation = run / "validation_results"
    prediction = validation / "predictions" / "PKU37"
    prediction.mkdir(parents=True)
    save_config(config, run / "resolved_config.yaml")
    identity = {
        "sample_id": "fixed_frame", "group_id": "pku_0006",
        "patient_id": "pku_0006", "dataset": "PKU37",
    }
    pd.DataFrame([{**identity, "vessel_dice": value}]).to_csv(
        validation / "frame_metrics.csv", index=False
    )
    pd.DataFrame([{**identity, "vessel_dice": value + .01}]).to_csv(
        validation / "group_metrics.csv", index=False
    )
    pd.DataFrame([{**identity, "component_id": 1, "coverage": value}]).to_csv(
        validation / "component_metrics.csv", index=False
    )
    pd.DataFrame([{**identity, "component_id": 1, "noisy_local_contrast": .2}]).to_csv(
        validation / "contrast_metrics.csv", index=False
    )
    pd.DataFrame([{**identity, "residual_structure_leakage": .01}]).to_csv(
        validation / "structure_leakage_metrics.csv", index=False
    )
    (prediction / "fixed_frame_vessel_prob.png").write_bytes(b"fixture")
    return run


def test_group_metrics_is_explicit_position_table_and_dose_arm_is_not_unknown(tmp_path):
    common = {"seed": 42, "evaluation": {"use_test": False}}
    d2 = _report_run(
        tmp_path, "d25", {
            **common, "train": {"stage": "denoise"},
            "d2": {"enabled": True, "arm": "D25"},
        }, .7
    )
    dose = _report_run(
        tmp_path, "d2_task_a050", {
            **common,
            "dose_response": {"enabled": True, "curve_type": "d2_task", "alpha": .5},
        }, .72,
    )
    output = tmp_path / "report"
    process = subprocess.run([
        sys.executable, str(ROOT / "tools/summarize_d2_seed42.py"),
        "--project-root", str(tmp_path), "--run-dirs", str(d2), str(dose),
        "--output", str(output), "--fixed-sample-ids", "fixed_frame",
    ], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    manifest = json.loads((output / "report_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "passed"
    assert manifest["position_level_source_filename"] == "group_metrics.csv"
    assert manifest["position_equal_aggregation"] is True
    positions = pd.read_csv(output / "metrics_by_position.csv")
    assert len(positions) == 2
    assert set(positions["arm"]) == {"D25", "d2_task:alpha=0.5"}
    assert set(positions["vessel_dice"].round(2)) == {.71, .73}
    frames = pd.read_csv(output / "metrics_by_image.csv")
    assert set(frames["vessel_dice"].round(2)) == {.70, .72}
