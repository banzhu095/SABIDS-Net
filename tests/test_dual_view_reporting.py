from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from tools.check_dual_view_gate import evaluate_gate


def _gate_table(overrides: dict[tuple[str, str], float] | None = None) -> pd.DataFrame:
    values = {
        (comparison, "vessel_dice"): 0.01
        for comparison in (
            "dual_vs_b0", "dual_vs_b1", "content_vs_b6", "pair_vs_c1", "ablation_vs_c5"
        )
    }
    values.update({
        ("dual_vs_b0", "vessel_recall"): -0.005,
        ("dual_vs_b0", "vessel_boundary_band_dice"): -0.005,
        ("dual_vs_b0", "vessel_component_small_recall_at_025"): -0.01,
        ("dual_vs_b0", "vessel_component_low_contrast_recall_at_025"): -0.01,
    })
    values.update(overrides or {})
    return pd.DataFrame([
        {
            "selection": "last", "comparison": comparison, "metric": metric,
            "mean_improvement": value,
        }
        for (comparison, metric), value in values.items()
    ])


def test_gate_passes_only_preregistered_fixed_final_thresholds() -> None:
    passed = evaluate_gate(_gate_table())
    assert passed["status"] == "passed"
    assert passed["formal_allowed"] is True
    failed = evaluate_gate(_gate_table({("dual_vs_b0", "vessel_recall"): -0.011}))
    assert failed["status"] == "failed"
    assert failed["formal_allowed"] is False
    check = next(
        item for item in failed["checks"]
        if item["comparison"] == "dual_vs_b0" and item["metric"] == "vessel_recall"
    )
    assert check["threshold"] == -0.01


def test_light_package_marks_failed_pilot_incomplete_and_excludes_checkpoints(tmp_path) -> None:
    root = tmp_path / "project"
    report = root / "reports/adaptive_denoising/dual_view_v1/demo/pilot"
    report.mkdir(parents=True)
    (report / "gate.json").write_text(
        json.dumps({"status": "failed", "formal_allowed": False}), encoding="utf-8"
    )
    run = root / "runs/adaptive_denoising/pku37_binary_v3/dual_view_v1/demo_pilot/b3_seed42"
    run.mkdir(parents=True)
    (run / "history.csv").write_text("epoch,value\n1,0.1\n", encoding="utf-8")
    (run / "last.pth").write_bytes(b"must not be packaged")
    requested = tmp_path / "bundle.zip"
    script = Path(__file__).resolve().parents[1] / "tools/package_dual_view_for_gpt.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--project-root", str(root), "--run-id", "demo",
         "--output", str(requested)],
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    output = Path(result["output"])
    assert output.name == "bundle_incomplete.zip"
    assert result["zip_readable"] is True
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert any(name.endswith("history.csv") for name in names)
    assert not any(name.endswith(".pth") for name in names)
    assert any(name.endswith("MANIFEST.csv") for name in names)
