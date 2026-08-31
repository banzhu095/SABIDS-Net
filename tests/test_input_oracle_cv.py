from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from tools.analyze_input_oracle_cv import orientation, scale_unit
from tools.build_input_oracle_folds import _balanced_assignment
from tools.input_oracle_cv_common import require_phase0
from tools.run_input_oracle_cv import _clean_once_loader
from tools.build_input_oracle_atlas import _tile


def test_error_orientation_and_units_are_improvement_positive() -> None:
    assert orientation("p0_vessel_dice") == 1.0
    assert orientation("lower_boundary_mae") == -1.0
    assert orientation("vessel_roi_fp_per_valid_pixel") == -1.0
    assert scale_unit("p0_vessel_dice") == (100.0, "percentage_points")
    assert scale_unit("thickness_mae") == (1.0, "px")


def test_balanced_assignment_is_deterministic_and_complete() -> None:
    table = pd.DataFrame({
        "group_id": [f"pku_{i:04d}" for i in range(16)],
        "vessel_fraction": np.linspace(0, 1, 16),
        "thickness": np.arange(16) % 5,
    })
    left = _balanced_assignment(table, 4, 20260831)
    right = _balanced_assignment(table, 4, 20260831)
    assert left == right
    assert set(left) == set(table["group_id"])
    assert sorted(list(left.values()).count(fold) for fold in range(4)) == [4, 4, 4, 4]


def test_blocked_phase0_cannot_authorize_training(tmp_path: Path) -> None:
    path = tmp_path / "audit/audit_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"status": "blocked", "training_authorized": False, "test_assets_opened": 0}))
    with pytest.raises(RuntimeError, match="blocked"):
        require_phase0(tmp_path, tmp_path)


class _CleanDataset(Dataset):
    def __init__(self) -> None:
        self.table = pd.DataFrame({"group_id": ["a", "a", "b", "b", "b"]})

    def __len__(self) -> int: return len(self.table)

    def __getitem__(self, index: int): return {"index": torch.tensor(index)}


def test_clean_validation_is_exactly_once_per_position() -> None:
    loader = DataLoader(_CleanDataset(), batch_size=1, shuffle=False)
    clean = _clean_once_loader(loader)
    observed = [int(batch["index"].item()) for batch in clean]
    assert observed == [0, 2]


def test_atlas_missing_asset_is_rendered_not_resized_from_empty(tmp_path: Path) -> None:
    tile, status = _tile(tmp_path / "absent.png", 0, 0, 32, 32)
    assert status == "missing"
    assert tile.shape == (144, 144, 3)


def test_three_arm_report_smoke(tmp_path: Path, monkeypatch) -> None:
    groups = ["pku_0001", "pku_0002", "pku_0003"]
    arm_offset = {"noisy": 0.0, "clean": 0.02, "denoised": 0.01}
    fold_root = tmp_path / "runs/input_oracle_cv/fold0"; fold_root.mkdir(parents=True)
    (fold_root / "d0_leakage_audit.json").write_text(json.dumps({"status": "passed", "test_assets_opened": 0}))
    (fold_root / "paired_data_plan_audit_fold0_seed42.json").write_text(json.dumps({"all_equal": True, "all_three_arms_present": True}))
    for arm, offset in arm_offset.items():
        directory = tmp_path / f"runs/input_oracle_cv/fold0/{arm}_seed42/final_validation"
        directory.mkdir(parents=True)
        table = pd.DataFrame({
            "sample_id": [f"{group}_f01" for group in groups], "group_id": groups,
            "dataset": "PKU37", "layer_dice": np.asarray([.8, .7, .6]) + offset,
            "p0_vessel_dice": np.asarray([.5, .4, .3]) + offset,
            "p0_vessel_precision": np.asarray([.55, .45, .35]) + offset,
            "p0_vessel_recall": np.asarray([.48, .38, .28]) + offset,
            "lower_boundary_mae": np.asarray([2.0, 3.0, 4.0]) - offset,
            "thickness_mae": np.asarray([3.0, 4.0, 5.0]) - offset,
        })
        table.to_csv(directory / "frame_metrics.csv", index=False)
        table.drop(columns="sample_id").to_csv(directory / "group_metrics.csv", index=False)
    output = tmp_path / "report"
    monkeypatch.setattr(sys, "argv", [
        "analyze_input_oracle_cv.py", "--project-root", str(tmp_path),
        "--runs", "runs/input_oracle_cv", "--output", str(output),
        "--folds", "0", "--seeds", "42",
    ])
    from tools.analyze_input_oracle_cv import main
    main()
    summary = pd.read_csv(output / "paired_gains_summary.csv")
    row = summary[(summary["comparison"] == "CLEAN-NOISY") & (summary["metric"] == "p0_vessel_dice")].iloc[0]
    assert row["mean"] == pytest.approx(2.0)
    assert json.loads((output / "missing_and_failure_checklist.json").read_text())["complete"] is True


def test_lightweight_package_excludes_checkpoints_and_float_cache(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "runs/input_oracle_cv/report"; report.mkdir(parents=True)
    (report / "missing_and_failure_checklist.json").write_text(json.dumps({"complete": True, "missing_inputs": []}))
    (report / "SUMMARY.md").write_text("ok")
    cache = tmp_path / "runs/input_oracle_cv/fold0/cache"; cache.mkdir(parents=True)
    (cache / "large.npy").write_bytes(b"forbidden")
    run = tmp_path / "runs/input_oracle_cv/fold0/noisy_seed42"; run.mkdir(parents=True)
    (run / "last.pth").write_bytes(b"forbidden")
    (run / "history.csv").write_text("epoch,loss\n1,1\n")
    output = tmp_path / "analysis.zip"
    monkeypatch.setattr(sys, "argv", [
        "package_input_oracle_analysis.py", "--project-root", str(tmp_path), "--output", str(output),
    ])
    from tools.package_input_oracle_analysis import main
    main()
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
    assert "MANIFEST.json" in names
    assert not any(name.endswith((".pth", ".npy")) for name in names)
