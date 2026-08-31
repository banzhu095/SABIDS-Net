from __future__ import annotations

import json
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
