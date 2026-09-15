from __future__ import annotations

import numpy as np
import pandas as pd

from tools.denoise_result_review.roi_metrics import compute_roi_metrics, vessel_stroma_pair_metrics
from tools.denoise_result_review.roi_statistics import aggregate_roi_metrics, position_bootstrap


def test_synthetic_roi_metrics_psnr_ssim_rmse_and_vitreous_fields():
    noisy = np.full((32, 32), 0.2, np.float32)
    reference = np.zeros((32, 32), np.float32)
    denoised = np.full((32, 32), 0.1, np.float32)
    result = compute_roi_metrics(noisy, reference, denoised, "vitreous", bright_outlier_threshold=0.15)
    assert np.isclose(result["rmse"], 0.1, atol=1e-7)
    assert np.isclose(result["mae"], 0.1, atol=1e-7)
    assert np.isclose(result["psnr"], 20.0, atol=1e-6)
    assert result["ssim_win_size"] == 7
    assert result["bright_outlier_fraction"] == 0.0
    assert "hf_residual_energy" in result


def test_vessel_stroma_cnr_and_expected_dark_polarity():
    vessel = np.array([0.1, 0.12, 0.08], dtype=np.float32)
    stroma = np.array([0.4, 0.42, 0.38], dtype=np.float32)
    result = vessel_stroma_pair_metrics(vessel, stroma, vessel, stroma)
    assert result["vessel_stroma_cnr"] > 0
    assert result["cnr_error_to_reference"] == 0
    assert result["polarity_preserved"] is True


def test_bootstrap_resamples_positions_not_roi_rows():
    rows = []
    for position, value in (("p1", 20.0), ("p2", 30.0)):
        for roi in range(3): rows.append({"dataset": "PKU37", "split": "test", "position_id": position, "sample_id": f"{position}_s", "tissue": "retina", "method_id": "nlm", "roi_id": f"r{roi}", "psnr": value, "seed": 0})
    per_roi = pd.DataFrame(rows)
    _, position = aggregate_roi_metrics(per_roi)
    result = position_bootstrap(position, iterations=500, seed=42, complete_position_count=2)
    assert result.iloc[0].n_positions == 2
    assert result.iloc[0].analysis_scope == "fixed_roi_confirmatory_full_test_positions"
    assert position_bootstrap(position, complete_position_count=3).empty
