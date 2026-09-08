from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tools.oct_denoise_benchmark.data import audit_protocol, load_protocol_manifest
from tools.oct_denoise_benchmark.cli import _candidate_grid, _complete_summary, _evaluate_candidates, _expand_non_bm3d, _position_macro_candidates, _select_candidate
from tools.oct_denoise_benchmark.metrics import ms_ssim
from tools.oct_denoise_benchmark.methods import AdapterContext, denoise
from tools.oct_denoise_benchmark.methods.dncnn import DnCNN
from tools.oct_denoise_benchmark.methods.ksvd_adapter import _dct_dictionary, _ksvd, omp
from tools.oct_denoise_benchmark.package_light import materialize_fixed_atlas
from tools.oct_denoise_benchmark.statistics import bootstrap_confidence_intervals


@pytest.mark.parametrize("config", [
    {"method_id": "noisy_identity"},
    {"method_id": "tv_chambolle", "weight": 0.03, "eps": 2e-4, "max_num_iter": 10},
    {"method_id": "nlm", "h_sigma_multiplier": 0.8, "patch_size": 3, "patch_distance": 2, "fast_mode": True, "provide_sigma": True},
])
def test_standard_adapter_contract(config):
    image = np.random.default_rng(42).random((31, 47), dtype=np.float32)
    first = denoise(image, config, AdapterContext())
    second = denoise(image, config, AdapterContext())
    assert first.shape == image.shape and first.dtype == np.float32
    assert np.isfinite(first).all() and 0 <= first.min() <= first.max() <= 1
    np.testing.assert_array_equal(first, second)


def test_bm3d_standard_rejects_lc_profile():
    image = np.zeros((8, 8), np.float32)
    with pytest.raises(ValueError, match="standard"):
        denoise(image, {"method_id": "bm3d_standard", "sigma_psd": 0.05, "profile": "lc"}, AdapterContext())


def test_ksvd_dictionary_sparse_svd_and_determinism():
    import tools.oct_denoise_benchmark.methods.ksvd_adapter as ksvd_module
    ksvd_module.SVD_UPDATE_CALLS = 0
    rng = np.random.default_rng(42)
    signals = rng.normal(size=(16, 30))
    dictionary = _dct_dictionary(4, 12)
    np.testing.assert_allclose(np.linalg.norm(dictionary, axis=0), 1, atol=1e-10)
    codes = omp(dictionary, signals, 3)
    assert np.max(np.count_nonzero(np.abs(codes) > 1e-12, axis=0)) <= 3
    first_dictionary, first_codes = _ksvd(signals, dictionary, 2, 3, 0, np.random.default_rng(7))
    second_dictionary, second_codes = _ksvd(signals, dictionary, 2, 3, 0, np.random.default_rng(7))
    np.testing.assert_allclose(np.linalg.norm(first_dictionary, axis=0), 1, atol=1e-10)
    np.testing.assert_array_equal(first_dictionary, second_dictionary)
    np.testing.assert_array_equal(first_codes, second_codes)
    assert ksvd_module.SVD_UPDATE_CALLS > 0


def test_ksvd_reconstructs_without_holes_and_has_no_clean_context():
    image = np.random.default_rng(42).random((19, 23), dtype=np.float32)
    config = {"method_id": "ksvd_self", "patch_size": 4, "dictionary_atoms": 12, "iterations": 1, "omp_max_nonzero": 2, "stride": 4, "max_training_patches": 32}
    output = denoise(image, config, AdapterContext(seed=42))
    assert output.shape == image.shape and np.isfinite(output).all()
    assert not hasattr(AdapterContext(), "clean") and not hasattr(AdapterContext(), "reference")


def test_ksvd_required_search_dimensions_vary_and_noise_weight_is_effective():
    grid = _candidate_grid("ksvd_self")
    for key in ("patch_size", "dictionary_atoms", "iterations", "omp_max_nonzero", "noise_weight", "stride", "aggregation_weight"):
        assert len({item[key] for item in grid}) > 1
    image = np.random.default_rng(8).random((16, 18), dtype=np.float32)
    config = {"method_id": "ksvd_self", "patch_size": 4, "dictionary_atoms": 12, "iterations": 1,
              "omp_max_nonzero": 2, "stride": 4, "max_training_patches": 24, "aggregation_weight": 1.0}
    identity = denoise(image, {**config, "noise_weight": 0.0}, AdapterContext(seed=3))
    sparse = denoise(image, {**config, "noise_weight": 1.0}, AdapterContext(seed=3))
    np.testing.assert_allclose(identity, image, atol=2e-7)
    assert not np.allclose(sparse, image)


def test_calibration_partial_file_skips_completed_work(tmp_path: Path, monkeypatch):
    calls = {"count": 0}
    monkeypatch.setattr("tools.oct_denoise_benchmark.cli.read_image", lambda path: (np.zeros((16, 16), np.float32), {}))
    def fake_denoise(image, config, context):
        calls["count"] += 1
        return image
    monkeypatch.setattr("tools.oct_denoise_benchmark.cli.denoise", fake_denoise)
    monkeypatch.setattr("tools.oct_denoise_benchmark.cli.compute_metrics", lambda *args: {"psnr": 1.0, "ssim": 0.5})
    rows = pd.DataFrame([{"sample_id": "a", "position_id": "p1", "image_path": "n", "clean_path": "r"},
                         {"sample_id": "b", "position_id": "p2", "image_path": "n", "clean_path": "r"}])
    partial = tmp_path / "partial.csv"
    config = [{"method_id": "tv_chambolle", "weight": 0.1}]
    assert len(_evaluate_candidates("tv_chambolle", config, rows, 0, "full", partial)) == 2
    assert len(_evaluate_candidates("tv_chambolle", config, rows, 0, "full", partial)) == 2
    assert calls["count"] == 2


def test_incomplete_calibration_candidate_cannot_be_selected():
    expected = pd.DataFrame([{"sample_id": "a", "position_id": "p1"}, {"sample_id": "b", "position_id": "p2"}])
    rows = [{"candidate_uid": "x", "candidate_json": "{}", "sample_id": "a", "position_id": "p1", "psnr": 30.0, "ssim": 0.9}]
    with pytest.raises(RuntimeError, match="every registered validation sample"):
        _complete_summary(rows, expected)


def test_fixed_atlas_materializer_does_not_open_reference_before_lock(tmp_path: Path, monkeypatch):
    run = tmp_path / "run"; (run / "audit").mkdir(parents=True)
    monkeypatch.setattr("tools.oct_denoise_benchmark.package_light.read_image", lambda path: pytest.fail("opened sealed image"))
    result = materialize_fixed_atlas(tmp_path, run, pd.DataFrame())
    assert result["status"] == "not_materialized_test_sealed"


def test_deep_adapter_never_uses_random_weights(tmp_path: Path):
    image = np.zeros((16, 16), np.float32)
    with pytest.raises(ValueError, match="checkpoint"):
        denoise(image, {"method_id": "dncnn_paired", "depth": 3, "features": 4}, AdapterContext())
    model = DnCNN(depth=3, features=4)
    checkpoint = tmp_path / "model.pth"
    torch.save({"architecture": "dncnn_paired", "model": model.state_dict()}, checkpoint)
    context = AdapterContext(checkpoint=checkpoint)
    output = denoise(image, {"method_id": "dncnn_paired", "depth": 3, "features": 4}, context)
    assert output.shape == image.shape
    loaded = context.extras["loaded_model"]
    denoise(image, {"method_id": "dncnn_paired", "depth": 3, "features": 4}, context)
    assert context.extras["loaded_model"] is loaded


def test_external_datasets_are_never_development_splits(tmp_path: Path):
    frame = pd.DataFrame([
        {"sample_id": "p1", "group_id": "p1", "dataset": "PKU37", "split": "train", "image_path": "a", "clean_path": "b"},
        {"sample_id": "d1", "group_id": "d1", "dataset": "Duke17", "split": "train", "image_path": "a", "clean_path": "b"},
    ])
    path = tmp_path / "manifest.csv"; frame.to_csv(path, index=False)
    loaded = load_protocol_manifest(tmp_path, path)
    assert loaded.loc[loaded.dataset == "Duke17", "split"].item() == "external_test"


def test_parameter_selection_is_position_macro_with_ssim_tolerance_tiebreak():
    rows = []
    # Candidate A has many high-scoring frames in one position but is worse
    # when independent positions receive equal weight.
    for index in range(10):
        rows.append({"candidate_uid": "a", "candidate_json": '{"name":"a"}', "position_id": "p1", "sample_id": f"a{index}", "psnr": 30.0, "ssim": 0.8})
    rows.append({"candidate_uid": "a", "candidate_json": '{"name":"a"}', "position_id": "p2", "sample_id": "a10", "psnr": 10.0, "ssim": 0.8})
    rows.extend([
        {"candidate_uid": "b", "candidate_json": '{"name":"b"}', "position_id": "p1", "sample_id": "b1", "psnr": 21.0, "ssim": 0.7},
        {"candidate_uid": "b", "candidate_json": '{"name":"b"}', "position_id": "p2", "sample_id": "b2", "psnr": 21.0, "ssim": 0.7},
    ])
    summary = _position_macro_candidates(rows)
    assert _select_candidate(summary, 1e-4).candidate_uid == "b"
    tied = pd.DataFrame([
        {"candidate_uid": "x", "candidate_json": "{}", "position_macro_psnr": 25.00000, "position_macro_ssim": 0.70},
        {"candidate_uid": "y", "candidate_json": "{}", "position_macro_psnr": 24.99995, "position_macro_ssim": 0.80},
    ])
    assert _select_candidate(tied, 1e-4).candidate_uid == "y"


def test_bootstrap_contains_paired_method_difference_ci():
    rows = []
    for position, baseline, method in [("p1", 20.0, 22.0), ("p2", 21.0, 22.0), ("p3", 19.0, 22.0)]:
        rows.extend([
            {"dataset": "PKU37", "position_id": position, "method_id": "noisy_identity", "psnr": baseline},
            {"dataset": "PKU37", "position_id": position, "method_id": "nlm", "psnr": method},
        ])
    result = bootstrap_confidence_intervals(pd.DataFrame(rows), iterations=1000, seed=42)
    paired = result[(result.comparison_type == "paired_difference") & (result.baseline_method == "noisy_identity") & (result.method_id == "nlm") & (result.metric == "psnr")]
    assert len(paired) == 1 and paired.iloc[0].ci95_low <= paired.iloc[0]["mean"] <= paired.iloc[0].ci95_high


def test_ms_ssim_identity_and_degradation():
    image = np.random.default_rng(4).random((128, 128), dtype=np.float32)
    assert ms_ssim(image, image) == pytest.approx(1.0, abs=1e-6)
    degraded = np.clip(image + 0.2, 0, 1)
    assert 0 <= ms_ssim(degraded, image) < 1


def test_tv_and_nlm_boundary_search_expand_without_changing_other_parameters():
    tv = [{"method_id": "tv_chambolle", "weight": w, "eps": 1e-4, "max_num_iter": 300} for w in (0.01, 0.24)]
    expanded, boundaries = _expand_non_bm3d("tv_chambolle", tv[-1], tv)
    assert "weight" in boundaries and any(item["weight"] > 0.24 for item in expanded)
    assert all(item["method_id"] == "tv_chambolle" for item in expanded)
    nlm = [{"method_id": "nlm", "h_sigma_multiplier": h, "patch_size": 5, "patch_distance": 6, "fast_mode": True, "provide_sigma": True} for h in (0.5, 1.5)]
    expanded, boundaries = _expand_non_bm3d("nlm", nlm[-1], nlm)
    assert "h_sigma_multiplier" in boundaries and any(item["h_sigma_multiplier"] > 1.5 for item in expanded)
