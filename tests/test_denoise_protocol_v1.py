from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tools.oct_denoise_benchmark.data import audit_protocol, load_protocol_manifest
from tools.oct_denoise_benchmark.methods import AdapterContext, denoise
from tools.oct_denoise_benchmark.methods.dncnn import DnCNN
from tools.oct_denoise_benchmark.methods.ksvd_adapter import _dct_dictionary, _ksvd, omp


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


def test_ksvd_reconstructs_without_holes_and_has_no_clean_context():
    image = np.random.default_rng(42).random((19, 23), dtype=np.float32)
    config = {"method_id": "ksvd_self", "patch_size": 4, "dictionary_atoms": 12, "iterations": 1, "omp_max_nonzero": 2, "stride": 4, "max_training_patches": 32}
    output = denoise(image, config, AdapterContext(seed=42))
    assert output.shape == image.shape and np.isfinite(output).all()
    assert not hasattr(AdapterContext(), "clean") and not hasattr(AdapterContext(), "reference")


def test_deep_adapter_never_uses_random_weights(tmp_path: Path):
    image = np.zeros((16, 16), np.float32)
    with pytest.raises(ValueError, match="checkpoint"):
        denoise(image, {"method_id": "dncnn_paired", "depth": 3, "features": 4}, AdapterContext())
    model = DnCNN(depth=3, features=4)
    checkpoint = tmp_path / "model.pth"
    torch.save({"architecture": "dncnn_paired", "model": model.state_dict()}, checkpoint)
    output = denoise(image, {"method_id": "dncnn_paired", "depth": 3, "features": 4}, AdapterContext(checkpoint=checkpoint))
    assert output.shape == image.shape


def test_external_datasets_are_never_development_splits(tmp_path: Path):
    frame = pd.DataFrame([
        {"sample_id": "p1", "group_id": "p1", "dataset": "PKU37", "split": "train", "image_path": "a", "clean_path": "b"},
        {"sample_id": "d1", "group_id": "d1", "dataset": "Duke17", "split": "train", "image_path": "a", "clean_path": "b"},
    ])
    path = tmp_path / "manifest.csv"; frame.to_csv(path, index=False)
    loaded = load_protocol_manifest(tmp_path, path)
    assert loaded.loc[loaded.dataset == "Duke17", "split"].item() == "external_test"
