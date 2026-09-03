from __future__ import annotations

import numpy as np
import pytest

from tools.oct_denoise_benchmark.adapters import AdapterUnavailable, denoise


@pytest.mark.parametrize(
    "config",
    [
        {"method_id": "noisy_identity"},
        {"method_id": "gaussian", "sigma": 0.8},
        {"method_id": "tv", "weight": 0.03, "max_num_iter": 20},
        {"method_id": "wavelet", "wavelet": "db2", "level": 1, "threshold_scale": 0.8},
        {"method_id": "nlm_speckle", "patch_size": 3, "search_radius": 1, "h": 0.15},
    ],
)
def test_adapter_contract_and_determinism(config):
    image = np.random.default_rng(42).random((31, 47), dtype=np.float32)
    first = denoise(image, config, {})
    second = denoise(image, config, {})
    assert first.shape == image.shape
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("method_id", ["msbtd", "ascibp"])
def test_blocked_matlab_adapters_are_explicit(method_id):
    image = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(AdapterUnavailable):
        denoise(image, {"method_id": method_id}, {})
