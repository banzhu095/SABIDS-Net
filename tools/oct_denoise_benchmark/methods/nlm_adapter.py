from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from skimage.restoration import denoise_nl_means, estimate_sigma
from .base import AdapterContext


def nlm_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    sigma = float(np.mean(estimate_sigma(image, channel_axis=None)))
    h = float(config.get("h", sigma * float(config.get("h_sigma_multiplier", 0.8))))
    provide_sigma = bool(config.get("provide_sigma", True))
    return denoise_nl_means(
        image,
        h=h,
        sigma=sigma if provide_sigma else 0.0,
        patch_size=int(config.get("patch_size", 5)),
        patch_distance=int(config.get("patch_distance", 6)),
        fast_mode=bool(config.get("fast_mode", True)),
        preserve_range=True,
        channel_axis=None,
    )
