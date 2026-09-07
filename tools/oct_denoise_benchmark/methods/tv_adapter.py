from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from .base import AdapterContext


def tv_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    return denoise_tv_chambolle(
        image,
        weight=float(config["weight"]),
        eps=float(config.get("eps", 2e-4)),
        max_num_iter=int(config.get("max_num_iter", 200)),
        channel_axis=None,
    )
