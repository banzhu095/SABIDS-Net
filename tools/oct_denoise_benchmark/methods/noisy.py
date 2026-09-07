from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from .base import AdapterContext


def noisy_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    return image.copy()
