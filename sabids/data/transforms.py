from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch


def _resize_pad(
    image: np.ndarray,
    target_size: Tuple[int, int],
    is_mask: bool = False,
) -> np.ndarray:
    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if scale > 1.0 and not is_mask:
        interpolation = cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    canvas = np.zeros((target_h, target_w), dtype=np.float32)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _percentile_normalize(image: np.ndarray, low: float, high: float) -> np.ndarray:
    lo, hi = np.percentile(image, [low, high])
    if hi <= lo + 1e-8:
        return np.clip(image, 0.0, 1.0)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


@dataclass
class JointOCTTransform:
    target_size: Tuple[int, int] = (512, 1024)
    training: bool = True
    horizontal_flip: float = 0.5
    normalization: str = "fixed"
    percentile_low: float = 0.5
    percentile_high: float = 99.5
    strong_private_only: bool = True
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.85, 1.15)
    speckle_std: float = 0.03
    blur_probability: float = 0.1

    def _normalize(self, image: np.ndarray) -> np.ndarray:
        if self.normalization == "percentile":
            return _percentile_normalize(
                image, self.percentile_low, self.percentile_high
            )
        if self.normalization == "fixed":
            return np.clip(image, 0.0, 1.0)
        if self.normalization == "zscore":
            mean = float(image.mean())
            std = float(image.std()) + 1e-6
            value = (image - mean) / std
            return np.clip((value + 3.0) / 6.0, 0.0, 1.0)
        raise ValueError(f"Unknown normalization mode: {self.normalization}")

    def _strong(self, image: np.ndarray) -> np.ndarray:
        gamma = np.random.uniform(*self.gamma_range)
        contrast = np.random.uniform(*self.contrast_range)
        value = np.clip(image, 0.0, 1.0) ** gamma
        value = np.clip((value - 0.5) * contrast + 0.5, 0.0, 1.0)
        if self.speckle_std > 0:
            noise = np.random.normal(0.0, self.speckle_std, value.shape).astype(
                np.float32
            )
            value = np.clip(value + value * noise, 0.0, 1.0)
        if np.random.rand() < self.blur_probability:
            value = cv2.GaussianBlur(value, (3, 3), sigmaX=0.5)
        return value

    @staticmethod
    def _tensor(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(image[None])).float()

    def __call__(
        self,
        arrays: Dict[str, Optional[np.ndarray]],
        masks: Dict[str, Optional[np.ndarray]],
        allow_strong: bool,
    ) -> Dict[str, torch.Tensor]:
        flip = self.training and np.random.rand() < self.horizontal_flip
        output: Dict[str, torch.Tensor] = {}

        for key, array in arrays.items():
            if array is None:
                continue
            value = _resize_pad(array, self.target_size, is_mask=False)
            value = self._normalize(value)
            if flip:
                value = np.flip(value, axis=1)
            weak = value.copy()
            strong = self._strong(value.copy()) if self.training and allow_strong else value
            output[key] = self._tensor(strong)
            output[f"{key}_weak"] = self._tensor(weak)

        for key, mask in masks.items():
            if mask is None:
                continue
            value = _resize_pad(mask, self.target_size, is_mask=True)
            if flip:
                value = np.flip(value, axis=1)
            output[key] = self._tensor((value > 0.5).astype(np.float32))
        return output

