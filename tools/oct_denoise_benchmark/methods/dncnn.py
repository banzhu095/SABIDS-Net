from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import torch
from torch import nn
from .base import AdapterContext
from .deep_common import load_checkpoint, tiled_forward


class DnCNN(nn.Module):
    """Classic grayscale DnCNN residual-noise predictor."""
    def __init__(self, depth: int = 17, features: int = 64):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(1, features, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers.extend([nn.Conv2d(features, features, 3, padding=1, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True)])
        layers.append(nn.Conv2d(features, 1, 3, padding=1, bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return noisy - self.body(noisy)


def dncnn_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    model = DnCNN(depth=int(config.get("depth", 17)), features=int(config.get("features", 64)))
    load_checkpoint(model, context, "dncnn_paired")
    return tiled_forward(model, image, context)
