from __future__ import annotations

import torch
from torch import nn


class SpatialExpertController(nn.Module):
    """Constrained noisy/mild/strong mixer for gated Stage-E experiments.

    The module predicts weights only; it cannot synthesize image content.
    A zero-initialized head starts from the explicitly configured prior.
    """

    def __init__(self, feature_channels: int = 7,
                 initial_weights: tuple[float, float, float] = (1.0, 0.0, 0.0)) -> None:
        super().__init__()
        if len(initial_weights) != 3 or any(value < 0 for value in initial_weights):
            raise ValueError("initial_weights must contain three non-negative values")
        total = float(sum(initial_weights))
        if total <= 0:
            raise ValueError("initial_weights must have positive mass")
        prior = torch.tensor(initial_weights, dtype=torch.float32) / total
        prior = prior.clamp_min(1e-6)
        self.body = nn.Sequential(
            nn.Conv2d(feature_channels, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, padding=1), nn.GELU(),
        )
        self.head = nn.Conv2d(16, 3, 1)
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(prior.log())

    def forward(self, features: torch.Tensor, noisy: torch.Tensor,
                mild: torch.Tensor, strong: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if noisy.shape != mild.shape or noisy.shape != strong.shape:
            raise ValueError("Controller expert geometry differs")
        if features.shape[0] != noisy.shape[0] or features.shape[-2:] != noisy.shape[-2:]:
            raise ValueError("Controller feature geometry differs")
        weights = torch.softmax(self.head(self.body(features)), dim=1)
        experts = torch.cat((noisy, mild, strong), dim=1)
        fused = (weights * experts).sum(dim=1, keepdim=True)
        return fused, weights
