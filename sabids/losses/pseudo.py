from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.nn import functional as F


@torch.no_grad()
def build_dual_source_pseudo_labels(
    image: torch.Tensor,
    teacher_vessel: torch.Tensor,
    layer_roi: torch.Tensor,
    eligible: torch.Tensor,
    dark_percentile: float = 5.0,
    positive_threshold: float = 0.9,
    negative_threshold: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    target = torch.zeros_like(teacher_vessel)
    confidence = torch.zeros_like(teacher_vessel)
    positive_pixels = 0
    negative_pixels = 0
    for index in range(image.shape[0]):
        if not bool(eligible[index]):
            continue
        roi = layer_roi[index] > 0.5
        values = image[index][roi]
        if values.numel() < 16:
            continue
        threshold = torch.quantile(values.float(), dark_percentile / 100.0)
        dark_prior = (image[index] <= threshold) & roi
        teacher = teacher_vessel[index]
        positive = (teacher >= positive_threshold) & dark_prior
        negative = (teacher <= negative_threshold) & (~dark_prior) & roi
        target[index][positive] = 1.0
        confidence[index][positive | negative] = 1.0
        positive_pixels += int(positive.sum().item())
        negative_pixels += int(negative.sum().item())
    statistics = {
        "pseudo_positive_pixels": float(positive_pixels),
        "pseudo_negative_pixels": float(negative_pixels),
        "pseudo_confident_pixels": float(confidence.sum().item()),
    }
    return target, confidence, statistics


def confidence_masked_bce(
    logits: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor
) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * confidence).sum() / confidence.sum().clamp_min(1.0)

