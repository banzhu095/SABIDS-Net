from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def charbonnier(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((prediction - target).pow(2) + eps * eps).mean()


def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    # Custom reductions are not automatically promoted by AMP.  At 512x512,
    # keeping probabilities/reductions in float16 can overflow or lose the
    # small gradients needed to recover from a saturated foreground mask.
    logits = logits.float()
    target = target.float()
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    score = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - score.mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    logits = logits.float()
    target = target.float()
    probability = torch.sigmoid(logits)
    dims = tuple(range(1, probability.ndim))
    tp = (probability * target).sum(dim=dims)
    fp = (probability * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - probability) * target).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return torch.pow(1.0 - tversky, gamma).mean()


def image_gradients(image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gx = image[..., :, 1:] - image[..., :, :-1]
    gy = image[..., 1:, :] - image[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return gx, gy


def edge_map(mask: torch.Tensor) -> torch.Tensor:
    gx, gy = image_gradients(mask)
    return torch.clamp(torch.abs(gx) + torch.abs(gy), 0.0, 1.0)


def multi_scale_ssim_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    levels: int = 3,
    window: int = 7,
) -> torch.Tensor:
    losses = []
    x, y = prediction, target
    for _ in range(levels):
        mu_x = F.avg_pool2d(x, window, stride=1, padding=window // 2)
        mu_y = F.avg_pool2d(y, window, stride=1, padding=window // 2)
        sigma_x = F.avg_pool2d(x * x, window, stride=1, padding=window // 2) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(y * y, window, stride=1, padding=window // 2) - mu_y.pow(2)
        sigma_x = sigma_x.clamp_min(0.0)
        sigma_y = sigma_y.clamp_min(0.0)
        sigma_xy = F.avg_pool2d(x * y, window, stride=1, padding=window // 2) - mu_x * mu_y
        c1, c2 = 0.01**2, 0.03**2
        numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
        losses.append(1.0 - (numerator / (denominator + 1e-8)).mean())
        if min(x.shape[-2:]) < 4:
            break
        x = F.avg_pool2d(x, 2, stride=2)
        y = F.avg_pool2d(y, 2, stride=2)
    return torch.stack(losses).mean()


def haar_components(image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    h = image.shape[-2] - image.shape[-2] % 2
    w = image.shape[-1] - image.shape[-1] % 2
    value = image[..., :h, :w]
    a = value[..., 0::2, 0::2]
    b = value[..., 0::2, 1::2]
    c = value[..., 1::2, 0::2]
    d = value[..., 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return ll, lh, hl, hh


def wavelet_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    predicted = haar_components(prediction)
    reference = haar_components(target)
    weights = (0.25, 1.0, 1.0, 1.0)
    return sum(
        weight * F.l1_loss(pred, ref)
        for weight, pred, ref in zip(weights, predicted, reference)
    ) / sum(weights)


def layer_boundary_targets(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, _, height, width = mask.shape
    target = torch.zeros((batch, 2, height, width), device=mask.device, dtype=mask.dtype)
    valid = torch.zeros_like(target)
    binary = mask[:, 0] > 0.5
    for b in range(batch):
        for x in range(width):
            indices = torch.nonzero(binary[b, :, x], as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            target[b, 0, indices[0], x] = 1.0
            target[b, 1, indices[-1], x] = 1.0
            valid[b, :, :, x] = 1.0
    return target, valid


class SegmentationLoss(nn.Module):
    def __init__(
        self,
        task: str,
        boundary_weight: float = 0.2,
        vessel_bce_weight: float = 0.5,
        vessel_fp_weight: float = 0.6,
        vessel_fn_weight: float = 0.4,
        vessel_tversky_gamma: float = 0.75,
        boundary_positive_weight_cap: float = 20.0,
    ):
        super().__init__()
        self.task = task
        self.boundary_weight = boundary_weight
        self.vessel_bce_weight = vessel_bce_weight
        self.vessel_fp_weight = vessel_fp_weight
        self.vessel_fn_weight = vessel_fn_weight
        self.vessel_tversky_gamma = vessel_tversky_gamma
        self.boundary_positive_weight_cap = boundary_positive_weight_cap

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        boundary_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = logits.float()
        target = target.float()
        if boundary_logits is not None:
            boundary_logits = boundary_logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target)
        dice = soft_dice_loss(logits, target)
        if self.task == "vessel":
            # In focal_tversky_loss, alpha multiplies false positives and beta
            # multiplies false negatives.  The previous 0.3/0.7 setting strongly
            # favoured recall and allowed the vessel head to fill most of the
            # choroid.  Make that trade-off explicit and add BCE calibration.
            region = dice + focal_tversky_loss(
                logits,
                target,
                alpha=self.vessel_fp_weight,
                beta=self.vessel_fn_weight,
                gamma=self.vessel_tversky_gamma,
            )
            region = region + self.vessel_bce_weight * bce
            predicted_edge = edge_map(torch.sigmoid(logits))
            target_edge = edge_map(target)
            boundary = F.l1_loss(predicted_edge, target_edge)
        else:
            region = dice + bce
            if boundary_logits is not None:
                boundary_target, valid = layer_boundary_targets(target)
                raw = F.binary_cross_entropy_with_logits(
                    boundary_logits, boundary_target, reduction="none"
                )
                positives = (boundary_target * valid).sum()
                negatives = ((1.0 - boundary_target) * valid).sum()
                positive_weight = (negatives / positives.clamp_min(1.0)).clamp(
                    1.0, self.boundary_positive_weight_cap
                )
                weights = 1.0 + boundary_target * (positive_weight - 1.0)
                boundary = (raw * valid * weights).sum() / (
                    valid * weights
                ).sum().clamp_min(1.0)
            else:
                boundary = F.l1_loss(edge_map(torch.sigmoid(logits)), edge_map(target))
        return region + self.boundary_weight * boundary
