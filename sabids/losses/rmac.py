from __future__ import annotations

from typing import Dict, Optional

import torch
from torch.nn import functional as F

from .common import multi_scale_ssim_loss


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    m = (0.5 * (p + q)).clamp(eps, 1.0 - eps)
    kl_p = p * torch.log(p / m) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - m))
    kl_q = q * torch.log(q / m) + (1.0 - q) * torch.log((1.0 - q) / (1.0 - m))
    return 0.5 * (kl_p + kl_q).mean()


def rmac_loss(
    output_a: Dict[str, torch.Tensor],
    output_b: Dict[str, torch.Tensor],
    valid: torch.Tensor,
    clean_output: Optional[Dict[str, torch.Tensor]] = None,
    clean_valid: Optional[torch.Tensor] = None,
    image_weight: float = 1.0,
    segmentation_weight: float = 1.0,
    layer_consistency_weight: float = 0.5,
    vessel_consistency_weight: float = 0.25,
    clean_teacher_weight: float = 0.25,
    feature_weight: float = 0.05,
) -> torch.Tensor:
    if not bool(valid.any()):
        return output_a["denoised_raw"].sum() * 0.0
    mask = valid.bool()
    denoised_a = output_a["denoised_raw"][mask]
    denoised_b = output_b["denoised_raw"][mask]
    image = F.l1_loss(denoised_a, denoised_b)
    image = image + 0.2 * multi_scale_ssim_loss(denoised_a, denoised_b)

    segmentation = layer_consistency_weight * js_divergence(
        output_a["layer_prob"][mask], output_b["layer_prob"][mask]
    ) + vessel_consistency_weight * js_divergence(
        output_a["vessel_prob"][mask], output_b["vessel_prob"][mask]
    )
    if clean_output is not None and clean_valid is not None and bool((valid & clean_valid).any()):
        clean_mask = valid.bool() & clean_valid.bool()
        clean_layer = clean_output["layer_prob"][clean_mask].detach()
        clean_vessel = clean_output["vessel_prob"][clean_mask].detach()
        segmentation = segmentation + clean_teacher_weight * (
            layer_consistency_weight
            * (
                js_divergence(output_a["layer_prob"][clean_mask], clean_layer)
                + js_divergence(output_b["layer_prob"][clean_mask], clean_layer)
            )
            + vessel_consistency_weight
            * (
                js_divergence(output_a["vessel_prob"][clean_mask], clean_vessel)
                + js_divergence(output_b["vessel_prob"][clean_mask], clean_vessel)
            )
        )

    embedding_a = output_a["anatomy_embedding"][mask]
    embedding_b = output_b["anatomy_embedding"][mask]
    feature = (1.0 - F.cosine_similarity(embedding_a, embedding_b, dim=1)).mean()
    return image_weight * image + segmentation_weight * segmentation + feature_weight * feature
