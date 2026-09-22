from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    edge_map,
    image_gradients,
    multi_scale_ssim_loss,
    multiscale_gradient_loss,
    multiscale_laplacian_loss,
    soft_dice_loss,
)


def _masked_l1_per_image(value: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    terms = []
    for index in range(value.shape[0]):
        current = mask[index].float()
        if bool(current.any()):
            terms.append(((value[index] - target[index]).abs() * current).sum() / current.sum())
    if not terms:
        return value.sum() * 0.0, 0
    return torch.stack(terms).mean(), len(terms)


def _boundary_band(mask: torch.Tensor, width: int) -> torch.Tensor:
    mask = mask.float()
    edge = edge_map(mask)
    kernel = 2 * max(1, int(width)) + 1
    return F.max_pool2d(edge, kernel, stride=1, padding=kernel // 2).clamp(0, 1)


def _masked_cnr(image: torch.Tensor, first: torch.Tensor, second: torch.Tensor, eps: float) -> tuple[list[torch.Tensor], int]:
    values = []
    for index in range(image.shape[0]):
        a = image[index][first[index].bool()]
        b = image[index][second[index].bool()]
        if a.numel() < 2 or b.numel() < 2:
            continue
        denominator = torch.sqrt(a.var(unbiased=False) + b.var(unbiased=False) + eps)
        values.append((a.mean() - b.mean()).abs() / denominator)
    return values, len(values)


class D2StructureLoss(nn.Module):
    """Modular task-structure preservation objective for an independent D2."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.weights = {str(k): float(v) for k, v in config.get("weights", {}).items()}

    def weight(self, key: str) -> float:
        return float(self.weights.get(key, 0.0))

    def forward(
        self,
        prediction: torch.Tensor,
        noisy: torch.Tensor,
        clean: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        teacher: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor | float]:
        prediction, noisy, clean = prediction.float(), noisy.float(), clean.float()
        spatial = batch["valid_mask"].float()
        vessel_valid = spatial * batch.get("vessel_valid_mask", spatial).float()
        label_valid = spatial * batch.get("label_valid_mask", spatial).float()
        vessel = batch["vessel_mask"].float() * vessel_valid
        layer = batch["layer_mask"].float() * label_valid
        has_vessel = batch["has_vessel"].float().view(-1, 1, 1, 1)
        has_layer = batch["has_layer"].float().view(-1, 1, 1, 1)
        vessel_roi = vessel * has_vessel
        stroma = layer * (1.0 - vessel) * has_layer * vessel_valid
        outside = (1.0 - layer) * spatial * label_valid * has_layer
        boundary = _boundary_band(vessel, int(self.config.get("boundary_width_pixels", 3))) * vessel_valid * has_vessel

        raw: Dict[str, torch.Tensor | float] = {}
        char_eps = float(self.config.get("charbonnier_epsilon", 1e-3))
        char_map = torch.sqrt((prediction - clean).pow(2) + char_eps * char_eps)
        raw["charbonnier"] = (char_map * spatial).sum() / spatial.sum().clamp_min(1.0)
        # Zeroing invalid pixels on both operands prevents padding/unknown image
        # values from entering the restoration graph. The spatial mask also
        # blocks gradients to prediction pixels outside the valid canvas.
        prediction_valid = prediction * spatial
        clean_valid = clean * spatial
        raw["ms_ssim"] = multi_scale_ssim_loss(prediction_valid, clean_valid)
        raw["gradient"] = multiscale_gradient_loss(
            prediction_valid, clean_valid, float(self.config.get("structure_beta", 2.0))
        )
        raw["laplacian"] = multiscale_laplacian_loss(prediction_valid, clean_valid)
        raw["vessel_roi"], vessel_count = _masked_l1_per_image(prediction, clean, vessel_roi)
        raw["stroma_roi"], stroma_count = _masked_l1_per_image(prediction, clean, stroma)
        raw["outside_roi"], outside_count = _masked_l1_per_image(prediction, clean, outside)
        pgx, pgy = image_gradients(prediction)
        cgx, cgy = image_gradients(clean)
        raw["boundary"], boundary_count = _masked_l1_per_image(
            pgx.abs() + pgy.abs(), cgx.abs() + cgy.abs(), boundary
        )
        pred_cnr, cnr_count = _masked_cnr(
            prediction, vessel_roi, stroma,
            float(self.config.get("epsilon", 1e-6)),
        )
        clean_cnr, _ = _masked_cnr(
            clean, vessel_roi, stroma,
            float(self.config.get("epsilon", 1e-6)),
        )
        if pred_cnr and len(pred_cnr) == len(clean_cnr):
            cap = float(self.config.get("cnr_error_cap", 5.0))
            raw["cnr"] = torch.stack([
                (left - right).abs().clamp_max(cap)
                for left, right in zip(pred_cnr, clean_cnr)
            ]).mean()
        else:
            raw["cnr"] = prediction.sum() * 0.0
            cnr_count = 0

        structure = (vessel_roi + boundary).clamp(0, 1)
        raw["leak"] = ((noisy - prediction).abs() * structure).sum() / structure.sum().clamp_min(1.0)
        raw["residual_amplitude"] = ((noisy - prediction).abs() * spatial).sum() / spatial.sum().clamp_min(1.0)
        raw["teacher_task"] = prediction.sum() * 0.0
        raw["teacher_consistency"] = prediction.sum() * 0.0
        if teacher is not None:
            task_terms = []
            if bool(batch["has_layer"].any()):
                valid = batch["has_layer"].bool()
                mask = label_valid[valid]
                logits = teacher["layer_logits"][valid]
                target = batch["layer_mask"][valid].float()
                bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
                task_terms.append(bce + soft_dice_loss(logits, target, mask))
            if bool(batch["has_vessel"].any()):
                valid = batch["has_vessel"].bool()
                mask = vessel_valid[valid]
                logits = teacher["vessel_logits"][valid]
                target = batch["vessel_mask"][valid].float()
                bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)
                task_terms.append(bce + soft_dice_loss(logits, target, mask))
            if task_terms:
                raw["teacher_task"] = torch.stack(task_terms).mean()
            consistency = []
            eps = float(self.config.get("epsilon", 1e-6))
            for task in ("layer", "vessel"):
                student = torch.sigmoid(teacher[f"{task}_logits"]).clamp(eps, 1 - eps)
                reference = teacher[f"clean_{task}_prob"].detach().clamp(eps, 1 - eps)
                mask = label_valid if task == "layer" else vessel_valid
                kl = reference * (reference.log() - student.log())
                kl += (1 - reference) * ((1 - reference).log() - (1 - student).log())
                consistency.append((kl * mask).sum() / mask.sum().clamp_min(1.0))
            raw["teacher_consistency"] = torch.stack(consistency).mean()

        result: Dict[str, torch.Tensor | float] = {}
        total = prediction.sum() * 0.0
        for name, value in raw.items():
            assert torch.is_tensor(value)
            weight = self.weight(name)
            weighted = value * weight
            result[f"d2_{name}_raw"] = value
            result[f"d2_{name}_weight"] = weight
            result[f"d2_{name}_weighted"] = weighted
            total = total + weighted
        result.update({
            "d2_vessel_valid_samples": float(vessel_count),
            "d2_stroma_valid_samples": float(stroma_count),
            "d2_outside_valid_samples": float(outside_count),
            "d2_boundary_valid_samples": float(boundary_count),
            "d2_cnr_valid_samples": float(cnr_count),
            "d2_total": total,
        })
        return result
