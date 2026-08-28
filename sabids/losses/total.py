from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .common import (
    SegmentationLoss,
    charbonnier,
    edge_map,
    image_gradients,
    multi_scale_ssim_loss,
    masked_bce_dice_loss,
    masked_negative_bce_loss,
    wavelet_loss,
)
from .pseudo import build_dual_source_pseudo_labels, confidence_masked_bce
from .rmac import rmac_loss


def _zero(output: Dict[str, torch.Tensor]) -> torch.Tensor:
    return output["denoised_raw"].sum() * 0.0


class SABIDSLoss(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.weights = config.get("weights", {})
        self.vessel_supervision_mode = str(
            config.get("vessel_supervision_mode", "composite")
        )
        if self.vessel_supervision_mode not in {
            "composite",
            "roi_bce_dice",
            "roi_bce_dice_outside",
        }:
            raise ValueError(
                "loss.vessel_supervision_mode must be composite, roi_bce_dice, "
                "or roi_bce_dice_outside"
            )
        common = {
            "boundary_weight": float(config.get("boundary_weight", 0.2)),
            "boundary_positive_weight_cap": float(
                config.get("boundary_positive_weight_cap", 20.0)
            ),
        }
        self.layer_loss = SegmentationLoss("layer", **common)
        self.vessel_loss = SegmentationLoss(
            "vessel",
            vessel_bce_weight=float(config.get("vessel_bce_weight", 0.5)),
            vessel_fp_weight=float(config.get("vessel_fp_weight", 0.6)),
            vessel_fn_weight=float(config.get("vessel_fn_weight", 0.4)),
            vessel_tversky_gamma=float(
                config.get("vessel_tversky_gamma", 0.75)
            ),
            **common,
        )

    def _weight(self, name: str, default: float = 0.0) -> float:
        return float(self.weights.get(name, default))

    def _restoration(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = batch["has_clean"].bool()
        if not bool(valid.any()):
            zero = _zero(output)
            return zero, zero
        prediction = output["denoised_raw"][valid]
        target = batch["clean"][valid]
        image = charbonnier(prediction, target)
        image = image + 0.2 * multi_scale_ssim_loss(prediction, target)
        image = image + 0.1 * wavelet_loss(prediction, target)

        layer_edge = edge_map(batch["layer_mask"][valid])
        vessel_edge = edge_map(batch["vessel_mask"][valid])
        has_layer = batch["has_layer"][valid].float().view(-1, 1, 1, 1)
        has_vessel = batch["has_vessel"][valid].float().view(-1, 1, 1, 1)
        boundary_weight = 1.0 + 1.0 * layer_edge * has_layer + 2.0 * vessel_edge * has_vessel
        pred_gx, pred_gy = image_gradients(prediction)
        target_gx, target_gy = image_gradients(target)
        edge = (
            boundary_weight * (torch.abs(pred_gx - target_gx) + torch.abs(pred_gy - target_gy))
        ).mean()
        image = image + 0.1 * edge

        residual_target = batch["image"][valid] - target
        residual = F.l1_loss(output["residual"][valid], residual_target)
        return image, residual

    def _segmentation(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        layer_valid = batch["has_layer"].bool()
        vessel_valid = batch["has_vessel"].bool()
        layer = _zero(output)
        vessel = _zero(output)
        vessel_stroma = _zero(output)
        vessel_area = _zero(output)
        vessel_outside = _zero(output)
        vessel_outside_valid_images = 0
        spatial_valid = batch["valid_mask"].float()
        layer_annotation_valid = batch.get(
            "label_valid_mask", spatial_valid
        ).float() * spatial_valid
        vessel_annotation_valid = batch.get("vessel_valid_mask", spatial_valid).float()
        vessel_annotation_valid = vessel_annotation_valid * spatial_valid
        if bool(layer_valid.any()):
            layer = self.layer_loss(
                output["layer_logits"][layer_valid],
                batch["layer_mask"][layer_valid],
                output["boundary_logits"][layer_valid],
                layer_annotation_valid[layer_valid],
            )
        if bool(vessel_valid.any()):
            if self.vessel_supervision_mode in {
                "roi_bce_dice",
                "roi_bce_dice_outside",
            }:
                roi = (
                    vessel_annotation_valid[vessel_valid]
                    * batch["layer_mask"][vessel_valid].float()
                )
                vessel = masked_bce_dice_loss(
                    output["vessel_logits"][vessel_valid],
                    batch["vessel_mask"][vessel_valid],
                    roi,
                )
                if self.vessel_supervision_mode == "roi_bce_dice_outside":
                    outside = vessel_annotation_valid[vessel_valid] * (
                        1.0 - batch["layer_mask"][vessel_valid].float()
                    )
                    vessel_outside, vessel_outside_valid_images = (
                        masked_negative_bce_loss(
                            output["vessel_logits"][vessel_valid], outside
                        )
                    )
            else:
                vessel = self.vessel_loss(
                    output["vessel_logits"][vessel_valid],
                    batch["vessel_mask"][vessel_valid],
                    valid_mask=vessel_annotation_valid[vessel_valid],
                )

        # The containment loss only forbids vessels outside the layer; by
        # itself it does not forbid predicting the *entire* layer as vessel.
        # These two supervised terms operate inside the labelled choroid:
        # (1) hard-negative BCE on annotated non-vessel stroma, and
        # (2) per-image vessel-area matching.  Padding is excluded.
        constrained = vessel_valid & layer_valid
        if bool(constrained.any()):
            vessel_logits = output["vessel_logits"][constrained].float()
            probability = torch.sigmoid(vessel_logits)
            vessel_target = batch["vessel_mask"][constrained].float()
            layer_target = batch["layer_mask"][constrained].float()
            valid_mask = vessel_annotation_valid[constrained]
            roi = layer_target * valid_mask
            stroma = roi * (1.0 - vessel_target)
            # BCE(target=0) == softplus(logit).  The former sigmoid/log/clamp
            # expression had zero gradient once sigmoid rounded to exactly 1,
            # which disabled the safeguard for saturated full-layer vessels.
            negative_log_likelihood = F.softplus(vessel_logits)
            vessel_stroma = (negative_log_likelihood * stroma).sum() / (
                stroma.sum().clamp_min(1.0)
            )
            reduce_dims = tuple(range(1, probability.ndim))
            roi_pixels = roi.sum(dim=reduce_dims).clamp_min(1.0)
            predicted_fraction = (probability * roi).sum(dim=reduce_dims) / roi_pixels
            true_fraction = (vessel_target * roi).sum(dim=reduce_dims) / roi_pixels
            vessel_area = F.smooth_l1_loss(
                predicted_fraction, true_fraction, beta=0.05
            )

        auxiliary_weight = float(self.config.get("auxiliary_weight", 0.1))
        if auxiliary_weight > 0:
            for auxiliary in output.get("auxiliary", []):
                if bool(layer_valid.any()):
                    target = F.interpolate(
                        batch["layer_mask"][layer_valid],
                        size=auxiliary["layer_logit"].shape[-2:],
                        mode="nearest",
                    )
                    layer = layer + auxiliary_weight * self.layer_loss(
                        auxiliary["layer_logit"][layer_valid],
                        target,
                        valid_mask=F.interpolate(
                            layer_annotation_valid[layer_valid],
                            size=auxiliary["layer_logit"].shape[-2:],
                            mode="nearest",
                        ),
                    )
                if bool(vessel_valid.any()):
                    target = F.interpolate(
                        batch["vessel_mask"][vessel_valid],
                        size=auxiliary["vessel_logit"].shape[-2:],
                        mode="nearest",
                    )
                    auxiliary_valid = F.interpolate(
                        vessel_annotation_valid[vessel_valid],
                        size=auxiliary["vessel_logit"].shape[-2:],
                        mode="nearest",
                    )
                    if self.vessel_supervision_mode in {
                        "roi_bce_dice",
                        "roi_bce_dice_outside",
                    }:
                        auxiliary_layer = F.interpolate(
                            batch["layer_mask"][vessel_valid],
                            size=auxiliary["vessel_logit"].shape[-2:],
                            mode="nearest",
                        )
                        auxiliary_loss = masked_bce_dice_loss(
                            auxiliary["vessel_logit"][vessel_valid],
                            target,
                            auxiliary_valid * auxiliary_layer,
                        )
                    else:
                        auxiliary_loss = self.vessel_loss(
                            auxiliary["vessel_logit"][vessel_valid],
                            target,
                            valid_mask=auxiliary_valid,
                        )
                    vessel = vessel + auxiliary_weight * auxiliary_loss
        return (
            layer,
            vessel,
            vessel_stroma,
            vessel_area,
            vessel_outside,
            vessel_outside_valid_images,
        )

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        stage: str,
        repeat_output: Optional[Dict[str, torch.Tensor]] = None,
        clean_output: Optional[Dict[str, torch.Tensor]] = None,
        teacher_output: Optional[Dict[str, torch.Tensor]] = None,
        ramp: float = 1.0,
    ) -> Dict[str, torch.Tensor | float]:
        losses: Dict[str, torch.Tensor | float] = {}
        zero = _zero(output)
        if stage in {"denoise", "warmup", "joint", "private"}:
            reconstruction, residual = self._restoration(output, batch)
        else:
            reconstruction, residual = zero, zero
        if stage in {"segment", "warmup", "joint", "private", "private_seg"}:
            (
                layer,
                vessel,
                vessel_stroma,
                vessel_area,
                vessel_outside,
                vessel_outside_valid_images,
            ) = self._segmentation(output, batch)
        else:
            layer, vessel, vessel_stroma, vessel_area, vessel_outside = (
                zero,
                zero,
                zero,
                zero,
                zero,
            )
            vessel_outside_valid_images = 0
        losses["reconstruction"] = reconstruction
        losses["residual"] = residual
        losses["layer"] = layer
        losses["vessel"] = vessel
        losses["vessel_stroma"] = vessel_stroma
        losses["vessel_area"] = vessel_area
        losses["vessel_outside"] = vessel_outside
        losses["vessel_outside_valid_images"] = float(
            vessel_outside_valid_images
        )

        containment = zero
        if stage in {"segment", "warmup", "joint", "private", "private_seg"}:
            has_layer = batch["has_layer"].view(-1, 1, 1, 1)
            layer_reference = torch.where(
                has_layer,
                batch["layer_mask"].float(),
                torch.sigmoid(output["layer_logits"].float()).detach(),
            ).float()
            valid_mask = batch["valid_mask"].float()
            vessel_probability = torch.sigmoid(output["vessel_logits"].float())
            containment = (
                vessel_probability * (1.0 - layer_reference) * valid_mask
            ).sum() / valid_mask.sum().clamp_min(1.0)
        losses["containment"] = containment

        identity = _zero(output)
        identity_valid = batch["has_clean"].bool() | batch["is_clean"].bool()
        if (
            stage in {"denoise", "warmup", "joint", "private"}
            and clean_output is not None
            and "denoised_raw" in clean_output
            and clean_output["denoised_raw"].requires_grad
            and bool(identity_valid.any())
        ):
            identity_target = torch.where(
                batch["has_clean"].view(-1, 1, 1, 1),
                batch["clean"],
                batch["image_weak"],
            )
            identity = F.l1_loss(
                clean_output["denoised_raw"][identity_valid],
                identity_target[identity_valid],
            )
        losses["identity"] = identity

        rmac = _zero(output)
        if stage in {"joint", "private"} and repeat_output is not None:
            rmac_config = self.config.get("rmac", {})
            rmac = rmac_loss(
                output,
                repeat_output,
                batch["has_repeat"].bool(),
                clean_output=clean_output,
                clean_valid=batch["has_clean"].bool(),
                image_weight=float(rmac_config.get("image_weight", 1.0)),
                segmentation_weight=float(
                    rmac_config.get("segmentation_weight", 1.0)
                ),
                layer_consistency_weight=float(
                    rmac_config.get("layer_consistency_weight", 0.5)
                ),
                vessel_consistency_weight=float(
                    rmac_config.get("vessel_consistency_weight", 0.25)
                ),
                clean_teacher_weight=float(
                    rmac_config.get("clean_teacher_weight", 0.25)
                ),
                feature_weight=float(rmac_config.get("feature_weight", 0.05)),
            )
        losses["rmac"] = rmac

        pseudo = _zero(output)
        pseudo_statistics: Dict[str, float] = {}
        if stage in {"private", "private_seg"} and teacher_output is not None:
            eligible = (~batch["has_vessel"].bool()) & batch["has_layer"].bool()
            layer_roi = torch.where(
                batch["has_layer"].view(-1, 1, 1, 1),
                batch["layer_mask"],
                teacher_output["layer_prob"],
            )
            target, confidence, pseudo_statistics = build_dual_source_pseudo_labels(
                batch["image_weak"],
                teacher_output["vessel_prob"].detach(),
                layer_roi.detach(),
                eligible,
                dark_percentile=float(self.config.get("dark_percentile", 5.0)),
                positive_threshold=float(self.config.get("pseudo_positive", 0.9)),
                negative_threshold=float(self.config.get("pseudo_negative", 0.1)),
            )
            if bool(confidence.any()):
                pseudo = confidence_masked_bce(
                    output["vessel_logits"], target, confidence
                )
        losses["pseudo"] = pseudo
        losses.update(pseudo_statistics)

        active = {
            "denoise": {"reconstruction", "residual", "identity"},
            "segment": {
                "layer", "vessel", "vessel_stroma", "vessel_area",
                "vessel_outside", "containment"
            },
            "warmup": {
                "reconstruction", "residual", "layer", "vessel",
                "vessel_stroma", "vessel_area", "vessel_outside",
                "containment", "identity"
            },
            "joint": {
                "reconstruction", "residual", "layer", "vessel",
                "vessel_stroma", "vessel_area", "vessel_outside",
                "containment", "identity", "rmac"
            },
            "private": {
                "reconstruction", "residual", "layer", "vessel",
                "vessel_stroma", "vessel_area", "vessel_outside",
                "containment", "identity", "rmac",
                "pseudo"
            },
            "private_seg": {
                "layer", "vessel", "vessel_stroma", "vessel_area",
                "vessel_outside", "containment",
                "pseudo"
            },
        }[stage]
        total = _zero(output)
        for name in active:
            multiplier = ramp if name in {"rmac", "pseudo"} else 1.0
            weight = self._weight(name) * multiplier
            weighted = weight * losses[name]
            losses[f"{name}_weight"] = float(weight)
            losses[f"{name}_weighted"] = weighted
            total = total + weighted
        losses["total"] = total
        return losses
