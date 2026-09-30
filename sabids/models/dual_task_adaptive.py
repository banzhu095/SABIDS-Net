from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn
from torch.nn import functional as F

from .dual_view_segmenter import NoisyMildDualViewSegmenter
from .sabids_net import SABIDSNet


def binary_entropy(probability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    value = probability.clamp(eps, 1.0 - eps)
    return -(value * value.log() + (1.0 - value) * (1.0 - value).log()) / math.log(2.0)


def gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    dx = F.pad(image[..., :, 1:] - image[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


class DualStrengthController(nn.Module):
    """Shared low-resolution context with independent bounded task heads."""

    maximum_strength = 1.25

    def __init__(self, hidden: int = 24, layer_init: float = 1.0,
                 vessel_init: float = 0.5) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(8, hidden, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
        )
        self.layer_head = nn.Conv2d(hidden, 1, 1)
        self.vessel_head = nn.Conv2d(hidden, 1, 1)
        self._initialize_head(self.layer_head, layer_init)
        self._initialize_head(self.vessel_head, vessel_init)

    @classmethod
    def _bias(cls, strength: float) -> float:
        if not 0.0 < strength < cls.maximum_strength:
            raise ValueError("Initial strength must lie strictly inside (0, 1.25)")
        probability = strength / cls.maximum_strength
        return math.log(probability / (1.0 - probability))

    @classmethod
    def _initialize_head(cls, head: nn.Conv2d, strength: float) -> None:
        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, cls._bias(float(strength)))

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.shared(features)
        layer = self.maximum_strength * torch.sigmoid(self.layer_head(context))
        vessel = self.maximum_strength * torch.sigmoid(self.vessel_head(context))
        return (
            F.interpolate(layer, size=output_size, mode="bilinear", align_corners=False),
            F.interpolate(vessel, size=output_size, mode="bilinear", align_corners=False),
        )


class TaskAuxiliaryFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.adapter = nn.Conv2d(3 * channels, channels, 1)
        self.gate = nn.Conv2d(3 * channels, channels, 1)
        self.gamma = nn.Parameter(torch.ones(()))

    def forward(self, noisy: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((noisy, auxiliary, noisy - auxiliary), dim=1)
        return noisy + self.gamma * torch.sigmoid(self.gate(joined)) * self.adapter(joined)


class DualTaskAdaptiveSegmenter(nn.Module):
    """Frozen D2/S0 coarse stage plus two bounded task-specific refiners."""

    def __init__(
        self,
        model_kwargs: Dict,
        fusion_levels: Iterable[int] = (3, 2, 1),
        layer_strength_init: float = 1.0,
        vessel_strength_init: float = 0.5,
        coarse_strength: float = 0.25,
        context_channels: int = 24,
    ) -> None:
        super().__init__()
        self.channels = list(model_kwargs.get("channels", (32, 64, 128, 256)))
        self.coarse_strength = float(coarse_strength)
        if not 0.0 <= self.coarse_strength <= 1.25:
            raise ValueError("coarse_strength must lie in [0,1.25]")
        base_kwargs = dict(model_kwargs)
        self.d2 = SABIDSNet(**base_kwargs)
        self.coarse_segmenter = NoisyMildDualViewSegmenter(
            **base_kwargs, dual_view_enabled=True, auxiliary_required=True,
            fusion_levels=tuple(fusion_levels), dual_scale_init=0.0,
        )
        self.fine_backbone = copy.deepcopy(self.coarse_segmenter)
        self.controller = DualStrengthController(
            hidden=context_channels,
            layer_init=layer_strength_init,
            vessel_init=vessel_strength_init,
        )
        self.fusion_levels = tuple(sorted({int(v) for v in fusion_levels}))
        invalid = [level for level in self.fusion_levels if level < 0 or level >= len(self.channels)]
        if invalid:
            raise ValueError(f"Invalid adaptive fusion levels: {invalid}")
        self.layer_aux_fusions = nn.ModuleDict({
            str(level): TaskAuxiliaryFusion(self.channels[level]) for level in self.fusion_levels
        })
        self.vessel_aux_fusions = nn.ModuleDict({
            str(level): TaskAuxiliaryFusion(self.channels[level]) for level in self.fusion_levels
        })
        final_channels = self.channels[0]
        self.layer_logit_correction = nn.Conv2d(final_channels + 2, 1, 1)
        self.vessel_logit_correction = nn.Conv2d(final_channels + 3, 1, 1)
        self.boundary_correction = nn.Conv2d(final_channels + 2, 2, 1)
        for module in (self.layer_logit_correction, self.vessel_logit_correction,
                       self.boundary_correction):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self._freeze_fixed_modules()

    def load_bound_checkpoints(self, d2_checkpoint: str | Path,
                               coarse_checkpoint: str | Path) -> None:
        d2 = torch.load(d2_checkpoint, map_location="cpu", weights_only=False)
        coarse = torch.load(coarse_checkpoint, map_location="cpu", weights_only=False)
        self.d2.load_state_dict(d2.get("model", d2), strict=True)
        state = coarse.get("model", coarse)
        self.coarse_segmenter.load_state_dict(state, strict=True)
        self.fine_backbone.load_state_dict(state, strict=True)
        self._freeze_fixed_modules()

    def _freeze_fixed_modules(self) -> None:
        for module in (self.d2, self.coarse_segmenter, self.fine_backbone):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

    def set_train_stage(self, stage: str, **_: object) -> None:
        if stage != "input_segment":
            raise ValueError("DualTaskAdaptiveSegmenter supports input_segment only")
        self._freeze_fixed_modules()

    def set_interaction_progress(self, _: float) -> None:
        return

    def enforce_frozen_eval(self) -> None:
        self._freeze_fixed_modules()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        self._freeze_fixed_modules()
        return self

    def _decode(self, task: str, noisy_features: List[torch.Tensor],
                auxiliary_features: List[torch.Tensor]) -> torch.Tensor:
        fusions = self.layer_aux_fusions if task == "layer" else self.vessel_aux_fusions
        fused = list(noisy_features)
        for level in self.fusion_levels:
            fused[level] = fusions[str(level)](noisy_features[level], auxiliary_features[level])
        deepest = len(self.channels) - 1
        value = self.fine_backbone.adapters[task][deepest](fused[deepest])
        for stage_index, level in enumerate(self.fine_backbone.decoder_levels):
            skip = self.fine_backbone.adapters[task][level](fused[level])
            value = self.fine_backbone.decoders[task][stage_index](value, skip)
        return value

    def forward(self, image: torch.Tensor, return_features: bool = True,
                return_auxiliary: bool = True, **extras: object) -> Dict[str, torch.Tensor | list]:
        forbidden = {"clean", "layer_mask", "vessel_mask", "ground_truth"} & set(extras)
        if forbidden:
            raise ValueError(f"Ground truth/clean inputs are forbidden in adaptive prediction: {sorted(forbidden)}")
        with torch.no_grad():
            d2_output = self.d2.forward_denoise_only(image)
            full_residual = (image - d2_output["denoised"]).detach()
            coarse_denoised = torch.clamp(image - self.coarse_strength * full_residual, 0.0, 1.0)
            coarse = self.coarse_segmenter(
                image, auxiliary_image=coarse_denoised,
                return_features=False, return_auxiliary=False,
            )
            coarse_layer_logits = coarse["layer_logits"].detach()
            coarse_vessel_logits = coarse["vessel_logits"].detach()
            coarse_boundary_logits = coarse["boundary_logits"].detach()
            layer_probability = torch.sigmoid(coarse_layer_logits)
            vessel_probability = torch.sigmoid(coarse_vessel_logits)
            layer_uncertainty = binary_entropy(layer_probability)
            vessel_uncertainty = binary_entropy(vessel_probability)

        context = torch.cat((
            image, coarse_denoised, image - coarse_denoised,
            layer_probability, vessel_probability,
            layer_uncertainty, vessel_uncertainty, gradient_magnitude(image),
        ), dim=1)
        layer_strength, vessel_strength = self.controller(context, image.shape[-2:])
        fine_layer_image = torch.clamp(image - layer_strength * full_residual, 0.0, 1.0)
        fine_vessel_image = torch.clamp(image - vessel_strength * full_residual, 0.0, 1.0)

        with torch.no_grad():
            noisy_features = self.fine_backbone.encode(image)
        layer_aux = self.fine_backbone.encode(fine_layer_image)
        vessel_aux = self.fine_backbone.encode(fine_vessel_image)
        layer_feature = self._decode("layer", noisy_features, layer_aux)
        vessel_feature = self._decode("vessel", noisy_features, vessel_aux)
        layer_input = torch.cat((layer_feature, layer_probability, layer_uncertainty), dim=1)
        vessel_input = torch.cat((vessel_feature, vessel_probability,
                                  vessel_uncertainty, layer_probability), dim=1)
        layer_logits = coarse_layer_logits + self.layer_logit_correction(layer_input)
        vessel_logits = coarse_vessel_logits + self.vessel_logit_correction(vessel_input)
        boundary_logits = coarse_boundary_logits + self.boundary_correction(layer_input)
        output: Dict[str, torch.Tensor | list] = {
            "denoised_raw": fine_vessel_image,
            "denoised": fine_vessel_image,
            "residual": image - fine_vessel_image,
            "layer_logits": layer_logits,
            "vessel_logits": vessel_logits,
            "layer_prob": torch.sigmoid(layer_logits),
            "vessel_prob": torch.sigmoid(vessel_logits),
            "boundary_logits": boundary_logits,
            "coarse_layer_logits": coarse_layer_logits,
            "coarse_vessel_logits": coarse_vessel_logits,
            "coarse_boundary_logits": coarse_boundary_logits,
            "coarse_layer_prob": layer_probability,
            "coarse_vessel_prob": vessel_probability,
            "coarse_denoised": coarse_denoised,
            "coarse_removed_residual": image - coarse_denoised,
            "full_d2_residual": full_residual,
            "layer_strength_map": layer_strength,
            "vessel_strength_map": vessel_strength,
            "strength_difference_map": layer_strength - vessel_strength,
            "fine_layer_denoised": fine_layer_image,
            "fine_vessel_denoised": fine_vessel_image,
            "layer_removed_residual": image - fine_layer_image,
            "vessel_removed_residual": image - fine_vessel_image,
            "auxiliary": [],
        }
        if return_features:
            output["anatomy_embedding"] = F.adaptive_avg_pool2d(
                torch.cat((layer_feature, vessel_feature), dim=1), 1
            ).flatten(1)
        return output
