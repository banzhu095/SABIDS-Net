from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List

import torch
from torch import nn
from torch.nn import functional as F

from .dual_task_adaptive import DualTaskAdaptiveSegmenter, binary_entropy, gradient_magnitude


class VesselResidualInjection(nn.Module):
    """Noisy-backed additive feature increment; zero gamma is an exact bypass."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.mapping = nn.Conv2d(3 * channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, noisy: torch.Tensor, adaptive: torch.Tensor) -> torch.Tensor:
        joined = torch.cat((noisy, adaptive, noisy - adaptive), dim=1)
        return noisy + self.gamma * self.mapping(joined)


class DualTaskAdaptiveV2Segmenter(nn.Module):
    """Frozen v1 Layer path plus a bounded, noisy-backed Vessel revision."""

    vessel_strength_cap = 0.5

    def __init__(self, model_kwargs: Dict, fusion_levels: Iterable[int] = (3, 2, 1),
                 coarse_strength: float = 0.25, context_channels: int = 24,
                 vessel_strength_init: float = 0.25) -> None:
        super().__init__()
        if not 0.0 < vessel_strength_init < self.vessel_strength_cap:
            raise ValueError("vessel_strength_init must lie inside (0, 0.5)")
        self.v1 = DualTaskAdaptiveSegmenter(
            model_kwargs=model_kwargs, fusion_levels=fusion_levels,
            coarse_strength=coarse_strength, context_channels=context_channels,
        )
        self.fusion_levels = tuple(sorted({int(x) for x in fusion_levels}))
        self.vessel_private_adapter = nn.Sequential(
            nn.Conv2d(context_channels, context_channels, 3, padding=1), nn.GELU())
        self.vessel_strength_head = nn.Conv2d(context_channels, 1, 1)
        nn.init.zeros_(self.vessel_strength_head.weight)
        p = vessel_strength_init / self.vessel_strength_cap
        nn.init.constant_(self.vessel_strength_head.bias, math.log(p / (1.0 - p)))
        channels = list(model_kwargs.get("channels", (32, 64, 128, 256)))
        self.vessel_increments = nn.ModuleDict({
            str(level): VesselResidualInjection(channels[level]) for level in self.fusion_levels
        })
        self.vessel_delta_head = nn.Conv2d(channels[0] + 3, 1, 1)
        self.vessel_logit_scale = nn.Parameter(torch.zeros(()))
        self._freeze_v1()

    def load_v1_checkpoint(self, path: str | Path) -> None:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        self.v1.load_state_dict(raw.get("model", raw), strict=True)
        self._freeze_v1()

    def load_bound_checkpoints(self, d2_checkpoint: str | Path,
                               coarse_checkpoint: str | Path,
                               v1_checkpoint: str | Path) -> None:
        self.v1.load_bound_checkpoints(d2_checkpoint, coarse_checkpoint)
        self.load_v1_checkpoint(v1_checkpoint)

    def _freeze_v1(self) -> None:
        for parameter in self.v1.parameters():
            parameter.requires_grad_(False)
        self.v1.eval()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode); self._freeze_v1(); return self

    def enforce_frozen_eval(self) -> None:
        self._freeze_v1()

    def set_train_stage(self, stage: str, **_: object) -> None:
        if stage != "input_segment":
            raise ValueError("DualTaskAdaptiveV2Segmenter supports input_segment only")
        self._freeze_v1()

    def set_interaction_progress(self, _: float) -> None:
        return

    def _decode_vessel(self, noisy: List[torch.Tensor], adaptive: List[torch.Tensor]):
        fused = list(noisy); rms = []
        for level in self.fusion_levels:
            value = self.vessel_increments[str(level)](noisy[level], adaptive[level])
            fused[level] = value
            rms.append((value - noisy[level]).square().mean().sqrt())
        backbone = self.v1.fine_backbone
        deepest = len(backbone.channels) - 1
        value = backbone.adapters["vessel"][deepest](fused[deepest])
        for stage_index, level in enumerate(backbone.decoder_levels):
            skip = backbone.adapters["vessel"][level](fused[level])
            value = backbone.decoders["vessel"][stage_index](value, skip)
        return value, rms

    def forward(self, image: torch.Tensor, return_features: bool = True,
                return_auxiliary: bool = True, vessel_adaptive_off: bool = False,
                **extras: object) -> Dict[str, torch.Tensor | list]:
        forbidden = {"clean", "layer_mask", "vessel_mask", "ground_truth"} & set(extras)
        if forbidden:
            raise ValueError(f"Ground truth/clean inputs are forbidden: {sorted(forbidden)}")
        with torch.no_grad():
            v1 = self.v1(image, return_features=False, return_auxiliary=False)
            coarse_logits = v1["coarse_vessel_logits"]
            layer_prob = v1["coarse_layer_prob"]
            vessel_prob = v1["coarse_vessel_prob"]
            context_input = torch.cat((
                image, v1["coarse_denoised"], image - v1["coarse_denoised"],
                layer_prob, vessel_prob, binary_entropy(layer_prob),
                binary_entropy(vessel_prob), gradient_magnitude(image)), dim=1)
            shared = self.v1.controller.shared(context_input)
            residual = v1["full_d2_residual"]
        private = self.vessel_private_adapter(shared)
        strength = self.vessel_strength_cap * torch.sigmoid(self.vessel_strength_head(private))
        strength = F.interpolate(strength, image.shape[-2:], mode="bilinear", align_corners=False)
        vessel_image = torch.clamp(image - strength * residual, 0.0, 1.0)
        with torch.no_grad():
            noisy_features = self.v1.fine_backbone.encode(image)
        adaptive_features = self.v1.fine_backbone.encode(vessel_image)
        feature, increment_rms = self._decode_vessel(noisy_features, adaptive_features)
        delta_input = torch.cat((feature, vessel_prob, binary_entropy(vessel_prob), layer_prob), 1)
        delta = self.vessel_delta_head(delta_input)
        vessel_logits = coarse_logits if vessel_adaptive_off else coarse_logits + self.vessel_logit_scale * delta
        output = dict(v1)
        output.update({
            "vessel_logits": vessel_logits, "vessel_prob": torch.sigmoid(vessel_logits),
            "vessel_strength_map": strength, "fine_vessel_denoised": vessel_image,
            "vessel_removed_residual": image - vessel_image, "denoised": vessel_image,
            "denoised_raw": vessel_image, "residual": image - vessel_image,
            "vessel_logit_delta": delta, "vessel_logit_scale": self.vessel_logit_scale,
            "vessel_increment_rms": torch.stack(increment_rms),
            "vessel_adaptive_off": torch.tensor(vessel_adaptive_off, device=image.device),
            "v1_vessel_prob": v1["vessel_prob"], "v1_vessel_logits": v1["vessel_logits"],
            "v1_layer_prob": v1["layer_prob"], "v1_layer_logits": v1["layer_logits"],
            "loss_zero_reference": vessel_logits,
        })
        for level, rms in zip(self.fusion_levels, increment_rms):
            output[f"vessel_increment_rms_level{level}"] = rms
        return output
