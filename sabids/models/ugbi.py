from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn


def binary_entropy(probability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )
    return entropy / 0.6931471805599453


class ConvGate(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        hidden = max(out_channels // 2, 8)
        super().__init__(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 3, padding=1),
        )


class UGBIBlock(nn.Module):
    """Uncertainty-gated bidirectional interaction at one decoder scale."""

    def __init__(
        self,
        channels: int,
        enable_seg_to_denoise: bool = True,
        enable_denoise_to_seg: bool = True,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        self.enable_seg_to_denoise = enable_seg_to_denoise
        self.enable_denoise_to_seg = enable_denoise_to_seg
        self.use_uncertainty = use_uncertainty
        self.layer_head = nn.Conv2d(channels, 1, 1)
        self.vessel_head = nn.Conv2d(channels, 1, 1)
        self.noise_head = nn.Conv2d(channels, 1, 1)

        self.layer_anatomy = nn.Conv2d(channels + 1, channels, 3, padding=1)
        self.vessel_anatomy = nn.Conv2d(channels + 1, channels, 3, padding=1)
        self.seg_to_denoise_gate = ConvGate(channels * 2, channels)

        self.restoration_context = nn.Conv2d(channels + 1, channels, 3, padding=1)
        self.denoise_to_layer_gate = ConvGate(channels * 2, channels)
        self.denoise_to_vessel_gate = ConvGate(channels * 2, channels)
        self.denoise_to_layer = nn.Conv2d(channels, channels, 1)
        self.denoise_to_vessel = nn.Conv2d(channels, channels, 1)

        self.seg_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.layer_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.vessel_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(
        self,
        denoise: torch.Tensor,
        layer: torch.Tensor,
        vessel: torch.Tensor,
        detach_cross: bool = False,
        detach_denoise_to_seg: bool = False,
        return_details: bool = True,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[Dict[str, torch.Tensor]],
    ]:
        source_d = denoise.detach() if detach_cross else denoise
        source_d_to_seg = (
            denoise.detach() if detach_cross or detach_denoise_to_seg else denoise
        )
        source_l = layer.detach() if detach_cross else layer
        source_v = vessel.detach() if detach_cross else vessel

        layer_logit = self.layer_head(source_l)
        vessel_logit = self.vessel_head(source_v)
        layer_prob = torch.sigmoid(layer_logit)
        vessel_prob = torch.sigmoid(vessel_logit)
        if self.use_uncertainty:
            layer_conf = 1.0 - binary_entropy(layer_prob)
            vessel_conf = 1.0 - binary_entropy(vessel_prob)
        else:
            layer_conf = torch.ones_like(layer_prob)
            vessel_conf = torch.ones_like(vessel_prob)

        layer_anatomy = self.layer_anatomy(torch.cat([source_l, layer_prob], dim=1))
        vessel_anatomy = self.vessel_anatomy(
            torch.cat([source_v, vessel_prob], dim=1)
        )
        anatomy = layer_conf * layer_anatomy + vessel_conf * vessel_anatomy
        s2d_gate = torch.sigmoid(
            self.seg_to_denoise_gate(torch.cat([source_d, anatomy], dim=1))
        )
        denoise_updated = (
            denoise + self.seg_scale * s2d_gate * anatomy
            if self.enable_seg_to_denoise
            else denoise
        )

        noise_hint = torch.abs(torch.tanh(self.noise_head(source_d_to_seg)))
        restoration = self.restoration_context(
            torch.cat([source_d_to_seg, noise_hint], dim=1)
        )
        d2l_gate = torch.sigmoid(
            self.denoise_to_layer_gate(torch.cat([source_l, restoration], dim=1))
        )
        d2v_gate = torch.sigmoid(
            self.denoise_to_vessel_gate(torch.cat([source_v, restoration], dim=1))
        )
        if self.enable_denoise_to_seg:
            layer_injection = (
                self.layer_scale * d2l_gate * self.denoise_to_layer(restoration)
            )
            vessel_injection = (
                self.vessel_scale * d2v_gate * self.denoise_to_vessel(restoration)
            )
            layer_updated = layer + layer_injection
            vessel_updated = vessel + vessel_injection
        else:
            layer_injection = torch.zeros_like(layer)
            vessel_injection = torch.zeros_like(vessel)
            layer_updated = layer
            vessel_updated = vessel

        auxiliary = None
        if return_details:
            auxiliary = {
                "layer_logit": layer_logit,
                "vessel_logit": vessel_logit,
                "layer_uncertainty": 1.0 - layer_conf,
                "vessel_uncertainty": 1.0 - vessel_conf,
                "noise_hint": noise_hint,
                "seg_to_denoise_gate": s2d_gate,
                "denoise_to_layer_gate": d2l_gate,
                "denoise_to_vessel_gate": d2v_gate,
                "denoise_to_layer_injection_abs_mean": layer_injection.detach().abs().mean(),
                "denoise_to_vessel_injection_abs_mean": vessel_injection.detach().abs().mean(),
                "layer_scale_abs_mean": self.layer_scale.detach().abs().mean(),
                "vessel_scale_abs_mean": self.vessel_scale.detach().abs().mean(),
            }
        return denoise_updated, layer_updated, vessel_updated, auxiliary
