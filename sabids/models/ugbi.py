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
        scale_init: float = 0.1,
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

        self.seg_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))
        self.layer_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))
        self.vessel_scale = nn.Parameter(torch.full((1, channels, 1, 1), float(scale_init)))

    def seg_to_denoise(
        self,
        denoise: torch.Tensor,
        layer: torch.Tensor,
        vessel: torch.Tensor,
        layer_probability: Optional[torch.Tensor] = None,
        vessel_probability: Optional[torch.Tensor] = None,
        detach_source: bool = False,
        strength: float = 1.0,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Inject trained segmentation guidance into denoising without a cycle."""
        source_l = layer.detach() if detach_source else layer
        source_v = vessel.detach() if detach_source else vessel
        layer_prob = torch.sigmoid(self.layer_head(source_l)) if layer_probability is None else layer_probability
        vessel_prob = torch.sigmoid(self.vessel_head(source_v)) if vessel_probability is None else vessel_probability
        if detach_source:
            layer_prob, vessel_prob = layer_prob.detach(), vessel_prob.detach()
        if layer_prob.shape[-2:] != source_l.shape[-2:]:
            layer_prob = torch.nn.functional.interpolate(layer_prob, source_l.shape[-2:], mode="bilinear", align_corners=False)
            vessel_prob = torch.nn.functional.interpolate(vessel_prob, source_v.shape[-2:], mode="bilinear", align_corners=False)
        if self.use_uncertainty:
            layer_conf = 1.0 - binary_entropy(layer_prob)
            vessel_conf = 1.0 - binary_entropy(vessel_prob)
        else:
            layer_conf, vessel_conf = torch.ones_like(layer_prob), torch.ones_like(vessel_prob)
        layer_anatomy = self.layer_anatomy(torch.cat([source_l, layer_prob], dim=1))
        vessel_anatomy = self.vessel_anatomy(torch.cat([source_v, vessel_prob], dim=1))
        anatomy = layer_conf * layer_anatomy + vessel_conf * vessel_anatomy
        gate = torch.sigmoid(self.seg_to_denoise_gate(torch.cat([denoise, anatomy], dim=1)))
        injection = strength * self.seg_scale * gate * anatomy if self.enable_seg_to_denoise else torch.zeros_like(denoise)
        details = {
            "seg_to_denoise_gate": gate, "seg_to_denoise_injection": injection,
            "seg_to_denoise_injection_relative_rms": injection.float().square().mean().sqrt()
            / (denoise.detach().float().square().mean().sqrt() + 1e-8),
            "seg_scale_abs_mean": self.seg_scale.detach().abs().mean(),
            "guidance_layer_probability_mean": layer_prob.detach().float().mean(),
            "guidance_vessel_probability_mean": vessel_prob.detach().float().mean(),
            "guidance_layer_probability_std": layer_prob.detach().float().std(),
            "guidance_vessel_probability_std": vessel_prob.detach().float().std(),
            "guidance_layer_probability_min": layer_prob.detach().float().amin(),
            "guidance_layer_probability_max": layer_prob.detach().float().amax(),
            "guidance_vessel_probability_min": vessel_prob.detach().float().amin(),
            "guidance_vessel_probability_max": vessel_prob.detach().float().amax(),
            "guidance_layer_confidence_mean": layer_conf.detach().float().mean(),
            "guidance_layer_confidence_std": layer_conf.detach().float().std(),
            "guidance_vessel_confidence_mean": vessel_conf.detach().float().mean(),
            "guidance_vessel_confidence_std": vessel_conf.detach().float().std(),
            "guidance_finite": (
                torch.isfinite(source_l).all() & torch.isfinite(source_v).all()
                & torch.isfinite(layer_prob).all() & torch.isfinite(vessel_prob).all()
            ).detach().float(),
            "seg_scale_signed_mean": self.seg_scale.detach().float().mean(),
            "seg_source_layer_rms": source_l.detach().float().square().mean().sqrt(),
            "seg_source_vessel_rms": source_v.detach().float().square().mean().sqrt(),
            "seg_transformed_anatomy_rms": anatomy.detach().float().square().mean().sqrt(),
            "denoise_receiver_rms": denoise.detach().float().square().mean().sqrt(),
            "s2d_gate_mean": gate.detach().float().mean(),
            "s2d_gate_std": gate.detach().float().std(),
            "s2d_gate_min": gate.detach().float().amin(),
            "s2d_gate_max": gate.detach().float().amax(),
            "s2d_gate_saturation_fraction": ((gate < 0.05) | (gate > 0.95)).detach().float().mean(),
            "s2d_gate_entropy": binary_entropy(gate.detach().float()).mean(),
        }
        return denoise + injection, details

    def denoise_to_seg(
        self,
        denoise: torch.Tensor,
        layer: torch.Tensor,
        vessel: torch.Tensor,
        detach_source: bool = False,
        strength: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Inject denoising features into segmentation after the S->D pass."""
        source_d = denoise.detach() if detach_source else denoise
        noise_hint = torch.abs(torch.tanh(self.noise_head(source_d)))
        restoration = self.restoration_context(torch.cat([source_d, noise_hint], dim=1))
        d2l_gate = torch.sigmoid(self.denoise_to_layer_gate(torch.cat([layer, restoration], dim=1)))
        d2v_gate = torch.sigmoid(self.denoise_to_vessel_gate(torch.cat([vessel, restoration], dim=1)))
        layer_injection = strength * self.layer_scale * d2l_gate * self.denoise_to_layer(restoration) if self.enable_denoise_to_seg else torch.zeros_like(layer)
        vessel_injection = strength * self.vessel_scale * d2v_gate * self.denoise_to_vessel(restoration) if self.enable_denoise_to_seg else torch.zeros_like(vessel)
        details = {
            "noise_hint": noise_hint, "denoise_to_layer_gate": d2l_gate,
            "denoise_to_vessel_gate": d2v_gate, "denoise_to_layer_injection": layer_injection,
            "denoise_to_vessel_injection": vessel_injection,
            "denoise_to_layer_injection_abs_mean": layer_injection.detach().abs().mean(),
            "denoise_to_vessel_injection_abs_mean": vessel_injection.detach().abs().mean(),
            "denoise_to_layer_injection_relative_rms": layer_injection.float().square().mean().sqrt()
            / (layer.detach().float().square().mean().sqrt() + 1e-8),
            "denoise_to_vessel_injection_relative_rms": vessel_injection.float().square().mean().sqrt()
            / (vessel.detach().float().square().mean().sqrt() + 1e-8),
            "layer_scale_abs_mean": self.layer_scale.detach().abs().mean(),
            "vessel_scale_abs_mean": self.vessel_scale.detach().abs().mean(),
            "denoise_guidance_mean": source_d.detach().float().mean(),
            "denoise_guidance_std": source_d.detach().float().std(),
            "denoise_guidance_finite": torch.isfinite(source_d).all().detach().float(),
            "layer_scale_signed_mean": self.layer_scale.detach().float().mean(),
            "vessel_scale_signed_mean": self.vessel_scale.detach().float().mean(),
            "denoise_source_rms": source_d.detach().float().square().mean().sqrt(),
            "restoration_transformed_rms": restoration.detach().float().square().mean().sqrt(),
            "layer_receiver_rms": layer.detach().float().square().mean().sqrt(),
            "vessel_receiver_rms": vessel.detach().float().square().mean().sqrt(),
            "d2l_gate_mean": d2l_gate.detach().float().mean(),
            "d2l_gate_std": d2l_gate.detach().float().std(),
            "d2l_gate_min": d2l_gate.detach().float().amin(),
            "d2l_gate_max": d2l_gate.detach().float().amax(),
            "d2l_gate_saturation_fraction": ((d2l_gate < 0.05) | (d2l_gate > 0.95)).detach().float().mean(),
            "d2l_gate_entropy": binary_entropy(d2l_gate.detach().float()).mean(),
            "d2v_gate_mean": d2v_gate.detach().float().mean(),
            "d2v_gate_std": d2v_gate.detach().float().std(),
            "d2v_gate_min": d2v_gate.detach().float().amin(),
            "d2v_gate_max": d2v_gate.detach().float().amax(),
            "d2v_gate_saturation_fraction": ((d2v_gate < 0.05) | (d2v_gate > 0.95)).detach().float().mean(),
            "d2v_gate_entropy": binary_entropy(d2v_gate.detach().float()).mean(),
        }
        return layer + layer_injection, vessel + vessel_injection, details

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
                "denoise_to_layer_injection_relative_rms": (
                    layer_injection.detach().float().square().mean().sqrt()
                    / (layer.detach().float().square().mean().sqrt() + 1e-8)
                ),
                "denoise_to_vessel_injection_relative_rms": (
                    vessel_injection.detach().float().square().mean().sqrt()
                    / (vessel.detach().float().square().mean().sqrt() + 1e-8)
                ),
                "layer_scale_abs_mean": self.layer_scale.detach().abs().mean(),
                "vessel_scale_abs_mean": self.vessel_scale.detach().abs().mean(),
            }
        return denoise_updated, layer_updated, vessel_updated, auxiliary
