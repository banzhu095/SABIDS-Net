from __future__ import annotations

from typing import Dict, Iterable, List

import torch
from torch import nn
from torch.nn import functional as F

from .sabids_net import SABIDSNet


class DualViewFusion(nn.Module):
    """Noisy-anchored residual fusion for one encoder scale."""

    def __init__(self, channels: int, scale_init: float = 0.0) -> None:
        super().__init__()
        self.adapter = nn.Conv2d(3 * channels, channels, 1)
        self.gate = nn.Conv2d(3 * channels, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(float(scale_init)))

    def forward(
        self, noisy: torch.Tensor, auxiliary: torch.Tensor
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        delta = self.adapter(torch.cat((noisy, auxiliary, noisy - auxiliary), dim=1))
        gate = torch.sigmoid(
            self.gate(torch.cat((noisy, auxiliary, (noisy - auxiliary).abs()), dim=1))
        )
        injection = self.gamma * gate * delta
        noisy_rms = noisy.float().square().mean().sqrt()
        delta_rms = delta.float().square().mean().sqrt()
        flat_gate = gate.float().flatten()
        quantiles = torch.quantile(flat_gate, flat_gate.new_tensor((0.25, 0.5, 0.75)))
        diagnostics = {
            "dual_gate_mean": gate.float().mean(),
            "dual_gate_std": gate.float().std(unbiased=False),
            "dual_gate_q25": quantiles[0],
            "dual_gate_q50": quantiles[1],
            "dual_gate_q75": quantiles[2],
            "dual_gamma": self.gamma.float(),
            "dual_noisy_feature_rms": noisy_rms,
            "dual_delta_rms": delta_rms,
            "dual_delta_to_noisy_rms": delta_rms / noisy_rms.clamp_min(1e-12),
            "dual_injection_rms": injection.float().square().mean().sqrt(),
        }
        return noisy + injection, diagnostics


class NoisyMildDualViewSegmenter(SABIDSNet):
    """Shared-weight Siamese segmentation encoder with a noisy identity path.

    The inherited denoising path exists only for checkpoint/API compatibility;
    it is frozen and is not evaluated by this forward.  Both encoder calls are
    segmentation-side calls and receive segmentation loss only.
    """

    def __init__(
        self,
        *args,
        dual_view_enabled: bool = True,
        auxiliary_required: bool = True,
        fusion_levels: Iterable[int] = (3, 2, 1),
        dual_scale_init: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.dual_view_enabled = bool(dual_view_enabled)
        self.auxiliary_required = bool(auxiliary_required)
        levels = tuple(int(level) for level in fusion_levels)
        invalid = [level for level in levels if level < 0 or level >= len(self.channels)]
        if invalid:
            raise ValueError(f"Invalid dual-view fusion levels: {invalid}")
        self.dual_fusion_levels = set(levels)
        self.dual_fusions = nn.ModuleDict(
            {
                str(level): DualViewFusion(self.channels[level], dual_scale_init)
                for level in levels
            }
        )

    def set_train_stage(self, stage: str, **kwargs) -> None:  # type: ignore[override]
        if stage != "input_segment":
            raise ValueError("NoisyMildDualViewSegmenter supports input_segment only")
        super().set_train_stage(stage, **kwargs)
        if not self.dual_view_enabled:
            self._set_module_trainable(self.dual_fusions, False)

    def forward(
        self,
        image: torch.Tensor,
        auxiliary_image: torch.Tensor | None = None,
        disable_auxiliary: bool = False,
        return_features: bool = True,
        return_auxiliary: bool = True,
        **_: object,
    ) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        use_auxiliary = self.dual_view_enabled and not bool(disable_auxiliary)
        if use_auxiliary and auxiliary_image is None:
            if self.auxiliary_required:
                raise ValueError("Dual-view forward requires auxiliary_image")
            use_auxiliary = False

        noisy_features = self.encode(image)
        diagnostics: List[Dict[str, torch.Tensor]] = []
        fused_features = list(noisy_features)
        if use_auxiliary:
            auxiliary_features = self.encode(auxiliary_image)  # shared weights
            for level in sorted(self.dual_fusion_levels):
                fused, details = self.dual_fusions[str(level)](
                    noisy_features[level], auxiliary_features[level]
                )
                details["level"] = image.new_tensor(level)
                fused_features[level] = fused
                diagnostics.append(details)

        deepest = len(self.channels) - 1
        layer = self.adapters["layer"][deepest](fused_features[deepest])
        vessel = self.adapters["vessel"][deepest](fused_features[deepest])
        anatomy_embedding = F.adaptive_avg_pool2d(
            torch.cat((layer, vessel), dim=1), 1
        ).flatten(1)
        for stage_index, level in enumerate(self.decoder_levels):
            layer = self.decoders["layer"][stage_index](
                layer, self.adapters["layer"][level](fused_features[level])
            )
            vessel = self.decoders["vessel"][stage_index](
                vessel, self.adapters["vessel"][level](fused_features[level])
            )

        layer_logits = self.layer_head(layer)
        vessel_logits = self.vessel_head(vessel)
        zero_residual = torch.zeros_like(image)
        output: Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]] = {
            "denoised_raw": image,
            "denoised": image,
            "residual": zero_residual,
            "layer_logits": layer_logits,
            "vessel_logits": vessel_logits,
            "layer_prob": torch.sigmoid(layer_logits),
            "vessel_prob": torch.sigmoid(vessel_logits),
            "boundary_logits": self.boundary_head(layer),
            "auxiliary": diagnostics if return_auxiliary else [],
        }
        if return_features:
            output["anatomy_embedding"] = anatomy_embedding
        return output
