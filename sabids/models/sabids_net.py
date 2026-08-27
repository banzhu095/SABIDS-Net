from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import BlockStack, DecoderStage, Downsample, TaskAdapter
from .ugbi import UGBIBlock


class SABIDSNet(nn.Module):
    """Sparse-annotation-aware bidirectional denoising and segmentation network."""

    def __init__(
        self,
        in_channels: int = 1,
        channels: Sequence[int] = (32, 64, 128, 256),
        encoder_depths: Sequence[int] = (2, 2, 4, 6),
        decoder_depth: int = 2,
        interaction_levels: Iterable[int] = (3, 2, 1),
        enable_seg_to_denoise: bool = True,
        enable_denoise_to_seg: bool = True,
        use_uncertainty: bool = True,
        detach_denoise_to_seg_source: bool = False,
        dropout: float = 0.0,
        residual_scale: float = 0.5,
    ) -> None:
        super().__init__()
        if len(channels) != len(encoder_depths):
            raise ValueError("channels and encoder_depths must have equal length")
        self.channels = list(channels)
        self.interaction_levels = set(int(level) for level in interaction_levels)
        self.residual_scale = residual_scale
        self.enable_seg_to_denoise = enable_seg_to_denoise
        self.enable_denoise_to_seg = enable_denoise_to_seg
        self.detach_denoise_to_seg_source = detach_denoise_to_seg_source

        self.stem = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.encoder_blocks = nn.ModuleList(
            [
                BlockStack(channel, depth, dropout=dropout)
                for channel, depth in zip(channels, encoder_depths)
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                Downsample(channels[index], channels[index + 1])
                for index in range(len(channels) - 1)
            ]
        )

        self.adapters = nn.ModuleDict(
            {
                task: nn.ModuleList([TaskAdapter(channel) for channel in channels])
                for task in ("denoise", "layer", "vessel")
            }
        )
        reversed_levels = list(range(len(channels) - 2, -1, -1))
        self.decoder_levels = reversed_levels
        self.decoders = nn.ModuleDict()
        for task in ("denoise", "layer", "vessel"):
            self.decoders[task] = nn.ModuleList(
                [
                    DecoderStage(channels[level + 1], channels[level], decoder_depth)
                    for level in reversed_levels
                ]
            )

        self.interactions = nn.ModuleDict(
            {
                str(level): UGBIBlock(
                    channels[level],
                    enable_seg_to_denoise=enable_seg_to_denoise,
                    enable_denoise_to_seg=enable_denoise_to_seg,
                    use_uncertainty=use_uncertainty,
                )
                for level in self.interaction_levels
            }
        )
        self.residual_head = nn.Conv2d(channels[0], 1, 3, padding=1)
        self.layer_head = nn.Conv2d(channels[0], 1, 1)
        self.boundary_head = nn.Conv2d(channels[0], 2, 1)
        self.vessel_head = nn.Conv2d(channels[0], 1, 1)

    def encode(self, image: torch.Tensor) -> List[torch.Tensor]:
        features: List[torch.Tensor] = []
        value = self.stem(image)
        for level, blocks in enumerate(self.encoder_blocks):
            value = blocks(value)
            features.append(value)
            if level < len(self.downsamples):
                value = self.downsamples[level](value)
        return features

    def _interact(
        self,
        level: int,
        denoise: torch.Tensor,
        layer: torch.Tensor,
        vessel: torch.Tensor,
        detach_cross: bool,
        auxiliary: Optional[List[Dict[str, torch.Tensor]]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if level not in self.interaction_levels:
            return denoise, layer, vessel
        denoise, layer, vessel, details = self.interactions[str(level)](
            denoise,
            layer,
            vessel,
            detach_cross=detach_cross,
            detach_denoise_to_seg=self.detach_denoise_to_seg_source,
            return_details=auxiliary is not None,
        )
        if auxiliary is not None and details is not None:
            details["level"] = torch.tensor(level, device=denoise.device)
            auxiliary.append(details)
        return denoise, layer, vessel

    def forward(
        self,
        image: torch.Tensor,
        detach_cross: bool = False,
        return_features: bool = True,
        return_auxiliary: bool = True,
    ) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        encoder_features = self.encode(image)
        deepest = len(self.channels) - 1
        denoise = self.adapters["denoise"][deepest](encoder_features[deepest])
        layer = self.adapters["layer"][deepest](encoder_features[deepest])
        vessel = self.adapters["vessel"][deepest](encoder_features[deepest])
        auxiliary: Optional[List[Dict[str, torch.Tensor]]] = (
            [] if return_auxiliary else None
        )
        denoise, layer, vessel = self._interact(
            deepest, denoise, layer, vessel, detach_cross, auxiliary
        )
        anatomy_embedding = F.adaptive_avg_pool2d(
            torch.cat([layer, vessel], dim=1), 1
        ).flatten(1)

        task_features = {"denoise": denoise, "layer": layer, "vessel": vessel}
        for stage_index, level in enumerate(self.decoder_levels):
            for task in task_features:
                skip = self.adapters[task][level](encoder_features[level])
                task_features[task] = self.decoders[task][stage_index](
                    task_features[task], skip
                )
            task_features["denoise"], task_features["layer"], task_features[
                "vessel"
            ] = self._interact(
                level,
                task_features["denoise"],
                task_features["layer"],
                task_features["vessel"],
                detach_cross,
                auxiliary,
            )

        residual = self.residual_scale * torch.tanh(
            self.residual_head(task_features["denoise"])
        )
        denoised_raw = image - residual
        layer_logits = self.layer_head(task_features["layer"])
        vessel_logits = self.vessel_head(task_features["vessel"])
        output: Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]] = {
            "denoised_raw": denoised_raw,
            "denoised": denoised_raw.clamp(0.0, 1.0),
            "residual": residual,
            "layer_logits": layer_logits,
            "vessel_logits": vessel_logits,
            "layer_prob": torch.sigmoid(layer_logits),
            "vessel_prob": torch.sigmoid(vessel_logits),
            "boundary_logits": self.boundary_head(task_features["layer"]),
            "auxiliary": auxiliary or [],
        }
        if return_features:
            output["anatomy_embedding"] = anatomy_embedding
        return output

    @staticmethod
    def _set_module_trainable(module: nn.Module, trainable: bool = True) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    def set_train_stage(
        self,
        stage: str,
        private_train_encoder_levels: Iterable[int] = (),
        freeze_shared_encoder: bool = False,
        train_denoise_to_seg: bool = False,
    ) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if stage == "denoise":
            for name, parameter in self.named_parameters():
                if any(token in name for token in ("layer", "vessel", "interactions")):
                    parameter.requires_grad_(False)
        elif stage == "segment":
            for name, parameter in self.named_parameters():
                if any(token in name for token in ("denoise", "residual_head", "interactions")):
                    parameter.requires_grad_(False)
            if freeze_shared_encoder:
                self._set_module_trainable(self.stem, False)
                self._set_module_trainable(self.encoder_blocks, False)
                self._set_module_trainable(self.downsamples, False)
            if train_denoise_to_seg:
                if self.enable_seg_to_denoise:
                    raise ValueError(
                        "Safe Stage 2 D->S requires model.enable_seg_to_denoise=false "
                        "so changing segmentation features cannot alter denoising."
                    )
                if not self.enable_denoise_to_seg:
                    raise ValueError(
                        "train_denoise_to_seg=true requires "
                        "model.enable_denoise_to_seg=true"
                    )
                for interaction in self.interactions.values():
                    for module in (
                        interaction.noise_head,
                        interaction.restoration_context,
                        interaction.denoise_to_layer_gate,
                        interaction.denoise_to_vessel_gate,
                        interaction.denoise_to_layer,
                        interaction.denoise_to_vessel,
                    ):
                        self._set_module_trainable(module)
                    interaction.layer_scale.requires_grad_(True)
                    interaction.vessel_scale.requires_grad_(True)
        elif stage == "private_seg":
            # Start from a fully frozen public model. Private adaptation then
            # updates only the layer/vessel pathways and D->S interaction path,
            # preserving the pretrained denoising encoder/decoder by default.
            for parameter in self.parameters():
                parameter.requires_grad_(False)

            for task in ("layer", "vessel"):
                self._set_module_trainable(self.adapters[task])
                self._set_module_trainable(self.decoders[task])
            self._set_module_trainable(self.layer_head)
            self._set_module_trainable(self.boundary_head)
            self._set_module_trainable(self.vessel_head)

            for interaction in self.interactions.values():
                # Intermediate segmentation heads support auxiliary supervision.
                self._set_module_trainable(interaction.layer_head)
                self._set_module_trainable(interaction.vessel_head)
                # Fixed denoising features are converted into restoration context
                # and injected only into the two segmentation branches.
                self._set_module_trainable(interaction.noise_head)
                self._set_module_trainable(interaction.restoration_context)
                self._set_module_trainable(interaction.denoise_to_layer_gate)
                self._set_module_trainable(interaction.denoise_to_vessel_gate)
                self._set_module_trainable(interaction.denoise_to_layer)
                self._set_module_trainable(interaction.denoise_to_vessel)
                interaction.layer_scale.requires_grad_(True)
                interaction.vessel_scale.requires_grad_(True)

            levels = sorted(set(int(level) for level in private_train_encoder_levels))
            for level in levels:
                if level < 0 or level >= len(self.encoder_blocks):
                    raise ValueError(
                        f"Invalid private encoder level {level}; expected 0..{len(self.encoder_blocks) - 1}"
                    )
                self._set_module_trainable(self.encoder_blocks[level])
                if level == 0:
                    self._set_module_trainable(self.stem)
                else:
                    self._set_module_trainable(self.downsamples[level - 1])
        elif stage in {"joint", "private", "warmup"}:
            return
        else:
            raise ValueError(f"Unsupported training stage: {stage}")

    def enforce_frozen_eval(self) -> None:
        """Keep the fixed denoising function deterministic during adaptation."""
        modules = (
            self.stem,
            self.encoder_blocks,
            self.downsamples,
            self.adapters["denoise"],
            self.decoders["denoise"],
            self.residual_head,
        )
        for module in modules:
            if not any(parameter.requires_grad for parameter in module.parameters()):
                module.eval()
