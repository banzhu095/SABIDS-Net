from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .base import AdapterContext
from .deep_common import tiled_forward


class TCFLGenerator(nn.Module):
    """Generator from TCFL-OCT: a 10-layer grayscale residual-noise DnCNN."""

    def __init__(self, channels: int = 1, num_layers: int = 10, features: int = 64):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(channels, features, 3, padding=1, bias=False), nn.LeakyReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers.extend([nn.Conv2d(features, features, 3, padding=1, bias=False), nn.BatchNorm2d(features), nn.LeakyReLU(inplace=True)])
        layers.append(nn.Conv2d(features, 1, 3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, sqrt(2.0 / n))
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return self.dncnn(noisy)


class TCFLDiscriminator(nn.Module):
    """Official TCFL-OCT PatchGAN discriminator."""

    def __init__(self, channels: int = 1):
        super().__init__()

        def block(in_channels: int, out_channels: int, normalization: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(channels, 64, False), *block(64, 128), *block(128, 256), *block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)), nn.Conv2d(512, 1, 4, padding=1, bias=False),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)


class _TCFLDenoiser(nn.Module):
    def __init__(self, generator: TCFLGenerator):
        super().__init__()
        self.generator = generator

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        return noisy - self.generator(noisy)


def tcfl_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    if context.checkpoint is None:
        raise ValueError("tcfl_dncnn requires a locked checkpoint")
    model = context.extras.get("loaded_model")
    if model is None:
        generator = TCFLGenerator(num_layers=int(config.get("num_layers", 10)), features=int(config.get("features", 64)))
        payload = torch.load(Path(context.checkpoint), map_location="cpu", weights_only=False)
        if isinstance(payload, dict):
            architecture = payload.get("architecture")
            if architecture and architecture != "tcfl_dncnn":
                raise ValueError(f"checkpoint architecture {architecture!r} != 'tcfl_dncnn'")
            state = payload.get("generator", payload.get("model", payload))
        else:
            state = payload
        generator.load_state_dict(state, strict=True)
        model = _TCFLDenoiser(generator).to(context.device).eval()
        context.extras["loaded_model"] = model
    return tiled_forward(model, image, context)
