from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import AdapterContext
from .deep_common import load_checkpoint, tiled_forward


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps) * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = x.chunk(2, dim=1)
        return left * right


class NAFBlock(nn.Module):
    """NAFNet's activation-free block, adapted only in surrounding I/O channels."""
    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2, drop_out_rate: float = 0.0):
        super().__init__()
        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, 1)
        self.conv2 = nn.Conv2d(dw_channels, dw_channels, 3, padding=1, groups=dw_channels)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw_channels // 2, dw_channels // 2, 1))
        self.conv3 = nn.Conv2d(dw_channels // 2, channels, 1)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channels, 1)
        self.conv5 = nn.Conv2d(ffn_channels // 2, channels, 1)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv3(self.sg(self.conv2(self.conv1(self.norm1(inp)))))
        x = x * self.sca(x)
        y = inp + self.dropout1(x) * self.beta
        x = self.conv5(self.sg(self.conv4(self.norm2(y))))
        return y + self.dropout2(x) * self.gamma


class NAFNet(nn.Module):
    def __init__(self, width: int = 32, enc_blocks: Sequence[int] = (1, 1, 1, 28), middle_blocks: int = 1, dec_blocks: Sequence[int] = (1, 1, 1, 1)):
        super().__init__()
        if len(enc_blocks) != len(dec_blocks):
            raise ValueError("encoder and decoder levels must match")
        self.intro = nn.Conv2d(1, width, 3, padding=1)
        self.ending = nn.Conv2d(width, 1, 3, padding=1)
        self.encoders, self.decoders = nn.ModuleList(), nn.ModuleList()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        channels = width
        for count in enc_blocks:
            self.encoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, 2))
            channels *= 2
        self.middle = nn.Sequential(*(NAFBlock(channels) for _ in range(middle_blocks)))
        for count in dec_blocks:
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels * 2, 1, bias=False), nn.PixelShuffle(2)))
            channels //= 2
            self.decoders.append(nn.Sequential(*(NAFBlock(channels) for _ in range(count))))
        self.padder_size = 2 ** len(enc_blocks)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        height, width = inp.shape[-2:]
        padded = F.pad(inp, (0, (-width) % self.padder_size, 0, (-height) % self.padder_size))
        x = self.intro(padded)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x); skips.append(x); x = down(x)
        x = self.middle(x)
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = decoder(up(x) + skip)
        return (self.ending(x) + padded)[..., :height, :width]


def _int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in value)


def nafnet_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    model = NAFNet(
        width=int(config.get("width", 32)),
        enc_blocks=_int_tuple(config.get("enc_blocks", [1, 1, 1, 28])),
        middle_blocks=int(config.get("middle_blocks", 1)),
        dec_blocks=_int_tuple(config.get("dec_blocks", [1, 1, 1, 1])),
    )
    load_checkpoint(model, context, "nafnet_paired")
    return tiled_forward(model, image, context)
