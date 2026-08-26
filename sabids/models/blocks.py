from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + self.eps) * self.weight + self.bias


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """A compact NAF-style restoration block suitable for OCT features."""

    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        if hidden % 2:
            hidden += 1
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.depthwise = nn.Conv2d(
            hidden, hidden, 3, padding=1, groups=hidden
        )
        self.gate = SimpleGate()
        gated = hidden // 2
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(gated, gated, 1)
        )
        self.conv2 = nn.Conv2d(gated, channels, 1)
        self.dropout1 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = LayerNorm2d(channels)
        self.ffn1 = nn.Conv2d(channels, hidden, 1)
        self.ffn2 = nn.Conv2d(hidden // 2, channels, 1)
        self.dropout2 = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.depthwise(self.conv1(self.norm1(x)))
        value = self.gate(value)
        value = value * self.channel_attention(value)
        value = self.dropout1(self.conv2(value))
        y = x + value * self.beta
        value = self.gate(self.ffn1(self.norm2(y)))
        value = self.dropout2(self.ffn2(value))
        return y + value * self.gamma


class BlockStack(nn.Sequential):
    def __init__(self, channels: int, depth: int, dropout: float = 0.0):
        super().__init__(*[NAFBlock(channels, dropout=dropout) for _ in range(depth)])


class TaskAdapter(nn.Module):
    def __init__(self, channels: int, bottleneck_ratio: int = 4):
        super().__init__()
        hidden = max(channels // bottleneck_ratio, 8)
        self.adapter = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
        )
        self.scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.adapter(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.body = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, depth: int = 2):
        super().__init__()
        self.fuse = nn.Conv2d(in_channels + skip_channels, skip_channels, 3, padding=1)
        self.blocks = BlockStack(skip_channels, depth)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.blocks(self.fuse(torch.cat([x, skip], dim=1)))

