"""Existing NAF encoder + two task decoders, with no restoration modules."""
from __future__ import annotations

import torch
from torch import nn

from sabids.models.blocks import BlockStack, DecoderStage, Downsample, TaskAdapter
from sabids.losses.common import masked_bce_dice_loss, masked_negative_bce_loss


class Segmenter(nn.Module):
    def __init__(self, channels=(32, 64, 128, 256), depths=(2, 2, 4, 6), decoder_depth=2):
        super().__init__()
        self.stem = nn.Conv2d(1, channels[0], 3, padding=1)
        self.encoder = nn.ModuleList([BlockStack(c, d) for c, d in zip(channels, depths)])
        self.down = nn.ModuleList([Downsample(a, b) for a, b in zip(channels, channels[1:])])
        self.levels = list(range(len(channels) - 2, -1, -1))
        self.adapters = nn.ModuleDict({t: nn.ModuleList([TaskAdapter(c) for c in channels]) for t in ['layer', 'vessel']})
        self.decoders = nn.ModuleDict({t: nn.ModuleList([DecoderStage(channels[i+1], channels[i], decoder_depth) for i in self.levels]) for t in ['layer', 'vessel']})
        self.heads = nn.ModuleDict({t: nn.Conv2d(channels[0], 1, 1) for t in ['layer', 'vessel']})

    def forward(self, image):
        x, features = self.stem(image), []
        for i, block in enumerate(self.encoder):
            x = block(x)
            features.append(x)
            if i < len(self.down):
                x = self.down[i](x)
        output = {}
        for task in ['layer', 'vessel']:
            x = self.adapters[task][-1](features[-1])
            for j, i in enumerate(self.levels):
                x = self.decoders[task][j](x, self.adapters[task][i](features[i]))
            output[task] = self.heads[task](x)
        return output


def objective(output, layer, vessel, valid, vessel_valid):
    outside, _ = masked_negative_bce_loss(output['vessel'], vessel_valid * (1-layer))
    return (masked_bce_dice_loss(output['layer'], layer, valid)
            + masked_bce_dice_loss(output['vessel'], vessel, vessel_valid * layer)
            + .5 * outside)
