from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .base import AdapterContext


def tiled_forward(model: nn.Module, image: np.ndarray, context: AdapterContext) -> np.ndarray:
    device = torch.device(context.device)
    tensor = torch.from_numpy(image)[None, None].to(device)
    tile = context.tile_size
    with torch.inference_mode():
        if not tile or (image.shape[0] <= tile and image.shape[1] <= tile):
            result = model(tensor)
        else:
            overlap = min(context.tile_overlap, tile // 2)
            step = tile - overlap
            result = torch.zeros_like(tensor)
            weight = torch.zeros_like(tensor)
            ys = list(range(0, max(image.shape[0] - tile + 1, 1), step))
            xs = list(range(0, max(image.shape[1] - tile + 1, 1), step))
            ys.append(max(image.shape[0] - tile, 0)); xs.append(max(image.shape[1] - tile, 0))
            for y in sorted(set(ys)):
                for x in sorted(set(xs)):
                    patch = tensor[..., y:y + tile, x:x + tile]
                    pred = model(patch)
                    result[..., y:y + pred.shape[-2], x:x + pred.shape[-1]] += pred
                    weight[..., y:y + pred.shape[-2], x:x + pred.shape[-1]] += 1
            result = result / weight.clamp_min(1)
    return result[0, 0].detach().cpu().numpy()


def load_checkpoint(model: nn.Module, context: AdapterContext, expected_architecture: str) -> nn.Module:
    if context.checkpoint is None:
        raise ValueError(f"{expected_architecture} requires a locked checkpoint")
    checkpoint = torch.load(Path(context.checkpoint), map_location="cpu", weights_only=False)
    architecture = checkpoint.get("architecture")
    if architecture and architecture != expected_architecture:
        raise ValueError(f"checkpoint architecture {architecture!r} != {expected_architecture!r}")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    return model.to(context.device).eval()
