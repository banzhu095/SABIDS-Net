from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from sabids.config import load_config
from sabids.engine.trainer import build_model

from .base import AdapterContext
from .deep_common import tiled_forward


class _DenoiseOnly(nn.Module):
    """Expose the current SABIDS Stage-1 denoiser as a plain image model."""

    def __init__(self, model: nn.Module, multiple: int = 8):
        super().__init__()
        self.model = model
        self.multiple = multiple

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        pad_h, pad_w = (-height) % self.multiple, (-width) % self.multiple
        padded = F.pad(image, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else image
        output = self.model.forward_denoise_only(padded)
        denoised = output["denoised_raw"]
        return denoised[..., :height, :width]


def _resolve_config(config: Mapping[str, Any], checkpoint: Path) -> Path | None:
    configured = config.get("config_path") or config.get("resolved_config")
    if configured:
        path = Path(str(configured)).expanduser()
        if not path.is_absolute():
            project_root = Path(str(config.get("project_root", "."))).resolve()
            path = project_root / path
        if path.is_file():
            return path.resolve()
        raise FileNotFoundError(f"SABIDS config does not exist: {path}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    embedded = payload.get("config") if isinstance(payload, dict) else None
    if isinstance(embedded, dict) and "model" in embedded:
        return None
    raise ValueError("sabids_current requires config_path/resolved_config or an embedded checkpoint config")


def sabids_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    if context.checkpoint is None:
        raise ValueError("sabids_current requires a locked Stage-1 checkpoint")
    model = context.extras.get("loaded_model")
    if model is None:
        checkpoint = Path(context.checkpoint).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config_path = _resolve_config(config, checkpoint)
        model_config = payload.get("config") if config_path is None else load_config(config_path)
        if not isinstance(model_config, dict) or "model" not in model_config:
            raise ValueError("SABIDS checkpoint/config does not describe a model")
        network = build_model(model_config)
        state = payload.get("model", payload)
        network.load_state_dict(state, strict=True)
        model = _DenoiseOnly(network).to(context.device).eval()
        context.extras.update({"loaded_model": model, "resolved_config_path": str(config_path) if config_path is not None else "embedded"})
    height, width = image.shape
    context.extras["last_padding"] = {"bottom": (-height) % 8, "right": (-width) % 8, "mode": "replicate"}
    return tiled_forward(model, image, context)
