from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping
import hashlib
import inspect

import numpy as np


@dataclass
class AdapterContext:
    device: str = "cpu"
    checkpoint: Path | None = None
    seed: int = 42
    tile_size: int | None = None
    tile_overlap: int = 32
    extras: dict[str, Any] = field(default_factory=dict)


Adapter = Callable[[np.ndarray, Mapping[str, Any], AdapterContext], np.ndarray]


def _registry() -> dict[str, Adapter]:
    from .bm3d_adapter import bm3d_adapter
    from .dncnn import dncnn_adapter
    from .ksvd_adapter import ksvd_adapter
    from .nafnet import nafnet_adapter
    from .nlm_adapter import nlm_adapter
    from .noisy import noisy_adapter
    from .tv_adapter import tv_adapter

    return {
        "noisy_identity": noisy_adapter,
        "bm3d_standard": bm3d_adapter,
        "bm3d_lc": bm3d_adapter,
        "tv_chambolle": tv_adapter,
        "nlm": nlm_adapter,
        "ksvd_self": ksvd_adapter,
        "dncnn_paired": dncnn_adapter,
        "nafnet_paired": nafnet_adapter,
    }


def adapter_source_sha256(method_id: str) -> str:
    adapter = _registry()[method_id]
    source = inspect.getsourcefile(adapter)
    if source is None:
        raise RuntimeError(f"cannot resolve adapter source for {method_id}")
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def denoise(image_float32_01: np.ndarray, method_config: Mapping[str, Any], context: AdapterContext | Mapping[str, Any] | None = None) -> np.ndarray:
    image = np.asarray(image_float32_01)
    if image.ndim != 2 or image.dtype != np.float32:
        raise TypeError(f"input must be single-channel float32 [0,1], got {image.shape} {image.dtype}")
    if not np.isfinite(image).all() or image.min(initial=0) < 0 or image.max(initial=1) > 1:
        raise ValueError("input contains NaN/Inf or lies outside [0,1]")
    method_id = str(method_config.get("method_id", ""))
    adapters = _registry()
    if method_id not in adapters:
        raise KeyError(f"unknown method_id: {method_id}")
    if context is None:
        ctx = AdapterContext()
    elif isinstance(context, AdapterContext):
        ctx = context
    else:
        known = {key: context[key] for key in ("device", "checkpoint", "seed", "tile_size", "tile_overlap") if key in context}
        if "checkpoint" in known and known["checkpoint"] is not None:
            known["checkpoint"] = Path(known["checkpoint"])
        ctx = AdapterContext(**known, extras={k: v for k, v in context.items() if k not in known})
    output = np.asarray(adapters[method_id](image, method_config, ctx))
    if output.shape != image.shape:
        raise ValueError(f"{method_id} changed shape {image.shape} -> {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError(f"{method_id} produced NaN/Inf")
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
