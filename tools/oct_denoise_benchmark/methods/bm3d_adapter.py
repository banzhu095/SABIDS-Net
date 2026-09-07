from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from .base import AdapterContext


def bm3d_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    import bm3d

    method_id = str(config.get("method_id", "bm3d_standard"))
    profile_name = str(config.get("profile", "standard" if method_id == "bm3d_standard" else "lc")).lower()
    if method_id == "bm3d_standard" and profile_name not in {"standard", "np"}:
        raise ValueError("bm3d_standard requires the standard BM3D profile")
    profile = bm3d.BM3DProfile() if profile_name in {"standard", "np"} else bm3d.BM3DProfileLC()
    if hasattr(profile, "num_threads"):
        profile.num_threads = int(config.get("num_threads", 1))
    stage_name = str(config.get("stage", "all")).lower()
    stage = bm3d.BM3DStages.HARD_THRESHOLDING if stage_name in {"hard", "hard_thresholding"} else bm3d.BM3DStages.ALL_STAGES
    if method_id == "bm3d_standard" and stage != bm3d.BM3DStages.ALL_STAGES:
        raise ValueError("bm3d_standard requires hard-thresholding and Wiener stages")
    return bm3d.bm3d(image, sigma_psd=float(config["sigma_psd"]), profile=profile, stage_arg=stage)
