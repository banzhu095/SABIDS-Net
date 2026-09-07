from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_image(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    encoded = np.fromfile(str(path), dtype=np.uint8)
    raw = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"cannot decode {path}")
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2GRAY if raw.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if np.issubdtype(raw.dtype, np.integer):
        maximum = np.iinfo(raw.dtype).max
        image = raw.astype(np.float32) / float(maximum)
        bit_depth = int(np.iinfo(raw.dtype).bits)
    else:
        image = raw.astype(np.float32)
        maximum = 1.0
        bit_depth = int(raw.dtype.itemsize * 8)
    if not np.isfinite(image).all() or image.min(initial=0) < 0 or image.max(initial=1) > 1:
        raise ValueError(f"decoded values outside [0,1]: {path}")
    return image, {"dtype": str(raw.dtype), "source_dtype": raw.dtype, "bit_depth": bit_depth, "width": raw.shape[1], "height": raw.shape[0], "maximum": maximum}


def save_image(path: Path, image: np.ndarray, metadata: dict[str, Any], preserve_bit_depth: bool = True) -> Path:
    path = Path(path)
    if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0, 1)
    dtype = np.dtype(metadata["source_dtype"]) if preserve_bit_depth else np.dtype("uint16")
    if np.issubdtype(dtype, np.integer):
        output = np.round(clipped * np.iinfo(dtype).max).astype(dtype)
    else:
        output = clipped.astype(dtype)
    ok, buffer = cv2.imencode(path.suffix, output)
    if not ok:
        raise RuntimeError(f"cannot encode {path}")
    buffer.tofile(str(path))
    return path
