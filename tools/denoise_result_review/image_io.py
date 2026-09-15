from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode_lossless(path: str | Path) -> np.ndarray:
    """Decode a PNG/TIFF from a Unicode-safe path without normalizing it."""
    source = Path(path)
    encoded = np.fromfile(source, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot decode image: {source}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        if image.shape[2] != 3 or not (
            np.array_equal(image[:, :, 0], image[:, :, 1])
            and np.array_equal(image[:, :, 0], image[:, :, 2])
        ):
            raise ValueError(f"non-grayscale multichannel image: {source}")
        image = image[:, :, 0]
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image: {source} shape={image.shape}")
    return np.ascontiguousarray(image)


def to_float01(image: np.ndarray, float_data_range: float | None = None) -> np.ndarray:
    """Convert by declared dtype range; never infer scale from image extrema."""
    value = np.asarray(image)
    if value.dtype == np.uint8:
        scale = 255.0
    elif value.dtype == np.uint16:
        scale = 65535.0
    elif np.issubdtype(value.dtype, np.floating):
        if float_data_range is None or not np.isfinite(float_data_range) or float_data_range <= 0:
            raise ValueError("float images require a positive manifest data_range")
        scale = float(float_data_range)
    else:
        raise TypeError(f"unsupported image dtype: {value.dtype}")
    result = value.astype(np.float32) / scale
    if not np.isfinite(result).all() or result.min(initial=0.0) < 0 or result.max(initial=0.0) > 1:
        raise ValueError("image values fall outside the declared [0, data_range]")
    return result


def read_float01(path: str | Path, float_data_range: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    raw = decode_lossless(path)
    bits = int(raw.dtype.itemsize * 8)
    return to_float01(raw, float_data_range), {
        "shape": list(raw.shape), "height": int(raw.shape[0]), "width": int(raw.shape[1]),
        "dtype": str(raw.dtype), "bit_depth": bits, "sha256": sha256_file(path),
    }


def display_uint8(image: np.ndarray, window: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    low, high = map(float, window)
    if not 0 <= low < high <= 1:
        raise ValueError("display window must satisfy 0 <= low < high <= 1")
    return np.round(np.clip((np.asarray(image) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
