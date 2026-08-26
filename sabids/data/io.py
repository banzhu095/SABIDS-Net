from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    """Read an OCT image through OpenCV with Windows/Chinese-path support."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    buffer = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"OpenCV failed to decode: {path}")
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_dtype = image.dtype
    image = image.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        maximum = float(np.iinfo(original_dtype).max)
        if maximum > 0:
            image /= maximum
    else:
        finite = image[np.isfinite(image)]
        if finite.size and finite.max() > 1.0:
            image /= float(finite.max())
    return np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)


def read_mask(path: str | Path) -> np.ndarray:
    mask = read_gray(path)
    return (mask > 0.5).astype(np.float32)


def write_gray(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(image, 0.0, 1.0)
    array = np.round(array * 255.0).astype(np.uint8)
    suffix = path.suffix if path.suffix else ".png"
    success, encoded = cv2.imencode(suffix, array)
    if not success:
        raise RuntimeError(f"OpenCV failed to encode: {path}")
    encoded.tofile(str(path))

