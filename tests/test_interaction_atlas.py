from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from tools.build_interaction_atlas import read_crop


def test_missing_atlas_asset_returns_nonempty_placeholder(tmp_path: Path):
    image, status = read_crop(tmp_path / "missing.png", 100, 200, 32, 24)
    assert status == "missing"
    assert image.ndim == 3
    assert image.size > 0


def test_out_of_bounds_atlas_crop_falls_back_without_empty_resize(tmp_path: Path):
    path = tmp_path / "image.png"
    source = np.full((20, 30), 127, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".png", source)
    assert encoded
    path.write_bytes(buffer.tobytes())
    image, status = read_crop(path, 100, 200, 16, 12)
    assert status == "crop_out_of_bounds"
    assert image.shape[:2] == (12, 16)


def test_valid_atlas_crop_uses_shared_original_grid_coordinates(tmp_path: Path):
    path = tmp_path / "image.png"
    source = np.arange(40 * 50, dtype=np.uint16).reshape(40, 50)
    encoded, buffer = cv2.imencode(".png", source)
    assert encoded
    path.write_bytes(buffer.tobytes())
    image, status = read_crop(path, 7, 9, 13, 11)
    assert status == "ok"
    assert image.shape[:2] == (11, 13)
