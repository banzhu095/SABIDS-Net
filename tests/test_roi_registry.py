from __future__ import annotations

from pathlib import Path

import pytest

from tools.denoise_result_review.roi_registry import ROIRegistry, require_locked, square_from_center


@pytest.mark.parametrize("size", [32, 48, 64])
def test_square_roi_sizes_and_original_coordinate_mapping(size: int):
    x0, y0, x1, y1 = square_from_center(100.0, 120.0, size, 640, 640)
    assert x1 - x0 == size and y1 - y0 == size
    assert (x0 + x1) / 2 == 100 and (y0 + y1) / 2 == 120


def test_square_roi_rejects_boundary_crossing():
    with pytest.raises(ValueError, match="boundary"):
        square_from_center(5, 5, 48, 640, 640)


def test_registry_save_lock_restore_version_and_explicit_unlock(tmp_path: Path):
    registry = ROIRegistry(tmp_path / "中文输出")
    registry.add(dataset="PKU37", split="test", position_id="pku_0006", sample_id="pku_0006_f26", tissue="retina", roi_size=48, center_x=100, center_y=100, width=640, height=640, selection_image="noisy_and_reference_only", selection_reason="manual blinded", reference_sha256="r", noisy_sha256="n")
    registry.save(); restored = ROIRegistry(registry.output_root)
    assert len(restored.frame) == 1 and not restored.locked
    restored.lock(); require_locked(restored.csv_path)
    with pytest.raises(RuntimeError): restored.add(dataset="PKU37", split="test", position_id="p", sample_id="s", tissue="custom", roi_size=48, center_x=100, center_y=100, width=640, height=640, selection_image="x", selection_reason="x", reference_sha256="r", noisy_sha256="n")
    restored.unlock("coordinate review correction")
    assert not restored.locked
    assert list(restored.version_dir.glob("unlock_reason_*.txt"))


def test_unlocked_registry_cannot_be_formally_evaluated(tmp_path: Path):
    registry = ROIRegistry(tmp_path)
    registry.add(dataset="PKU37", split="test", position_id="p", sample_id="s", tissue="custom", roi_size=32, center_x=50, center_y=50, width=100, height=100, selection_image="x", selection_reason="x", reference_sha256="r", noisy_sha256="n")
    registry.save()
    with pytest.raises(RuntimeError, match="locked"):
        require_locked(registry.csv_path)
