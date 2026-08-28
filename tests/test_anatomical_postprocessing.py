from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from sabids.engine.evaluator import evaluate_model
from sabids.postprocessing import clean_layer_mask, hard_contain_vessel, regularize_lower_boundary
from tools.prepare_current_data import split_multiclass_labels


def test_multiclass_vessel_is_in_layer_and_ignore_is_excluded(tmp_path: Path):
    source = tmp_path / "Label" / "voc_seg"
    reference = tmp_path / "Label" / "voc_jpg"
    source.mkdir(parents=True)
    reference.mkdir(parents=True)
    label = np.array([[0, 1], [2, 255]], dtype=np.uint8)
    Image.fromarray(label).save(source / "0001.png")
    labels, _ = split_multiclass_labels(tmp_path, 1, 2, 255, True, False)
    layer = np.asarray(Image.open(tmp_path / labels["0001"]["layer_mask_path"])) > 0
    vessel = np.asarray(Image.open(tmp_path / labels["0001"]["vessel_mask_path"])) > 0
    valid = np.asarray(Image.open(tmp_path / labels["0001"]["label_valid_mask_path"])) > 0
    assert layer.tolist() == [[False, True], [True, False]]
    assert vessel.tolist() == [[False, False], [True, False]]
    assert valid.tolist() == [[True, True], [True, False]]


def test_p1_preserves_invalid_region_and_does_not_synthesize_empty():
    valid = np.ones((8, 8), dtype=bool)
    valid[:, 0] = False
    result, stats = clean_layer_mask(np.zeros_like(valid), valid)
    assert not result.any()
    assert stats["p1_empty_input"] == 1.0
    assert stats["p1_cleanup_failed"] == 1.0


def test_p1_reports_fragmented_failure_and_fills_main_hole():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:7, 1:5] = True
    mask[4, 3] = False
    mask[2:5, 8:11] = True
    cleaned, stats = clean_layer_mask(mask, np.ones_like(mask), minimum_main_fraction=0.75)
    assert cleaned[4, 3]
    assert not cleaned[:, 8:11].any()
    assert stats["p1_component_count"] == 2.0
    assert stats["p1_cleanup_failed"] == 1.0


def test_p2_keeps_upper_boundary_and_limits_displacement():
    mask = np.zeros((20, 10), dtype=bool)
    for column, lower in enumerate([9, 9, 10, 16, 10, 9, 10, 9, 10, 9]):
        mask[4:lower + 1, column] = True
    result, stats = regularize_lower_boundary(mask, np.ones_like(mask), 5.0, 3)
    assert all(np.flatnonzero(result[:, column])[0] == 4 for column in range(10))
    assert stats["p2_max_abs_displacement"] <= 3


def test_p3_is_strict_containment_and_accounts_removed_tp_fp():
    vessel = np.array([[1, 1, 1, 0]], dtype=bool)
    layer = np.array([[1, 0, 0, 0]], dtype=bool)
    valid = np.array([[1, 1, 1, 0]], dtype=bool)
    target = np.array([[1, 1, 0, 0]], dtype=bool)
    final, stats = hard_contain_vessel(vessel, layer, valid, target)
    assert np.array_equal(final, np.array([[1, 0, 0, 0]], dtype=bool))
    assert not np.any(final & ~layer)
    assert stats["p3_removed_tp"] == 1.0
    assert stats["p3_removed_fp"] == 1.0


def test_p3_does_not_apply_largest_component_cleanup_to_vessels():
    vessel = np.zeros((6, 10), dtype=bool)
    vessel[2:4, 1:3] = True
    vessel[2, 8] = True
    final, _ = hard_contain_vessel(vessel, np.ones_like(vessel), np.ones_like(vessel))
    assert np.array_equal(final, vessel)


def test_v0_evaluator_restores_geometry_and_reports_selected_tasks(tmp_path: Path):
    class OneSample(Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            image = torch.linspace(0, 1, 24).reshape(1, 4, 6)
            layer = torch.zeros_like(image)
            layer[:, 1:4, :] = 1
            vessel = torch.zeros_like(image)
            vessel[:, 2, 1:5] = 1
            return {
                "image": image, "clean": image, "layer_mask": layer,
                "vessel_mask": vessel, "valid_mask": torch.ones_like(image),
                "vessel_valid_mask": torch.ones_like(image),
                "has_clean": True, "has_layer": True, "has_vessel": True,
                "sample_id": "s1", "group_id": "g1", "patient_id": "p1",
                "dataset": "synthetic", "scan_protocol": "repeat",
                "original_path": "image.png", "clean_path": "clean.png",
                "layer_mask_path": "layer.png", "vessel_mask_path": "vessel.png",
                "original_height": 8, "original_width": 12,
                "manifest_group_frames": 1,
            }

    class IdentityModel(torch.nn.Module):
        def forward(self, image, **kwargs):
            return {"denoised": image, "layer_prob": torch.ones_like(image) * 0.8,
                    "vessel_prob": torch.ones_like(image) * 0.2, "auxiliary": []}

    summary = evaluate_model(
        IdentityModel(), DataLoader(OneSample(), batch_size=1), torch.device("cpu"),
        output_dir=tmp_path / "evaluation", tasks=("denoise", "layer", "vessel"),
        postprocess_modes=("p0", "p1", "p2", "p3"), restore_original_geometry=True,
    )
    assert summary["evaluated_tasks"] == ["denoise", "layer", "vessel"]
    assert summary["restored_original_geometry"] is True
    frame = (tmp_path / "evaluation" / "frame_metrics.csv").read_text(encoding="utf-8-sig")
    assert "evaluation_height,evaluation_width" in frame
    assert "p3_vessel_dice" in frame
    denoise_dir = tmp_path / "denoise_only"
    denoise_summary = evaluate_model(
        IdentityModel(), DataLoader(OneSample(), batch_size=1), torch.device("cpu"),
        output_dir=denoise_dir, tasks=("denoise",), restore_original_geometry=True,
    )
    assert denoise_summary["evaluated_tasks"] == ["denoise"]
    assert "vessel_stroma_cnr_denoised" in (denoise_dir / "frame_metrics.csv").read_text(encoding="utf-8-sig")
