from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from sabids.experiments.seg_guided import (
    apply_spatial_dose, audit_residual_identity, block_shuffle,
    destroy_spatial_structure, deterministic_wrong_guides,
    grouped_four_fold_assignment, random_histogram_matched_map,
    shift_without_wrap, signed_residual, spatial_alpha_map,
)
from sabids.models import NoisyMildDualViewSegmenter, SpatialExpertController


def _model(residual_enabled: bool = True) -> NoisyMildDualViewSegmenter:
    torch.manual_seed(17)
    model = NoisyMildDualViewSegmenter(
        channels=(4, 8, 16, 32), encoder_depths=(1, 1, 1, 1),
        decoder_depth=1, interaction_levels=(), enable_seg_to_denoise=False,
        enable_denoise_to_seg=False, fusion_levels=(3, 2, 1), dual_scale_init=0.0,
        residual_aware=True, residual_enabled=residual_enabled,
    )
    model.set_train_stage("input_segment")
    return model


def test_interventions_are_deterministic_and_destroy_pairing() -> None:
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    assert np.array_equal(shift_without_wrap(image, 0), image)
    assert not np.array_equal(shift_without_wrap(image, 2), image)
    shuffled = block_shuffle(image, 2, 42, "sample")
    assert shuffled.shape == image.shape
    assert np.array_equal(shuffled, block_shuffle(image, 2, 42, "sample"))
    destroyed = destroy_spatial_structure(image, 42, "sample")
    assert np.array_equal(np.sort(destroyed.ravel()), np.sort(image.ravel()))
    assert not np.array_equal(destroyed, image)


def test_wrong_guides_stay_in_split_and_change_group() -> None:
    rows = [{"sample_id": f"s{i}", "group_id": f"g{i}", "split": "val"} for i in range(4)]
    mapped = deterministic_wrong_guides(rows, 42, 3)
    assert len(mapped) == 12
    assert all(row["group_id"] != row["auxiliary_group_id"] for row in mapped)
    assert all(row["sample_id"] != row["auxiliary_sample_id"] for row in mapped)


def test_grouped_cv_has_each_development_group_once_and_rejects_sealed() -> None:
    rows = []
    for group in range(16):
        for frame in range(group % 3 + 1):
            rows.append({"sample_id": f"g{group}_f{frame}", "group_id": f"g{group}",
                         "split": "train", "vessel_mask_path": "mask.npy"})
    result = grouped_four_fold_assignment(pd.DataFrame(rows), ["sealed"], seed=7)
    validation = [group for fold in result["folds"].values() for group in fold["val_groups"]]
    assert len(validation) == len(set(validation)) == 16
    assert all(len(fold["val_groups"]) == 4 for fold in result["folds"].values())
    bad = pd.DataFrame(rows + [{"sample_id": "x", "group_id": "sealed", "split": "val",
                               "vessel_mask_path": "mask.npy"}])
    with pytest.raises(ValueError, match="Sealed"):
        grouped_four_fold_assignment(bad, ["sealed"], expected_groups=17)


def test_signed_residual_and_spatial_dose_identities() -> None:
    noisy = np.array([[.2, .8], [.6, .1]], np.float32)
    strong = np.array([[.1, .9], [.4, .2]], np.float32)
    residual = signed_residual(noisy, strong)
    assert residual.min() < 0 < residual.max()
    assert audit_residual_identity(noisy, strong, residual) == 0
    assert np.array_equal(apply_spatial_dose(noisy, strong, np.zeros_like(noisy)), noisy)
    expected = .75 * noisy + .25 * strong
    assert np.allclose(apply_spatial_dose(noisy, strong, np.full_like(noisy, .25)), expected)


def test_spatial_map_constraints_and_random_control_histogram() -> None:
    layer = np.ones((8, 9), bool); vessel = np.zeros_like(layer); vessel[3:5, 4:6] = True
    valid = np.ones_like(layer); valid[0] = False
    alpha, regions = spatial_alpha_map(layer, vessel, valid, 0.0, 0.1, 0.5, 0.25, 1)
    assert alpha.min() >= 0 and alpha.max() <= 1 and np.all(alpha[~valid] == 0)
    assert np.all(alpha[vessel] == 0.0) and regions["vessel_pixels"] == 4
    random = random_histogram_matched_map(alpha, valid, 3, "x")
    assert np.array_equal(np.sort(random[valid]), np.sort(alpha[valid]))
    with pytest.raises(ValueError):
        spatial_alpha_map(layer, vessel, valid, .6, .1, .5, .5)


def test_b3r_zero_init_is_noisy_identity_and_later_mapping_gets_gradient() -> None:
    model = _model(True).train()
    noisy = torch.rand(1, 1, 32, 48)
    mild = torch.rand_like(noisy)
    with torch.no_grad():
        fused = model(noisy, auxiliary_image=mild)
        disabled = model(noisy, auxiliary_image=mild, disable_auxiliary=True)
    assert torch.equal(fused["vessel_logits"], disabled["vessel_logits"])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model(noisy, auxiliary_image=mild)
        output["vessel_logits"].square().mean().backward()
        optimizer.step()
    assert any(fusion.gamma.grad is not None for fusion in model.dual_fusions.values())
    assert any(fusion.adapter.weight.grad is not None and fusion.adapter.weight.grad.abs().sum() > 0
               for fusion in model.dual_fusions.values())
    assert model(noisy, auxiliary_image=mild)["vessel_logits"].shape == noisy.shape


def test_b6r_uses_zero_residual_and_controller_is_convex() -> None:
    model = _model(False).eval()
    noisy = torch.rand(1, 1, 24, 40)
    with torch.no_grad():
        output = model(noisy, auxiliary_image=noisy)
    assert all(float(item["dual_residual_rms"]) == 0.0 for item in output["auxiliary"])
    controller = SpatialExpertController()
    features = torch.rand(1, 7, 24, 40)
    mild, strong = noisy * .8, noisy * .5
    fused, weights = controller(features, noisy, mild, strong)
    assert torch.all(weights >= 0)
    assert torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]))
    assert fused.min() >= torch.minimum(strong, torch.minimum(noisy, mild)).min()
    assert fused.max() <= torch.maximum(strong, torch.maximum(noisy, mild)).max()
