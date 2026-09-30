from __future__ import annotations

import copy

import pytest
import torch

from sabids.models.dual_task_adaptive import (
    DualStrengthController,
    DualTaskAdaptiveSegmenter,
    binary_entropy,
)
from sabids.losses import SABIDSLoss
from sabids.engine.trainer import build_model


def tiny_model() -> DualTaskAdaptiveSegmenter:
    return DualTaskAdaptiveSegmenter(
        model_kwargs={
            "in_channels": 1,
            "channels": (4, 8, 12, 16),
            "encoder_depths": (1, 1, 1, 1),
            "decoder_depth": 1,
            "interaction_levels": (3, 2, 1),
            "enable_seg_to_denoise": False,
            "enable_denoise_to_seg": False,
            "use_uncertainty": True,
            "detach_denoise_to_seg_source": True,
            "dropout": 0.0,
            "residual_scale": 0.5,
            "causal_interaction_experiment": False,
            "detach_seg_to_denoise_source": True,
            "interaction_scale_init": 0.1,
            "s2d_source_mode": "cross",
            "d2s_source_mode": "cross",
            "strong_s2d_rho": None,
            "strong_d2s_rho": None,
        },
        fusion_levels=(3, 2, 1),
        context_channels=4,
    )


def test_shapes_finite_bounds_and_non_square_input() -> None:
    model = tiny_model().eval()
    image = torch.rand(1, 1, 24, 32)
    with torch.no_grad():
        output = model(image)
    for key in ("layer_prob", "vessel_prob", "layer_strength_map", "vessel_strength_map"):
        assert output[key].shape == image.shape
        assert torch.isfinite(output[key]).all()
    assert output["layer_strength_map"].min() >= 0
    assert output["vessel_strength_map"].max() <= 1.25


def test_initial_gate_values_and_zero_correction_equal_coarse() -> None:
    model = tiny_model().eval()
    with torch.no_grad():
        output = model(torch.rand(1, 1, 24, 32))
    assert torch.allclose(output["layer_strength_map"], torch.ones_like(output["layer_strength_map"]), atol=1e-6)
    assert torch.allclose(output["vessel_strength_map"], torch.full_like(output["vessel_strength_map"], 0.5), atol=1e-6)
    assert torch.equal(output["layer_logits"], output["coarse_layer_logits"])
    assert torch.equal(output["vessel_logits"], output["coarse_vessel_logits"])


def test_dose_formula_and_coarse_alpha_identity() -> None:
    model = tiny_model().eval()
    image = torch.rand(1, 1, 24, 32)
    with torch.no_grad():
        output = model(image)
    residual = output["full_d2_residual"]
    assert torch.allclose(output["coarse_denoised"], (image - 0.25 * residual).clamp(0, 1))
    assert torch.allclose(output["fine_layer_denoised"], (image - output["layer_strength_map"] * residual).clamp(0, 1))
    assert torch.allclose(output["fine_vessel_denoised"], (image - output["vessel_strength_map"] * residual).clamp(0, 1))


def test_shared_context_and_independent_task_heads() -> None:
    controller = DualStrengthController(hidden=4)
    assert controller.layer_head is not controller.vessel_head
    assert controller.shared is controller.shared
    context = torch.rand(2, 8, 16, 20)
    layer, vessel = controller(context, (64, 80))
    assert layer.shape == vessel.shape == (2, 1, 64, 80)
    assert not torch.allclose(layer, vessel)


def test_frozen_anchors_receive_no_grad_and_trainable_parts_update() -> None:
    torch.manual_seed(7)
    model = tiny_model().train()
    frozen_before = {name: value.detach().clone() for name, value in model.named_parameters() if not value.requires_grad}
    trainable_before = {name: value.detach().clone() for name, value in model.named_parameters() if value.requires_grad}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(3):
        output = model(torch.rand(1, 1, 24, 32))
        target = torch.rand_like(output["layer_prob"])
        loss = (output["layer_prob"] - target).square().mean()
        loss = loss + (output["vessel_prob"] - target).square().mean()
        loss = loss + 0.01 * output["fine_layer_denoised"].mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    named = dict(model.named_parameters())
    assert all(parameter.grad is None for name, parameter in named.items() if name.startswith(("d2.", "coarse_segmenter.", "fine_backbone.")))
    assert all(torch.equal(named[name], value) for name, value in frozen_before.items())
    changed = [name for name, value in trainable_before.items() if not torch.equal(named[name], value)]
    assert any(name.startswith("controller.") for name in changed)
    assert any("logit_correction" in name for name in changed)
    assert any(name.startswith(("layer_aux_fusions.", "vessel_aux_fusions.")) for name in changed)


def test_clean_and_ground_truth_are_rejected_from_prediction() -> None:
    model = tiny_model()
    image = torch.rand(1, 1, 24, 32)
    with pytest.raises(ValueError, match="forbidden"):
        model(image, clean=image)
    with pytest.raises(ValueError, match="forbidden"):
        model(image, vessel_mask=image)


def test_state_dict_resume_is_exact() -> None:
    model = tiny_model()
    clone = tiny_model()
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    image = torch.rand(1, 1, 24, 32)
    model.eval(); clone.eval()
    with torch.no_grad():
        first, second = model(image), clone(image)
    assert torch.equal(first["layer_prob"], second["layer_prob"])
    assert torch.equal(first["vessel_strength_map"], second["vessel_strength_map"])


def test_binary_entropy_is_finite_at_probability_extremes() -> None:
    value = binary_entropy(torch.tensor([0.0, 0.5, 1.0]))
    assert torch.isfinite(value).all()
    assert value[1] > value[0]


def test_gate_regularizers_ignore_padding() -> None:
    loss = SABIDSLoss({"zero_source": "final_segmentation", "auxiliary_weight": 0.0,
                       "gate": {"tv_weight": 1.0, "reconstruction_weight": 1.0}})
    shape = (1, 1, 8, 10)
    valid = torch.zeros(shape); valid[..., 2:6, 2:8] = 1
    base_map = torch.rand(shape)
    output = {
        "denoised_raw": torch.zeros(shape), "layer_logits": torch.zeros(shape),
        "vessel_logits": torch.zeros(shape), "boundary_logits": torch.zeros(1, 2, 8, 10),
        "layer_strength_map": base_map.clone(), "vessel_strength_map": base_map.clone(),
        "fine_layer_denoised": torch.rand(shape), "fine_vessel_denoised": torch.rand(shape),
        "auxiliary": [],
    }
    batch = {
        "has_clean": torch.tensor([True]), "clean": torch.rand(shape),
        "has_layer": torch.tensor([False]), "has_vessel": torch.tensor([False]),
        "is_clean": torch.tensor([False]), "image_weak": torch.zeros(shape),
        "layer_mask": torch.zeros(shape), "vessel_mask": torch.zeros(shape),
        "valid_mask": valid, "label_valid_mask": valid, "vessel_valid_mask": valid,
    }
    first = loss(output, batch, "input_segment")
    changed = {key: value.clone() if torch.is_tensor(value) else value for key, value in output.items()}
    invalid = valid == 0
    changed["layer_strength_map"][invalid] = 100
    changed["vessel_strength_map"][invalid] = -100
    changed["fine_layer_denoised"][invalid] = 100
    changed["fine_vessel_denoised"][invalid] = -100
    second = loss(changed, batch, "input_segment")
    assert torch.allclose(first["gate_tv_loss"], second["gate_tv_loss"])
    assert torch.allclose(first["gate_reconstruction_loss"], second["gate_reconstruction_loss"])


def test_zero_gate_weights_do_not_require_clean_or_add_gradients() -> None:
    loss = SABIDSLoss({"zero_source": "final_segmentation", "auxiliary_weight": 0.0})
    shape = (1, 1, 4, 6)
    gate = torch.rand(shape, requires_grad=True)
    logits = torch.zeros(shape, requires_grad=True)
    output = {"denoised_raw": logits, "layer_logits": logits, "vessel_logits": logits,
              "boundary_logits": torch.zeros(1, 2, 4, 6), "layer_strength_map": gate,
              "vessel_strength_map": gate, "fine_layer_denoised": gate,
              "fine_vessel_denoised": gate, "auxiliary": []}
    valid = torch.ones(shape)
    batch = {"has_clean": torch.tensor([False]), "clean": torch.zeros(shape),
             "has_layer": torch.tensor([False]), "has_vessel": torch.tensor([False]),
             "is_clean": torch.tensor([False]), "image_weak": torch.zeros(shape),
             "layer_mask": torch.zeros(shape), "vessel_mask": torch.zeros(shape),
             "valid_mask": valid, "label_valid_mask": valid, "vessel_valid_mask": valid}
    result = loss(output, batch, "input_segment")
    assert result["gate_tv_loss"].item() == 0
    assert result["gate_reconstruction_loss"].item() == 0
    result["total"].backward()
    assert gate.grad is None


def test_nonzero_gate_reconstruction_fails_without_paired_clean() -> None:
    loss = SABIDSLoss({"zero_source": "final_segmentation", "auxiliary_weight": 0.0,
                       "gate": {"reconstruction_weight": 0.1}})
    shape = (1, 1, 4, 6)
    value = torch.zeros(shape)
    output = {"denoised_raw": value, "layer_logits": value, "vessel_logits": value,
              "boundary_logits": torch.zeros(1, 2, 4, 6), "layer_strength_map": value,
              "vessel_strength_map": value, "fine_layer_denoised": value,
              "fine_vessel_denoised": value, "auxiliary": []}
    batch = {"has_clean": torch.tensor([False]), "clean": value,
             "has_layer": torch.tensor([False]), "has_vessel": torch.tensor([False]),
             "is_clean": torch.tensor([False]), "image_weak": value,
             "layer_mask": value, "vessel_mask": value, "valid_mask": torch.ones(shape),
             "label_valid_mask": torch.ones(shape), "vessel_valid_mask": torch.ones(shape)}
    with pytest.raises(ValueError, match="paired clean"):
        loss(output, batch, "input_segment")


def test_legacy_and_adaptive_models_build_unique_optimizer_parameters() -> None:
    base = {"model": {"in_channels": 1, "channels": [4, 8, 12, 16],
                      "encoder_depths": [1, 1, 1, 1], "decoder_depth": 1,
                      "interaction_levels": [3, 2, 1], "d2s_enabled": False,
                      "s2d_enabled": False}}
    single = build_model({**base, "dual_view": {"enabled": False}})
    single.set_train_stage("segment")
    dual = build_model({**base, "dual_view": {"enabled": True, "use_auxiliary": True,
                                               "fusion_levels": [3, 2, 1]}})
    dual.set_train_stage("input_segment")
    adaptive = build_model({**base, "dual_view": {"enabled": False},
                            "dual_task_adaptive": {"enabled": True,
                                                   "load_bound_checkpoints": False,
                                                   "fusion_levels": [3, 2, 1],
                                                   "context_channels": 4}})
    adaptive.set_train_stage("input_segment")
    for model in (single, dual, adaptive):
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        assert parameters and len(ids) == len(set(ids))
    adaptive_trainable = {name for name, value in adaptive.named_parameters() if value.requires_grad}
    assert adaptive_trainable
    assert all(name.startswith(("controller.", "layer_aux_fusions.", "vessel_aux_fusions.",
                                "layer_logit_correction.", "vessel_logit_correction.",
                                "boundary_correction.")) for name in adaptive_trainable)
    assert all(not value.requires_grad for name, value in adaptive.named_parameters()
               if name.startswith(("d2.", "coarse_segmenter.", "fine_backbone.")))
