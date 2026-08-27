import torch

from sabids.losses import SABIDSLoss
from sabids.metrics import vessel_diagnostic_metrics
from sabids.models import SABIDSNet


def test_forward_and_backward():
    model = SABIDSNet(
        channels=(8, 16, 32, 64),
        encoder_depths=(1, 1, 1, 1),
        decoder_depth=1,
        interaction_levels=(3, 2, 1),
    )
    image = torch.rand(2, 1, 64, 128)
    output = model(image)
    assert output["denoised"].shape == image.shape
    assert output["layer_prob"].shape == image.shape
    assert output["vessel_prob"].shape == image.shape
    compact = model(image, return_features=False, return_auxiliary=False)
    assert compact["auxiliary"] == []
    assert "anatomy_embedding" not in compact

    batch = {
        "image": image,
        "image_weak": image,
        "repeat": torch.rand_like(image),
        "clean": torch.rand_like(image),
        "layer_mask": (torch.rand_like(image) > 0.5).float(),
        "vessel_mask": (torch.rand_like(image) > 0.8).float(),
        "valid_mask": torch.ones_like(image),
        "has_clean": torch.tensor([True, True]),
        "has_layer": torch.tensor([True, True]),
        "has_vessel": torch.tensor([True, True]),
        "has_repeat": torch.tensor([True, True]),
        "is_clean": torch.tensor([False, False]),
    }
    repeat_output = model(batch["repeat"])
    clean_output = model(batch["clean"])
    criterion = SABIDSLoss(
        {
            "boundary_weight": 0.2,
            "auxiliary_weight": 0.1,
            "weights": {
                "reconstruction": 1.0,
                "residual": 0.5,
                "layer": 1.0,
                "vessel": 1.0,
                "vessel_stroma": 0.25,
                "vessel_area": 0.2,
                "containment": 0.1,
                "rmac": 0.3,
                "identity": 0.05,
                "pseudo": 0.5,
            },
        }
    )
    losses = criterion(
        output,
        batch,
        stage="joint",
        repeat_output=repeat_output,
        clean_output=clean_output,
    )
    losses["total"].backward()
    assert torch.isfinite(losses["total"])


def test_full_layer_vessel_prediction_is_penalized():
    height, width = 32, 32
    layer = torch.zeros(1, 1, height, width)
    layer[:, :, 8:24] = 1.0
    vessel = torch.zeros_like(layer)
    vessel[:, :, 11:15, 8:16] = 1.0
    good_vessel_logits = torch.where(
        vessel > 0.5, torch.full_like(vessel, 4.0), torch.full_like(vessel, -4.0)
    )
    full_layer_logits = torch.where(
        layer > 0.5, torch.full_like(layer, 4.0), torch.full_like(layer, -4.0)
    )

    criterion = SABIDSLoss(
        {
            "vessel_bce_weight": 0.5,
            "vessel_fp_weight": 0.6,
            "vessel_fn_weight": 0.4,
            "weights": {
                "layer": 1.0,
                "vessel": 1.0,
                "vessel_stroma": 0.25,
                "vessel_area": 0.2,
                "containment": 0.1,
            },
        }
    )
    batch = {
        "layer_mask": layer,
        "vessel_mask": vessel,
        "valid_mask": torch.ones_like(layer),
        "has_layer": torch.tensor([True]),
        "has_vessel": torch.tensor([True]),
        "has_clean": torch.tensor([False]),
        "has_repeat": torch.tensor([False]),
        "is_clean": torch.tensor([False]),
        "image": torch.zeros_like(layer),
        "image_weak": torch.zeros_like(layer),
        "clean": torch.zeros_like(layer),
    }

    def make_output(vessel_logits):
        layer_logits = torch.where(
            layer > 0.5,
            torch.full_like(layer, 4.0),
            torch.full_like(layer, -4.0),
        )
        return {
            "denoised_raw": torch.zeros_like(layer, requires_grad=True),
            "residual": torch.zeros_like(layer),
            "layer_logits": layer_logits,
            "vessel_logits": vessel_logits,
            "layer_prob": torch.sigmoid(layer_logits),
            "vessel_prob": torch.sigmoid(vessel_logits),
            "boundary_logits": torch.zeros(1, 2, height, width),
            "auxiliary": [],
        }

    good = criterion(make_output(good_vessel_logits), batch, stage="segment")
    collapsed = criterion(make_output(full_layer_logits), batch, stage="segment")
    assert collapsed["vessel_stroma"] > good["vessel_stroma"]
    assert collapsed["vessel_area"] > good["vessel_area"]
    assert collapsed["total"] > good["total"]


def test_saturated_stroma_logits_keep_a_finite_corrective_gradient():
    height, width = 16, 16
    layer = torch.zeros(1, 1, height, width)
    layer[:, :, 4:12] = 1.0
    vessel = torch.zeros_like(layer)
    vessel[:, :, 6:8, 5:11] = 1.0
    # Half precision reproduces the saturation regime seen under CUDA AMP.
    vessel_logits = torch.full(
        layer.shape, -20.0, dtype=torch.float16, requires_grad=True
    )
    with torch.no_grad():
        vessel_logits[layer.bool()] = 20.0
    layer_logits = torch.where(layer > 0.5, 20.0, -20.0)
    output = {
        "denoised_raw": torch.zeros_like(layer, requires_grad=True),
        "residual": torch.zeros_like(layer),
        "layer_logits": layer_logits,
        "vessel_logits": vessel_logits,
        "layer_prob": torch.sigmoid(layer_logits),
        "vessel_prob": torch.sigmoid(vessel_logits),
        "boundary_logits": torch.zeros(1, 2, height, width),
        "auxiliary": [],
    }
    batch = {
        "layer_mask": layer,
        "vessel_mask": vessel,
        "valid_mask": torch.ones_like(layer),
        "has_layer": torch.tensor([True]),
        "has_vessel": torch.tensor([True]),
        "has_clean": torch.tensor([False]),
        "has_repeat": torch.tensor([False]),
        "is_clean": torch.tensor([False]),
        "image": torch.zeros_like(layer),
        "image_weak": torch.zeros_like(layer),
        "clean": torch.zeros_like(layer),
    }
    criterion = SABIDSLoss(
        {
            "weights": {
                "layer": 0.0,
                "vessel": 0.0,
                "vessel_stroma": 1.0,
                "vessel_area": 0.0,
                "containment": 0.0,
            }
        }
    )
    losses = criterion(output, batch, stage="segment")
    losses["total"].backward()
    stroma = (layer > 0.5) & (vessel < 0.5)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(vessel_logits.grad).all()
    assert vessel_logits.grad[stroma].mean() > 0


def test_private_seg_freezes_public_denoising_path():
    model = SABIDSNet(
        channels=(8, 16, 32, 64),
        encoder_depths=(1, 1, 1, 1),
        decoder_depth=1,
        interaction_levels=(3, 2, 1),
        enable_seg_to_denoise=False,
        enable_denoise_to_seg=True,
    )
    model.set_train_stage("private_seg")

    trainable = {name for name, value in model.named_parameters() if value.requires_grad}
    assert any(name.startswith("decoders.layer") for name in trainable)
    assert any(name.startswith("decoders.vessel") for name in trainable)
    assert any("denoise_to_layer" in name for name in trainable)
    assert not any(name.startswith("stem") for name in trainable)
    assert not any(name.startswith("encoder_blocks") for name in trainable)
    assert not any(name.startswith("decoders.denoise") for name in trainable)
    assert not any(name.startswith("residual_head") for name in trainable)

    image = torch.rand(1, 1, 64, 128)
    before = model(image)["denoised"].detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    output = model(image)
    loss = output["layer_prob"].mean() + output["vessel_prob"].mean()
    loss.backward()
    optimizer.step()
    after = model(image)["denoised"].detach()
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_optional_private_deep_encoder_unfreezing():
    model = SABIDSNet(
        channels=(8, 16, 32, 64),
        encoder_depths=(1, 1, 1, 1),
        decoder_depth=1,
    )
    model.set_train_stage("private_seg", private_train_encoder_levels=(3,))
    assert any(parameter.requires_grad for parameter in model.encoder_blocks[3].parameters())
    assert any(parameter.requires_grad for parameter in model.downsamples[2].parameters())
    assert not any(parameter.requires_grad for parameter in model.encoder_blocks[2].parameters())


def test_safe_stage2_preserves_complete_denoising_function():
    model = SABIDSNet(
        channels=(8, 16, 32, 64),
        encoder_depths=(1, 1, 1, 1),
        decoder_depth=1,
        enable_seg_to_denoise=False,
        enable_denoise_to_seg=False,
        dropout=0.2,
    )
    model.set_train_stage("segment", freeze_shared_encoder=True)
    assert not any(parameter.requires_grad for parameter in model.stem.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.encoder_blocks.parameters()
    )
    assert any(
        parameter.requires_grad for parameter in model.decoders["vessel"].parameters()
    )

    image = torch.rand(1, 1, 64, 128)
    model.eval()
    before = model(image)["denoised"].detach().clone()
    model.train()
    model.enforce_frozen_eval()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    output = model(image)
    (output["layer_prob"].mean() + output["vessel_prob"].mean()).backward()
    optimizer.step()
    model.eval()
    after = model(image)["denoised"].detach()
    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)


def test_safe_stage2_denoise_to_seg_has_gradients_without_denoise_drift():
    model = SABIDSNet(
        channels=(8, 16, 32, 64),
        encoder_depths=(1, 1, 1, 1),
        decoder_depth=1,
        enable_seg_to_denoise=False,
        enable_denoise_to_seg=True,
        detach_denoise_to_seg_source=True,
    )
    model.set_train_stage(
        "segment", freeze_shared_encoder=True, train_denoise_to_seg=True
    )
    image = torch.rand(1, 1, 64, 128)
    model.eval()
    denoised_before = model(image)["denoised"].detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    for _ in range(2):
        optimizer.zero_grad()
        model.train()
        model.enforce_frozen_eval()
        output = model(image)
        (output["layer_prob"].mean() + output["vessel_prob"].mean()).backward()
        optimizer.step()
    assert any(
        interaction.layer_scale.grad is not None
        and torch.isfinite(interaction.layer_scale.grad).all()
        for interaction in model.interactions.values()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for interaction in model.interactions.values()
        for parameter in interaction.denoise_to_vessel.parameters()
    )
    model.eval()
    final = model(image)
    torch.testing.assert_close(
        denoised_before, final["denoised"].detach(), rtol=0.0, atol=0.0
    )
    assert any(
        float(item["denoise_to_vessel_injection_abs_mean"]) > 0.0
        for item in final["auxiliary"]
    )


def test_roi_vessel_loss_ignores_unknown_pixels_and_penalizes_mislocalization():
    height, width = 16, 16
    layer = torch.ones(1, 1, height, width)
    target = torch.zeros_like(layer)
    target[:, :, 4:8, 3:7] = 1.0
    valid = torch.ones_like(layer)
    valid[:, :, :, 12:] = 0.0
    good_logits = torch.where(target > 0.5, 6.0, -6.0)
    shifted = torch.zeros_like(target)
    shifted[:, :, 8:12, 3:7] = 1.0
    shifted_logits = torch.where(shifted > 0.5, 6.0, -6.0)
    changed_only_in_unknown = good_logits.clone()
    changed_only_in_unknown[:, :, :, 12:] = 20.0

    criterion = SABIDSLoss(
        {
            "vessel_supervision_mode": "roi_bce_dice",
            "weights": {
                "layer": 0.0,
                "vessel": 1.0,
                "vessel_stroma": 0.0,
                "vessel_area": 0.0,
                "containment": 0.0,
            },
        }
    )

    def compute(logits):
        output = {
            "denoised_raw": torch.zeros_like(layer, requires_grad=True),
            "layer_logits": torch.full_like(layer, 6.0),
            "vessel_logits": logits,
            "layer_prob": torch.ones_like(layer),
            "vessel_prob": torch.sigmoid(logits),
            "boundary_logits": torch.zeros(1, 2, height, width),
            "residual": torch.zeros_like(layer),
            "auxiliary": [],
        }
        batch = {
            "layer_mask": layer,
            "vessel_mask": target,
            "vessel_valid_mask": valid,
            "valid_mask": torch.ones_like(layer),
            "has_layer": torch.tensor([True]),
            "has_vessel": torch.tensor([True]),
            "has_clean": torch.tensor([False]),
            "has_repeat": torch.tensor([False]),
            "is_clean": torch.tensor([False]),
            "image": torch.zeros_like(layer),
            "image_weak": torch.zeros_like(layer),
            "clean": torch.zeros_like(layer),
        }
        return criterion(output, batch, stage="segment")["vessel"]

    good = compute(good_logits)
    unknown_changed = compute(changed_only_in_unknown)
    wrong_location = compute(shifted_logits)
    torch.testing.assert_close(good, unknown_changed)
    assert wrong_location > good


def test_vessel_diagnostics_separate_full_roi_and_outside_errors():
    layer = torch.zeros(8, 8, dtype=torch.bool).numpy()
    layer[2:6, 1:7] = True
    target = torch.zeros(8, 8, dtype=torch.bool).numpy()
    target[3:5, 2:4] = True
    vessel_probability = target.astype("float32")
    vessel_probability[0, 0] = 1.0
    metrics = vessel_diagnostic_metrics(
        vessel_probability,
        layer.astype("float32"),
        target,
        layer,
        valid=torch.ones(8, 8, dtype=torch.bool).numpy(),
    )
    assert metrics["vessel_roi_dice"] > metrics["vessel_dice"]
    assert metrics["vessel_outside_gt_layer_fraction"] > 0.0
    assert metrics["whole_layer_baseline_vessel_dice"] < 1.0
    assert metrics["vessel_error_outside_gt_and_pred_layer_pixels"] == 1.0
    assert metrics["vessel_oracle_gt_layer_dice"] == 1.0


def test_roi_outside_bce_penalizes_saturated_false_positive_and_ignores_unknown():
    shape = (2, 1, 12, 12)
    layer = torch.zeros(shape)
    layer[:, :, 3:9, 2:10] = 1.0
    vessel = torch.zeros_like(layer)
    vessel[:, :, 5:7, 4:8] = 1.0
    valid = torch.ones_like(layer)
    valid[0, :, :3] = 0.0
    # Image 1 has no outside region and must be skipped safely.
    layer[1] = 1.0
    logits = torch.full(shape, -20.0, requires_grad=True)
    with torch.no_grad():
        logits[0, :, 10:, :] = 20.0  # valid, high-confidence outside FP
        logits[0, :, :3, :] = 20.0   # unknown, must have no effect

    output = {
        "denoised_raw": torch.zeros_like(layer, requires_grad=True),
        "residual": torch.zeros_like(layer),
        "layer_logits": torch.where(layer > 0.5, 20.0, -20.0),
        "vessel_logits": logits,
        "layer_prob": layer,
        "vessel_prob": torch.sigmoid(logits),
        "boundary_logits": torch.zeros(2, 2, 12, 12),
        "auxiliary": [],
    }
    batch = {
        "layer_mask": layer,
        "vessel_mask": vessel,
        "vessel_valid_mask": valid,
        "valid_mask": torch.ones_like(layer),
        "has_layer": torch.tensor([True, True]),
        "has_vessel": torch.tensor([True, True]),
        "has_clean": torch.tensor([False, False]),
        "has_repeat": torch.tensor([False, False]),
        "is_clean": torch.tensor([False, False]),
        "image": torch.zeros_like(layer),
        "image_weak": torch.zeros_like(layer),
        "clean": torch.zeros_like(layer),
    }
    criterion = SABIDSLoss(
        {
            "vessel_supervision_mode": "roi_bce_dice_outside",
            "weights": {
                "layer": 0.0,
                "vessel": 0.0,
                "vessel_stroma": 0.0,
                "vessel_area": 0.0,
                "vessel_outside": 0.5,
                "containment": 0.0,
            },
        }
    )
    losses = criterion(output, batch, stage="segment")
    losses["total"].backward()
    assert torch.isfinite(losses["vessel_outside"])
    assert losses["vessel_outside_valid_images"] == 1.0
    assert losses["vessel_outside_weight"] == 0.5
    torch.testing.assert_close(
        losses["vessel_outside_weighted"], losses["vessel_outside"] * 0.5
    )
    assert logits.grad[0, :, 10:, :].mean() > 0
    assert logits.grad[0, :, :3, :].abs().sum() == 0
