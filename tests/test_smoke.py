import torch

from sabids.losses import SABIDSLoss
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
