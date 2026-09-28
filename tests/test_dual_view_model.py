from __future__ import annotations

import torch

from sabids.models import NoisyMildDualViewSegmenter


def _model() -> NoisyMildDualViewSegmenter:
    torch.manual_seed(7)
    model = NoisyMildDualViewSegmenter(
        channels=(4, 8, 16, 32), encoder_depths=(1, 1, 1, 1),
        decoder_depth=1, interaction_levels=(), enable_seg_to_denoise=False,
        enable_denoise_to_seg=False, fusion_levels=(3, 2, 1), dual_scale_init=0.0,
    )
    model.set_train_stage("input_segment")
    return model


def test_zero_init_and_disabled_auxiliary_equal_noisy_path_non_square() -> None:
    model = _model().eval()
    noisy = torch.rand(2, 1, 32, 48)
    mild = torch.rand_like(noisy)
    with torch.no_grad():
        dual = model(noisy, auxiliary_image=mild)
        disabled = model(noisy, auxiliary_image=mild, disable_auxiliary=True)
        different_aux = model(noisy, auxiliary_image=torch.zeros_like(mild))
    for key in ("layer_logits", "vessel_logits", "boundary_logits"):
        assert torch.equal(dual[key], disabled[key])
        assert torch.equal(dual[key], different_aux[key])
        assert torch.isfinite(dual[key]).all()
    assert dual["layer_logits"].shape == noisy.shape
    assert dual["boundary_logits"].shape == (2, 2, 32, 48)


def test_siamese_encoder_is_registered_once_and_segmentation_gradients_start_fusion() -> None:
    model = _model().train()
    encoder_names = [
        name for name, _ in model.named_parameters()
        if name.startswith(("stem.", "encoder_blocks.", "downsamples."))
    ]
    assert len(encoder_names) == len(set(encoder_names))
    assert not any("auxiliary_encoder" in name for name, _ in model.named_parameters())
    noisy = torch.rand(1, 1, 32, 48)
    mild = torch.rand_like(noisy)
    output = model(noisy, auxiliary_image=mild)
    (output["layer_logits"].mean() + output["vessel_logits"].mean()).backward()
    assert all(fusion.gamma.grad is not None for fusion in model.dual_fusions.values())
    assert all(torch.isfinite(fusion.gamma.grad).all() for fusion in model.dual_fusions.values())
    # Zero gamma intentionally blocks adapter weights on the first step.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    output = model(noisy, auxiliary_image=mild)
    output["vessel_logits"].square().mean().backward()
    assert any(
        fusion.adapter.weight.grad is not None
        and float(fusion.adapter.weight.grad.abs().sum()) > 0
        for fusion in model.dual_fusions.values()
    )
    # The inherited denoising branch is excluded from this experiment.
    assert all(not parameter.requires_grad for parameter in model.adapters["denoise"].parameters())
    assert all(not parameter.requires_grad for parameter in model.decoders["denoise"].parameters())


def test_checkpoint_round_trip_preserves_dual_output(tmp_path) -> None:
    model = _model().eval()
    noisy = torch.rand(1, 1, 32, 48)
    mild = torch.rand_like(noisy)
    with torch.no_grad():
        expected = model(noisy, auxiliary_image=mild)["vessel_logits"]
    path = tmp_path / "checkpoint.pth"
    torch.save({"model": model.state_dict()}, path)
    restored = _model().eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=False)["model"], strict=True)
    with torch.no_grad():
        actual = restored(noisy, auxiliary_image=mild)["vessel_logits"]
    assert torch.equal(expected, actual)
