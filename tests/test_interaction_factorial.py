from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sabids.config import load_config
from sabids.models import SABIDSNet


def make_model(d2s: bool, s2d: bool, detach_d2s: bool = True, detach_s2d: bool = True, scale: float = 0.0):
    return SABIDSNet(
        channels=(4, 8), encoder_depths=(1, 1), decoder_depth=1,
        interaction_levels=(1, 0), enable_denoise_to_seg=d2s,
        enable_seg_to_denoise=s2d, causal_interaction_experiment=True,
        detach_denoise_to_seg_source=detach_d2s,
        detach_seg_to_denoise_source=detach_s2d, interaction_scale_init=scale,
    )


def nonzero_gradient(model: SABIDSNet, prefix: str) -> bool:
    return any(
        parameter.grad is not None and bool((parameter.grad.detach().abs() > 0).any())
        for name, parameter in model.named_parameters() if name.startswith(prefix)
    )


def test_causal_factorial_shapes_finite_and_no_target_inputs():
    image = torch.rand(2, 1, 16, 24)
    for d2s, s2d in ((False, False), (True, False), (False, True), (True, True)):
        model = make_model(d2s, s2d)
        output = model(image)
        for key in ("denoised", "layer_prob", "vessel_prob", "base_layer_prob", "base_vessel_prob"):
            assert output[key].shape == image.shape
            assert torch.isfinite(output[key]).all()
    with pytest.raises(TypeError):
        make_model(False, False)(image, clean=image)  # type: ignore[call-arg]


def test_j00_and_zero_gates_are_exactly_baseline_equivalent():
    torch.manual_seed(7)
    baseline = SABIDSNet(
        channels=(4, 8), encoder_depths=(1, 1), decoder_depth=1,
        interaction_levels=(1, 0), enable_denoise_to_seg=False,
        enable_seg_to_denoise=False, causal_interaction_experiment=False,
        interaction_scale_init=0.0,
    ).eval()
    causal = make_model(False, False).eval()
    causal.load_state_dict(baseline.state_dict())
    image = torch.rand(1, 1, 16, 24)
    with torch.no_grad():
        expected, actual = baseline(image), causal(image)
    for key in ("denoised_raw", "layer_logits", "vessel_logits", "boundary_logits"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)

    both = make_model(True, True, scale=0.0).eval()
    both.load_state_dict(baseline.state_dict())
    with torch.no_grad():
        zero = both(image)
    for key in ("denoised_raw", "layer_logits", "vessel_logits", "boundary_logits"):
        torch.testing.assert_close(zero[key], expected[key], rtol=0.0, atol=0.0)


def test_detached_sources_block_cross_task_gradients_but_mapping_trains():
    image = torch.rand(1, 1, 16, 24)
    s2d = make_model(False, True, detach_s2d=True, scale=0.1)
    s2d(image)["denoised_raw"].mean().backward()
    assert not nonzero_gradient(s2d, "decoders.layer")
    assert not nonzero_gradient(s2d, "layer_head")
    assert nonzero_gradient(s2d, "interactions.1.layer_anatomy")

    d2s = make_model(True, False, detach_d2s=True, scale=0.1)
    d2s(image)["vessel_logits"].mean().backward()
    assert not nonzero_gradient(d2s, "decoders.denoise")
    assert not nonzero_gradient(d2s, "residual_head")
    assert nonzero_gradient(d2s, "interactions.1.denoise_to_vessel")


def test_open_sources_receive_corresponding_task_gradient():
    image = torch.rand(1, 1, 16, 24)
    s2d = make_model(False, True, detach_s2d=False, scale=0.1)
    s2d(image)["denoised_raw"].mean().backward()
    assert nonzero_gradient(s2d, "decoders.layer")
    assert nonzero_gradient(s2d, "layer_head")

    d2s = make_model(True, False, detach_d2s=False, scale=0.1)
    d2s(image)["vessel_logits"].mean().backward()
    assert nonzero_gradient(d2s, "decoders.denoise")


def test_zero_scale_starts_then_mapping_receives_gradient_after_one_step():
    model = make_model(False, True, scale=0.0)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1)
    image = torch.rand(1, 1, 16, 24)
    model(image)["denoised_raw"].mean().backward()
    assert nonzero_gradient(model, "interactions.1.seg_scale")
    assert not nonzero_gradient(model, "interactions.1.layer_anatomy")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model(image)["denoised_raw"].mean().backward()
    assert nonzero_gradient(model, "interactions.1.layer_anatomy")


def test_frozen_encoder_and_identical_computation_counts():
    image = torch.rand(1, 1, 16, 24)
    counts = []
    for d2s, s2d in ((False, False), (True, False), (False, True), (True, True)):
        torch.manual_seed(123)
        model = make_model(d2s, s2d)
        model.set_train_stage("interaction", freeze_shared_encoder=True)
        model.train()
        model.enforce_frozen_eval()
        assert model.stem.training is False
        assert model.encoder_blocks.training is False
        assert model.downsamples.training is False
        frozen_before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters()
            if name.startswith(("stem", "encoder_blocks", "downsamples"))
        }
        call_count = {"layer": 0, "vessel": 0, "denoise": 0}
        hooks = []
        for task in call_count:
            for module in model.decoders[task]:
                hooks.append(module.register_forward_hook(lambda _m, _i, _o, task=task: call_count.__setitem__(task, call_count[task] + 1)))
        output = model(image)
        (output["denoised_raw"].mean() + output["layer_logits"].mean() + output["vessel_logits"].mean()).backward()
        optimizer_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        assert len({id(parameter) for parameter in optimizer_parameters}) == len(optimizer_parameters)
        torch.optim.SGD(optimizer_parameters, lr=0.01).step()
        for name, before in frozen_before.items():
            torch.testing.assert_close(dict(model.named_parameters())[name], before, rtol=0.0, atol=0.0)
        counts.append(call_count)
        for hook in hooks:
            hook.remove()
    assert counts == [{"layer": 2, "vessel": 2, "denoise": 1}] * 4


def test_same_seed_common_and_interaction_initialization_match_all_variants():
    states = []
    for d2s, s2d in ((False, False), (True, False), (False, True), (True, True)):
        torch.manual_seed(99)
        states.append(make_model(d2s, s2d).state_dict())
    for name in states[0]:
        for state in states[1:]:
            torch.testing.assert_close(states[0][name], state[name], rtol=0.0, atol=0.0)


def test_factorial_configs_fix_protocol_and_do_not_enable_auxiliary_losses():
    root = Path(__file__).resolve().parents[1]
    expected = {"j00": (False, False), "j10": (True, False), "j01": (False, True), "j11": (True, True)}
    for variant, (d2s, s2d) in expected.items():
        config = load_config(root / "configs" / "current" / f"interaction_{variant}_fold0.yaml")
        assert config["train"]["stage"] == "interaction"
        assert config["train"]["epochs"] == 20
        assert config["train"]["monitor_denoise_drift"] is False
        assert config["model"]["d2s_enabled"] is d2s
        assert config["model"]["s2d_enabled"] is s2d
        assert config["model"]["freeze_shared_encoder"] is True
        assert config["model"]["detach_d2s_source"] is True
        assert config["model"]["detach_s2d_source"] is True
        assert config["loss"]["auxiliary_weight"] == 0.0
        for key in ("rmac", "pseudo", "identity"):
            assert config["loss"]["weights"][key] == 0.0
