from pathlib import Path

import numpy as np
import pytest
import torch

from sabids.config import load_config
from sabids.engine.trainer import build_model
from sabids.experiments.dose_response import (
    ALPHAS, alpha_code, augmentation_plan_sha, cache_key, deterministic_flip,
    dose_deterministic_algorithms, dose_input, read_binary_mask, save_cache, write_strict_json,
)
from sabids.losses import SABIDSLoss

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("enabled,warn_only", [(False, False), (True, True), (True, False)])
@pytest.mark.parametrize("failure", [False, True])
def test_deterministic_scope_restores_global_flags(enabled, warn_only, failure):
    original = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        def run():
            with dose_deterministic_algorithms():
                assert torch.are_deterministic_algorithms_enabled()
                assert not torch.is_deterministic_algorithms_warn_only_enabled()
                if failure:
                    raise RuntimeError("fixture failure")
        if failure:
            with pytest.raises(RuntimeError, match="fixture failure"):
                run()
        else:
            run()
        assert (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled()) == (enabled, warn_only)
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])


@pytest.mark.parametrize("dose,failure", [(False, False), (False, True), (True, False), (True, True)])
def test_fit_only_enables_determinism_for_dose_and_restores_on_failure(dose, failure):
    from sabids.engine.trainer import Trainer
    original = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        torch.use_deterministic_algorithms(False)
        trainer = object.__new__(Trainer)
        trainer.config = {"dose_response": {"enabled": dose}}
        def impl():
            assert torch.are_deterministic_algorithms_enabled() is dose
            if failure:
                raise RuntimeError("fixture failure")
        trainer._fit_impl = impl
        if failure:
            with pytest.raises(RuntimeError, match="fixture failure"):
                trainer.fit()
        else:
            trainer.fit()
        assert not torch.are_deterministic_algorithms_enabled()
    finally:
        torch.use_deterministic_algorithms(original[0], warn_only=original[1])


def metadata():
    return {"protocol_id": "unit", "sample_id": "frame1", "group_id": "position1",
            "curve_type": "oracle", "alpha": 0., "source_noisy_path": "noisy.npy",
            "source_noisy_sha256": "n", "source_clean_path": "clean.npy", "source_clean_sha256": "c",
            "checkpoint_path": "explicit.pth", "checkpoint_sha256": "d",
            "resolved_config_sha256": "r", "restoration_mode": "structure_d1",
            "geometry_sha256": "g", "code_version": "v", "split": "train"}


@pytest.mark.parametrize("curve", ["oracle", "d1"])
@pytest.mark.parametrize("alpha", ALPHAS)
def test_formula_dtype_range_and_repeatability(curve, alpha):
    x = np.array([[.05, .9], [.7, .1]], np.float32)
    ref = np.array([[.95, .1], [.2, .8]], np.float32)
    result, stats = dose_input(x, ref, alpha, curve)
    if alpha == 0:
        assert np.array_equal(result, x)
    elif alpha == 1:
        assert np.array_equal(result, ref)
        assert stats["preclip_min"] == float(ref.min())
        assert stats["preclip_max"] == float(ref.max())
    else:
        assert np.allclose(result, np.clip(x + alpha * (ref - x), 0, 1), atol=1e-7)
    assert result.dtype == np.float32 and np.isfinite(result).all()
    assert 0 <= result.min() <= result.max() <= 1
    again, again_stats = dose_input(x, ref, alpha, curve)
    assert np.array_equal(result, again) and stats == again_stats


def test_clip_fractions_exclude_padding():
    x = np.array([[1, 0, .5, .5, 0]], np.float32)
    ref = np.array([[0, 1, .5, .5, 1]], np.float32)
    value, stats = dose_input(x, ref, 1.25, "d1", np.array([[1, 1, 1, 1, 0]]))
    assert stats["below_zero_fraction"] == .25
    assert stats["above_one_fraction"] == .25
    assert stats["total_clip_fraction"] == .5
    assert stats["alpha_description"] == "extrapolation/residual amplification"
    assert value.min() == 0 and value.max() == 1


def test_codes_and_cache_key_identity():
    assert [alpha_code(a) for a in ALPHAS] == ["a000", "a025", "a050", "a075", "a100", "a125"]
    m = metadata()
    keys = {cache_key({**m, "curve_type": c, "alpha": a}) for c in ("oracle", "d1") for a in ALPHAS}
    assert len(keys) == 12
    for field in ("protocol_id", "sample_id", "checkpoint_sha256", "source_noisy_sha256",
                  "source_clean_sha256", "resolved_config_sha256", "geometry_sha256", "code_version", "split"):
        assert cache_key({**m, field: "different"}) != cache_key(m)
    with pytest.raises(ValueError, match="Missing"):
        cache_key({k: v for k, v in m.items() if k != "checkpoint_sha256"})


@pytest.mark.parametrize("bad", [-.25, .3, 2, float("nan"), float("inf"), True])
def test_illegal_alpha(bad):
    with pytest.raises(ValueError):
        alpha_code(bad)


@pytest.mark.parametrize("bad", [np.array([[np.nan]]), np.array([[1.1]]), np.zeros((0, 1)), np.zeros(4)])
def test_invalid_image(bad):
    with pytest.raises(ValueError):
        dose_input(bad, bad, .5, "oracle")


def test_cache_never_overwrites_mismatch(tmp_path):
    p = tmp_path / "input.npy"
    value = np.ones((4, 4), np.float32) * .5
    m = metadata()
    save_cache(p, value, m)
    before = p.read_bytes()
    save_cache(p, value, {**m, "generation_time_utc": "later"})
    assert before == p.read_bytes()
    with pytest.raises(FileExistsError):
        save_cache(p, value, {**m, "alpha": .5})
    with pytest.raises(FileExistsError):
        save_cache(p, value * .5, m)
    assert before == p.read_bytes()


def test_augmentation_is_independent_of_global_rng():
    ids = ["one", "two", "three"]
    first = augmentation_plan_sha(42, 20, ids, .5)
    np.random.seed(1)
    np.random.rand(1000)
    torch.manual_seed(98)
    torch.rand(1000)
    assert augmentation_plan_sha(42, 20, ids, .5) == first
    assert augmentation_plan_sha(43, 20, ids, .5) != first
    assert deterministic_flip(42, 0, "one", 0) is False
    assert deterministic_flip(42, 0, "one", 1) is True


def test_strict_json(tmp_path):
    p = tmp_path / "report.json"
    write_strict_json(p, {"bad": float("nan"), "infinite": float("inf"), "scalar": np.float32(.5)})
    text = p.read_text()
    assert "NaN" not in text and "Infinity" not in text and "null" in text


@pytest.mark.parametrize("foreground", [1, 255])
def test_binary_png_encoding_preserved(tmp_path, foreground):
    import cv2
    value = np.array([[0, foreground], [foreground, 0]], np.uint8)
    ok, encoded = cv2.imencode(".png", value)
    assert ok
    path = tmp_path / "mask.png"
    encoded.tofile(str(path))
    assert np.array_equal(read_binary_mask(path), (value > 0).astype(np.float32))


def test_multiclass_is_not_silently_read_as_binary(tmp_path):
    p = tmp_path / "invalid.npy"
    np.save(p, np.array([[0, 1, 2, 255]], np.float32))
    with pytest.raises(ValueError, match="not multiclass"):
        read_binary_mask(p)


def small_config(curve="oracle"):
    cfg = load_config(ROOT / f"configs/adaptive_denoising/dose_{curve}.yaml")
    cfg["model"].update(channels=[4, 8], encoder_depths=[1, 1], decoder_depth=1, interaction_levels=[1])
    return cfg


def test_loss_paths_and_independent_initialization(monkeypatch):
    torch.set_num_threads(1)
    trainable_sets = []
    states = []
    for curve in ("oracle", "d1"):
        for alpha in ALPHAS:
            cfg = small_config(curve)
            torch.manual_seed(42)
            model = build_model(cfg)
            model.set_train_stage("input_segment")
            states.append({n: p.clone() for n, p in model.state_dict().items()})
            trainable_sets.append({n for n, p in model.named_parameters() if p.requires_grad})
    assert all(s == trainable_sets[0] for s in trainable_sets)
    assert all(all(torch.equal(s[n], states[0][n]) for n in s) for s in states)
    assert not any(n.startswith("interactions.") or n.startswith("residual_head") for n in trainable_sets[0])
    assert cfg["loss"]["auxiliary_weight"] == 0
    assert all(not i.enable_denoise_to_seg and not i.enable_seg_to_denoise for i in model.interactions.values())
    image = torch.rand(1, 1, 24, 32)
    batch = {"image": image, "clean": torch.rand_like(image), "layer_mask": torch.ones_like(image),
             "vessel_mask": (image > .7).float(), "valid_mask": torch.ones_like(image),
             "label_valid_mask": torch.ones_like(image), "vessel_valid_mask": torch.ones_like(image),
             "has_layer": torch.tensor([True]), "has_vessel": torch.tensor([True]),
             "has_clean": torch.tensor([True]), "has_repeat": torch.tensor([False]), "is_clean": torch.tensor([False])}
    batch["layer_mask"][:, :, :3] = 0
    loss_fn = SABIDSLoss(cfg["loss"])
    def forbidden(*args, **kwargs):
        raise AssertionError("Inactive restoration path called")
    monkeypatch.setattr(loss_fn, "_restoration", forbidden)
    output = model(image, return_auxiliary=False, return_features=False)
    assert output["layer_logits"].shape == image.shape
    assert output["vessel_logits"].shape == image.shape
    assert all(torch.isfinite(v).all() for v in output.values() if torch.is_tensor(v))
    baseline = loss_fn(output, batch, "input_segment")
    # Poison auxiliary heads; weight zero means they cannot enter the objective.
    output["auxiliary"] = [{"layer_logit": torch.full_like(image, float("nan")),
                            "vessel_logit": torch.full_like(image, float("nan"))}]
    altered = loss_fn(output, batch, "input_segment")
    assert torch.equal(baseline["total"], altered["total"])
    inactive_poisoned = {**output, "denoised_raw": torch.full_like(image, float("nan"))}
    assert torch.equal(loss_fn(inactive_poisoned, batch, "input_segment")["total"], baseline["total"])
    for k in ("reconstruction", "residual", "identity", "rmac", "pseudo"):
        assert float(altered[k].detach()) == 0
    altered["total"].backward()
    assert all(p.grad is None for n, p in model.named_parameters() if n.startswith("interactions."))
    assert all(p.grad is None for n, p in model.named_parameters()
               if n.startswith(("residual_head", "adapters.denoise", "decoders.denoise")))
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.startswith("vessel_head"))


def test_unknown_and_padding_have_zero_supervision_gradient():
    cfg = small_config()
    image = torch.rand(1, 1, 16, 24)
    valid = torch.ones_like(image)
    valid[:, :, :2] = 0
    annotation = torch.ones_like(image)
    annotation[:, :, 6:9, 8:12] = 0
    layer = torch.zeros_like(image)
    layer[:, :, 4:13] = 1
    output = {"denoised_raw": image, "layer_logits": torch.zeros_like(image, requires_grad=True),
              "vessel_logits": torch.zeros_like(image, requires_grad=True),
              "boundary_logits": torch.zeros(1, 2, 16, 24, requires_grad=True), "auxiliary": []}
    batch = {"has_layer": torch.tensor([True]), "has_vessel": torch.tensor([True]),
             "has_clean": torch.tensor([False]), "is_clean": torch.tensor([False]),
             "layer_mask": layer, "vessel_mask": layer * (image > .5), "valid_mask": valid,
             "label_valid_mask": annotation, "vessel_valid_mask": annotation}
    loss = SABIDSLoss(cfg["loss"])(output, batch, "input_segment")["total"]
    loss.backward()
    excluded = (valid * annotation) == 0
    assert torch.count_nonzero(output["layer_logits"].grad[excluded]) == 0
    assert torch.count_nonzero(output["vessel_logits"].grad[excluded]) == 0
    assert torch.count_nonzero(output["boundary_logits"].grad[:, :, :, 8:12]) == 0
    altered_batch = {**batch, "layer_mask": batch["layer_mask"].clone(), "vessel_mask": batch["vessel_mask"].clone()}
    altered_batch["layer_mask"][excluded] = 1 - altered_batch["layer_mask"][excluded]
    altered_batch["vessel_mask"][excluded] = 1 - altered_batch["vessel_mask"][excluded]
    altered_loss = SABIDSLoss(cfg["loss"])(output, altered_batch, "input_segment")["total"]
    assert torch.equal(loss, altered_loss)
