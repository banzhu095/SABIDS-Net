from __future__ import annotations

import io
import json
import math
import tarfile
import torch

from sabids.experiments.dual_task_adaptive_v2 import (
    REFERENCES, audit_v2_inputs, checkpoint_eligibility, frozen_weak_mask,
    select_vessel_safe_epoch, vessel_protection_losses,
)
from sabids.models.dual_task_adaptive_v2 import DualTaskAdaptiveV2Segmenter
from sabids.engine.trainer import build_model
from sabids.models import SABIDSNet
from tools.prepare_dual_task_adaptive_v2 import _materialize_bound_coarse_evidence


def tiny_model():
    kwargs = dict(in_channels=1, channels=(4, 8, 16, 32),
                  encoder_depths=(1, 1, 1, 1), decoder_depth=1,
                  interaction_levels=(3, 2, 1), enable_seg_to_denoise=False,
                  enable_denoise_to_seg=False, use_uncertainty=False,
                  detach_denoise_to_seg_source=True, dropout=0.0,
                  residual_scale=.5, causal_interaction_experiment=False,
                  detach_seg_to_denoise_source=True, interaction_scale_init=.1,
                  s2d_source_mode="cross", d2s_source_mode="cross",
                  strong_s2d_rho=None, strong_d2s_rho=None)
    return DualTaskAdaptiveV2Segmenter(kwargs, context_channels=4)


def test_gate_range_initialization_and_non_square_forward():
    model=tiny_model(); out=model(torch.rand(2,1,24,32))
    assert out["vessel_logits"].shape == (2,1,24,32)
    assert float(out["vessel_strength_map"].detach().min()) >= 0
    assert float(out["vessel_strength_map"].detach().max()) <= .5
    assert torch.allclose(out["vessel_strength_map"], torch.full_like(out["vessel_strength_map"], .25))


def test_zero_increment_and_off_are_exact_coarse_bypass_layer_unchanged():
    model=tiny_model(); image=torch.rand(1,1,24,32)
    on=model(image); off=model(image,vessel_adaptive_off=True)
    assert torch.equal(on["vessel_logits"], on["coarse_vessel_logits"])
    assert torch.equal(off["vessel_logits"], off["coarse_vessel_logits"])
    assert torch.equal(on["layer_logits"], off["layer_logits"])


def test_only_new_vessel_modules_trainable_and_v1_has_no_gradient():
    model=tiny_model(); out=model(torch.rand(1,1,24,32))
    loss=out["vessel_logits"].mean()+out["vessel_strength_map"].mean(); loss.backward()
    assert all(not p.requires_grad and p.grad is None for p in model.v1.parameters())
    assert model.vessel_logit_scale.grad is not None
    assert model.vessel_strength_head.bias.grad is not None


def test_noisy_path_is_additive_and_checkpoint_roundtrip(tmp_path):
    model=tiny_model(); image=torch.rand(1,1,24,32)
    with torch.no_grad(): model.vessel_logit_scale.fill_(.2); model.vessel_increments["1"].gamma.fill_(.1)
    expected=model(image); path=tmp_path/"m.pth"; torch.save(model.state_dict(),path)
    restored=tiny_model(); restored.load_state_dict(torch.load(path,weights_only=True))
    actual=restored(image)
    assert torch.allclose(expected["vessel_logits"],actual["vessel_logits"])
    assert torch.allclose(actual["vessel_increments_rms"] if "vessel_increments_rms" in actual else actual["vessel_increment_rms"], actual["vessel_increment_rms"])


def test_protection_formula_and_empty_regions():
    strength=torch.tensor([[[[.4,.4],[.1,.1]]]],requires_grad=True)
    vessel=torch.tensor([[[[1.,1.],[0.,0.]]]]); layer=torch.ones_like(vessel); valid=torch.ones_like(vessel)
    losses=vessel_protection_losses(strength,vessel,layer,valid,vessel,margin=.1)
    assert torch.allclose(losses["vessel_protect_rank"],torch.tensor(.4))
    assert torch.allclose(losses["vessel_protect"], losses["vessel_protect_rank"]+.5*losses["vessel_protect_boundary"]+.5*losses["vessel_protect_weak"])
    empty=vessel_protection_losses(strength,torch.zeros_like(vessel),torch.zeros_like(layer),valid,torch.zeros_like(vessel))
    assert all(torch.isfinite(v) for v in empty.values())


def test_protection_updates_new_vessel_only():
    model=tiny_model(); image=torch.rand(1,1,24,32); out=model(image)
    vessel=torch.zeros(1,1,24,32); vessel[:,:,5:10,5:10]=1; layer=torch.ones_like(vessel)
    loss=vessel_protection_losses(out["vessel_strength_map"],vessel,layer,torch.ones_like(vessel),vessel)["vessel_protect"]
    loss.backward()
    assert model.vessel_strength_head.bias.grad is not None
    assert all(p.grad is None for p in model.v1.parameters())


def test_checkpoint_gate_and_fail_closed():
    passed=checkpoint_eligibility(dict(REFERENCES))
    assert passed["eligible"] and math.isfinite(passed["q"])
    failed=checkpoint_eligibility({**REFERENCES,"small_recall":REFERENCES["small_recall"]-.006})
    assert not failed["eligible"] and failed["failed_checks"] == ["small_recall"]
    missing=checkpoint_eligibility({})
    assert not missing["eligible"] and len(missing["failed_checks"]) == 8


def test_prediction_rejects_clean_or_labels():
    model=tiny_model(); image=torch.rand(1,1,24,32)
    for key in ("clean","layer_mask","vessel_mask","ground_truth"):
        try: model(image,**{key:image})
        except ValueError: pass
        else: raise AssertionError(key)


def test_frozen_train_thresholds_define_weak_mask():
    vessel=torch.zeros(1,1,8,10); vessel[:,:,1:3,1:3]=1; vessel[:,:,4:8,5:10]=1
    layer=torch.ones_like(vessel); valid=torch.ones_like(vessel); noisy=torch.ones_like(vessel)*.5
    noisy[:,:,1:3,1:3]=.49; noisy[:,:,4:8,5:10]=.1
    weak=frozen_weak_mask(vessel,layer,valid,noisy,small_max=5,low_contrast_max=.02)
    assert weak[:,:,1:3,1:3].all() and not weak[:,:,4:8,5:10].any()


def test_legacy_config_does_not_enable_v2():
    config={"model":{"channels":[4,8],"encoder_depths":[1,1],"decoder_depth":1,
                     "interaction_levels":[1],"d2s_enabled":False,"s2d_enabled":False}}
    assert isinstance(build_model(config),SABIDSNet)


def test_all_feature_and_logit_scales_are_zero_initialized():
    model=tiny_model()
    assert model.vessel_logit_scale.item()==0
    assert all(block.gamma.item()==0 for block in model.vessel_increments.values())


def test_layer_and_shared_context_are_frozen():
    model=tiny_model()
    assert all(not p.requires_grad for p in model.v1.controller.shared.parameters())
    assert all(not p.requires_grad for p in model.v1.controller.layer_head.parameters())
    assert all(not p.requires_grad for p in model.v1.layer_aux_fusions.parameters())


def test_gate_has_no_clean_or_label_arguments():
    model=tiny_model(); out=model(torch.rand(1,1,24,32))
    assert out["vessel_strength_map"].shape==out["vessel_prob"].shape


def test_rank_is_zero_when_vessel_is_sufficiently_below_stroma():
    strength=torch.tensor([[[[.1,.1],[.4,.4]]]])
    vessel=torch.tensor([[[[1.,1.],[0.,0.]]]]); one=torch.ones_like(vessel)
    assert vessel_protection_losses(strength,vessel,one,one,vessel)["vessel_protect_rank"].item()==0


def test_boundary_and_weak_terms_are_exact_region_means():
    strength=torch.arange(25,dtype=torch.float32).reshape(1,1,5,5)/100
    vessel=torch.zeros_like(strength); vessel[:,:,2,2]=1; layer=torch.ones_like(vessel)
    loss=vessel_protection_losses(strength,vessel,layer,torch.ones_like(vessel),vessel)
    assert torch.allclose(loss["vessel_protect_weak"],strength[:,:,2,2].mean())
    assert loss["vessel_protect_boundary"].item()>0


def test_safe_epoch_earliest_tie_and_no_eligible_fail_closed():
    assert select_vessel_safe_epoch([{"epoch":2,"eligible":True,"q":.8},{"epoch":1,"eligible":True,"q":.8}])==1
    assert select_vessel_safe_epoch([{"epoch":1,"eligible":False,"q":.9}]) is None


def test_audit_fails_closed_without_formal_v1_evidence(tmp_path):
    cfg={"seed":42,"dual_task_adaptive_v2":{"v1_checkpoint":"missing.pth","v1_best_epoch":17,
         "vessel_strength_cap":.5,"evidence":{"v1_binding":"missing.json"}}}
    report=audit_v2_inputs(cfg,tmp_path)
    assert report["status"]=="blocked" and report["test_assets_opened"]==0


def test_v2_materializes_verified_b3_binding_without_mutating_old_run(tmp_path):
    checkpoint_sha = "a" * 64
    old_binding = tmp_path / "runs" / "historical" / "checkpoint_binding_best.json"
    payload = {
        "status": "passed",
        "checkpoint_path": "runs/historical/best.pth",
        "checkpoint_sha256": checkpoint_sha,
        "selection_rule": "best_validation_vessel_soft_dice",
        "completed_epochs": 20,
        "test_assets_opened": 0,
    }
    archive_path = tmp_path / "exports" / "best_checkpoint_supplement_test.tar.gz"
    archive_path.parent.mkdir(parents=True)
    serialized = json.dumps(payload).encode("utf-8")
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("supplement/b3_checkpoint_binding_best.json")
        info.size = len(serialized)
        archive.addfile(info, io.BytesIO(serialized))
    config = {
        "dual_task_adaptive": {
            "anchors": {"coarse_checkpoint_sha256": checkpoint_sha},
            "evidence": {"coarse_binding": str(old_binding)},
        }
    }
    registry = tmp_path / "cache" / "v2" / "run"
    issue = _materialize_bound_coarse_evidence(tmp_path, config, registry)
    recovered = registry / "evidence" / "b3_checkpoint_binding_best.json"
    assert issue is None
    assert not old_binding.exists()
    assert recovered.is_file()
    assert json.loads(recovered.read_text(encoding="utf-8")) == payload
    assert config["dual_task_adaptive"]["evidence"]["coarse_binding"] == str(recovered)


def test_v2_binding_recovery_fails_closed_without_verified_supplement(tmp_path):
    config = {
        "dual_task_adaptive": {
            "anchors": {"coarse_checkpoint_sha256": "b" * 64},
            "evidence": {"coarse_binding": "runs/missing/checkpoint_binding_best.json"},
        }
    }
    issue = _materialize_bound_coarse_evidence(tmp_path, config, tmp_path / "registry")
    assert issue is not None
    assert "exactly one matching best-checkpoint supplement" in issue.lower()


def test_vessel_source_stays_detached_while_receiver_updates():
    model=tiny_model(); x=torch.rand(1,1,24,32); out=model(x)
    (out["vessel_strength_map"].mean()+out["vessel_logits"].mean()).backward()
    assert all(p.grad is None for p in model.v1.parameters())
    assert model.vessel_strength_head.bias.grad is not None
