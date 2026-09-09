import json
import copy
import random
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml

from sabids.engine.trainer import build_model
from sabids.engine.trainer import Trainer
from sabids.config import load_config
from sabids.data.io import write_gray
from sabids.experiments.atlas import build_report_atlas
from sabids.experiments.protocol_lock import d1_protocol_evidence_rows, extract_active_protocol_lock, find_d1_runs, load_protocol_lock, validate_checkpoint_config
from sabids.losses.total import SABIDSLoss
from sabids.losses.common import multiscale_gradient_loss, multiscale_laplacian_loss
from sabids.models.ugbi import UGBIBlock
from sabids.training.phase_state_machine import PhaseStateMachine
from tools.package_next_stage_for_gpt import forbidden
from tools.prepare_interaction_shuffle import make_mapping
from tools.run_next_stage_suite import SUITES

def test_d1_structure_losses_finite_backward():
    x=torch.rand(1,1,32,32,requires_grad=True); y=torch.rand_like(x)
    loss=multiscale_gradient_loss(x,y)+multiscale_laplacian_loss(x,y)
    assert torch.isfinite(loss); loss.backward(); assert torch.isfinite(x.grad).all()

def test_rms_update_and_zero_equivalence():
    update=torch.randn(2,4,8,8,requires_grad=True); receiver=torch.randn_like(update)
    assert torch.equal(UGBIBlock.rms_scaled_update(update,receiver,0),torch.zeros_like(update))
    scaled=UGBIBlock.rms_scaled_update(update,receiver,0.01)
    ratio=scaled.square().mean((1,2,3)).sqrt()/receiver.square().mean((1,2,3)).sqrt()
    assert torch.allclose(ratio,torch.full_like(ratio,0.01),atol=1e-5)
    scaled.sum().backward(); assert update.grad is not None

def test_detached_source_blocks_source_gradient_but_mapping_trains():
    block=UGBIBlock(4,enable_seg_to_denoise=False,enable_denoise_to_seg=True)
    source=torch.randn(1,4,8,8,requires_grad=True); layer=torch.randn_like(source,requires_grad=True); vessel=torch.randn_like(source,requires_grad=True)
    lo,vo,_=block.denoise_to_seg(source,layer,vessel,detach_source=True,rms_rho=.01)
    (lo.mean()+vo.mean()).backward()
    assert source.grad is None and layer.grad is not None
    assert block.denoise_to_layer.weight.grad is not None

def test_interaction_diagnostics_are_not_auxiliary_heads():
    cfg={"auxiliary_weight":0.1,"weights":{"layer":1,"vessel":1}}
    loss=SABIDSLoss(cfg)
    output={"denoised_raw":torch.zeros(1,1,8,8,requires_grad=True),"layer_logits":torch.zeros(1,1,8,8,requires_grad=True),"vessel_logits":torch.zeros(1,1,8,8,requires_grad=True),"boundary_logits":torch.zeros(1,2,8,8,requires_grad=True),"auxiliary":[{"requested_rho":torch.tensor(0.01)}]}
    batch={"has_clean":torch.tensor([False]),"has_layer":torch.tensor([True]),"has_vessel":torch.tensor([True]),"layer_mask":torch.zeros(1,1,8,8),"vessel_mask":torch.zeros(1,1,8,8),"valid_mask":torch.ones(1,1,8,8),"is_clean":torch.tensor([False])}
    result=loss(output,batch,"interaction")
    assert torch.isfinite(result["total"])


def _write_d1_run(root: Path, seed: int, plan: str = "plan") -> Path:
    run = root / "runs" / "current" / f"d1_denoise_struct_pku37_v2_fold0_seed{seed}"
    run.mkdir(parents=True)
    config = {
        "protocol_id": "pku37_binary_v2", "manifest_root": "Manifests/pku37_binary_v2",
        "data_plan_sha256": plan, "label_inventory_sha256": "labels",
        "dataset_inventory_sha256": "dataset", "split_contract_sha256": "split",
        "seed": seed, "fold": 0, "data": {"target_size": [512, 512], "normalization": "fixed"},
        "train": {"stage": "denoise", "epochs": 60},
        "loss": {"restoration_mode": "structure_d1", "char_weight": 1.0},
    }
    (run / "resolved_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (run / "data_plan_audit.json").write_text(json.dumps({"train_positions": ["p1"], "validation_positions": ["p2"], "test_positions": ["p3"], "git_commit": "abc"}), encoding="utf-8")
    return run


def test_active_protocol_lock_is_extracted_from_d1_evidence(tmp_path):
    for seed in (42, 43, 44): _write_d1_run(tmp_path, seed)
    runs = find_d1_runs(tmp_path)
    lock = extract_active_protocol_lock(runs)
    assert lock["source"] == "active_d1_run"
    assert lock["protocol_id"] == "pku37_binary_v2"
    assert lock["validation_positions"] == ["p2"]
    assert lock["test_assets_opened"] == 0


def test_d1_seed_sha_disagreement_blocks_lock(tmp_path):
    _write_d1_run(tmp_path, 42, "one"); _write_d1_run(tmp_path, 43, "two")
    with pytest.raises(RuntimeError, match="data_plan_sha256"):
        extract_active_protocol_lock(find_d1_runs(tmp_path))


def test_protocol_lock_does_not_confuse_seed_sampler_hash_with_protocol_hash(tmp_path):
    for seed in (42, 43, 44):
        run = _write_d1_run(tmp_path, seed, "stable-protocol-plan")
        (run / "initialization_audit.json").write_text(
            json.dumps({"data_plan_sha256": f"seed-dependent-sampler-{seed}"}),
            encoding="utf-8",
        )
    runs = find_d1_runs(tmp_path)
    lock = extract_active_protocol_lock(runs)
    evidence = d1_protocol_evidence_rows(runs)
    assert lock["data_plan_sha256"] == "stable-protocol-plan"
    assert {row["sampler_plan_sha256"] for row in evidence} == {
        "seed-dependent-sampler-42",
        "seed-dependent-sampler-43",
        "seed-dependent-sampler-44",
    }


def test_incomplete_or_test_tainted_protocol_lock_is_rejected(tmp_path):
    path = tmp_path / "lock.json"
    path.write_text(json.dumps({"protocol_id": "p", "manifest_root": "m", "data_plan_sha256": "d", "label_inventory_sha256": "l", "dataset_inventory_sha256": "i", "split_contract_sha256": "s", "test_assets_opened": 1}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="zero test access"):
        load_protocol_lock(path)


def test_old_label_checkpoint_cannot_enter_active_protocol():
    lock = {"protocol_id": "p", "data_plan_sha256": "new-data", "label_inventory_sha256": "new-label"}
    checkpoint = {"config": {"protocol_id": "p", "data_plan_sha256": "new-data", "label_inventory_sha256": "old-label"}}
    with pytest.raises(RuntimeError, match="label_inventory_sha256 mismatch"):
        validate_checkpoint_config(checkpoint, lock)


def test_continuous_phase_machine_restores_global_step_and_rng():
    machine = PhaseStateMachine("order_ds", (2, 2, 2))
    machine.record_epoch_phase(0)
    for _ in range(7): machine.step()
    state = machine.snapshot(1)
    restored = PhaseStateMachine("order_ds", (2, 2, 2))
    restored.restore(state)
    assert restored.global_step == 7
    assert restored.phase(0) == "denoise" and restored.phase(2) == "segment" and restored.phase(4) == "joint"
    with pytest.raises(RuntimeError, match="schedule_epochs"):
        PhaseStateMachine("order_ds", (1, 1, 1)).restore(state)


def test_alt_uses_whole_optimizer_steps_and_equal_budget():
    machine = PhaseStateMachine("order_alt", (40, 20))
    counts = {"denoise": 0, "segment": 0}
    for _ in range(100):
        phase = machine.phase(0)
        assert machine.phase(0, 99) == phase
        counts[phase] += 1; machine.step()
    assert counts == {"denoise": 50, "segment": 50}


def test_phase_boundary_resume_matches_uninterrupted_optimizer_and_rng():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Linear(2, 2)
            self.residual_head = torch.nn.Linear(2, 1)
            self.layer_head = torch.nn.Linear(2, 1)
            self.vessel_head = torch.nn.Linear(2, 1)

        def objective(self, x, phase):
            shared = torch.tanh(self.stem(x))
            denoise = self.residual_head(shared).square().mean()
            segment = self.layer_head(shared).square().mean() + self.vessel_head(shared).square().mean()
            return denoise if phase == "denoise" else segment if phase == "segment" else denoise + segment

    def initialize():
        random.seed(8); torch.manual_seed(8)
        model = Tiny(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6)
        return model, optimizer, scheduler, PhaseStateMachine("order_ds", (2, 2, 2))

    def advance(model, optimizer, scheduler, machine, start, stop):
        for epoch in range(start, stop):
            phase = machine.phase(epoch); PhaseStateMachine.set_trainable(model, phase)
            optimizer.zero_grad(); model.objective(torch.randn(4, 2), phase).backward(); optimizer.step(); scheduler.step(); machine.step()

    full_model, full_optimizer, full_scheduler, full_machine = initialize()
    advance(full_model, full_optimizer, full_scheduler, full_machine, 0, 6)
    part_model, part_optimizer, part_scheduler, part_machine = initialize()
    advance(part_model, part_optimizer, part_scheduler, part_machine, 0, 2)
    checkpoint = {"model": copy.deepcopy(part_model.state_dict()), "optimizer": copy.deepcopy(part_optimizer.state_dict()), "scheduler": copy.deepcopy(part_scheduler.state_dict()), "phase": part_machine.snapshot(1)}
    resumed_model, resumed_optimizer, resumed_scheduler, resumed_machine = initialize()
    resumed_model.load_state_dict(checkpoint["model"]); resumed_optimizer.load_state_dict(checkpoint["optimizer"]); resumed_scheduler.load_state_dict(checkpoint["scheduler"]); resumed_machine.restore(checkpoint["phase"])
    advance(resumed_model, resumed_optimizer, resumed_scheduler, resumed_machine, 2, 6)
    for name, expected in full_model.state_dict().items():
        assert torch.equal(expected, resumed_model.state_dict()[name]), name
    assert full_machine.global_step == resumed_machine.global_step == 6
    assert full_scheduler.state_dict() == resumed_scheduler.state_dict()


def test_continuous_order_trainer_smoke_writes_phase_checkpoint(tmp_path):
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        sample = f"s{index}"; image = torch.linspace(0, 1, 32 * 32).reshape(32, 32).numpy()
        layer = torch.zeros(32, 32).numpy(); layer[8:25] = 1
        vessel = torch.zeros(32, 32).numpy(); vessel[14:18, 10:22] = 1
        paths = {}
        for name, value in (("image", image), ("clean", image * .95), ("layer", layer), ("vessel", vessel)):
            path = tmp_path / f"{sample}_{name}.png"; write_gray(path, value); paths[name] = str(path)
        rows.append({"sample_id": sample, "group_id": f"g{index}", "patient_id": f"g{index}", "dataset": "PKU37", "split": split, "image_path": paths["image"], "clean_path": paths["clean"], "layer_mask_path": paths["layer"], "vessel_mask_path": paths["vessel"]})
    manifest = tmp_path / "manifest.csv"; pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = load_config("configs/next_stage_v3/order_ds_pku37_v3.yaml")
    cfg.update({"device": "cpu", "deterministic": True})
    cfg["model"].update({"channels": [4, 8], "encoder_depths": [1, 1], "decoder_depth": 1, "interaction_levels": [1], "d2s_enabled": False, "s2d_enabled": False})
    cfg["data"].update({"manifest": str(manifest), "root": str(tmp_path), "target_size": [32, 32], "max_train_samples": 2, "max_val_samples": 1, "train_datasets": ["PKU37"], "val_datasets": ["PKU37"]})
    cfg["train"].update({"output_dir": str(tmp_path / "run"), "epochs": 3, "schedule_epochs": [1, 1, 1], "batch_size": 1, "gradient_accumulation_steps": 1, "num_workers": 0, "amp": False, "early_stopping_patience": 4, "pretrained": None, "resume": None})
    cfg["evaluation"]["num_workers"] = 0
    trainer = Trainer(cfg); trainer.fit()
    checkpoint = torch.load(tmp_path / "run" / "last.pth", map_location="cpu", weights_only=False)
    assert checkpoint["phase_state"]["current_phase"] == "joint"
    assert checkpoint["phase_state"]["global_step"] == 6
    assert checkpoint["optimizer"] and checkpoint["scheduler"]


def test_phase_freezing_changes_only_requested_parameter_groups():
    cfg = {"model": {"channels": [4, 8], "encoder_depths": [1, 1], "decoder_depth": 1, "interaction_levels": [1], "d2s_enabled": False, "s2d_enabled": False}}
    model = build_model(cfg)
    active = PhaseStateMachine.set_trainable(model, "denoise")
    assert any(name.startswith("stem") for name in active)
    assert any(name.startswith("decoders.denoise") for name in active)
    assert not any(name.startswith("decoders.layer") for name in active)
    assert not model.decoders["layer"].training


def test_all_seven_strength_arms_and_percentages_are_exact():
    names = SUITES["decoder_interaction_strength"]
    assert len(names) == 7
    values = sorted(float(yaml.safe_load((Path("configs/next_stage_v3") / name).read_text(encoding="utf-8"))["model"].get("strong_d2s_rho", 0.0) or yaml.safe_load((Path("configs/next_stage_v3") / name).read_text(encoding="utf-8"))["model"].get("strong_s2d_rho", 0.0)) for name in names)
    assert values == [0.0, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02]


def test_requested_rho_diagnostics_match_actual_rho():
    block = UGBIBlock(4, enable_seg_to_denoise=True, enable_denoise_to_seg=False)
    target = torch.randn(2, 4, 8, 8)
    source = torch.randn_like(target)
    output, details = block.seg_to_denoise(target, source, source, detach_source=True, rms_rho=.02)
    assert torch.isfinite(output).all()
    assert details["requested_rho"].item() == pytest.approx(.02)
    assert details["actual_rho_mean"].item() == pytest.approx(.02, abs=1e-5)


def test_shuffle_mapping_is_reproducible_cross_position_and_within_split():
    table = pd.DataFrame({"sample_id": ["a1", "a2", "b1", "b2", "c1", "d1"], "group_id": ["a", "a", "b", "b", "c", "d"], "split": ["train", "train", "train", "train", "val", "val"]})
    left, right = make_mapping(table, 42), make_mapping(table, 42)
    pd.testing.assert_frame_equal(left, right)
    assert not (left.source_group_id == left.guidance_group_id).any()
    assert (left["split"] == right["split"]).all()


def test_self_adapter_keeps_identical_parameter_budget():
    base = {"channels": [4, 8], "encoder_depths": [1, 1], "decoder_depth": 1, "interaction_levels": [1], "d2s_enabled": True, "s2d_enabled": True, "causal_interaction_experiment": True}
    cross = build_model({"model": {**base, "d2s_source_mode": "cross", "s2d_source_mode": "cross"}})
    self_adapter = build_model({"model": {**base, "d2s_source_mode": "receiver_capacity", "s2d_source_mode": "receiver_capacity"}})
    assert sum(p.numel() for p in cross.parameters()) == sum(p.numel() for p in self_adapter.parameters())


def test_batch_one_shuffle_uses_explicit_cached_guidance_without_batch_shuffle():
    cfg = {"model": {"channels": [4, 8], "encoder_depths": [1, 1], "decoder_depth": 1, "interaction_levels": [1], "d2s_enabled": True, "s2d_enabled": False, "causal_interaction_experiment": True, "d2s_source_mode": "shuffled_cross", "detach_d2s_source": True, "strong_d2s_rho": .01}}
    model = build_model(cfg)
    image, guidance = torch.rand(1, 1, 16, 16), torch.rand(1, 1, 16, 16)
    output = model(image, interaction_guidance_image=guidance)
    assert output["vessel_prob"].shape == image.shape
    assert torch.isfinite(output["vessel_prob"]).all()
    with pytest.raises(ValueError, match="interaction_guidance_image"):
        model(image)


def test_light_archive_filter_rejects_sensitive_or_heavy_members():
    for value in ("runs/a/best.pth", "runs/cache/value.npy", "Data/patient.png", "runs/report/test_results/x.csv"):
        assert forbidden(Path(value))
    assert not forbidden(Path("runs/reports/fixed_atlas/image.png"))


def test_missing_atlas_images_render_placeholders_instead_of_crashing(tmp_path):
    report = tmp_path / "input_image_pku37_v2"; report.mkdir()
    pd.DataFrame([{"run_id": "run", "arm": "I-NOISY", "group_id": "p1", "sample_id": "s1"}]).to_csv(report / "metrics_by_frame.csv", index=False)
    pd.DataFrame([{"group_id": "p1", "selection_rule": "fixed"}]).to_csv(report / "atlas_selection.csv", index=False)
    missing = build_report_atlas(report)
    assert missing and (report / "fixed_atlas_sheets" / "p1_s1_atlas.png").is_file()


def test_no_next_stage_config_trains_on_duke():
    for path in Path("configs/next_stage_v3").glob("*.yaml"):
        text = path.read_text(encoding="utf-8").lower()
        assert "duke17" not in text and "duke28" not in text
