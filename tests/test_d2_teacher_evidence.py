import copy
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
import yaml
import cv2
import numpy as np

from sabids.config import load_config, save_config
from sabids.engine import Trainer
from sabids.engine.trainer import build_model
from sabids.experiments.d2_teacher import (
    audit_teacher_protocol,
    audit_teacher_binding,
    bind_derived_legacy_teacher,
    bind_native_teacher,
)
from sabids.experiments.dose_response import asset_inventory, effective_split_sha, stable_sha
from sabids.experiments.protocol_lock import sha256_file


def _teacher_fixture(tmp_path: Path) -> dict:
    run = tmp_path / "teacher_run"
    run.mkdir()
    assets = tmp_path / "assets"; assets.mkdir()
    rows = []
    for index, split in enumerate(("train", "train", "val")):
        paths = {}
        for column in ("image", "clean", "layer_mask", "vessel_mask"):
            path = assets / f"s{index}_{column}.bin"
            path.write_bytes(f"{index}-{column}".encode())
            paths[f"{column}_path"] = str(path)
        rows.append({
            "sample_id": f"s{index}", "group_id": f"g{index}",
            "patient_id": f"g{index}", "dataset": "PKU37", "split": split,
            **paths,
        })
    manifest = tmp_path / "train_segment.csv"
    table = pd.DataFrame(rows); table.to_csv(manifest, index=False)
    split_contract = tmp_path / "split.yaml"
    split_contract.write_text(
        "protocol_id: fixture\nvalidation_positions: [g2]\ntest_positions: [g9]\n",
        encoding="utf-8",
    )
    lock = {
        "protocol_id": "fixture", "manifest_root": str(tmp_path),
        "data_plan_sha256": "data-plan", "label_inventory_sha256": "labels",
        "dataset_inventory_sha256": "dataset", "split_contract_sha256": sha256_file(split_contract),
        "train_positions": ["g0", "g1", "g_unlabelled"],
        "validation_positions": ["g2"],
        "sealed_test_positions": ["g9"], "input_resolution": [16, 16],
        "normalization": "fixed", "test_assets_opened": 0,
    }
    lock_path = tmp_path / "active_protocol_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    config = {
        "protocol_id": "fixture", "label_type": "binary", "seed": 42,
        "model": {"d2s_enabled": False, "s2d_enabled": False,
                  "enable_denoise_to_seg": False, "enable_seg_to_denoise": False,
                  "stage2_freeze_shared_encoder": True,
                  "stage2_train_denoise_to_seg": False},
        "data": {"manifest": str(manifest), "root": str(tmp_path),
                 "train_split": "train", "val_split": "val",
                 "target_size": [16, 16], "normalization": "fixed"},
        "train": {"stage": "segment", "epochs": 3,
                  "monitor": "vessel_soft_dice",
                  "checkpoint_selection_rule": "best_validation_vessel_soft_dice",
                  "output_dir": str(run)},
        "loss": {"weights": {}},
        "formal_d2_teacher": {
            "enabled": True,
            "expected_train_positions": ["g0", "g1"],
            "expected_validation_positions": ["g2"],
        },
        "runtime": {"active_protocol_lock": lock,
                    "manifest_sha256": sha256_file(manifest),
                    "effective_split_sha256": effective_split_sha(table)},
    }
    config_path = run / "resolved_config.yaml"; save_config(config, config_path)
    config = load_config(config_path)
    checkpoint = run / "best.pth"
    torch.save({
        "epoch": 1, "best_metric": .8, "global_optimizer_step": 4,
        "model": {"weight": torch.ones(1)}, "optimizer": {"state": {}},
        "config": config,
    }, checkpoint)
    history = run / "history.csv"
    pd.DataFrame({"epoch": [1, 2, 3],
                  "val_vessel_soft_dice": [.5, .8, .7]}).to_csv(history, index=False)
    metadata = {
        "run_id": run.name, "best_checkpoint": str(checkpoint.resolve()),
        "best_checkpoint_sha256": sha256_file(checkpoint), "best_epoch": 2,
        "best_metric": .8, "monitor": "vessel_soft_dice",
        "selection_rule": "best_validation_vessel_soft_dice",
    }
    metadata_path = run / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    records = asset_inventory(tmp_path, table, include_labels=True)
    initial_inventory = run / "training_asset_inventory_initial.json"
    initial_inventory.write_text(json.dumps({
        "recorded_at_training": True, "recorded_before_optimizer_step": True,
        "manifest_sha256": sha256_file(manifest),
        "effective_split_sha256": effective_split_sha(table),
        "records": records, "records_sha256": stable_sha(records),
        "test_assets_opened": 0,
    }), encoding="utf-8")
    initial_checkpoint = run / "initial.pth"
    torch.save({"epoch": -1, "global_optimizer_step": 0, "model": {"weight": torch.zeros(1)},
                "optimizer": {"state": {}}, "config": config}, initial_checkpoint)
    initialization_audit = run / "initialization_audit.json"
    initialization_audit.write_text(json.dumps({"model_state_sha256": "initial"}), encoding="utf-8")
    parameter_audit = run / "formal_teacher_parameter_audit.json"
    parameter_audit.write_text(json.dumps({
        "status": "passed", "optimizer_parameter_count": 1,
        "changed_trainable_parameter_count": 1, "changed_frozen_parameter_count": 0,
        "optimizer_steps": 4,
    }), encoding="utf-8")
    historical = run / "historical_protocol_evidence.json"
    historical.write_text(json.dumps({
        "recorded_at_training": True, "recorded_before_optimizer_step": True,
        "evidence_origin": "immutable_training_record", "run_id": run.name,
        "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_epoch": 2,
        "selection_rule": "best_validation_vessel_soft_dice",
        "selection_metric": "val_vessel_soft_dice",
        **{key: lock[key] for key in (
            "protocol_id", "data_plan_sha256", "label_inventory_sha256",
            "dataset_inventory_sha256", "split_contract_sha256")},
        "manifest_sha256": sha256_file(manifest),
        "effective_split_sha256": effective_split_sha(table),
        "train_positions": ["g0", "g1"], "validation_positions": ["g2"],
        "input_resolution": [16, 16], "normalization": "fixed",
        "label_semantics": "binary", "test_assets_opened": 0,
    }), encoding="utf-8")
    return locals()


def _native(e: dict, output: Path):
    return bind_native_teacher(
        e["tmp_path"], e["checkpoint"], e["history"], e["config_path"],
        e["metadata_path"], e["lock_path"], e["split_contract"],
        e["initial_inventory"], e["initial_checkpoint"],
        e["initialization_audit"], e["parameter_audit"], output,
    )


def _derived(e: dict, output: Path):
    return bind_derived_legacy_teacher(
        e["tmp_path"], e["checkpoint"], e["history"], e["config_path"],
        e["metadata_path"], e["lock_path"], e["split_contract"],
        e["historical"], e["parameter_audit"], output,
    )


def test_native_protocol_binding_passes_and_is_hash_auditable(tmp_path):
    e = _teacher_fixture(tmp_path)
    result = _native(e, tmp_path / "native.json")
    assert result["evidence_type"] == "native_protocol_binding"
    assert audit_teacher_binding(tmp_path / "native.json", e["checkpoint"])["status"] == "passed"
    assert result["train_positions"] == ["g0", "g1"]


def test_teacher_protocol_accepts_registered_label_subset_but_rejects_drift(tmp_path):
    e = _teacher_fixture(tmp_path)
    protocol = audit_teacher_protocol(
        tmp_path, load_config(e["config_path"]), e["lock_path"], e["split_contract"]
    )
    assert protocol["train_positions"] == ["g0", "g1"]
    assert protocol["protocol_train_positions"] == ["g0", "g1", "g_unlabelled"]
    assert protocol["teacher_train_subset_of_protocol"] is True

    config = load_config(e["config_path"])
    config["formal_d2_teacher"]["expected_train_positions"] = ["g0"]
    with pytest.raises(ValueError, match="registered label-eligible cohort"):
        audit_teacher_protocol(tmp_path, config, e["lock_path"], e["split_contract"])

    config = load_config(e["config_path"])
    table = pd.read_csv(e["manifest"])
    table.loc[table["group_id"].eq("g1"), "group_id"] = "outside_lock"
    table.to_csv(e["manifest"], index=False)
    config["runtime"]["manifest_sha256"] = sha256_file(e["manifest"])
    config["runtime"]["effective_split_sha256"] = effective_split_sha(table)
    with pytest.raises(ValueError, match="exceed the active-lock train cohort"):
        audit_teacher_protocol(tmp_path, config, e["lock_path"], e["split_contract"])


def test_complete_immutable_legacy_binding_passes(tmp_path):
    e = _teacher_fixture(tmp_path)
    result = _derived(e, tmp_path / "derived.json")
    assert result["evidence_type"] == "derived_legacy_binding"


@pytest.mark.parametrize("failure,match", [
    ("protocol_only_patch", "Legacy protocol evidence mismatch"),
    ("checkpoint_sha", "checkpoint SHA mismatch"),
    ("history_best_epoch", "history best epoch"),
    ("checkpoint_epoch", "history best epoch"),
    ("selection_metric", "selection metric/rule"),
    ("run_id", "run ID mismatch"),
    ("train_positions", "train positions"),
    ("split_contract", "split_contract_sha256"),
    ("label_asset", "label_inventory_sha256"),
    ("data_plan", "data_plan_sha256"),
    ("missing_evidence", "Missing derived teacher source"),
    ("test_reference", "sealed test assets"),
])
def test_legacy_binding_fails_closed_for_incomplete_or_changed_evidence(
    tmp_path, failure, match
):
    e = _teacher_fixture(tmp_path)
    historical = json.loads(e["historical"].read_text())
    if failure == "protocol_only_patch":
        historical = {"recorded_at_training": True,
                      "recorded_before_optimizer_step": True,
                      "evidence_origin": "immutable_training_record",
                      "protocol_id": "fixture"}
    elif failure == "checkpoint_sha": historical["checkpoint_sha256"] = "changed"
    elif failure == "history_best_epoch":
        pd.DataFrame({"epoch": [1, 2, 3],
                      "val_vessel_soft_dice": [.9, .8, .7]}).to_csv(e["history"], index=False)
    elif failure == "checkpoint_epoch":
        raw = torch.load(e["checkpoint"], weights_only=False); raw["epoch"] = 2
        torch.save(raw, e["checkpoint"])
        historical["checkpoint_sha256"] = sha256_file(e["checkpoint"])
        e["metadata"]["best_checkpoint_sha256"] = sha256_file(e["checkpoint"])
        e["metadata_path"].write_text(json.dumps(e["metadata"]))
    elif failure == "selection_metric": historical["selection_metric"] = "vessel_dice"
    elif failure == "run_id": historical["run_id"] = "another_run"
    elif failure == "train_positions": historical["train_positions"] = ["g0"]
    elif failure == "split_contract": historical["split_contract_sha256"] = "changed"
    elif failure == "label_asset": historical["label_inventory_sha256"] = "changed"
    elif failure == "data_plan": historical["data_plan_sha256"] = "changed"
    elif failure == "missing_evidence": e["historical"].unlink()
    elif failure == "test_reference": historical["test_asset_paths"] = ["sealed/test.png"]
    if e["historical"].exists():
        e["historical"].write_text(json.dumps(historical))
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        _derived(e, tmp_path / f"{failure}.json")


def test_existing_teacher_binding_is_never_silently_overwritten(tmp_path):
    e = _teacher_fixture(tmp_path); output = tmp_path / "binding.json"
    _native(e, output)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        _native(e, output)
    assert output.read_bytes() == original


def test_native_binding_rejects_test_row_before_opening_assets(tmp_path):
    e = _teacher_fixture(tmp_path)
    table = pd.read_csv(e["manifest"])
    table.loc[len(table)] = {
        "sample_id": "sealed", "group_id": "g9", "patient_id": "g9",
        "dataset": "PKU37", "split": "test",
        "image_path": str(tmp_path / "must_not_open.png"), "clean_path": "",
        "layer_mask_path": "", "vessel_mask_path": "",
    }
    table.to_csv(e["manifest"], index=False)
    config = load_config(e["config_path"])
    config["runtime"]["manifest_sha256"] = sha256_file(e["manifest"])
    save_config(config, e["config_path"])
    raw = torch.load(e["checkpoint"], weights_only=False)
    raw["config"] = load_config(e["config_path"]); torch.save(raw, e["checkpoint"])
    e["metadata"]["best_checkpoint_sha256"] = sha256_file(e["checkpoint"])
    e["metadata_path"].write_text(json.dumps(e["metadata"]))
    with pytest.raises(ValueError, match="non-development split"):
        _native(e, tmp_path / "blocked.json")
    assert not (tmp_path / "must_not_open.png").exists()


def test_formal_teacher_cpu_smoke_writes_initial_best_last_and_freeze_audits(tmp_path):
    protocol = tmp_path / "Manifests" / "fixture"; protocol.mkdir(parents=True)
    rows = []
    for index, split in enumerate(("train", "val")):
        image = np.full((16, 16), .4, np.float32)
        clean = np.full((16, 16), .45, np.float32)
        layer = np.zeros((16, 16), np.uint8); layer[2:14, 2:14] = 255
        vessel = np.zeros((16, 16), np.uint8); vessel[6:9, 6:9] = 255
        image_path = tmp_path / f"{split}_image.npy"; np.save(image_path, image)
        clean_path = tmp_path / f"{split}_clean.npy"; np.save(clean_path, clean)
        layer_path = tmp_path / f"{split}_layer.png"; assert cv2.imwrite(str(layer_path), layer)
        vessel_path = tmp_path / f"{split}_vessel.png"; assert cv2.imwrite(str(vessel_path), vessel)
        rows.append({
            "sample_id": f"s{index}", "group_id": f"g{index}",
            "patient_id": f"g{index}", "dataset": "PKU37", "split": split,
            "image_path": str(image_path), "clean_path": str(clean_path),
            "layer_mask_path": str(layer_path), "vessel_mask_path": str(vessel_path),
        })
    manifest = protocol / "train_segment.csv"
    table = pd.DataFrame(rows); table.to_csv(manifest, index=False)
    split = tmp_path / "split.yaml"
    split.write_text("protocol_id: fixture\nvalidation_positions: [g1]\ntest_positions: [g9]\n")
    lock = {
        "protocol_id": "fixture", "manifest_root": str(protocol),
        "data_plan_sha256": "data", "label_inventory_sha256": "labels",
        "dataset_inventory_sha256": "dataset", "split_contract_sha256": sha256_file(split),
        "train_positions": ["g0"], "validation_positions": ["g1"],
        "sealed_test_positions": ["g9"], "input_resolution": [16, 16],
        "normalization": "fixed", "test_assets_opened": 0,
    }
    lock_path = protocol / "active_protocol_lock.json"
    lock_path.write_text(json.dumps(lock))
    cfg = load_config(Path(__file__).parents[1] / "configs/base.yaml")
    cfg.update(device="cpu", deterministic=True, seed=42, protocol_id="fixture",
               manifest_root=str(protocol), data_plan_sha256="data",
               label_inventory_sha256="labels", label_type="binary")
    cfg["model"].update(
        channels=[2, 4], encoder_depths=[1, 1], decoder_depth=1,
        interaction_levels=[1], d2s_enabled=False, s2d_enabled=False,
        enable_denoise_to_seg=False, enable_seg_to_denoise=False,
        stage2_freeze_shared_encoder=True, stage2_train_denoise_to_seg=False,
    )
    cfg["data"].update(
        manifest=str(manifest), root=str(tmp_path), target_size=[16, 16],
        train_datasets=["PKU37"], val_datasets=["PKU37"],
        samples_per_epoch=1, load_segmentation_labels=True,
    )
    cfg["train"].update(
        stage="segment", output_dir=str(tmp_path / "formal_teacher"), epochs=1,
        batch_size=1, gradient_accumulation_steps=1, num_workers=0,
        learning_rate=1e-3, amp=False, monitor="vessel_soft_dice",
        checkpoint_selection_rule="best_validation_vessel_soft_dice",
        early_stopping_patience=2, scheduler="cosine", evaluate_epoch0=False,
        train_eval_every=0, monitor_denoise_drift=False, resume=None,
    )
    cfg["loss"].update(auxiliary_weight=0.0)
    cfg["loss"]["weights"].update(
        layer=1.0, vessel=1.0, vessel_stroma=.25, vessel_area=.2,
        vessel_outside=0.0, containment=.1,
    )
    cfg["runtime"].update(active_protocol_lock=lock,
                          manifest_sha256=sha256_file(manifest),
                          effective_split_sha256=effective_split_sha(table))
    d1 = tmp_path / "d1_best.pth"
    d1_config = copy.deepcopy(cfg)
    d1_config.pop("formal_d2_teacher", None)
    torch.save({"epoch": 0, "model": build_model(cfg).state_dict(),
                "config": d1_config}, d1)
    cfg["train"].update(pretrained=str(d1), strict_pretrained=False)
    cfg["training_asset_evidence"] = {
        "enabled": True, "project_root": str(tmp_path),
        "protocol_lock": str(lock_path),
    }
    cfg["formal_d2_teacher"] = {
        "enabled": True, "template_only": False, "run_mode": "smoke",
        "split_contract": str(split),
        "expected_train_positions": ["g0"],
        "expected_validation_positions": ["g1"],
    }
    run = tmp_path / "formal_teacher"
    trainer = Trainer(cfg)
    save_config(trainer.config, run / "resolved_config.yaml")
    trainer.fit()
    for name in (
        "initial.pth", "best.pth", "last.pth", "history.csv",
        "training_asset_inventory_initial.json", "training_asset_inventory_last.json",
        "initialization_audit.json", "formal_teacher_parameter_audit.json",
        "formal_teacher_training_summary.json", "run_metadata.json",
    ):
        assert (run / name).is_file(), name
    audit = json.loads((run / "formal_teacher_parameter_audit.json").read_text())
    assert audit["status"] == "passed"
    assert audit["changed_trainable_parameter_count"] > 0
    assert audit["changed_frozen_parameter_count"] == 0
    binding = bind_native_teacher(
        tmp_path, run / "best.pth", run / "history.csv",
        run / "resolved_config.yaml", run / "run_metadata.json",
        lock_path, split, run / "training_asset_inventory_initial.json",
        run / "initial.pth", run / "initialization_audit.json",
        run / "formal_teacher_parameter_audit.json", run / "teacher_binding.json",
    )
    assert binding["evidence_type"] == "native_protocol_binding"
