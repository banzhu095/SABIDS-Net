import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sabids.config import load_config, save_config
from sabids.engine import Trainer
from sabids.experiments import dose_response as dose


ROOT = Path(__file__).resolve().parents[1]


def _assets(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    rows = []
    for index, split in enumerate(("train", "train", "val", "test")):
        image = tmp_path / f"{split}_{index}_noisy.npy"
        clean = tmp_path / f"{split}_{index}_clean.npy"
        if split != "test":
            np.save(image, np.full((16, 16), 0.2 + index * 0.1, np.float32))
            np.save(clean, np.full((16, 16), 0.3 + index * 0.1, np.float32))
        rows.append({
            "sample_id": f"sample_{index}",
            "group_id": f"group_{index}",
            "dataset": "PKU37" if index != 1 else "OTHER",
            "split": split,
            "image_path": str(image),
            "clean_path": str(clean),
        })
    manifest = tmp_path / "manifest.csv"
    table = pd.DataFrame(rows)
    table.to_csv(manifest, index=False)
    return manifest, table


def _config(tmp_path: Path, manifest: Path) -> dict:
    cfg = load_config(ROOT / "configs/base.yaml")
    cfg.update(seed=42, deterministic=True, device="cpu", protocol_id="fixture")
    cfg["model"].update(
        channels=[2, 4], encoder_depths=[1, 1], decoder_depth=1,
        interaction_levels=[1], d2s_enabled=False, s2d_enabled=False,
    )
    cfg["data"].update(
        manifest=str(manifest), root=str(tmp_path), target_size=[16, 16],
        train_datasets=["PKU37"], val_datasets=["PKU37"],
    )
    cfg["train"].update(
        stage="denoise", output_dir=str(tmp_path / "run"), epochs=1,
        batch_size=1, gradient_accumulation_steps=1, num_workers=0,
        amp=False, early_stopping_patience=2, monitor="psnr", resume=None,
    )
    cfg["loss"].update(restoration_mode="structure_d1")
    cfg["training_asset_evidence"] = {"enabled": True, "project_root": str(tmp_path)}
    return cfg


def test_training_inventory_uses_filtered_train_val_only_and_is_deterministic(tmp_path, monkeypatch):
    manifest, table = _assets(tmp_path)
    filtered = table[
        table["split"].isin(["train", "val"]) & table["dataset"].eq("PKU37")
    ].iloc[::-1].reset_index(drop=True)
    cfg = _config(tmp_path, manifest)
    opened = []
    original = dose.sha256_file

    def guarded(path):
        opened.append(str(path))
        assert "test_3" not in str(path)
        return original(path)

    monkeypatch.setattr(dose, "sha256_file", guarded)
    first = dose.create_training_asset_evidence(
        tmp_path, cfg, filtered, tmp_path / "first.json"
    )
    second = dose.create_training_asset_evidence(
        tmp_path, cfg, filtered.iloc[::-1], tmp_path / "second.json"
    )
    assert first["records"] == second["records"]
    assert first["records_sha256"] == second["records_sha256"]
    assert [(r["sample_id"], r["column"]) for r in first["records"]] == [
        ("sample_0", "clean_path"), ("sample_0", "image_path"),
        ("sample_2", "clean_path"), ("sample_2", "image_path"),
    ]
    assert first["recorded_before_optimizer_step"] is True
    assert first["test_assets_opened"] == 0
    assert opened and not any("test_3" in path for path in opened)


def test_pixel_or_manifest_drift_and_retroactive_binding_are_rejected(tmp_path):
    manifest, table = _assets(tmp_path)
    filtered = table[table["split"].isin(["train", "val"]) & table["dataset"].eq("PKU37")]
    cfg = _config(tmp_path, manifest)
    initial = tmp_path / "initial.json"
    evidence = dose.create_training_asset_evidence(tmp_path, cfg, filtered, initial)
    checkpoint = tmp_path / "last.pth"
    torch.save({"epoch": 0, "model": {"weight": torch.ones(1)}, "config": cfg}, checkpoint)
    records = dose.asset_inventory(tmp_path, filtered, False)
    with pytest.raises(ValueError, match="training-start"):
        dose.bind_training_asset_evidence(
            tmp_path / "missing.json", tmp_path / "bound.json", checkpoint,
            manifest, records, 1, 1,
        )

    np.save(filtered.iloc[0].image_path, np.zeros((16, 16), np.float32))
    changed = dose.asset_inventory(tmp_path, filtered, False)
    assert dose.stable_sha(changed) != evidence["records_sha256"]
    with pytest.raises(ValueError, match="assets changed"):
        dose.bind_training_asset_evidence(
            initial, tmp_path / "bound.json", checkpoint, manifest, changed, 1, 1
        )

    # Restore the pixel, then prove manifest bytes are also immutable.
    np.save(filtered.iloc[0].image_path, np.full((16, 16), 0.2, np.float32))
    records = dose.asset_inventory(tmp_path, filtered, False)
    manifest.write_text(manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest changed"):
        dose.bind_training_asset_evidence(
            initial, tmp_path / "bound.json", checkpoint, manifest, records, 1, 1
        )


def test_last_checkpoint_is_sha_bound_to_initial_chain(tmp_path):
    manifest, table = _assets(tmp_path)
    filtered = table[table["split"].isin(["train", "val"]) & table["dataset"].eq("PKU37")]
    cfg = _config(tmp_path, manifest)
    initial = tmp_path / "training_asset_inventory_initial.json"
    dose.create_training_asset_evidence(tmp_path, cfg, filtered, initial)
    checkpoint = tmp_path / "last.pth"
    torch.save({"epoch": 0, "model": {"weight": torch.ones(1)}, "config": cfg}, checkpoint)
    records = dose.asset_inventory(tmp_path, filtered, False)
    bound_path = tmp_path / "training_asset_inventory_last.json"
    bound = dose.bind_training_asset_evidence(
        initial, bound_path, checkpoint, manifest, records, 1, 1
    )
    assert bound["checkpoint_sha256"] == dose.sha256_file(checkpoint)
    assert bound["source_initial_evidence_sha256"] == dose.sha256_file(initial)
    assert dose.audit_bound_training_asset_evidence(
        bound_path, dose.sha256_file(checkpoint), dose.sha256_file(manifest),
        records, "fixed_final", 1,
    )["records"] == records


def test_forged_training_marker_without_initial_chain_fails(tmp_path):
    forged = tmp_path / "training_asset_inventory_last.json"
    dose.write_strict_json(forged, {
        "recorded_at_training": True,
        "checkpoint_sha256": "checkpoint",
        "manifest_sha256": "manifest",
        "records": [],
        "records_sha256": dose.stable_sha([]),
        "train_val_noisy_clean_asset_sha256": dose.stable_sha([]),
        "completed_epochs": 60,
        "selection_rule": "fixed_final",
    })
    with pytest.raises(ValueError, match="initial evidence chain"):
        dose.audit_bound_training_asset_evidence(
            forged, "checkpoint", "manifest", [], "fixed_final", 60
        )


def test_cpu_smoke_writes_initial_and_last_evidence_but_default_is_opt_in(tmp_path):
    manifest, _ = _assets(tmp_path)
    cfg = _config(tmp_path, manifest)
    trainer = Trainer(cfg)
    initial = Path(cfg["train"]["output_dir"]) / "training_asset_inventory_initial.json"
    assert initial.is_file()
    assert not (initial.parent / "training_asset_inventory_last.json").exists()
    trainer.fit()
    bound = json.loads(
        (initial.parent / "training_asset_inventory_last.json").read_text(encoding="utf-8")
    )
    assert bound["completed_epochs"] == 1
    assert bound["checkpoint_sha256"] == dose.sha256_file(initial.parent / "last.pth")

    legacy = copy.deepcopy(cfg)
    legacy.pop("training_asset_evidence")
    legacy["train"]["output_dir"] = str(tmp_path / "legacy_run")
    legacy_trainer = Trainer(legacy)
    legacy_trainer.writer.close()
    assert not (tmp_path / "legacy_run" / "training_asset_inventory_initial.json").exists()


def test_existing_run_or_resume_cannot_gain_retroactive_evidence(tmp_path):
    manifest, _ = _assets(tmp_path)
    cfg = _config(tmp_path, manifest)
    run = Path(cfg["train"]["output_dir"])
    run.mkdir(parents=True)
    (run / "history.csv").write_text("epoch\n1\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="retroactive"):
        Trainer(cfg)
    (run / "history.csv").unlink()
    cfg["train"]["resume"] = str(tmp_path / "old.pth")
    with pytest.raises(ValueError, match="retroactively"):
        Trainer(cfg)


def test_reproduction_audit_allows_only_output_and_evidence_config(tmp_path):
    manifest, _ = _assets(tmp_path)
    reference = _config(tmp_path, manifest)
    reference.pop("training_asset_evidence")
    reference["train"]["output_dir"] = str(tmp_path / "old_run")
    reference["train"]["resume"] = str(tmp_path / "old_run" / "last.pth")
    reference_path = tmp_path / "old_resolved.yaml"
    save_config(reference, reference_path)
    candidate = copy.deepcopy(reference)
    candidate["train"]["output_dir"] = str(tmp_path / "new_run")
    candidate["train"]["resume"] = None
    candidate["training_asset_evidence"] = {
        "enabled": True,
        "project_root": str(tmp_path),
        "reference_resolved_config": str(reference_path),
        "allowed_semantic_differences": [
            "train.output_dir", "training_asset_evidence"
        ],
    }
    audit = dose.audit_d1_reproduction_config(tmp_path, candidate)
    assert audit["status"] == "passed"
    assert {item["path"] for item in audit["semantic_differences"]} == {
        "train.output_dir", "training_asset_evidence"
    }
    assert audit["operational_differences"] == [{
        "path": "train.resume",
        "reference": str(tmp_path / "old_run" / "last.pth"),
        "candidate": None,
        "classification": "run_control_not_training_semantics",
        "reason": "fresh reproduction must not resume the historical run",
    }]
    candidate["train"]["learning_rate"] = 9e-4
    with pytest.raises(ValueError, match="learning_rate"):
        dose.audit_d1_reproduction_config(tmp_path, candidate)


def test_trainer_binds_stale_template_hashes_from_active_lock_before_audit(tmp_path):
    manifest, _ = _assets(tmp_path)
    reference = _config(tmp_path, manifest)
    reference.pop("training_asset_evidence")
    reference.update(
        manifest_root="protocol",
        data_plan_sha256="active-data",
        label_inventory_sha256="active-label",
    )
    reference["train"].update(
        output_dir=str(tmp_path / "old_run"),
        resume=str(tmp_path / "old_run" / "last.pth"),
        fixed_epoch=1,
        checkpoint_selection_rule="fixed_final_primary",
    )
    reference_path = tmp_path / "old_resolved.yaml"
    save_config(reference, reference_path)
    lock = {
        "protocol_id": "fixture",
        "manifest_root": "protocol",
        "data_plan_sha256": "active-data",
        "label_inventory_sha256": "active-label",
        "dataset_inventory_sha256": "active-dataset",
        "split_contract_sha256": "active-contract",
        "train_positions": ["group_0"],
        "validation_positions": ["group_2"],
        "sealed_test_positions": ["group_3"],
        "input_resolution": [16, 16],
        "normalization": "fixed",
        "test_assets_opened": 0,
    }
    lock_path = tmp_path / "active_protocol_lock.json"
    dose.write_strict_json(lock_path, lock)
    candidate = copy.deepcopy(reference)
    candidate.update(data_plan_sha256="stale-data", label_inventory_sha256="stale-label")
    candidate["train"].update(
        output_dir=str(tmp_path / "new_run"), resume=None, early_stopping_patience=2
    )
    candidate["training_asset_evidence"] = {
        "enabled": True,
        "project_root": str(tmp_path),
        "protocol_lock": str(lock_path),
        "reference_resolved_config": str(reference_path),
        "allowed_semantic_differences": [
            "train.output_dir", "training_asset_evidence"
        ],
    }
    trainer = Trainer.__new__(Trainer)
    trainer.config = candidate
    trainer.output_dir = Path(candidate["train"]["output_dir"])
    trainer.output_dir.mkdir(parents=True)
    trainer._prepare_training_asset_evidence_config()
    assert candidate["data_plan_sha256"] == "active-data"
    assert candidate["label_inventory_sha256"] == "active-label"
    report = json.loads(
        (trainer.output_dir / "d1_reproduction_semantic_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert {item["path"] for item in report["protocol_lock_bindings"]} == {
        "data_plan_sha256", "label_inventory_sha256"
    }
    assert {item["path"] for item in report["semantic_differences"]} == {
        "train.output_dir", "training_asset_evidence"
    }
