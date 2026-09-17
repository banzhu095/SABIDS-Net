"""Fabricated provenance fixtures test gates, never scientific checkpoints."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sabids.config import load_config, save_config
from sabids.engine.trainer import build_model
from sabids.experiments import dose_response as dose

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence(tmp_path):
    protocol = tmp_path / "Manifests/fixture_protocol"
    protocol.mkdir(parents=True)
    source = tmp_path / "assets"
    source.mkdir()
    rows, labels = [], []
    for index, split in enumerate(("train", "val", "test")):
        group = f"pku_{index + 1:04d}"
        row = {"sample_id": f"{group}_f01", "group_id": group, "patient_id": group,
               "dataset": "PKU37", "split": split}
        labelrow = {"group_id": group}
        for role in ("image", "clean", "layer_mask", "vessel_mask"):
            p = source / f"{index}_{role}.npy" if split != "test" else tmp_path / "sealed" / f"{index}_{role}.npy"
            if split != "test":
                np.save(p, np.ones((24, 32), np.float32) * (.5 if role in {"image", "clean"} else 1))
            row[f"{role}_path"] = str(p)
            if role in {"layer_mask", "vessel_mask"}:
                task = role.split("_")[0]
                labelrow[f"{task}_path"] = str(p)
                labelrow[f"{task}_sha256"] = dose.sha256_file(p) if split != "test" else "sealed-metadata-only"
        rows.append(row)
        labels.append(labelrow)
    allrows = pd.DataFrame(rows)
    joint = allrows[allrows.split.ne("test")].copy()
    sealed = allrows[allrows.split.eq("test")].copy()
    inventory = allrows[["group_id", "split", "clean_path", "layer_mask_path", "vessel_mask_path"]]
    labeltable = pd.DataFrame(labels)
    contract = tmp_path / "contract.yaml"
    contract.write_text("protocol_id: fixture_protocol\nvalidation_positions: [pku_0002]\ntest_positions: [pku_0003]\n")
    for name, df in {"train_denoise": joint, "train_segment": joint, "train_joint": joint,
                     "test_sealed": sealed, "label_inventory": labeltable, "dataset_inventory": inventory}.items():
        df.to_csv(protocol / f"{name}.csv", index=False)
    sha_table = lambda df: hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    lock = {"protocol_id": "fixture_protocol", "manifest_root": str(protocol),
            "data_plan_sha256": sha_table(allrows), "label_inventory_sha256": sha_table(labeltable),
            "dataset_inventory_sha256": sha_table(inventory), "split_contract_sha256": dose.sha256_file(contract),
            "train_positions": ["pku_0001"], "validation_positions": ["pku_0002"],
            "sealed_test_positions": ["pku_0003"], "input_resolution": [32, 32],
            "git_commit_at_d1_start": "fabricated-unit-fixture", "test_assets_opened": 0}
    lp = protocol / "active_protocol_lock.json"
    dose.write_strict_json(lp, lock)
    dose.write_strict_json(protocol / "protocol_audit.json", {**lock, "status": "passed"})
    cfg = load_config(ROOT / "configs/base.yaml")
    cfg["model"].update(channels=[4, 8], encoder_depths=[1, 1], decoder_depth=1, interaction_levels=[1])
    cfg["train"].update(stage="denoise", epochs=60, monitor="psnr", output_dir=str(tmp_path / "formal_run"))
    cfg["loss"]["restoration_mode"] = "structure_d1"
    cfg["data"].update(manifest=str(protocol / "train_denoise.csv"), root=str(tmp_path), target_size=[32, 32], normalization="fixed")
    cfg["runtime"].update(active_protocol_lock=lock,
        manifest_sha256=dose.sha256_file(protocol / "train_denoise.csv"),
        effective_split_sha256=hashlib.sha256(b"train:pku_0001\nval:pku_0002").hexdigest(),
        train_val_noisy_clean_asset_sha256=dose.stable_sha(dose.asset_inventory(tmp_path, joint, False)))
    run = tmp_path / "formal_run"
    run.mkdir()
    save_config(cfg, run / "resolved_config.yaml")
    cp = run / "last.pth"
    torch.save({"model": build_model(cfg).state_dict(), "config": cfg, "epoch": 59}, cp)
    pd.DataFrame([{"epoch": i, "val_psnr": 30.} for i in range(1, 61)]).to_csv(run / "history.csv", index=False)
    return {"root": tmp_path, "checkpoint": cp, "lock_path": lp, "contract": contract,
            "cfg": cfg, "lock": lock, "joint": joint, "protocol": protocol}


def audit(e):
    return dose.formal_preflight(e["root"], str(e["checkpoint"]), str(e["lock_path"]),
                                 str(e["contract"]), "fixed_final")


def rewrite_checkpoint(e, cfg):
    raw = torch.load(e["checkpoint"], map_location="cpu", weights_only=False)
    raw["config"] = cfg
    torch.save(raw, e["checkpoint"])
    save_config(cfg, e["checkpoint"].parent / "resolved_config.yaml")


def test_formal_pass_never_opens_sealed_assets(evidence, monkeypatch):
    original = dose.sha256_file
    def guarded(path):
        assert "sealed" not in Path(path).parts, f"Forbidden test asset opened: {path}"
        return original(path)
    monkeypatch.setattr(dose, "sha256_file", guarded)
    result = audit(evidence)
    assert result["status"] == "passed", result
    assert result["test_assets_opened"] == 0
    assert result["d1_epoch"] == 60


@pytest.mark.parametrize("kind,match", [
    ("missing_checkpoint", "Missing checkpoint"), ("auto", "auto forbidden"),
    ("smoke", "Smoke/pilot"), ("missing_lock", "missing/nonunique"),
    ("duplicate_lock", "missing/nonunique"), ("protocol", "protocol_id mismatch"),
    ("data_sha", "data_plan_sha256 mismatch"), ("missing_manifest", "Missing checkpoint manifest"),
    ("manifest_sha", "manifest SHA"), ("untraceable", "untraceable"),
    ("split_leak", "leakage"), ("label_sha", "label_inventory.csv SHA"),
    ("pixel_sha_missing", "pixel fingerprint"), ("pixel_drift", "pixel fingerprint"),
    ("short_budget", "Smoke/short"), ("selection_missing", "Freeze selection"),
    ("resolved_missing", "resolved_config"), ("contract_sha", "contract SHA"),
    ("wrong_source_asset", "D1 manifest row differs"),
    ("incomplete_history", "complete fixed budget"),
])
def test_formal_refuses_invalid_evidence(evidence, kind, match):
    e = evidence
    cfg = copy.deepcopy(e["cfg"])
    cp, lp, rule = str(e["checkpoint"]), str(e["lock_path"]), "fixed_final"
    if kind == "missing_checkpoint":
        cp = str(e["root"] / "absent.pth")
    elif kind == "auto":
        cp = "auto"
    elif kind == "smoke":
        new = e["root"] / "smoke.pth"
        new.write_bytes(e["checkpoint"].read_bytes())
        cp = str(new)
    elif kind == "missing_lock":
        e["lock_path"].unlink()
    elif kind == "duplicate_lock":
        dose.write_strict_json(e["root"] / "Manifests/other/active_protocol_lock.json", e["lock"])
    elif kind in {"protocol", "data_sha"}:
        cfg["runtime"]["active_protocol_lock"]["protocol_id" if kind == "protocol" else "data_plan_sha256"] = "different"
        rewrite_checkpoint(e, cfg)
    elif kind == "missing_manifest":
        Path(cfg["data"]["manifest"]).unlink()
    elif kind == "manifest_sha":
        cfg["runtime"]["manifest_sha256"] = "wrong"
        rewrite_checkpoint(e, cfg)
    elif kind == "untraceable":
        torch.save({"model": {"random": torch.ones(1)}}, e["checkpoint"])
    elif kind == "split_leak":
        table = e["joint"].copy()
        table.loc[table.split.eq("val"), "group_id"] = "pku_0001"
        table.to_csv(cfg["data"]["manifest"], index=False)
        cfg["runtime"]["manifest_sha256"] = dose.sha256_file(cfg["data"]["manifest"])
        rewrite_checkpoint(e, cfg)
    elif kind == "label_sha":
        p = e["protocol"] / "label_inventory.csv"
        p.write_text(p.read_text() + "\nanything\n")
    elif kind == "pixel_sha_missing":
        cfg["runtime"].pop("train_val_noisy_clean_asset_sha256")
        rewrite_checkpoint(e, cfg)
    elif kind == "pixel_drift":
        np.save(e["joint"].iloc[0].image_path, np.zeros((24, 32), np.float32))
    elif kind == "short_budget":
        cfg["train"]["epochs"] = 2
        rewrite_checkpoint(e, cfg)
    elif kind == "selection_missing":
        rule = None
    elif kind == "resolved_missing":
        (e["checkpoint"].parent / "resolved_config.yaml").unlink()
    elif kind == "contract_sha":
        e["contract"].write_text("different")
    elif kind == "wrong_source_asset":
        table = e["joint"].copy()
        table.loc[0, "image_path"] = table.loc[1, "image_path"]
        table.to_csv(cfg["data"]["manifest"], index=False)
        cfg["runtime"]["manifest_sha256"] = dose.sha256_file(cfg["data"]["manifest"])
        rewrite_checkpoint(e, cfg)
    elif kind == "incomplete_history":
        pd.DataFrame([{"epoch": 60, "val_psnr": 30.}]).to_csv(e["checkpoint"].parent / "history.csv", index=False)
    result = dose.formal_preflight(e["root"], cp, lp, str(e["contract"]), rule)
    assert result["status"] == "blocked"
    assert any(match in issue for issue in result["issues"]), result
    assert result["test_assets_opened"] == 0


def test_segmentation_manifest_cannot_override_locked_cohort(evidence):
    table = evidence["joint"].copy()
    table.loc[0, "image_path"] = "unregistered_asset.npy"
    table.to_csv(evidence["protocol"] / "train_segment.csv", index=False)
    result = audit(evidence)
    assert result["status"] == "blocked"
    assert "differs from locked data plan" in result["issues"][0]


def test_best_selection_is_frozen_and_sha_bound(evidence):
    e = evidence
    cp = e["checkpoint"].with_name("best.pth")
    raw = torch.load(e["checkpoint"], map_location="cpu", weights_only=False)
    raw["epoch"] = 2
    torch.save(raw, cp)
    pd.DataFrame([{"epoch": i, "val_psnr": 32. if i == 3 else 30.} for i in range(1, 61)]).to_csv(cp.parent / "history.csv", index=False)
    dose.write_strict_json(cp.parent / "run_metadata.json", {"best_checkpoint_sha256": dose.sha256_file(cp)})
    result = dose.formal_preflight(e["root"], str(cp), str(e["lock_path"]), str(e["contract"]), "best_validation_psnr")
    assert result["status"] == "passed", result
    dose.write_strict_json(cp.parent / "run_metadata.json", {"best_checkpoint_sha256": "incorrect"})
    result = dose.formal_preflight(e["root"], str(cp), str(e["lock_path"]), str(e["contract"]), "best_validation_psnr")
    assert result["status"] == "blocked" and "provenance SHA" in result["issues"][0]
