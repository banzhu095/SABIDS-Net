from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from sabids.experiments.dual_task_adaptive import audit_adaptive_inputs, sha256_file


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def valid_fixture(tmp_path: Path) -> dict:
    d2_checkpoint, coarse_checkpoint = tmp_path / "d2.pth", tmp_path / "coarse.pth"
    d2_checkpoint.write_bytes(b"d2")
    coarse_checkpoint.write_bytes(b"coarse")
    d2_hash, coarse_hash = sha256_file(d2_checkpoint), sha256_file(coarse_checkpoint)
    d2_binding, coarse_binding = tmp_path / "d2.json", tmp_path / "coarse.json"
    write_json(d2_binding, {"status": "passed", "checkpoint_sha256": d2_hash, "test_assets_opened": 0})
    write_json(coarse_binding, {
        "status": "passed", "checkpoint_sha256": coarse_hash,
        "selection_rule": "best_validation_vessel_soft_dice", "completed_epochs": 20,
        "test_assets_opened": 0,
    })
    inventory = tmp_path / "inventory.json"
    write_json(inventory, {"recorded_at_training": True})
    lock = tmp_path / "lock.json"
    write_json(lock, {"protocol_id": "pku37_binary_v3", "test_assets_opened": 0})
    split = tmp_path / "split.yaml"
    split.write_text(yaml.safe_dump({"test_positions": ["pku0024", "pku0031", "pku0037", "pku0039"]}), encoding="utf-8")
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([
        {"sample_id": "a", "group_id": "pku_0001", "split": "train",
         "layer_mask_path": "layer.png", "vessel_mask_path": "vessel.png"},
        {"sample_id": "b", "group_id": "pku_0006", "split": "val",
         "layer_mask_path": "layer.png", "vessel_mask_path": "vessel.png"},
    ]).to_csv(manifest, index=False)
    return {
        "seed": 42,
        "data": {"manifest": str(manifest)},
        "evaluation": {"threshold": 0.5, "layer_threshold": 0.5, "vessel_threshold": 0.5},
        "dual_task_adaptive": {
            "coarse_strength": 0.25,
            "anchors": {"d2_checkpoint": str(d2_checkpoint), "d2_checkpoint_sha256": d2_hash,
                        "coarse_checkpoint": str(coarse_checkpoint), "coarse_checkpoint_sha256": coarse_hash},
            "evidence": {
                "d2_binding": str(d2_binding), "d2_binding_sha256": sha256_file(d2_binding),
                "coarse_binding": str(coarse_binding),
                "d2_inventory": str(inventory), "d2_inventory_sha256": sha256_file(inventory),
                "protocol_lock": str(lock), "protocol_lock_sha256": sha256_file(lock),
                "split_contract": str(split), "split_contract_sha256": sha256_file(split),
                "manifest_sha256": sha256_file(manifest),
            },
        },
    }


def test_metadata_only_preflight_passes_without_opening_assets(tmp_path: Path) -> None:
    report = audit_adaptive_inputs(valid_fixture(tmp_path), tmp_path)
    assert report["status"] == "passed"
    assert report["test_assets_opened"] == 0


def test_preflight_fails_closed_on_changed_checkpoint(tmp_path: Path) -> None:
    config = valid_fixture(tmp_path)
    Path(config["dual_task_adaptive"]["anchors"]["d2_checkpoint"]).write_bytes(b"changed")
    report = audit_adaptive_inputs(config, tmp_path)
    assert report["status"] == "blocked"
    assert "BLOCKED" in report["blocked_message"]


def test_preflight_rejects_sealed_test_group_in_manifest(tmp_path: Path) -> None:
    config = valid_fixture(tmp_path)
    manifest = Path(config["data"]["manifest"])
    table = pd.read_csv(manifest)
    table.loc[len(table)] = ["x", "pku_0024", "val", "layer.png", "vessel.png"]
    table.to_csv(manifest, index=False)
    config["dual_task_adaptive"]["evidence"]["manifest_sha256"] = sha256_file(manifest)
    report = audit_adaptive_inputs(config, tmp_path)
    assert report["status"] == "blocked"
    assert not report["checks"]["manifest_excludes_test_groups"]
