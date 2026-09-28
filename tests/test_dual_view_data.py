from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sabids.data import OCTManifestDataset
from sabids.data.transforms import JointOCTTransform
from sabids.experiments.dual_view import (
    BLOCKED, audit_formal_input_evidence, build_fixed_component_inventory,
    deterministic_shuffle, evaluate_fixed_components,
)
from sabids.experiments.protocol_lock import sha256_file
from tools.prepare_dual_view_inputs import _recover_generation_time


def _write(path: Path, value: np.ndarray) -> str:
    np.save(path, value.astype(np.float32), allow_pickle=False)
    return str(path)


def test_auxiliary_gets_identical_geometry_validity_and_flip(tmp_path) -> None:
    shape = (16, 24)
    noisy = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / np.prod(shape)
    mild = 1.0 - noisy
    ones = np.ones(shape, np.float32)
    paths = {
        "image_path": _write(tmp_path / "source.npy", noisy),
        "primary_path": _write(tmp_path / "noisy.npy", noisy),
        "auxiliary_path": _write(tmp_path / "mild.npy", mild),
        "clean_path": _write(tmp_path / "clean.npy", noisy),
        "layer_mask_path": _write(tmp_path / "layer.npy", ones),
        "vessel_mask_path": _write(tmp_path / "vessel.npy", ones),
        "label_valid_mask_path": _write(tmp_path / "label_valid.npy", ones),
        "vessel_valid_mask_path": _write(tmp_path / "vessel_valid.npy", ones),
        "spatial_valid_mask_path": _write(tmp_path / "spatial_valid.npy", ones),
    }
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "sample_id": "s1", "group_id": "g1", "dataset": "synthetic", "split": "train", **paths
    }]).to_csv(manifest, index=False)
    dataset = OCTManifestDataset(
        manifest, "train", JointOCTTransform(shape, training=True, horizontal_flip=1.0),
        sample_repeat=False, root=tmp_path, image_column="primary_path",
        auxiliary_image_column="auxiliary_path", auxiliary_mode="column",
        pretransformed_model_grid=True, deterministic_augmentation_seed=42,
    )
    item = dataset[0]
    assert item["augmentation_flip"] is True
    assert item["image"].shape == item["auxiliary_image"].shape == item["valid_mask"].shape
    assert np.array_equal(item["image"][0].numpy(), np.fliplr(noisy))
    assert np.array_equal(item["auxiliary_image"][0].numpy(), np.fliplr(mild))


def test_legacy_dataset_does_not_emit_dual_view_keys(tmp_path) -> None:
    shape = (8, 12)
    image = np.ones(shape, np.float32)
    manifest = tmp_path / "legacy.csv"
    pd.DataFrame([{
        "sample_id": "legacy", "group_id": "g", "dataset": "legacy", "split": "train",
        "image_path": _write(tmp_path / "image.npy", image),
        "clean_path": "", "layer_mask_path": "", "vessel_mask_path": "",
        "label_valid_mask_path": "", "vessel_valid_mask_path": "",
    }]).to_csv(manifest, index=False)
    dataset = OCTManifestDataset(
        manifest, "train", JointOCTTransform(shape, training=False),
        sample_repeat=False, root=tmp_path, load_segmentation_labels=False,
    )
    item = dataset[0]
    assert "auxiliary_image" not in item
    assert "has_auxiliary_image" not in item


def test_shuffle_is_deterministic_cross_group_and_breaks_pairing() -> None:
    rows = [
        {"sample_id": f"{split}_s{i}", "group_id": f"g{i % 3}", "split": split}
        for split in ("train", "val") for i in range(6)
    ]
    first = deterministic_shuffle(rows, 42)
    second = deterministic_shuffle(rows, 42)
    assert first == second
    by_id = {row["sample_id"]: row for row in rows}
    assert all(by_id[source]["split"] == by_id[target]["split"] for source, target in first.items())
    assert all(by_id[source]["group_id"] != by_id[target]["group_id"] for source, target in first.items())


def test_shuffle_supports_highly_imbalanced_groups_without_a_bijection() -> None:
    rows = [
        {"sample_id": f"major_{index}", "group_id": "major", "split": "train"}
        for index in range(9)
    ] + [{"sample_id": "minor_0", "group_id": "minor", "split": "train"}]
    mapping = deterministic_shuffle(rows, 42)
    by_id = {row["sample_id"]: row for row in rows}
    assert len(mapping) == len(rows)
    assert all(
        by_id[source]["group_id"] != by_id[target]["group_id"]
        for source, target in mapping.items()
    )
    # Reuse is expected and valid because a 9:1 cross-group bijection cannot exist.
    assert len(set(mapping.values())) < len(mapping)


def test_shuffle_rejects_ambiguous_duplicate_sample_ids() -> None:
    rows = [
        {"sample_id": "same", "group_id": "g1", "split": "train"},
        {"sample_id": "same", "group_id": "g2", "split": "train"},
    ]
    with pytest.raises(ValueError, match="globally unique"):
        deterministic_shuffle(rows, 42)


def test_interrupted_preparation_reuses_one_verified_timestamp(tmp_path) -> None:
    evidence = {
        "protocol_id": "p",
        "sha256": {
            "split_contract": "split", "d1_checkpoint": "checkpoint",
            "checkpoint_binding": "binding",
        },
    }
    sidecar = tmp_path / "residuals/train/sample/noisy_minus_mild.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({
        "version": "noisy-mild-dual-view-v1", "protocol_id": "p",
        "split_contract_sha256": "split",
        "denoiser_checkpoint_sha256": "checkpoint",
        "checkpoint_binding_sha256": "binding",
        "created_at": "2026-09-28T00:00:00+00:00",
        "test_assets_opened": 0,
    }), encoding="utf-8")
    assert _recover_generation_time(tmp_path, evidence) == "2026-09-28T00:00:00+00:00"
    changed = json.loads(sidecar.read_text(encoding="utf-8"))
    changed["checkpoint_binding_sha256"] = "wrong"
    sidecar.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(FileExistsError, match="identity conflict"):
        _recover_generation_time(tmp_path, evidence)


def test_missing_formal_evidence_fails_closed(tmp_path) -> None:
    paths = [tmp_path / f"missing_{index}.json" for index in range(6)]
    result = audit_formal_input_evidence(*paths)
    assert result["status"] == "blocked"
    assert result["message"] == BLOCKED
    assert result["test_assets_opened"] == 0


def test_formal_d1_evidence_chain_passes_without_any_d2_asset(tmp_path) -> None:
    split = tmp_path / "split.yaml"
    split.write_text("protocol_id: p\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"trusted-checkpoint-fixture")
    inventory = tmp_path / "training_asset_inventory_initial.json"
    inventory.write_text(json.dumps({
        "recorded_at_training": True,
        "recorded_before_optimizer_step": True,
    }), encoding="utf-8")
    lock = tmp_path / "active_protocol_lock.json"
    lock.write_text(json.dumps({
        "protocol_id": "p", "split_contract_sha256": sha256_file(split),
    }), encoding="utf-8")
    binding = tmp_path / "checkpoint_binding.json"
    binding.write_text(json.dumps({
        "status": "passed", "selection_rule": "best_validation_psnr",
        "checkpoint_sha256": sha256_file(checkpoint), "test_assets_opened": 0,
        "source_sha256": {
            "initial_inventory": sha256_file(inventory),
            "protocol_lock": sha256_file(lock), "split_contract": sha256_file(split),
        },
    }), encoding="utf-8")
    registry = tmp_path / "preparation_registry.json"
    registry.write_text(json.dumps({
        "preflight": {
            "mode": "formal", "status": "passed",
            "selection_rule": "best_validation_psnr",
        },
        "identity": {
            "protocol_id": "p", "checkpoint_sha256": sha256_file(checkpoint),
            "curves": ["d1"], "alphas": [0, 0.25, 1],
        },
        "test_assets_opened": 0,
    }), encoding="utf-8")
    result = audit_formal_input_evidence(
        lock, split, checkpoint, binding, inventory, registry
    )
    assert result["status"] == "passed"
    assert result["protocol_id"] == "p"
    assert not any("d2" in key.lower() for key in result["checks"])


def _sample(sample_id: str, split: str, offset: int = 0) -> dict:
    vessel = np.zeros((20, 24), bool)
    vessel[3:6, 4 + offset:7 + offset] = True
    vessel[10:15, 12:17] = True
    layer = np.ones_like(vessel)
    noisy = np.full(vessel.shape, 0.6, np.float32)
    noisy[vessel] = 0.2
    return {
        "sample_id": sample_id, "group_id": f"g_{sample_id}", "split": split,
        "vessel": vessel, "layer": layer, "valid": np.ones_like(vessel),
        "noisy": noisy, "clean": None,
    }


def test_fixed_component_membership_does_not_depend_on_arm_input() -> None:
    inventory = build_fixed_component_inventory(
        [_sample("train", "train")], [_sample("val", "val")]
    )
    val = _sample("val", "val")
    rows_b0 = evaluate_fixed_components("val", val["vessel"], val["vessel"], val["valid"], inventory)
    rows_b3 = evaluate_fixed_components("val", np.zeros_like(val["vessel"]), val["vessel"], val["valid"], inventory)
    assert [(row["sample_id"], row["component_id"], row["size_bin"], row["contrast_bin"]) for row in rows_b0] == [
        (row["sample_id"], row["component_id"], row["size_bin"], row["contrast_bin"]) for row in rows_b3
    ]
    assert all(row["recall_at_050"] == 1 for row in rows_b0)
    assert all(row["completely_missed"] == 1 for row in rows_b3)
    assert all(row["clean_contrast_bin"] == "unknown" for row in rows_b3)
    changed = _sample("val", "val", offset=1)
    with pytest.raises(ValueError, match="GT component changed"):
        evaluate_fixed_components("val", changed["vessel"], changed["vessel"], changed["valid"], inventory)
