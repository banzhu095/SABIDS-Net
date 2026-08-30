from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.data.io import read_gray, write_gray
from sabids.data.transforms import JointOCTTransform
from sabids.losses import SABIDSLoss
from sabids.models import SABIDSNet
from tools.prepare_input_factorial import atomic_save_npy, audit_d0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _small_model() -> SABIDSNet:
    return SABIDSNet(
        channels=(4, 8), encoder_depths=(1, 1), decoder_depth=1,
        interaction_levels=(1, 0), enable_denoise_to_seg=False,
        enable_seg_to_denoise=False,
    )


def test_float_cache_is_not_renormalized_and_rejects_invalid_values(tmp_path: Path):
    expected = np.linspace(0.11, 0.89, 20, dtype=np.float32).reshape(4, 5)
    cache = tmp_path / "image.npy"
    atomic_save_npy(cache, expected)
    actual = read_gray(cache)
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, expected)

    for name, value in (
        ("integer.npy", np.zeros((2, 2), dtype=np.uint8)),
        ("nonfinite.npy", np.array([[np.nan]], dtype=np.float32)),
        ("outside.npy", np.array([[1.1]], dtype=np.float32)),
        ("three_dimensional.npy", np.zeros((1, 2, 2), dtype=np.float32)),
    ):
        path = tmp_path / name
        np.save(path, value)
        with pytest.raises(ValueError):
            read_gray(path)


def test_dataset_uses_selected_cache_column_with_same_spatial_transform(tmp_path: Path):
    noisy = np.arange(24, dtype=np.float32).reshape(4, 6) / 24.0
    denoised = noisy * 0.5 + 0.2
    np.save(tmp_path / "noisy.npy", noisy)
    np.save(tmp_path / "denoised.npy", denoised)
    mask = np.zeros((4, 6), dtype=np.float32)
    mask[:, :2] = 1.0
    write_gray(tmp_path / "layer.png", mask)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{
        "sample_id": "s0", "group_id": "g0", "dataset": "synthetic",
        "split": "train", "image_path": "noisy.npy",
        "noisy_cache_path": "noisy.npy", "denoised_cache_path": "denoised.npy",
        "layer_mask_path": "layer.png",
    }]).to_csv(manifest, index=False)
    transform = JointOCTTransform(
        target_size=(4, 6), training=True, horizontal_flip=1.0,
        normalization="fixed", gamma_range=(1.0, 1.0),
        contrast_range=(1.0, 1.0), speckle_std=0.0, blur_probability=0.0,
    )
    noisy_item = OCTManifestDataset(
        manifest, "train", transform, sample_repeat=False, root=tmp_path,
        image_column="noisy_cache_path",
    )[0]
    denoised_item = OCTManifestDataset(
        manifest, "train", transform, sample_repeat=False, root=tmp_path,
        image_column="denoised_cache_path",
    )[0]
    np.testing.assert_allclose(noisy_item["image"][0].numpy(), np.fliplr(noisy), atol=2e-8)
    np.testing.assert_allclose(denoised_item["image"][0].numpy(), np.fliplr(denoised), atol=2e-8)
    torch.testing.assert_close(noisy_item["layer_mask"], denoised_item["layer_mask"])


def test_input_segment_parameter_boundary_is_identical_and_excludes_denoising():
    state = _small_model().state_dict()
    trainable_sets = []
    for _ in range(2):
        model = _small_model()
        model.load_state_dict(state)
        model.set_train_stage("input_segment")
        trainable = {name for name, value in model.named_parameters() if value.requires_grad}
        trainable_sets.append(trainable)
        assert any(name.startswith("stem") for name in trainable)
        assert any(name.startswith("decoders.layer") for name in trainable)
        assert any(name.startswith("decoders.vessel") for name in trainable)
        assert not any(name.startswith("decoders.denoise") for name in trainable)
        assert not any(name.startswith("adapters.denoise") for name in trainable)
        assert not any(name.startswith("residual_head") for name in trainable)
        assert not any(name.startswith("interactions") for name in trainable)
    assert trainable_sets[0] == trainable_sets[1]


def test_stage1_denoise_only_matches_disabled_interaction_full_forward():
    torch.manual_seed(17)
    model = _small_model().eval()
    image = torch.rand(1, 1, 16, 24)
    with torch.no_grad():
        full = model(image, return_features=False, return_auxiliary=False)
        compact = model.forward_denoise_only(image)
    for key in ("denoised_raw", "denoised", "residual"):
        torch.testing.assert_close(compact[key], full[key], rtol=0.0, atol=0.0)
    assert "layer_logits" not in compact and "vessel_logits" not in compact


def test_fold_stage1_uses_memory_safe_equivalent_batch_and_no_interaction():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/current/stage1_denoise_fold0.yaml")
    assert config["train"]["batch_size"] == 1
    assert config["train"]["gradient_accumulation_steps"] == 2
    assert config["train"]["amp"] is True
    assert config["loss"]["weights"]["identity"] > 0
    assert config["model"]["enable_seg_to_denoise"] is False
    assert config["model"]["enable_denoise_to_seg"] is False


def test_input_segment_loss_has_no_denoising_rmac_or_pseudo_terms():
    shape = (1, 1, 8, 8)
    layer_logits = torch.zeros(shape, requires_grad=True)
    vessel_logits = torch.zeros(shape, requires_grad=True)
    output = {
        "denoised_raw": torch.rand(shape, requires_grad=True),
        "residual": torch.rand(shape, requires_grad=True),
        "layer_logits": layer_logits, "vessel_logits": vessel_logits,
        "layer_prob": torch.sigmoid(layer_logits), "vessel_prob": torch.sigmoid(vessel_logits),
        "boundary_logits": torch.zeros((1, 2, 8, 8)), "auxiliary": [],
    }
    batch = {
        "image": torch.rand(shape), "image_weak": torch.rand(shape),
        "clean": torch.rand(shape), "repeat": torch.rand(shape),
        "layer_mask": torch.ones(shape), "vessel_mask": torch.zeros(shape),
        "valid_mask": torch.ones(shape), "label_valid_mask": torch.ones(shape),
        "vessel_valid_mask": torch.ones(shape),
        "has_clean": torch.tensor([True]), "has_repeat": torch.tensor([True]),
        "has_layer": torch.tensor([True]), "has_vessel": torch.tensor([True]),
        "is_clean": torch.tensor([False]),
    }
    criterion = SABIDSLoss({"weights": {
        "reconstruction": 1.0, "residual": 1.0, "layer": 1.0,
        "vessel": 1.0, "containment": 0.1, "rmac": 1.0,
        "pseudo": 1.0, "identity": 1.0,
    }})
    losses = criterion(output, batch, stage="input_segment", repeat_output=output, teacher_output=output)
    for name in ("reconstruction", "residual", "rmac", "pseudo", "identity"):
        assert float(losses[name].detach()) == 0.0
        assert f"{name}_weighted" not in losses
    losses["total"].backward()
    assert layer_logits.grad is not None
    assert vessel_logits.grad is not None
    assert output["denoised_raw"].grad is None or not bool(
        (output["denoised_raw"].grad != 0).any()
    )


def test_input_configs_are_paired_and_use_fixed_full_budget():
    root = Path(__file__).resolve().parents[1]
    noisy = load_config(root / "configs/current/input_noisy_fold0.yaml")
    denoised = load_config(root / "configs/current/input_denoised_fold0.yaml")
    assert noisy["data"]["input_column"] == "noisy_cache_path"
    assert denoised["data"]["input_column"] == "denoised_cache_path"
    assert noisy["train"]["stage"] == denoised["train"]["stage"] == "input_segment"
    assert noisy["train"]["epochs"] == denoised["train"]["epochs"] == 60
    assert noisy["train"]["pretrained"] == denoised["train"]["pretrained"]
    assert noisy["model"]["d2s_enabled"] is noisy["model"]["s2d_enabled"] is False
    assert denoised["model"]["d2s_enabled"] is denoised["model"]["s2d_enabled"] is False
    for key in ("reconstruction", "residual", "identity", "rmac", "pseudo"):
        assert noisy["loss"]["weights"][key] == denoised["loss"]["weights"][key] == 0.0


def test_d0_audit_proves_manifest_identity_and_blocks_group_leakage(tmp_path: Path):
    manifest_root = tmp_path / "Manifests"
    (manifest_root / "joint_folds").mkdir(parents=True)
    (manifest_root / "segmentation_folds").mkdir(parents=True)
    write_gray(tmp_path / "train_noisy.png", np.zeros((2, 3), dtype=np.float32))
    write_gray(tmp_path / "train_clean.png", np.ones((2, 3), dtype=np.float32))
    stage1_manifest = manifest_root / "joint_folds/manifest_joint_fold0.csv"
    pd.DataFrame([{
        "sample_id": "train0", "group_id": "train_group", "dataset": "x",
        "split": "train", "image_path": "train_noisy.png", "clean_path": "train_clean.png",
    }, {
        # A nonexistent test asset must never be opened by the audit.
        "sample_id": "test0", "group_id": "test_group", "dataset": "x",
        "split": "test", "image_path": "must_not_be_opened.png", "clean_path": "",
    }]).to_csv(stage1_manifest, index=False)
    segmentation_manifest = manifest_root / "segmentation_folds/manifest_seg_fold0.csv"
    pd.DataFrame([{
        "sample_id": "val0", "group_id": "val_group", "dataset": "x",
        "split": "val", "image_path": "must_not_be_opened_val.png",
    }, {
        "sample_id": "test0", "group_id": "test_group", "dataset": "x",
        "split": "test", "image_path": "must_not_be_opened_test.png",
    }]).to_csv(segmentation_manifest, index=False)

    def save_checkpoint(path: Path, train_groups: list[str]) -> None:
        torch.save({
            "epoch": 2,
            "config": {"runtime": {
                "manifest_sha256": _sha256(stage1_manifest),
                "effective_split_sha256": "known-split-fingerprint",
                "effective_groups": {"train": train_groups},
            }},
        }, path)

    checkpoint = tmp_path / "stage1.pth"
    save_checkpoint(checkpoint, ["train_group"])
    passed = audit_d0(tmp_path, checkpoint, tmp_path / "passed.json")
    assert passed["status"] == "passed"
    assert passed["test_assets_opened"] == 0
    assert passed["current_train_val_asset_count"] == 2

    save_checkpoint(checkpoint, ["train_group", "val_group"])
    blocked = audit_d0(tmp_path, checkpoint, tmp_path / "blocked.json")
    assert blocked["status"] == "blocked"
    assert blocked["held_out_overlap_with_d0_train"] == ["val_group"]


def test_d0_audit_accepts_only_sha_linked_complete_run_metadata(tmp_path: Path):
    manifest_root = tmp_path / "Manifests"
    (manifest_root / "joint_folds").mkdir(parents=True)
    (manifest_root / "segmentation_folds").mkdir(parents=True)
    write_gray(tmp_path / "noisy.png", np.zeros((2, 2), dtype=np.float32))
    write_gray(tmp_path / "clean.png", np.ones((2, 2), dtype=np.float32))
    stage1 = manifest_root / "joint_folds/manifest_joint_fold0.csv"
    pd.DataFrame([{
        "sample_id": "s", "group_id": "train_group", "dataset": "x",
        "split": "train", "image_path": "noisy.png", "clean_path": "clean.png",
    }]).to_csv(stage1, index=False)
    pd.DataFrame([{
        "sample_id": "v", "group_id": "val_group", "dataset": "x",
        "split": "val", "image_path": "not_opened.png",
    }]).to_csv(manifest_root / "segmentation_folds/manifest_seg_fold0.csv", index=False)
    checkpoint = tmp_path / "best.pth"
    torch.save({"epoch": 1, "config": {"runtime": {}, "data": {}}}, checkpoint)
    metadata = {
        "best_checkpoint_sha256": _sha256(checkpoint),
        "manifest_sha256": _sha256(stage1),
        "effective_split_sha256": "linked-split",
        "effective_groups": {"train": ["train_group"], "val": []},
    }
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    passed = audit_d0(tmp_path, checkpoint, tmp_path / "linked.json")
    assert passed["status"] == "passed"
    assert passed["provenance_source"] == "sha256_linked_run_metadata"

    metadata["best_checkpoint_sha256"] = "not-the-checkpoint"
    (tmp_path / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    blocked = audit_d0(tmp_path, checkpoint, tmp_path / "mismatch.json")
    assert blocked["status"] == "blocked"
    assert blocked["sidecar_status"] == "checkpoint_sha_mismatch"
