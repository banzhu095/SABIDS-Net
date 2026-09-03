from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from sabids.data.io import write_gray
from tools.export_input_oracle_visualizations import (
    _archive_output,
    _materialize_sample,
    overlay_mask,
)


def test_overlay_uses_green_layer_red_vessel_and_preserves_background() -> None:
    image = np.full((3, 4), 0.5, np.float32)
    layer = np.zeros_like(image, bool); layer[0, 0] = True; layer[1, 1] = True
    vessel = np.zeros_like(image, bool); vessel[1, 1] = True; vessel[2, 2] = True
    result = overlay_mask(image, layer, vessel)
    assert np.array_equal(result[0, 3], np.array((0.5, 0.5, 0.5), np.float32))
    assert result[0, 0, 1] > result[0, 0, 0] == result[0, 0, 2]
    assert result[2, 2, 0] > result[2, 2, 1] == result[2, 2, 2]
    assert result[1, 1, 0] > result[1, 1, 1] > result[1, 1, 2]


def test_archive_existing_preserves_previous_output(tmp_path: Path) -> None:
    output = tmp_path / "result"
    output.mkdir(); (output / "old.txt").write_text("old")
    archived = _archive_output(output, archive_existing=True)
    assert archived is not None and (archived / "old.txt").read_text() == "old"
    assert output.is_dir() and not any(output.iterdir())


def test_materialize_sample_exports_float_probabilities_and_exact_overlays(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "result"
    source.mkdir(); sample = "pku_0001_f01"
    image = np.full((8, 9), 0.4, np.float32)
    layer = np.zeros((8, 9), np.float32); layer[2:7, 1:8] = 1
    vessel = np.zeros((8, 9), np.float32); vessel[4:6, 3:6] = 1
    for suffix, value in {
        "noisy.png": image, "layer_prob.png": layer * 0.8, "vessel_prob.png": vessel * 0.9,
        "layer_mask.png": layer, "vessel_mask.png": vessel, "layer_gt.png": layer,
        "vessel_gt.png": vessel,
    }.items():
        write_gray(source / f"{sample}_{suffix}", value)
    np.save(source / f"{sample}_layer_prob_float32.npy", layer * 0.812345, allow_pickle=False)
    np.save(source / f"{sample}_vessel_prob_float32.npy", vessel * 0.912345, allow_pickle=False)
    paths, missing = _materialize_sample(source, destination, {
        "sample_id": sample, "group_id": "pku_0001", "fold": 0, "arm": "I-NOISY",
        "expected_original_height": 8, "expected_original_width": 9,
    })
    assert not missing
    assert set(paths) >= {"input_path", "layer_prob_path", "layer_prob_float_path", "combined_overlay_path"}
    assert np.load(destination / "layer_prob.npy", allow_pickle=False).dtype == np.float32
    assert np.load(destination / "layer_prob.npy", allow_pickle=False)[3, 3] == np.float32(0.812345)
    overlay = cv2.cvtColor(cv2.imread(str(destination / "combined_overlay.png")), cv2.COLOR_BGR2RGB)
    assert overlay[4, 4, 0] > overlay[4, 4, 1] > overlay[4, 4, 2]
    assert np.array_equal(overlay[0, 0], np.array((102, 102, 102), np.uint8))


def _write_prediction_set(directory: Path, sample: str, offset: float) -> None:
    image = np.full((12, 14), 0.4 + offset, np.float32)
    layer = np.zeros((12, 14), np.float32); layer[2:10, 1:13] = 1
    vessel = np.zeros((12, 14), np.float32); vessel[5:8, 4:10] = 1
    for suffix, value in {
        "noisy.png": image, "layer_prob.png": layer * 0.8, "vessel_prob.png": vessel * 0.8,
        "layer_mask.png": layer, "vessel_mask.png": vessel, "layer_gt.png": layer,
        "vessel_gt.png": vessel,
    }.items():
        write_gray(directory / f"{sample}_{suffix}", value)
    np.save(directory / f"{sample}_layer_prob_float32.npy", layer * 0.8, allow_pickle=False)
    np.save(directory / f"{sample}_vessel_prob_float32.npy", vessel * 0.8, allow_pickle=False)


def test_one_position_fixed_final_smoke_builds_atlas_and_safe_bundle(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs/input_oracle_cv"
    (runs / "audit").mkdir(parents=True)
    (runs / "audit/audit_summary.json").write_text(json.dumps({
        "status": "passed", "training_authorized": True, "sealed_test_groups": ["pku_9999"],
        "test_assets_opened": 0,
    }))
    (runs / "splits").mkdir()
    (runs / "splits/split_audit.json").write_text(json.dumps({
        "status": "passed", "folds": {"0": {"val_groups": ["pku_0001"]}},
        "sealed_test_groups_metadata_only": ["pku_9999"], "test_assets_opened": 0,
    }))
    protocol = tmp_path / "configs/current/input_oracle_cv/protocol.yaml"
    protocol.parent.mkdir(parents=True)
    protocol.write_text("sealed_test_groups: [pku_9999]\n")
    manifest = tmp_path / "manifest.csv"
    sample = "pku_0001_f01"
    pd.DataFrame([{
        "sample_id": sample, "group_id": "pku_0001", "dataset": "PKU37", "split": "val",
        "noisy_cache_path": "cache/noisy.npy", "denoised_cache_path": "cache/den.npy",
        "clean_cache_path": "cache/clean.npy", "image_path": "unused.png",
    }]).to_csv(manifest, index=False)
    initial = {"stem.weight": torch.zeros(1), "adapters.denoise.weight": torch.zeros(1)}
    for arm, offset in (("noisy", 0.0), ("denoised", 0.05), ("clean", 0.1)):
        run = runs / f"fold0/{arm}_seed42"; run.mkdir(parents=True)
        init_path = runs / "fold0/preseg_initialization.pth"
        if not init_path.exists():
            torch.save({"model": initial, "epoch": -1}, init_path)
        config = {
            "seed": 42, "model": {"d2s_enabled": False, "s2d_enabled": False, "stage2_freeze_shared_encoder": True},
            "data": {"manifest": str(manifest), "val_split": "val", "input_column": f"{arm}_cache_path"},
            "train": {"pretrained": str(init_path), "epochs": 60, "stage": "input_segment"},
        }
        (run / "resolved_config.yaml").write_text(yaml.safe_dump(config))
        final = {"stem.weight": torch.ones(1), "adapters.denoise.weight": torch.zeros(1)}
        torch.save({"model": final, "epoch": 59, "optimizer": {"param_groups": [{"params": [0]}]}}, run / "last.pth")
        (run / "parameter_audit.json").write_text(json.dumps({
            "trainable": ["stem.weight"], "frozen": ["adapters.denoise.weight"],
            "optimizer_parameter_count": 1,
        }))
        (run / "initialization_audit.json").write_text(json.dumps({
            "model_state_sha256": "shared", "common_state_sha256": "shared",
            "data_plan_sha256": "plan",
        }))
        validation = run / "final_validation"; prediction = validation / "predictions/PKU37"
        prediction.mkdir(parents=True)
        _write_prediction_set(prediction, sample, offset)
        pd.DataFrame([{
            "sample_id": sample, "group_id": "pku_0001", "dataset": "PKU37",
            "original_height": 12, "original_width": 14, "evaluation_height": 12,
            "evaluation_width": 14, "layer_dice": 0.9, "upper_boundary_mae": 1,
            "lower_boundary_mae": 1, "thickness_mae": 1, "p0_vessel_dice": 0.8,
            "p0_vessel_precision": 0.8, "p0_vessel_recall": 0.8, "vessel_roi_dice": 0.8,
            "vessel_roi_fp_pixels": 0, "vessel_roi_fn_pixels": 0,
            "vessel_outside_gt_layer_fraction": 0, "vessel_area_fraction_pred": 0.2,
            "vessel_area_fraction_mae": 0,
        }]).to_csv(validation / "frame_metrics.csv", index=False)

    output = tmp_path / "visualization"
    monkeypatch.setattr(sys, "argv", [
        "export_input_oracle_visualizations.py", "--project-root", str(tmp_path),
        "--runs-root", str(runs), "--output-root", str(output), "--folds", "0",
        "--smoke-test", "--all-outer-val-frames", "--make-atlas", "--make-gpt-bundle",
    ])
    from tools.export_input_oracle_visualizations import main
    main()
    assert (output / "atlas/panels/pku_0001_three_input_comparison.png").is_file()
    assert (output / "per_position/pku_0001/I-CLEAN/input.png").is_file()
    assert not (output / "per_frame/fold0/pku_0001/I-CLEAN").exists()
    selection = pd.read_csv(output / "atlas/atlas_selection.csv")
    assert selection.loc[0, "noisy_sample_id"] == sample
    archive = next(output.glob("SABIDS_I_NOISY_I_DENOISED_I_CLEAN_*.zip"))
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        assert handle.testzip() is None
    assert not any(name.endswith((".pth", ".npy", ".npz")) for name in names)
    assert not any("pku_9999" in name for name in names)
