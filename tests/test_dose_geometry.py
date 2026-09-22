import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from sabids.data.dataset import OCTManifestDataset
from sabids.data.transforms import JointOCTTransform, _resize_pad
from sabids.engine.evaluator import evaluate_model
from sabids.experiments.d2 import derive_vessel_strata
from sabids.experiments.dose_response import to_model_grid


def test_non_square_common_transform_nearest_and_padding():
    layer = np.zeros((20, 40), np.float32)
    layer[6:16, 10:30] = 1
    vessel = np.zeros_like(layer)
    vessel[9:12, 16:20] = 1
    arrays = {"noisy": layer.copy(), "clean": layer.copy(), "layer": layer,
              "vessel": vessel, "label_valid": np.ones_like(layer), "vessel_valid": np.ones_like(layer)}
    values, geometry = to_model_grid(arrays, (32, 32))
    assert geometry["original_noisy_shape"] == [20, 40]
    assert geometry["resize_shape"] == [16, 32]
    assert geometry["pad_top_bottom_left_right"] == [8, 8, 0, 0]
    assert geometry["metric_evaluation_shape"] == [16, 32]
    assert geometry["label_interpolation"] == "nearest"
    assert set(np.unique(values["layer"])) == {0, 1}
    assert np.array_equal(values["layer"], _resize_pad(layer, (32, 32), is_mask=True))
    assert np.all(values["vessel"] <= values["layer"])
    assert not values["spatial_valid"][:8].any() and values["spatial_valid"].sum() == 512
    assert np.array_equal(values["noisy"], values["clean"])
    with pytest.raises(ValueError, match="geometry"):
        to_model_grid({**arrays, "clean": np.ones((32, 32), np.float32)}, (32, 32))


def dataset_fixture(tmp_path):
    original = np.ones((20, 40), np.float32) * .5
    layer = np.zeros_like(original)
    layer[6:16, 10:30] = 1
    vessel = np.zeros_like(original)
    vessel[9:12, 16:20] = 1
    annotation = np.ones_like(original)
    annotation[9:12, 24:28] = 0
    grid, _ = to_model_grid({"noisy": original, "clean": original, "layer": layer,
        "vessel": vessel, "label_valid": annotation, "vessel_valid": annotation}, (32, 32))
    row = {"sample_id": "one", "group_id": "anatomy", "patient_id": "person", "dataset": "SYNTHETIC", "split": "val"}
    for column, value in {"image_path": original, "dose_path": grid["noisy"], "clean_path": grid["clean"],
                         "layer_mask_path": grid["layer"], "vessel_mask_path": grid["vessel"],
                         "label_valid_mask_path": grid["label_valid"], "vessel_valid_mask_path": grid["vessel_valid"],
                         "spatial_valid_mask_path": grid["spatial_valid"]}.items():
        p = tmp_path / f"{column}.npy"
        np.save(p, value)
        row[column] = str(p)
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([row]).to_csv(manifest, index=False)
    return OCTManifestDataset(manifest, "val", JointOCTTransform((32, 32), training=False),
                              sample_repeat=False, image_column="dose_path", pretransformed_model_grid=True), grid


def test_original_geometry_not_cache_geometry_and_evaluation_excludes_invalid(tmp_path):
    dataset, grid = dataset_fixture(tmp_path)
    item = dataset[0]
    assert (item["original_height"], item["original_width"]) == (20, 40)
    assert item["image"].shape[-2:] == (32, 32)
    assert item["valid_mask"].sum() == 512
    class FixedModel(torch.nn.Module):
        def forward(self, image, **kwargs):
            # Deliberately wrong in padding AND annotation-invalid pixels.
            invalid = (grid["spatial_valid"] * grid["label_valid"]) == 0
            l, v = grid["layer"].copy(), grid["vessel"].copy()
            l[invalid] = 1
            v[invalid] = 1
            return {"denoised": image, "layer_prob": torch.tensor(l[None, None]),
                    "vessel_prob": torch.tensor(v[None, None])}
    out = tmp_path / "metrics"
    summary = evaluate_model(FixedModel(), DataLoader(dataset, batch_size=1), torch.device("cpu"),
                             output_dir=out, stage="input_segment", tasks=("layer", "vessel"),
                             model_grid_contract=True, restore_original_geometry=False)
    assert summary["layer_dice"] == pytest.approx(1)
    assert summary["vessel_dice"] == pytest.approx(1)
    frame = pd.read_csv(out / "frame_metrics.csv")
    assert (frame.iloc[0].evaluation_height, frame.iloc[0].evaluation_width) == (16, 32)
    assert (frame.iloc[0].original_height, frame.iloc[0].original_width) == (20, 40)
    assert np.isnan(frame.iloc[0].upper_boundary_mae)
    assert summary["original_resolution_boundary_metrics"] == "NOT IMPLEMENTED"
    groups = pd.read_csv(out / "group_metrics.csv")
    assert "repeat_dose_input_mae" in groups and "repeat_denoised_mae" not in groups
    json.loads((out / "summary.json").read_text(encoding="utf-8"), parse_constant=lambda v: pytest.fail(v))


def test_model_grid_refuses_original_restoration(tmp_path):
    dataset, _ = dataset_fixture(tmp_path)
    with pytest.raises(ValueError, match="contract"):
        evaluate_model(torch.nn.Identity(), DataLoader(dataset), torch.device("cpu"),
                       model_grid_contract=True, restore_original_geometry=True, tasks=("layer", "vessel"))


def test_opt_in_d2_diagnostics_write_region_and_frozen_strata_csv(tmp_path):
    dataset, grid = dataset_fixture(tmp_path)
    definition = derive_vessel_strata([{
        "split": "train", "vessel": grid["vessel"].astype(bool),
        "layer": grid["layer"].astype(bool), "valid": grid["spatial_valid"].astype(bool),
        "noisy": grid["noisy"], "clean": grid["clean"],
    }])

    class FixedDenoiser(torch.nn.Module):
        def forward_denoise_only(self, image):
            denoised = torch.clamp(image * .9, 0, 1)
            return {"denoised": denoised, "denoised_raw": denoised}

    output = tmp_path / "d2_diagnostics"
    summary = evaluate_model(
        FixedDenoiser(), DataLoader(dataset, batch_size=1), torch.device("cpu"),
        output_dir=output, stage="denoise", tasks=("denoise",),
        postprocess_modes=("p0",), restore_original_geometry=False,
        d2_diagnostics=True, vessel_strata_definition=definition,
    )
    assert summary["n_frames"] == 1
    leakage = pd.read_csv(output / "structure_leakage_metrics.csv")
    for column in (
        "residual_vessel_mean_abs", "residual_vessel_boundary_mean_abs",
        "residual_small_vessel_mean_abs", "residual_low_contrast_vessel_mean_abs",
        "residual_small_low_contrast_vessel_mean_abs",
        "residual_layer_stroma_mean_abs", "residual_layer_outside_mean_abs",
    ):
        assert column in leakage
    assert not (output / "component_metrics.csv").exists()


def test_explicit_augmentation_plan_applied_without_global_rng(tmp_path):
    dataset, grid = dataset_fixture(tmp_path)
    dataset.transform.training = True
    dataset.transform.horizontal_flip = 1
    dataset.deterministic_augmentation_seed = 42
    dataset.set_epoch(2)
    before = np.random.get_state()
    item = dataset[0]
    after = np.random.get_state()
    assert item["augmentation_flip"] is True
    assert np.array_equal(item["layer_mask"].numpy()[0], np.fliplr(grid["layer"]))
    assert np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.fixture
def prepared_project(tmp_path):
    from tools.prepare_dose_response_inputs import prepare, synthetic_assets
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "sabids", tmp_path / "sabids", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(root / "configs/adaptive_denoising", tmp_path / "configs/adaptive_denoising")
    shutil.copyfile(root / "configs/base.yaml", tmp_path / "configs/base.yaml")
    shutil.copyfile(root / "evaluate.py", tmp_path / "evaluate.py")
    (tmp_path / "tools").mkdir()
    shutil.copyfile(root / "tools/prepare_dose_response_inputs.py", tmp_path / "tools/prepare_dose_response_inputs.py")
    torch.set_num_threads(1)
    pf = synthetic_assets(tmp_path, "unit_fixture")
    result = prepare(tmp_path, pf, curves=["oracle", "d1"], alphas=[0., .5], seeds=[42],
                     budget="smoke", tag="unit_fixture", device_name="cpu")
    return tmp_path, result


def test_cpu_smoke_checkpoints_metadata_pairing_and_no_test(prepared_project):
    from tools.prepare_dose_response_inputs import cpu_smoke
    root, result = prepared_project
    report = cpu_smoke(result["configs"], root / "runs/cpu_report.json")
    assert report["status"] == "passed" and report["config_count"] == 4
    assert all(report["paired_checks"].values()) and report["test_assets_opened"] == 0
    assert report["notice"] == "NOT FOR SCIENTIFIC EVALUATION"
    assert report["scientific_evaluation"] is False
    for p in result["configs"]:
        from sabids.config import load_config
        cfg = load_config(p)
        output = Path(cfg["train"]["output_dir"])
        raw = torch.load(output / "last.pth", weights_only=False, map_location="cpu")
        assert raw["epoch"] == 0 and raw["config"]["dose_response"]["scientific_evaluation"] is False
        history = pd.read_csv(output / "history.csv")
        assert len(history) == 1 and np.isfinite(history.train_total).all()
        metadata = json.loads((output / "dose_training_metadata.json").read_text(encoding="utf-8"), parse_constant=lambda v: pytest.fail(v))
        assert metadata["completed_optimizer_steps"] == metadata["expected_optimizer_steps"] == 2
        assert metadata["changed_trainable_parameter_names"] and not metadata["changed_frozen_parameter_names"]


def test_preparation_reuse_is_exact_and_drift_rejected(prepared_project):
    from tools.prepare_dose_response_inputs import prepare
    root, result = prepared_project
    again = prepare(root, result["preflight"], curves=["oracle", "d1"], alphas=[0., .5], seeds=[42],
                    budget="smoke", tag="unit_fixture", device_name="cpu")
    assert again == result
    with pytest.raises(FileExistsError, match="identity mismatch"):
        prepare(root, result["preflight"], curves=["oracle", "d1"], alphas=[0., .75], seeds=[42],
                budget="smoke", tag="unit_fixture", device_name="cpu")


def test_d2_curve_type_is_written_explicitly_in_generated_config(prepared_project):
    from sabids.config import load_config
    from tools.prepare_dose_response_inputs import prepare
    root, result = prepared_project
    prepared = prepare(
        root, result["preflight"], curves=["d2_task"], alphas=[0.], seeds=[42],
        budget="smoke", tag="d2_curve_fixture", device_name="cpu",
    )
    config = load_config(prepared["configs"][0])
    assert config["dose_response"]["curve_type"] == "d2_task"
    assert config["dose_response"]["alpha"] == 0.0


@pytest.mark.parametrize("key", ["auxiliary", "interaction", "pretrained", "existing_output", "unknown_cache"])
def test_registered_training_guards(prepared_project, key):
    from sabids.config import load_config
    from sabids.experiments.dose_response import validate_dose_config
    root, result = prepared_project
    cfg = load_config(result["configs"][0])
    validate_dose_config(cfg)
    if key == "auxiliary":
        cfg["loss"]["auxiliary_weight"] = .1
    elif key == "interaction":
        cfg["model"]["d2s_enabled"] = True
    elif key == "pretrained":
        cfg["train"]["pretrained"] = "any.pth"
    elif key == "existing_output":
        output = Path(cfg["train"]["output_dir"])
        output.mkdir(parents=True)
        (output / "do_not_touch.txt").write_text("keep")
    elif key == "unknown_cache":
        table = pd.read_csv(cfg["data"]["manifest"])
        np.save(table.iloc[0].label_valid_mask_path, np.zeros(tuple(cfg["data"]["target_size"]), np.float32))
    with pytest.raises(ValueError):
        validate_dose_config(cfg)


def test_old_dataset_default_unchanged(tmp_path):
    dataset, _ = dataset_fixture(tmp_path)
    legacy = OCTManifestDataset(dataset.manifest, "val", JointOCTTransform((32, 32), training=False), sample_repeat=False)
    item = legacy[0]
    assert item["model_input_path"].endswith("image_path.npy")
    assert (item["original_height"], item["original_width"]) == (20, 40)
    assert "metric_coordinate_system" not in item
