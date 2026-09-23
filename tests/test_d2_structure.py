import copy
import json
import math
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import cv2

from sabids.config import load_config, save_config
from sabids.engine import Trainer
from sabids.experiments.d2 import (
    audit_d2_checkpoint_binding,
    aggregate_component_rows,
    derive_vessel_strata,
    evaluate_vessel_components,
    local_component_contrast,
    select_d2_checkpoints,
)
from sabids.experiments.dose_response import sha256_file
from sabids.losses.d2 import D2StructureLoss


ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def test_memory_safe_teacher_keeps_source_gradient_and_detaches_clean_reference():
    class TinyTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mapping = torch.nn.Conv2d(1, 2, 1, bias=False)

        def forward(self, image, return_features=False, return_auxiliary=False):
            logits = self.mapping(image)
            return {
                "layer_logits": logits[:, :1],
                "vessel_logits": logits[:, 1:],
                "layer_prob": torch.sigmoid(logits[:, :1]),
                "vessel_prob": torch.sigmoid(logits[:, 1:]),
            }

    trainer = Trainer.__new__(Trainer)
    trainer.config = {"train": {"memory_safe_d2_teacher": True}}
    trainer.d2_teacher = TinyTeacher().eval()
    for parameter in trainer.d2_teacher.parameters():
        parameter.requires_grad_(False)

    batch = {"clean": torch.full((1, 1, 8, 8), 0.5)}
    clean = trainer._precompute_d2_teacher_clean_outputs(batch)
    assert clean is not None
    assert not clean["clean_layer_prob"].requires_grad
    assert not clean["clean_vessel_prob"].requires_grad

    denoised = torch.full((1, 1, 8, 8), 0.4, requires_grad=True)
    output = {"denoised_raw": denoised}
    trainer._attach_d2_teacher_outputs(output, batch, clean)
    assert output["d2_teacher_layer_logits"].requires_grad
    assert output["d2_teacher_vessel_logits"].requires_grad
    (output["d2_teacher_layer_logits"].sum()
     + output["d2_teacher_vessel_logits"].sum()).backward()
    assert denoised.grad is not None and torch.isfinite(denoised.grad).all()
    assert all(parameter.grad is None for parameter in trainer.d2_teacher.parameters())


def _sample(low=False, adjacent=False):
    noisy = np.full((16, 16), 0.5, np.float32)
    clean = noisy.copy()
    layer = np.zeros((16, 16), bool); layer[2:14, 2:14] = True
    vessel = np.zeros_like(layer); vessel[5:7, 5:7] = True
    if adjacent:
        vessel[5:7, 8:10] = True
    noisy[vessel] = 0.48 if low else 0.2
    clean[vessel] = 0.15
    return {"split": "train", "vessel": vessel, "layer": layer,
            "valid": np.ones_like(layer), "noisy": noisy, "clean": clean}


def test_train_only_strata_and_small_low_contrast_component_recall():
    definition = derive_vessel_strata([_sample(), _sample(low=True), _sample(adjacent=True)])
    assert definition["training_component_count"] == 4
    assert definition["primary_contrast_source"] == "noisy"
    sample = _sample(low=True)
    prediction = np.zeros_like(sample["vessel"])
    prediction[5, 5] = True  # exactly one quarter of a 2x2 component
    rows = evaluate_vessel_components(
        prediction, sample["vessel"], sample["layer"], sample["valid"],
        sample["noisy"], definition, sample["clean"],
    )
    assert rows[0]["coverage"] == pytest.approx(0.25)
    assert rows[0]["any_overlap"] == 1
    assert rows[0]["recall_at_025"] == 1
    assert rows[0]["recall_at_050"] == 0
    summary = aggregate_component_rows(rows)
    assert summary["recall_at_025"] == 1
    assert summary["mean_coverage"] == pytest.approx(0.25)


def test_component_ring_excludes_invalid_other_vessels_and_empty_is_nan():
    sample = _sample(adjacent=True)
    valid = sample["valid"].copy(); valid[:, :3] = False
    component = np.zeros_like(valid); component[5:7, 5:7] = True
    contrast, count = local_component_contrast(
        sample["noisy"], component, sample["vessel"], sample["layer"], valid, 2
    )
    assert math.isfinite(contrast) and contrast > 0 and count > 0
    bright = sample["noisy"].copy(); bright[component] = .8
    bright_contrast, _ = local_component_contrast(
        bright, component, sample["vessel"], sample["layer"], valid, 2
    )
    assert bright_contrast < 0  # signed ring-minus-vessel definition
    empty = aggregate_component_rows([])
    assert math.isnan(empty["recall_at_025"])
    assert empty["component_metric_reason"] == "no_valid_gt_components"
    with pytest.raises(ValueError, match="train samples"):
        derive_vessel_strata([{**sample, "split": "val"}])


def _batch():
    vessel = torch.zeros(2, 1, 16, 16); vessel[0, :, 5:7, 5:7] = 1
    layer = torch.zeros_like(vessel); layer[:, :, 2:14, 2:14] = 1
    return {
        "valid_mask": torch.ones_like(vessel), "vessel_valid_mask": torch.ones_like(vessel),
        "label_valid_mask": torch.ones_like(vessel), "vessel_mask": vessel,
        "layer_mask": layer, "has_vessel": torch.tensor([True, False]),
        "has_layer": torch.tensor([True, True]),
    }


def test_d2_losses_are_finite_empty_safe_and_task_gradient_only_hits_d2_input():
    prediction = torch.rand(2, 1, 16, 16, requires_grad=True)
    noisy, clean = torch.rand_like(prediction), torch.rand_like(prediction)
    teacher_weight = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
    teacher = {
        "layer_logits": prediction * teacher_weight,
        "vessel_logits": prediction * teacher_weight,
        "clean_layer_prob": torch.sigmoid(clean).detach(),
        "clean_vessel_prob": torch.sigmoid(clean).detach(),
    }
    loss = D2StructureLoss({"weights": {
        "charbonnier": 1, "ms_ssim": .2, "gradient": .1, "laplacian": .05,
        "vessel_roi": .5, "stroma_roi": .1, "outside_roi": .05,
        "boundary": .2, "cnr": .1, "teacher_task": .2,
        "teacher_consistency": .05, "leak": .1, "residual_amplitude": .01,
    }})
    values = loss(prediction, noisy, clean, _batch(), teacher)
    assert all(math.isfinite(float(value.detach())) for value in values.values() if torch.is_tensor(value))
    values["d2_total"].backward()
    assert prediction.grad is not None and prediction.grad.abs().sum() > 0
    assert teacher_weight.grad is None
    assert values["d2_vessel_valid_samples"] == 1


def test_d2_reconstruction_excludes_padding_and_unknown_outside_roi():
    prediction = torch.rand(1, 1, 16, 16, requires_grad=True)
    noisy, clean = torch.rand_like(prediction), torch.rand_like(prediction)
    batch = {key: (value[:1].clone() if torch.is_tensor(value) else value)
             for key, value in _batch().items()}
    batch["valid_mask"][:, :, :, :4] = 0
    batch["label_valid_mask"][:, :, 6:10, 6:10] = 0
    batch["vessel_valid_mask"][:, :, 6:10, 6:10] = 0
    loss = D2StructureLoss({"weights": {
        "charbonnier": 1.0, "ms_ssim": .2, "gradient": .1, "laplacian": .05,
        "outside_roi": .5,
    }})
    first = loss(prediction, noisy, clean, batch)["d2_total"]
    altered = clean.clone()
    altered[:, :, :, :4] = 100.0
    second = loss(prediction, noisy, altered, batch)["d2_total"]
    assert torch.allclose(first, second)
    first.backward()
    assert torch.count_nonzero(prediction.grad[:, :, :, :4]) == 0


def test_d2_selection_is_hierarchical_and_earliest_tie():
    table = pd.DataFrame({
        "epoch": [1, 2, 3, 4], "val_psnr": [30.0, 30.2, 30.2, 30.05],
        "val_teacher_task_preservation": [.8, .79, .79, .9],
    })
    selected = select_d2_checkpoints(table, psnr_noninferiority_db=.2)
    assert selected["best_pixel_epoch"] == 2
    assert selected["best_task_preserving_epoch"] == 4
    bad = table.copy(); bad.loc[0, "val_psnr"] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        select_d2_checkpoints(bad)


def test_vessel_strata_tool_hashes_train_labels_without_opening_val_or_test(tmp_path):
    image = np.full((16, 16), .5, np.float32)
    layer = np.zeros((16, 16), np.uint8); layer[2:14, 2:14] = 255
    vessel = np.zeros((16, 16), np.uint8); vessel[6:8, 6:8] = 255
    image_path = tmp_path / "train_image.npy"; np.save(image_path, image)
    clean_path = tmp_path / "train_clean.npy"; np.save(clean_path, image)
    layer_path = tmp_path / "train_layer.png"; assert cv2.imwrite(str(layer_path), layer)
    vessel_path = tmp_path / "train_vessel.png"; assert cv2.imwrite(str(vessel_path), vessel)
    rows = [{
        "sample_id": "train", "group_id": "train", "patient_id": "train",
        "dataset": "PKU37", "split": "train", "image_path": str(image_path),
        "clean_path": str(clean_path), "layer_mask_path": str(layer_path),
        "vessel_mask_path": str(vessel_path),
    }]
    for split in ("val", "test"):
        rows.append({
            "sample_id": split, "group_id": split, "patient_id": split,
            "dataset": "PKU37", "split": split,
            "image_path": str(tmp_path / "sealed" / f"{split}_image.npy"),
            "clean_path": str(tmp_path / "sealed" / f"{split}_clean.npy"),
            "layer_mask_path": str(tmp_path / "sealed" / f"{split}_layer.png"),
            "vessel_mask_path": str(tmp_path / "sealed" / f"{split}_vessel.png"),
        })
    manifest = tmp_path / "manifest.csv"; pd.DataFrame(rows).to_csv(manifest, index=False)
    config = load_config(ROOT / "configs/base.yaml")
    config["data"].update(manifest=str(manifest), root=str(tmp_path), target_size=[16, 16])
    config_path = tmp_path / "strata.yaml"; save_config(config, config_path)

    def run(output):
        process = subprocess.run([
            sys.executable, str(ROOT / "tools/prepare_vessel_strata.py"),
            "--project-root", str(tmp_path), "--config", str(config_path),
            "--output", str(output),
        ], cwd=ROOT, capture_output=True, text=True)
        assert process.returncode == 0, process.stderr
        return json.loads(output.read_text(encoding="utf-8"))

    first = run(tmp_path / "strata_first.json")
    assert first["selected_split"] == "train"
    assert first["validation_assets_opened"] == first["test_assets_opened"] == 0
    assert {row["sample_id"] for row in first["training_label_asset_records"]} == {"train"}
    vessel[8, 8] = 255; assert cv2.imwrite(str(vessel_path), vessel)
    second = run(tmp_path / "strata_second.json")
    assert first["training_label_asset_sha256"] != second["training_label_asset_sha256"]


def test_d20_cpu_end_to_end_smoke_writes_audits_and_bindings(tmp_path):
    rows = []
    for index, split in enumerate(("train", "val")):
        image = np.full((16, 16), .4, np.float32)
        clean = np.full((16, 16), .45, np.float32)
        layer = np.zeros((16, 16), np.float32); layer[2:14, 2:14] = 1
        vessel = np.zeros((16, 16), np.float32); vessel[6:8, 6:8] = 1
        paths = {}
        for name, value in (("image", image), ("clean", clean),
                            ("layer_mask", layer), ("vessel_mask", vessel)):
            if name.endswith("mask"):
                path = tmp_path / f"{split}_{name}.png"
                assert cv2.imwrite(str(path), (value * 255).astype(np.uint8))
            else:
                path = tmp_path / f"{split}_{name}.npy"; np.save(path, value)
            paths[name] = str(path)
        rows.append({"sample_id": f"s{index}", "group_id": f"g{index}",
                     "patient_id": f"g{index}", "dataset": "PKU37", "split": split,
                     **{f"{name}_path": value for name, value in paths.items()}})
    manifest = tmp_path / "manifest.csv"; pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = load_config(ROOT / "configs/base.yaml")
    cfg.update(device="cpu", deterministic=True, seed=42, protocol_id="smoke")
    cfg["model"].update(channels=[2, 4], encoder_depths=[1, 1], decoder_depth=1,
                        interaction_levels=[1], d2s_enabled=False, s2d_enabled=False)
    cfg["data"].update(manifest=str(manifest), root=str(tmp_path), target_size=[16, 16],
                       load_segmentation_labels=True, samples_per_epoch=1)
    cfg["train"].update(stage="denoise", output_dir=str(tmp_path / "d20"), epochs=1,
                        fixed_epoch=1, checkpoint_selection_rule="d2_hierarchical_fixed_budget_v1",
                        batch_size=1, gradient_accumulation_steps=1, num_workers=0, amp=False,
                        monitor="psnr", early_stopping_patience=2, pretrained=None, resume=None)
    cfg["loss"].update(restoration_mode="structure_d2", definition_version="d20-smoke",
                       d2={"weights": {"charbonnier": 1.0, "ms_ssim": .2,
                                       "gradient": .1, "laplacian": .05}})
    cfg["loss"]["weights"].update(reconstruction=1.0, residual=0.0, identity=0.0)
    cfg["d2"] = {"enabled": True, "arm": "D20", "run_mode": "smoke",
                 "scientific_evaluation": False, "notice": "NOT FOR SCIENTIFIC EVALUATION",
                 "selection": {"psnr_noninferiority_db": .2},
                 "teacher": {"enabled": False}}
    cfg["training_asset_evidence"] = {"enabled": True, "project_root": str(tmp_path)}
    trainer = Trainer(cfg); trainer.fit()
    run = tmp_path / "d20"
    for name in ("best_pixel.pth", "best_task_preserving.pth", "last.pth",
                 "checkpoint_binding_best_pixel.json", "checkpoint_binding_best_task_preserving.json",
                 "checkpoint_binding_last.json", "parameter_audit.json", "teacher_audit.json",
                 "cost_profile.json", "training_asset_inventory_initial.json",
                 "training_asset_inventory_last.json"):
        assert (run / name).is_file(), name
    cost = json.loads((run / "cost_profile.json").read_text(encoding="utf-8"))
    assert cost["scientific_evaluation"] is False
    assert cost["notice"] == "NOT FOR SCIENTIFIC EVALUATION"
    teacher_audit = run / "teacher_audit.json"
    teacher_audit.write_text(teacher_audit.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="teacher_audit"):
        audit_d2_checkpoint_binding(
            run / "checkpoint_binding_best_pixel.json", run / "best_pixel.pth", "best_pixel"
        )


def test_d25_cpu_smoke_updates_d2_and_keeps_independent_teacher_frozen(tmp_path):
    # Reuse the completed synthetic D20 closure only as a structurally valid
    # teacher checkpoint fixture. This is explicitly not scientific evidence.
    test_d20_cpu_end_to_end_smoke_writes_audits_and_bindings(tmp_path)
    d20 = tmp_path / "d20"
    teacher_checkpoint = d20 / "last.pth"
    teacher_evidence = tmp_path / "synthetic_teacher_evidence.json"
    selection_rule = "fixed_final"
    training_data = "synthetic-train-val-only-fixture"
    teacher_evidence.write_text(json.dumps({
        "schema_version": "synthetic-d2-teacher-test-v1",
        "status": "passed",
        "checkpoint_sha256": sha256_file(teacher_checkpoint),
        "selection_rule": selection_rule,
        "training_data": training_data,
        "split": "development_train_val",
        "test_assets_opened": 0,
        "scientific_evaluation": False,
    }), encoding="utf-8")

    raw = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    cfg = copy.deepcopy(raw["config"])
    cfg["train"].update(output_dir=str(tmp_path / "d25"), pretrained=None, resume=None)
    cfg["loss"].update(
        restoration_mode="structure_d2",
        definition_version="d25-cpu-smoke-not-scientific",
        d2={"weights": {
            "charbonnier": 1.0, "ms_ssim": .2, "gradient": .1, "laplacian": .05,
            "vessel_roi": .5, "stroma_roi": .1, "outside_roi": .05,
            "boundary": .2, "cnr": .1, "teacher_task": .2,
            "teacher_consistency": .05, "leak": .1, "residual_amplitude": .01,
        }},
    )
    cfg["loss"]["weights"].update(reconstruction=1.0, residual=0.0, identity=.05)
    cfg["d2"] = {
        "enabled": True, "arm": "D25", "run_mode": "smoke",
        "scientific_evaluation": False, "notice": "NOT FOR SCIENTIFIC EVALUATION",
        "selection": {"psnr_noninferiority_db": .2},
        "teacher": {
            "enabled": True, "checkpoint": str(teacher_checkpoint),
            "sha256": sha256_file(teacher_checkpoint),
            "evidence": str(teacher_evidence),
            "evidence_sha256": sha256_file(teacher_evidence),
            "selection_rule": selection_rule, "training_data": training_data,
            "split": "development_train_val",
        },
    }
    Trainer(cfg).fit()
    run = tmp_path / "d25"
    history = pd.read_csv(run / "history.csv", low_memory=False)
    assert len(history) == 1 and np.isfinite(history.select_dtypes("number")).all().all()
    teacher_audit = json.loads((run / "teacher_audit.json").read_text(encoding="utf-8"))
    assert teacher_audit["status"] == "passed"
    assert teacher_audit["changed_parameter_count"] == 0
    assert teacher_audit["requires_grad_parameter_count"] == 0
    parameter_audit = json.loads((run / "parameter_audit.json").read_text(encoding="utf-8"))
    assert parameter_audit["status"] == "passed"
    assert parameter_audit["changed_trainable_parameter_count"] > 0
    assert parameter_audit["changed_frozen_parameter_count"] == 0
    for kind in ("best_pixel", "best_task_preserving", "last"):
        audit_d2_checkpoint_binding(
            run / f"checkpoint_binding_{kind}.json", run / f"{kind}.pth", kind
        )
