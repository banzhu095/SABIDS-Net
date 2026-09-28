#!/usr/bin/env python
"""Run the complete five-arm dual-view matrix on synthetic CPU train/val data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config, save_config
from sabids.engine.trainer import Trainer, build_model
from sabids.experiments.dose_response import tensor_sha, write_strict_json
from sabids.utils import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="runs/adaptive_denoising/synthetic_dual_view_smoke_v1")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
    assets = output / "assets"
    assets.mkdir(parents=True)
    height, width = 32, 48
    yy, xx = np.mgrid[:height, :width]
    layer = ((yy >= 8) & (yy < 26)).astype(np.float32)
    vessel = (((yy - 16) ** 2 + (xx - 20) ** 2) < 20).astype(np.float32)
    clean = np.clip(0.2 + 0.35 * layer - 0.22 * vessel + xx / 400, 0, 1).astype(np.float32)
    rng = np.random.default_rng(42)
    rows = []
    for index in range(4):
        noisy = np.clip(clean + rng.normal(0, 0.05, clean.shape), 0, 1).astype(np.float32)
        mild = np.ascontiguousarray(noisy - 0.25 * (noisy - clean), dtype=np.float32)
        paths = {}
        for name, value in {
            "image": noisy, "primary": noisy, "auxiliary": mild, "clean": clean,
            "layer_mask": layer, "vessel_mask": vessel,
            "label_valid_mask": np.ones_like(clean), "vessel_valid_mask": np.ones_like(clean),
            "spatial_valid_mask": np.ones_like(clean),
        }.items():
            path = assets / f"s{index}_{name}.npy"
            np.save(path, value.astype(np.float32), allow_pickle=False)
            paths[f"{name}_path"] = str(path)
        rows.append({
            "sample_id": f"s{index}", "group_id": f"g{index}", "patient_id": f"p{index}",
            "dataset": "SYNTHETIC_DUAL_VIEW", "split": "train" if index < 2 else "val",
            **paths,
        })
    base = pd.DataFrame(rows)
    templates = {
        "B0": "b0_noisy.yaml", "B1": "b1_mild.yaml", "B3": "b3_noisy_mild.yaml",
        "B6": "b6_noisy_noisy.yaml", "C1": "c1_shuffled_mild.yaml",
    }
    manifests = {}
    for arm in templates:
        table = base.copy()
        if arm == "B1":
            table["primary_path"] = table["auxiliary_path"]
        elif arm == "B6":
            table["auxiliary_path"] = table["primary_path"]
        elif arm == "C1":
            for split in ("train", "val"):
                indices = table.index[table.split.eq(split)].tolist()
                table.loc[indices, "auxiliary_path"] = table.loc[indices[::-1], "auxiliary_path"].to_numpy()
        path = output / f"manifest_{arm.lower()}.csv"
        table.to_csv(path, index=False)
        manifests[arm] = path
    seed_everything(42, True, use_cuda=False)
    init_config = load_config(root / "configs/adaptive_denoising/dual_view/b3_noisy_mild.yaml")
    init_config.update(seed=42, device="cpu", protocol_id="synthetic_dual_view_smoke_v1")
    init_config["model"].update(channels=[4, 8, 16, 32], encoder_depths=[1, 1, 1, 1], decoder_depth=1)
    initial_model = build_model(init_config)
    initial_sha = {name: tensor_sha(parameter) for name, parameter in initial_model.named_parameters()}
    init_path = output / "common_initialization.pth"
    torch.save({"model": initial_model.state_dict(), "config": init_config, "epoch": -1}, init_path)
    audits = []
    for arm, template in templates.items():
        config = load_config(root / "configs/adaptive_denoising/dual_view" / template)
        config.update(seed=42, device="cpu", protocol_id="synthetic_dual_view_smoke_v1")
        config["model"].update(channels=[4, 8, 16, 32], encoder_depths=[1, 1, 1, 1], decoder_depth=1)
        config["data"].update(
            manifest=str(manifests[arm]), root=str(root), target_size=[height, width],
            samples_per_epoch=2, deterministic_augmentation=True,
        )
        config["train"].update(
            output_dir=str(output / arm.lower()), epochs=1, early_stopping_patience=2,
            batch_size=1, gradient_accumulation_steps=1, num_workers=0, amp=False,
            pretrained=str(init_path),
        )
        config["dual_view"].update(
            budget="smoke", scientific_evaluation=False, notice="NOT FOR SCIENTIFIC EVALUATION"
        )
        trainer = Trainer(config)
        save_config(config, trainer.output_dir / "resolved_config.yaml")
        actual_initial = {
            name: tensor_sha(parameter) for name, parameter in trainer.model.named_parameters()
        }
        if actual_initial != initial_sha:
            raise AssertionError(f"Common initialization mismatch for {arm}")
        trainer.fit()
        metadata = json.loads((trainer.output_dir / "dual_view_training_metadata.json").read_text(encoding="utf-8"))
        initialization_audit = json.loads(
            (trainer.output_dir / "initialization_audit.json").read_text(encoding="utf-8")
        )
        audits.append({
            "arm": arm, "completed_epochs": metadata["completed_epochs"],
            "frozen_changes": metadata["changed_frozen_parameter_names"],
            "test_assets_opened": metadata["test_assets_opened"],
            "model_state_sha256": initialization_audit["model_state_sha256"],
            "sampler_plan_sha256": initialization_audit["sampler_plan_sha256"],
            "actual_augmentation_plan_sha256": initialization_audit["actual_augmentation_plan_sha256"],
            "paired_cohort_sha256": initialization_audit["paired_cohort_sha256"],
        })
    paired_fields = (
        "model_state_sha256", "sampler_plan_sha256",
        "actual_augmentation_plan_sha256", "paired_cohort_sha256",
    )
    paired_checks = {
        field: len({audit[field] for audit in audits}) == 1 for field in paired_fields
    }
    if not all(paired_checks.values()):
        raise AssertionError(f"Paired smoke audit failed: {paired_checks}")
    result = {
        "status": "passed", "arms": audits,
        "common_initialization_verified": True,
        "paired_data_and_augmentation_verified": paired_checks,
        "non_square_model_grid": [height, width],
        "scientific_evaluation": False, "test_assets_opened": 0,
    }
    write_strict_json(output / "smoke_report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
