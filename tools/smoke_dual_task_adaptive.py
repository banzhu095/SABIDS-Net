from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.engine import Trainer
from sabids.config import save_config


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sabids_dual_task_smoke_") as temporary:
        root = Path(temporary)
        rows = []
        for index in range(6):
            split = "train" if index < 4 else "val"
            group = f"smoke_{index // 2}"
            yy, xx = np.mgrid[:24, :32]
            clean = np.clip(0.25 + 0.5 * xx / 31 + 0.1 * np.sin(yy / 3), 0, 1)
            noisy = np.clip(clean + np.random.default_rng(index).normal(0, 0.03, clean.shape), 0, 1)
            layer = ((yy >= 7) & (yy <= 19)).astype(np.uint8)
            vessel = (((xx - 8 - index) ** 2 + (yy - 13) ** 2) < 10).astype(np.uint8) * layer
            paths = {}
            for name, value in (("image", noisy), ("clean", clean), ("layer", layer), ("vessel", vessel)):
                path = root / f"s{index}_{name}.png"
                cv2.imwrite(str(path), (value * 255).astype(np.uint8))
                paths[name] = str(path)
            rows.append({"sample_id": f"s{index}", "group_id": group, "dataset": "smoke",
                         "split": split, "image_path": paths["image"], "clean_path": paths["clean"],
                         "layer_mask_path": paths["layer"], "vessel_mask_path": paths["vessel"]})
        import pandas as pd
        manifest = root / "manifest.csv"
        pd.DataFrame(rows).to_csv(manifest, index=False)
        output = root / "run"
        config = {
            "seed": 42, "deterministic": True, "device": "cpu",
            "model": {"in_channels": 1, "channels": [4, 8, 12, 16],
                      "encoder_depths": [1, 1, 1, 1], "decoder_depth": 1,
                      "interaction_levels": [3, 2, 1], "d2s_enabled": False,
                      "s2d_enabled": False, "dropout": 0.0},
            "data": {"manifest": str(manifest), "root": str(root), "train_split": "train",
                     "val_split": "val", "target_size": [24, 32], "normalization": "fixed",
                     "samples_per_epoch": 4, "load_segmentation_labels": True,
                     "augmentation": {"horizontal_flip": 0.0}},
            "train": {"stage": "input_segment", "output_dir": str(output), "epochs": 1,
                      "early_stopping_patience": 2, "batch_size": 1,
                      "gradient_accumulation_steps": 1, "num_workers": 0,
                      "learning_rate": 1e-3, "minimum_learning_rate": 1e-5,
                      "weight_decay": 0.0, "gradient_clip": 1.0, "amp": False,
                      "monitor": "joint_soft_dice", "evaluate_epoch0": False,
                      "pretrained": None, "resume": None, "use_ema": False},
            "loss": {"definition_version": "dual-task-adaptive-smoke-v1",
                     "auxiliary_weight": 0.0, "zero_source": "final_segmentation",
                     "exclude_annotation_invalid_containment": True,
                     "exclude_annotation_invalid_boundary": True,
                     "vessel_supervision_mode": "roi_bce_dice_outside",
                     "gate": {"tv_weight": 0.001, "reconstruction_weight": 0.02},
                     "weights": {"layer": 1.0, "vessel": 1.0, "vessel_outside": 0.5,
                                 "containment": 0.1}},
            "evaluation": {"threshold": 0.5, "layer_threshold": 0.5,
                           "vessel_threshold": 0.5, "batch_size": 1, "num_workers": 0},
            "dual_view": {"enabled": False},
            "dual_task_adaptive": {"enabled": True, "load_bound_checkpoints": False,
                                   "coarse_strength": 0.25, "layer_strength_init": 1.0,
                                   "vessel_strength_init": 0.5, "context_channels": 4,
                                   "fusion_levels": [3, 2, 1], "anchors": {},
                                   "run_mode": "cpu_smoke", "test_assets_opened": 0},
        }
        trainer = Trainer(config)
        trainer.fit()
        metadata = json.loads((output / "dual_task_adaptive_training_metadata.json").read_text())
        binding = json.loads((output / "checkpoint_binding_best_joint.json").read_text())
        changed = metadata["changed_trainable_parameter_names"]
        gate_heads_updated = (
            any(name.startswith("controller.layer_head.") for name in changed)
            and any(name.startswith("controller.vessel_head.") for name in changed)
        )
        history = pd.read_csv(output / "history.csv")
        required_metrics = {
            "val_layer_soft_dice", "val_vessel_soft_dice", "val_joint_soft_dice",
            "val_layer_gate_mean", "val_vessel_gate_mean", "train_gate_tv_loss",
            "train_gate_reconstruction_loss",
        }
        if (metadata["status"] != "passed" or binding["status"] != "passed"
                or not gate_heads_updated or not required_metrics <= set(history.columns)
                or not (output / "best_joint.pth").is_file()):
            raise RuntimeError("Adaptive CPU smoke did not close the training/checkpoint loop")
        component_inventory = root / "fixed_component_inventory.json"
        component_inventory.write_text(json.dumps({
            "schema_version": "dual-task-adaptive-fixed-components-v1",
            "validation_components": [
                {"sample_id": f"s{index}", "group_id": f"smoke_{index // 2}",
                 "split": "val", "component_id": 1, "area_original_px": 25,
                 "contrast_original": 0.2, "small": True, "low_contrast": True,
                 "small_low_contrast": True}
                for index in (4, 5)
            ], "test_assets_opened": 0,
        }), encoding="utf-8")
        config["dual_task_adaptive"].setdefault("evidence", {})[
            "fixed_component_inventory"
        ] = str(component_inventory)
        config_path = root / "config.yaml"
        save_config(config, config_path)
        evaluation = root / "evaluation"
        subprocess.run([
            sys.executable, str(ROOT / "tools/evaluate_dual_task_adaptive.py"),
            "--config", str(config_path), "--checkpoint", str(output / "best_joint.pth"),
            "--output", str(evaluation), "--device", "cpu", "--save-atlas",
        ], check=True)
        required_outputs = {
            "metrics_by_image.csv", "metrics_by_position.csv", "denoising_results.csv",
            "segmentation_results.csv", "coarse_vs_fine_deltas.csv",
            "gate_metrics_by_image.csv", "gate_metrics_by_position.csv",
            "gate_region_summary.csv", "gate_difference_summary.csv", "RESULTS_TABLE.csv",
        }
        if not all((evaluation / name).is_file() for name in required_outputs):
            raise RuntimeError("Adaptive CPU smoke evaluation outputs are incomplete")
        print(json.dumps({"status": "passed", "non_square_input": [24, 32],
                          "best_joint": True, "both_gate_heads_updated": True,
                          "coarse_fine_outputs_validated": True,
                          "test_assets_opened": 0}, indent=2))


if __name__ == "__main__":
    main()
