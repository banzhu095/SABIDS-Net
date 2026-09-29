#!/usr/bin/env python
"""One-epoch non-square CPU smoke for B3R/B6R (not scientific evidence)."""
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.project_root).resolve(), Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing existing smoke output: {output}")
    assets = output / "assets"; assets.mkdir(parents=True)
    h, w = 32, 48
    yy, xx = np.mgrid[:h, :w]
    layer = ((yy > 6) & (yy < 27)).astype(np.float32)
    vessel = (((yy - 17) ** 2 + (xx - 22) ** 2) < 22).astype(np.float32)
    clean = np.clip(.2 + .3 * layer - .18 * vessel + xx / 500, 0, 1).astype(np.float32)
    rng = np.random.default_rng(20260929); rows = []
    for index in range(4):
        noisy = np.clip(clean + rng.normal(0, .04, clean.shape), 0, 1).astype(np.float32)
        mild = (.75 * noisy + .25 * clean).astype(np.float32)
        values = {"noisy": noisy, "mild": mild, "strong": clean, "primary": noisy,
                  "auxiliary": mild, "clean": clean, "layer_mask": layer,
                  "vessel_mask": vessel, "label_valid_mask": np.ones_like(clean),
                  "vessel_valid_mask": np.ones_like(clean), "spatial_valid_mask": np.ones_like(clean)}
        paths = {}
        for name, value in values.items():
            path = assets / f"s{index}_{name}.npy"; np.save(path, value, allow_pickle=False)
            paths[f"{name}_path"] = str(path)
        rows.append({"sample_id": f"s{index}", "group_id": f"g{index}",
                     "patient_id": f"p{index}", "dataset": "SYNTHETIC_SEG_GUIDED",
                     "split": "train" if index < 2 else "val",
                     "image_path": paths["noisy_path"], **paths})
    base = pd.DataFrame(rows)
    templates = {"B3R": "b3r_noisy_mild_residual.yaml", "B6R": "b6r_noisy_noisy_zero_residual.yaml"}
    configs = {}
    for arm, name in templates.items():
        table = base.copy()
        if arm == "B6R": table["auxiliary_path"] = table["noisy_path"]
        manifest = output / f"manifest_{arm.lower()}.csv"; table.to_csv(manifest, index=False)
        config = load_config(root / "configs/adaptive_denoising/seg_guided_v1" / name)
        config.update(seed=42, device="cpu", protocol_id="synthetic_seg_guided_smoke")
        config["model"].update(channels=[4, 8, 16, 32], encoder_depths=[1, 1, 1, 1], decoder_depth=1)
        config["data"].update(manifest=str(manifest), root=str(root), target_size=[h, w], samples_per_epoch=2)
        config["train"].update(output_dir=str(output / arm.lower()), epochs=1,
                               fixed_checkpoint_epochs=[1], early_stopping_patience=2,
                               batch_size=1, gradient_accumulation_steps=1, num_workers=0, amp=False)
        config.setdefault("seg_guided", {}).update(enabled=True, project_root=str(root),
                                                   run_id="synthetic", primary_epoch=1)
        configs[arm] = config
    seed_everything(42, True, use_cuda=False)
    init = build_model(configs["B3R"])
    init_sha = {name: tensor_sha(value) for name, value in init.named_parameters()}
    checkpoint = output / "common_initialization.pth"
    torch.save({"model": init.state_dict(), "epoch": -1, "config": configs["B3R"]}, checkpoint)
    audits = []
    for arm, config in configs.items():
        config["train"].update(pretrained=str(checkpoint), strict_pretrained=True)
        trainer = Trainer(config); save_config(config, trainer.output_dir / "resolved_config.yaml")
        if {name: tensor_sha(value) for name, value in trainer.model.named_parameters()} != init_sha:
            raise AssertionError(f"Common initialization mismatch for {arm}")
        trainer.fit()
        if not (trainer.output_dir / "epoch001.pth").is_file():
            raise AssertionError("Fixed endpoint checkpoint missing")
        history = pd.read_csv(trainer.output_dir / "history.csv")
        audits.append({"arm": arm, "completed_epochs": int(history["epoch"].max()),
                       "fixed_endpoint_present": True})
    result = {"status": "passed", "arms": audits, "non_square_model_grid": [h, w],
              "fixed_endpoint_saved": True, "scientific_evaluation": False,
              "test_assets_opened": 0}
    write_strict_json(output / "smoke_report.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
