from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.engine.trainer import _make_transform, build_model
from sabids.utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Same-checkpoint interaction dependence/perturbation diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test":
        raise ValueError("Reserved test evaluation is prohibited for interaction diagnostics")
    config = load_config(args.config)
    if args.device:
        config["device"] = args.device
    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    load_checkpoint(args.checkpoint, model, strict=True, map_location=device)
    model.set_train_stage(
        str(config.get("train", {}).get("stage", "interaction")),
        freeze_shared_encoder=bool(config.get("model", {}).get("freeze_shared_encoder", False)),
    )
    model.eval()
    dataset = OCTManifestDataset(
        config["data"]["manifest"], split=args.split, transform=_make_transform(config, False),
        sample_repeat=False, root=config["data"].get("root"),
    )
    groups = sorted(dataset.groups)
    if len(groups) < 2:
        raise ValueError("Diagnostics require two distinct anatomical positions")
    indices = [dataset.groups[groups[0]][0], dataset.groups[groups[1]][0]]
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=2, shuffle=False, num_workers=0)))
    image = batch["image"].to(device)
    height, width = image.shape[-2:]
    cases = {
        "reference": {},
        "d2s_off": {"d2s_strength": 0.0},
        "s2d_off": {"s2d_strength": 0.0},
        "both_off": {"d2s_strength": 0.0, "s2d_strength": 0.0},
        "spatial_misalignment": {"guidance_roll": (height // 4, width // 4)},
        "other_position_guidance": {"other_position_guidance": True},
    }
    for strength in (0.25, 0.5, 1.5, 2.0):
        cases[f"strength_{strength:g}"] = {"d2s_strength": strength, "s2d_strength": strength}
    outputs, timings = {}, {}
    with torch.no_grad():
        _ = model(image, return_features=False, return_auxiliary=False)
        for name, diagnostic in cases.items():
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            output = model(image, return_features=False, return_auxiliary=True, interaction_diagnostic=diagnostic)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings[name] = {
                "seconds_for_two_frames": time.perf_counter() - start,
                "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            }
            outputs[name] = output
    reference = outputs["reference"]
    rows = []
    for name, output in outputs.items():
        row = {"case": name, **timings[name]}
        for key in ("denoised", "layer_prob", "vessel_prob"):
            row[f"{key}_mean_abs_delta"] = float((output[key] - reference[key]).abs().mean().item())
            row[f"{key}_finite"] = bool(torch.isfinite(output[key]).all().item())
        for item in output.get("auxiliary", []):
            level = int(item["level"].item())
            for key in (
                "seg_scale_abs_mean", "layer_scale_abs_mean", "vessel_scale_abs_mean",
                "seg_to_denoise_injection_relative_rms", "denoise_to_layer_injection_relative_rms",
                "denoise_to_vessel_injection_relative_rms", "guidance_layer_probability_mean",
                "guidance_vessel_probability_mean",
            ):
                if key in item:
                    row[f"level{level}_{key}"] = float(item[key].detach().float().mean().item())
        rows.append(row)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(output_dir / "interaction_dependence_diagnostics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "classification": "dependence/perturbation diagnostic; not a retraining gain",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "group_ids": groups[:2],
        "batch_size": 2,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count_from_config": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "notes": "Other-position guidance uses a deterministic two-position cyclic permutation; no batch-size-1 shuffle is used.",
    }
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
