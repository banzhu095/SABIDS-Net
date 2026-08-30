from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cv2
from scipy.ndimage import distance_transform_edt, label as connected_components
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
    parser.add_argument("--groups", nargs="+", default=None)
    parser.add_argument(
        "--component-size-thresholds", type=int, nargs=2, default=None,
        metavar=("SMALL_MAX", "MEDIUM_MAX"),
        help="Pre-registered component areas in original label-grid pixels.",
    )
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
    if args.groups:
        requested = [str(value).lower() for value in args.groups]
        selected_groups = [
            group for group in groups
            if str(group).lower() in requested
        ]
        missing_groups = sorted(set(requested) - {str(group).lower() for group in selected_groups})
        if missing_groups:
            raise ValueError(f"Requested validation groups are absent: {missing_groups}")
        groups = selected_groups
    if len(groups) < 2:
        raise ValueError("Diagnostics require two distinct anatomical positions")
    indices = [dataset.groups[group][0] for group in groups]
    batch = next(iter(DataLoader(Subset(dataset, indices), batch_size=len(indices), shuffle=False, num_workers=0)))
    image = batch["image"].to(device)
    height, width = image.shape[-2:]
    cases = {
        "aligned_learned_scale": {},
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
        learned = model(image, return_features=False, return_auxiliary=True)
        current_d2s = np.mean([
            max(float(item.get("denoise_to_layer_injection_relative_rms", 0.0)),
                float(item.get("denoise_to_vessel_injection_relative_rms", 0.0)))
            for item in learned.get("auxiliary", [])
        ])
        current_s2d = np.mean([
            float(item.get("seg_to_denoise_injection_relative_rms", 0.0))
            for item in learned.get("auxiliary", [])
        ])
        for target in (0.001, 0.0025, 0.005, 0.01):
            cases[f"target_injection_rms_{100 * target:g}pct"] = {
                "d2s_strength": target / current_d2s if current_d2s > 0 else 0.0,
                "s2d_strength": target / current_s2d if current_s2d > 0 else 0.0,
            }
        cases["self_feature_capacity_control"] = {"source_mode": "receiver_capacity"}
        for name, diagnostic in cases.items():
            diagnostic = dict(diagnostic)
            source_mode = diagnostic.pop("source_mode", "cross")
            model.s2d_source_mode = source_mode
            model.d2s_source_mode = source_mode
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
        model.s2d_source_mode = "cross"
        model.d2s_source_mode = "cross"
    reference = outputs["aligned_learned_scale"]
    rows = []
    for name, output in outputs.items():
        row = {"case": name, **timings[name]}
        for key in ("denoised", "layer_prob", "vessel_prob"):
            row[f"{key}_mean_abs_delta"] = float((output[key] - reference[key]).abs().mean().item())
            row[f"{key}_finite"] = bool(torch.isfinite(output[key]).all().item())
        for item in output.get("auxiliary", []):
            level = int(item["level"].item())
            for key, value in item.items():
                if key in {"level", "direction"} or not torch.is_tensor(value) or value.numel() != 1:
                    continue
                row[f"level{level}_{key}"] = float(value.detach().float().item())
        rows.append(row)
    output_dir = Path(args.output)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    pd.DataFrame(rows).to_csv(output_dir / "interaction_dependence_diagnostics.csv", index=False, encoding="utf-8-sig")
    pixel_rows = []
    for case, output in outputs.items():
        for index, sample_id in enumerate(batch["sample_id"]):
            row = {
                "case": case, "sample_id": sample_id, "group_id": batch["group_id"][index],
                "layer_logit_mean_abs_delta": float((output["layer_logits"][index] - reference["layer_logits"][index]).abs().mean()),
                "vessel_logit_mean_abs_delta": float((output["vessel_logits"][index] - reference["vessel_logits"][index]).abs().mean()),
                "layer_probability_mean_abs_delta": float((output["layer_prob"][index] - reference["layer_prob"][index]).abs().mean()),
                "vessel_probability_mean_abs_delta": float((output["vessel_prob"][index] - reference["vessel_prob"][index]).abs().mean()),
                "layer_pixels_probability_0_45_to_0_55": int(((output["layer_prob"][index] >= 0.45) & (output["layer_prob"][index] <= 0.55)).sum()),
                "vessel_pixels_probability_0_45_to_0_55": int(((output["vessel_prob"][index] >= 0.45) & (output["vessel_prob"][index] <= 0.55)).sum()),
            }
            if bool(batch["has_vessel"][index]):
                valid = batch["vessel_valid_mask"][index].to(device) > 0.5
                target = batch["vessel_mask"][index].to(device) > 0.5
                prediction = output["vessel_prob"][index] >= 0.5
                ref_prediction = reference["vessel_prob"][index] >= 0.5
                for key, mask in (
                    ("tp", prediction & target & valid), ("fp", prediction & ~target & valid),
                    ("fn", ~prediction & target & valid),
                ):
                    ref_mask = {
                        "tp": ref_prediction & target & valid,
                        "fp": ref_prediction & ~target & valid,
                        "fn": ~ref_prediction & target & valid,
                    }[key]
                    row[f"vessel_{key}_change"] = int(mask.sum() - ref_mask.sum())
                layer = batch["layer_mask"][index].to(device) > 0.5
                row["predicted_vessel_inside_layer_change"] = int(((prediction & layer & valid).sum() - (ref_prediction & layer & valid).sum()))
                row["predicted_vessel_outside_layer_change"] = int(((prediction & ~layer & valid).sum() - (ref_prediction & ~layer & valid).sum()))
                stroma = layer & ~target & valid
                row["stroma_false_positive_change"] = int(
                    (prediction & stroma).sum() - (ref_prediction & stroma).sum()
                )
                target_np = target[0].detach().cpu().numpy()
                valid_np = valid[0].detach().cpu().numpy()
                surface = target_np & ~cv2.erode(
                    target_np.astype(np.uint8), np.ones((3, 3), np.uint8)
                ).astype(bool)
                original_h = int(batch["original_height"][index])
                original_w = int(batch["original_width"][index])
                linear_scale = np.sqrt(float(valid_np.sum()) / max(float(original_h * original_w), 1.0))
                distance = distance_transform_edt(~surface) if surface.any() else np.full(surface.shape, np.inf)
                boundary_band = torch.from_numpy(
                    valid_np & (distance <= max(3.0 * linear_scale, 1.0))
                ).to(device=device)[None]
                for key, mask in (
                    ("fp", ~target), ("fn", target),
                ):
                    current_error = (prediction if key == "fp" else ~prediction) & mask & boundary_band
                    reference_error = (ref_prediction if key == "fp" else ~ref_prediction) & mask & boundary_band
                    row[f"boundary_band_{key}_change"] = int(current_error.sum() - reference_error.sum())
                if args.component_size_thresholds:
                    small_max_original = int(args.component_size_thresholds[0])
                    labelled, count = connected_components(target_np & valid_np)
                    area_scale = float(valid_np.sum()) / max(float(original_h * original_w), 1.0)
                    small_max_model = max(1.0, small_max_original * area_scale)
                    small_ids = [
                        component_id for component_id in range(1, count + 1)
                        if float((labelled == component_id).sum()) <= small_max_model
                    ]
                    small_mask = torch.from_numpy(np.isin(labelled, small_ids)).to(device=device)[None]
                    row["small_vessel_true_positive_change"] = int(
                        (prediction & small_mask).sum() - (ref_prediction & small_mask).sum()
                    )
                    row["small_vessel_gt_pixels_model_grid"] = int(small_mask.sum())
            pixel_rows.append(row)
    pd.DataFrame(pixel_rows).to_csv(output_dir / "selected_position_perturbation_metrics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "classification": "dependence/perturbation diagnostic; not a retraining gain",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "group_ids": groups,
        "batch_size": len(groups),
        "learned_mean_d2s_injection_relative_rms": float(current_d2s),
        "learned_mean_s2d_injection_relative_rms": float(current_s2d),
        "component_size_thresholds_original_pixels": args.component_size_thresholds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count_from_config": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "notes": "Other-position guidance uses a deterministic two-position cyclic permutation; no batch-size-1 shuffle is used.",
    }
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
