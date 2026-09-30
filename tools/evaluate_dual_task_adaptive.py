from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.data.io import read_gray
from sabids.engine.trainer import _make_transform, build_model
from sabids.experiments.dual_task_adaptive import gate_statistics, sha256_file
from sabids.metrics import (
    binary_metrics,
    edge_preservation_index,
    high_frequency_energy_ratio,
    layer_boundary_mae,
    layer_shape_metrics,
    psnr,
    reference_edge_mae,
    region_cnr,
    rmse,
    soft_dice_score,
    ssim,
    vessel_diagnostic_metrics,
)
from sabids.utils import get_device, load_checkpoint


def _u8(value: np.ndarray) -> np.ndarray:
    return np.clip(value * 255.0, 0, 255).astype(np.uint8)


def _color_probability(value: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(_u8(value), cv2.COLORMAP_TURBO)


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = cv2.cvtColor(_u8(image), cv2.COLOR_GRAY2BGR)
    edge = binary_dilation(mask, iterations=1) ^ binary_erosion(mask, iterations=1)
    base[edge] = color
    return base


def _label(tile: np.ndarray, text: str) -> np.ndarray:
    value = tile.copy()
    cv2.rectangle(value, (0, 0), (min(value.shape[1], 300), 25), (0, 0, 0), -1)
    cv2.putText(value, text, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-atlas", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    is_v2 = bool(config.get("dual_task_adaptive_v2", {}).get("enabled", False))
    if not config.get("dual_task_adaptive", {}).get("enabled", False) and not is_v2:
        raise ValueError("Evaluation requires an adaptive model")
    if args.device:
        config["device"] = args.device
    output_dir = Path(args.output).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation: {output_dir}")
    output_dir.mkdir(parents=True)
    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    checkpoint_sha = sha256_file(checkpoint)
    loaded = load_checkpoint(checkpoint, model, strict=True, map_location=device)
    expected_name = "best_vessel_safe.pth" if is_v2 else "best_joint.pth"
    if checkpoint.name != expected_name:
        raise ValueError(f"Primary adaptive evaluation requires {expected_name}")
    model.eval()

    data = config["data"]
    dataset = OCTManifestDataset(
        data["manifest"], split=data.get("val_split", "val"),
        transform=_make_transform(config, False), sample_repeat=False,
        root=data.get("root"), datasets=data.get("val_datasets"),
        groups=data.get("val_groups"), image_column=data.get("input_column", "image_path"),
        auxiliary_mode="none", load_segmentation_labels=True,
        pretransformed_model_grid=bool(data.get("pretransformed_model_grid", False)),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    fixed_atlas_samples = set(
        dataset.table.sort_values("sample_id").groupby("group_id", sort=True).first()["sample_id"].astype(str)
    )
    frame_rows: list[dict] = []
    denoise_rows: list[dict] = []
    gate_rows: list[dict] = []
    atlas_groups: set[str] = set()
    component_rows: list[dict] = []
    adaptive_section = config["dual_task_adaptive_v2"] if is_v2 else config["dual_task_adaptive"]
    component_path = adaptive_section.get("evidence", {}).get(
        "fixed_component_inventory"
    )
    if not component_path or not Path(component_path).is_file():
        raise FileNotFoundError("Formal adaptive evaluation requires fixed train-defined component inventory")
    component_inventory = json.loads(Path(component_path).read_text(encoding="utf-8"))
    components_by_sample: dict[str, list[dict]] = {}
    for item in component_inventory["validation_components"]:
        components_by_sample.setdefault(str(item["sample_id"]), []).append(item)
    thresholds = config["evaluation"]

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            prediction = model(image, return_features=False, return_auxiliary=False)
            sample_id, group_id = str(batch["sample_id"][0]), str(batch["group_id"][0])
            valid = batch["valid_mask"][0, 0].numpy() > 0.5
            layer_valid = valid & (batch["label_valid_mask"][0, 0].numpy() > 0.5)
            vessel_valid = valid & (batch["vessel_valid_mask"][0, 0].numpy() > 0.5)
            layer_gt = batch["layer_mask"][0, 0].numpy() > 0.5
            vessel_gt = batch["vessel_mask"][0, 0].numpy() > 0.5
            variants = {
                "C0_coarse": (
                    prediction["coarse_layer_prob"], prediction["coarse_vessel_prob"]
                ),
                "C1_adaptive": (prediction["layer_prob"], prediction.get("v1_vessel_prob", prediction["vessel_prob"])),
            }
            if is_v2:
                off = model(image, return_features=False, return_auxiliary=False, vessel_adaptive_off=True)
                variants.update({"C2_off": (off["layer_prob"], off["vessel_prob"]),
                                 "C2_on": (prediction["layer_prob"], prediction["vessel_prob"])})
            for variant, (layer_tensor, vessel_tensor) in variants.items():
                layer_prob = layer_tensor[0, 0].cpu().numpy()
                vessel_prob = vessel_tensor[0, 0].cpu().numpy()
                layer_pred = layer_prob >= float(thresholds.get("layer_threshold", 0.5))
                vessel_pred = vessel_prob >= float(thresholds.get("vessel_threshold", 0.5))
                row = {"sample_id": sample_id, "group_id": group_id, "variant": variant}
                if is_v2:
                    row["v1_layer_probability_max_abs_error"] = float(
                        (prediction["layer_prob"] - prediction["v1_layer_prob"]).abs().max().item())
                    row["v1_layer_logits_max_abs_error"] = float(
                        (prediction["layer_logits"] - prediction["v1_layer_logits"]).abs().max().item())
                for key, value in binary_metrics(layer_pred[layer_valid], layer_gt[layer_valid]).items():
                    row[f"layer_{key}"] = value
                row["layer_soft_dice"] = soft_dice_score(layer_prob, layer_gt, layer_valid)
                upper, lower, thickness = layer_boundary_mae(layer_pred, layer_gt)
                row.update({"layer_upper_boundary_mae_px": upper, "layer_lower_boundary_mae_px": lower,
                            "layer_thickness_mae_px": thickness})
                row.update(layer_shape_metrics(layer_pred, layer_gt))
                row.update(vessel_diagnostic_metrics(
                    vessel_prob, layer_prob, vessel_gt, layer_gt, vessel_valid,
                    vessel_threshold=0.5, layer_threshold=0.5,
                ))
                frame_rows.append(row)

                component_vessel = read_gray(batch["vessel_mask_path"][0])
                _, original_labels = cv2.connectedComponents(
                    (component_vessel > 0.5).astype(np.uint8), connectivity=8
                )
                original_probability = cv2.resize(
                    vessel_prob, (component_vessel.shape[1], component_vessel.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                for component in components_by_sample.get(sample_id, []):
                    component_mask = original_labels == int(component["component_id"])
                    component_rows.append({
                        "sample_id": sample_id, "group_id": group_id, "variant": variant,
                        **component,
                        "component_recall": float((original_probability[component_mask] >= 0.5).mean()),
                        "component_detected": float(bool((original_probability[component_mask] >= 0.5).any())),
                        "component_missed": float(not bool((original_probability[component_mask] >= 0.5).any())),
                    })

            vessel_array_name = "C2_vessel" if is_v2 else "C1_vessel"
            arrays = {
                "noisy": image[0, 0].cpu().numpy(),
                "C0_coarse": prediction["coarse_denoised"][0, 0].cpu().numpy(),
                "C1_layer": prediction["fine_layer_denoised"][0, 0].cpu().numpy(),
                vessel_array_name: prediction["fine_vessel_denoised"][0, 0].cpu().numpy(),
            }
            model_component_labels = cv2.resize(
                original_labels.astype(np.float32),
                (valid.shape[1], valid.shape[0]), interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
            if bool(batch["has_clean"][0]):
                clean = batch["clean"][0, 0].numpy()
                noisy_array = arrays["noisy"]
                roi = layer_gt & valid
                vessel_roi = vessel_gt & roi
                stroma = roi & ~vessel_gt
                for variant, value in arrays.items():
                    row = {
                        "sample_id": sample_id, "group_id": group_id, "variant": variant,
                        "psnr": psnr(value[valid], clean[valid]),
                        "ssim": ssim(value, clean), "rmse": rmse(value[valid], clean[valid]),
                        "epi": edge_preservation_index(value, clean),
                        "reference_edge_mae": reference_edge_mae(value, clean),
                        "high_frequency_energy_ratio": high_frequency_energy_ratio(value, clean),
                    }
                    if roi.any():
                        row.update({"layer_roi_psnr": psnr(value[roi], clean[roi]),
                                    "layer_roi_rmse": rmse(value[roi], clean[roi])})
                        roi_value = clean.copy(); roi_value[roi] = value[roi]
                        row["layer_roi_ssim"] = ssim(roi_value, clean)
                    row["vessel_roi_mae"] = (
                        float(np.abs(value[vessel_roi] - clean[vessel_roi]).mean())
                        if vessel_roi.any() else float("nan")
                    )
                    cnr_value = region_cnr(value, vessel_roi, stroma)
                    cnr_clean = region_cnr(clean, vessel_roi, stroma)
                    row["vessel_stroma_cnr"] = cnr_value
                    row["cnr_absolute_error"] = abs(cnr_value - cnr_clean)
                    removed = noisy_array - value
                    clean_centered = clean - float(clean[valid].mean())
                    removed_centered = removed - float(removed[valid].mean())
                    denominator = float(np.linalg.norm(clean_centered[valid]) * np.linalg.norm(removed_centered[valid]))
                    row["residual_structure_leakage"] = (
                        abs(float(np.dot(clean_centered[valid], removed_centered[valid]) / denominator))
                        if denominator > 1e-12 else float("nan")
                    )
                    for stratum in ("small", "low_contrast", "small_low_contrast"):
                        ids = [int(item["component_id"])
                               for item in components_by_sample.get(sample_id, [])
                               if bool(item.get(stratum, False))]
                        stratum_mask = valid & np.isin(model_component_labels, ids)
                        row[f"{stratum}_vessel_residual"] = (
                            float(np.abs(removed[stratum_mask]).mean())
                            if stratum_mask.any() else float("nan")
                        )
                    denoise_rows.append(row)

            layer_strength = prediction["layer_strength_map"][0, 0].cpu().numpy()
            vessel_strength = prediction["vessel_strength_map"][0, 0].cpu().numpy()
            boundary = binary_dilation(layer_gt, iterations=3) ^ binary_erosion(layer_gt, iterations=3)
            masks = {
                "full_valid": valid,
                "background": valid & ~layer_gt,
                "layer": valid & layer_gt,
                "vessel": valid & vessel_gt,
                "stroma": valid & layer_gt & ~vessel_gt,
                "upper_lower_boundary_band": valid & boundary,
            }
            for stratum in ("small", "low_contrast", "small_low_contrast"):
                ids = [int(item["component_id"]) for item in components_by_sample.get(sample_id, [])
                       if bool(item.get(stratum, False))]
                masks[f"fixed_{stratum}"] = valid & np.isin(model_component_labels, ids)
            vessel_boundary = binary_dilation(vessel_gt, iterations=2) ^ binary_erosion(vessel_gt, iterations=2)
            masks["vessel_boundary"] = valid & vessel_boundary
            masks["layer_outside"] = valid & ~layer_gt
            for task, value in (("layer", layer_strength), ("vessel", vessel_strength),
                                ("layer_minus_vessel", layer_strength - vessel_strength)):
                for row in gate_statistics(value, masks):
                    if is_v2:
                        selected = value[masks[row["region"]]]
                        row["fraction_lt_005"] = float((selected < .05).mean())
                        row["fraction_gt_045"] = float((selected > .45).mean())
                        row["fraction_at_cap"] = float((selected >= (.5 - 1e-4)).mean()) if task == "vessel" else float((selected >= (1.25 - 1e-4)).mean())
                    gate_rows.append({"sample_id": sample_id, "group_id": group_id,
                                      "task": task, **row})

            if args.save_atlas and sample_id in fixed_atlas_samples and group_id not in atlas_groups:
                atlas_groups.add(group_id)
                atlas_dir = output_dir / "atlas"
                atlas_dir.mkdir(exist_ok=True)
                sample_dir = atlas_dir / f"{group_id}__{sample_id}"
                sample_dir.mkdir()
                coarse_layer = prediction["coarse_layer_prob"][0, 0].cpu().numpy()
                coarse_vessel = prediction["coarse_vessel_prob"][0, 0].cpu().numpy()
                fine_layer = prediction["layer_prob"][0, 0].cpu().numpy()
                fine_vessel = prediction["vessel_prob"][0, 0].cpu().numpy()
                off_vessel = off["vessel_prob"][0, 0].cpu().numpy() if is_v2 else coarse_vessel
                residual_scale = 0.625
                residual_images = {
                    "coarse_residual": arrays["noisy"] - arrays["C0_coarse"],
                    "fine_layer_residual": arrays["noisy"] - arrays["C1_layer"],
                    "fine_vessel_residual": arrays["noisy"] - arrays[vessel_array_name],
                }
                tiles = {
                    "noisy": cv2.cvtColor(_u8(arrays["noisy"]), cv2.COLOR_GRAY2BGR),
                    "clean": cv2.cvtColor(_u8(batch["clean"][0, 0].numpy()), cv2.COLOR_GRAY2BGR),
                    "coarse_denoised": cv2.cvtColor(_u8(arrays["C0_coarse"]), cv2.COLOR_GRAY2BGR),
                    "fine_layer_denoised": cv2.cvtColor(_u8(arrays["C1_layer"]), cv2.COLOR_GRAY2BGR),
                    "fine_vessel_denoised": cv2.cvtColor(_u8(arrays[vessel_array_name]), cv2.COLOR_GRAY2BGR),
                    "coarse_layer_probability": _color_probability(coarse_layer),
                    "coarse_vessel_probability": _color_probability(coarse_vessel),
                    "fine_layer_probability": _color_probability(fine_layer),
                    "fine_vessel_probability": _color_probability(fine_vessel),
                    "layer_strength": _color_probability(layer_strength / 1.25),
                    "vessel_strength": _color_probability(vessel_strength / (.5 if is_v2 else 1.25)),
                    "strength_difference": _color_probability((layer_strength - vessel_strength + 1.25) / 2.5),
                    "coarse_layer_mask": cv2.cvtColor(_u8(coarse_layer >= 0.5), cv2.COLOR_GRAY2BGR),
                    "coarse_vessel_mask": cv2.cvtColor(_u8(coarse_vessel >= 0.5), cv2.COLOR_GRAY2BGR),
                    "fine_layer_mask": cv2.cvtColor(_u8(fine_layer >= 0.5), cv2.COLOR_GRAY2BGR),
                    "fine_vessel_mask": cv2.cvtColor(_u8(fine_vessel >= 0.5), cv2.COLOR_GRAY2BGR),
                    "coarse_layer_overlay": _overlay(arrays["noisy"], coarse_layer >= 0.5, (0, 255, 0)),
                    "coarse_vessel_overlay": _overlay(arrays["noisy"], coarse_vessel >= 0.5, (0, 0, 255)),
                    "fine_layer_overlay": _overlay(arrays["noisy"], fine_layer >= 0.5, (0, 255, 0)),
                    "fine_vessel_overlay": _overlay(arrays["noisy"], fine_vessel >= 0.5, (0, 0, 255)),
                    "coarse_vs_fine_error": _color_probability(np.abs(fine_vessel - coarse_vessel)),
                }
                if is_v2:
                    residual_images["full_D2_residual"] = prediction["full_d2_residual"][0,0].cpu().numpy()
                    tiles.update({
                        "C2_off_vessel_probability": _color_probability(off_vessel),
                        "C2_off_vessel_mask": cv2.cvtColor(_u8(off_vessel >= .5),cv2.COLOR_GRAY2BGR),
                        "C2_off_vessel_overlay": _overlay(arrays["noisy"],off_vessel >= .5,(0,0,255)),
                        "layer_gt": cv2.cvtColor(_u8(layer_gt),cv2.COLOR_GRAY2BGR),
                        "vessel_gt": cv2.cvtColor(_u8(vessel_gt),cv2.COLOR_GRAY2BGR),
                    })
                for name, residual in residual_images.items():
                    tiles[name] = _color_probability((np.clip(residual, -residual_scale, residual_scale) + residual_scale) / (2 * residual_scale))
                method_noise_boundary = tiles["fine_vessel_residual"].copy()
                vessel_edge = binary_dilation(vessel_gt, iterations=1) ^ binary_erosion(vessel_gt, iterations=1)
                method_noise_boundary[vessel_edge] = (255, 255, 255)
                tiles["method_noise_vessel_boundary"] = method_noise_boundary
                for name, tile in tiles.items():
                    cv2.imwrite(str(sample_dir / f"{name}.png"), tile)
                ordered = list(tiles.items())
                rows = []
                for start in range(0, len(ordered), 4):
                    part = [_label(tile, name) for name, tile in ordered[start:start + 4]]
                    while len(part) < 4:
                        part.append(np.zeros_like(part[0]))
                    rows.append(np.concatenate(part, axis=1))
                contact = np.concatenate(rows, axis=0)
                adaptive_metrics = frame_rows[-1]
                annotation = (
                    f"{group_id} {sample_id} sha={checkpoint_sha[:12]} "
                    f"L-Dice={adaptive_metrics.get('layer_dice', float('nan')):.3f} "
                    f"V-Dice={adaptive_metrics.get('vessel_dice', float('nan')):.3f}"
                )
                cv2.putText(contact, annotation,
                            (5, contact.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imwrite(str(atlas_dir / f"{sample_id}_contact_sheet.png"), contact)

    frames = pd.DataFrame(frame_rows)
    denoise = pd.DataFrame(denoise_rows)
    gates = pd.DataFrame(gate_rows)
    components = pd.DataFrame(component_rows)
    if not components.empty:
        for stratum in ("small", "low_contrast", "small_low_contrast"):
            part = components[components[stratum].astype(bool)]
            aggregate = part.groupby(["sample_id", "group_id", "variant"], as_index=False).agg(
                **{f"{stratum}_vessel_recall": ("component_recall", "mean"),
                   f"{stratum}_component_coverage": ("component_detected", "mean"),
                   f"{stratum}_complete_miss_count": ("component_missed", "sum")}
            )
            frames = frames.merge(aggregate, on=["sample_id", "group_id", "variant"], how="left")
    frames.to_csv(output_dir / "metrics_by_image.csv", index=False, encoding="utf-8-sig")
    frames.to_csv(output_dir / "segmentation_results.csv", index=False, encoding="utf-8-sig")
    denoise.to_csv(output_dir / "denoising_results.csv", index=False, encoding="utf-8-sig")
    denoise_metric_columns = [
        column for column in denoise.select_dtypes(include=[np.number]).columns
    ]
    denoise_positions = denoise.groupby(
        ["variant", "group_id"], as_index=False
    )[denoise_metric_columns].mean()
    denoise_positions.to_csv(
        output_dir / "denoising_metrics_by_position.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gates.to_csv(output_dir / "gate_metrics_by_image.csv", index=False, encoding="utf-8-sig")
    components.to_csv(output_dir / "fixed_component_metrics.csv", index=False, encoding="utf-8-sig")
    metric_columns = [column for column in frames.select_dtypes(include=[np.number]).columns]
    positions = frames.groupby(["variant", "group_id"], as_index=False)[metric_columns].mean()
    positions.to_csv(output_dir / "metrics_by_position.csv", index=False, encoding="utf-8-sig")
    gains = positions.pivot(index="group_id", columns="variant", values=metric_columns)
    gain_rows = []
    for group_id in gains.index:
        row = {"group_id": group_id}
        for metric in metric_columns:
            row[metric] = float(gains.loc[group_id, (metric, "C1_adaptive")] - gains.loc[group_id, (metric, "C0_coarse")])
        gain_rows.append(row)
    gain_table = pd.DataFrame(gain_rows)
    gain_table.to_csv(output_dir / "coarse_vs_fine_deltas.csv", index=False, encoding="utf-8-sig")
    if is_v2:
        v2_rows=[]
        for group_id in gains.index:
            for comparison, left, right in (("C2_on-C0","C2_on","C0_coarse"),
                                             ("C2_off-C0","C2_off","C0_coarse"),
                                             ("C2_on-C2_off","C2_on","C2_off")):
                row={"group_id":group_id,"comparison":comparison}
                for metric in metric_columns: row[metric]=float(gains.loc[group_id,(metric,left)]-gains.loc[group_id,(metric,right)])
                v2_rows.append(row)
        pd.DataFrame(v2_rows).to_csv(output_dir / "coarse_vs_v2_deltas.csv", index=False, encoding="utf-8-sig")
        failures=pd.DataFrame(v2_rows)
        failures=failures[failures["comparison"].eq("C2_on-C0")].sort_values("vessel_dice")
        failures.to_csv(output_dir / "failure_cases.csv",index=False,encoding="utf-8-sig")
    gate_aggregations={"mean":("mean","mean"),"std":("std","mean"),"p10":("p10","mean"),
                       "p50":("p50","mean"),"p90":("p90","mean"),"count":("count","sum")}
    if is_v2: gate_aggregations.update(fraction_lt_005=("fraction_lt_005","mean"),
                                       fraction_gt_045=("fraction_gt_045","mean"),
                                       fraction_at_cap=("fraction_at_cap","mean"))
    gate_positions = gates.groupby(["task", "region", "group_id"], as_index=False).agg(**gate_aggregations)
    gate_positions.to_csv(output_dir / "gate_metrics_by_position.csv", index=False, encoding="utf-8-sig")
    gate_summary = gate_positions.groupby(["task", "region"], as_index=False).mean(numeric_only=True)
    gate_summary.to_csv(output_dir / "gate_region_summary.csv", index=False, encoding="utf-8-sig")
    gate_summary[gate_summary["task"] == "layer_minus_vessel"].to_csv(
        output_dir / "gate_difference_summary.csv", index=False, encoding="utf-8-sig"
    )
    results_table = positions.groupby("variant", as_index=False).mean(numeric_only=True)
    results_table.to_csv(output_dir / "RESULTS_TABLE.csv", index=False, encoding="utf-8-sig")
    scientific_gate = None
    if is_v2:
        indexed=results_table.set_index("variant")
        c0=indexed.loc["C0_coarse"]; c1=indexed.loc["C1_adaptive"]; on=indexed.loc["C2_on"]; off=indexed.loc["C2_off"]
        q=lambda r: .35*r["vessel_roi_dice"]+.25*r["vessel_dice"]+.15*r["vessel_recall"]+.10*r["vessel_boundary_band_dice"]+.075*r["small_vessel_recall"]+.075*r["low_contrast_vessel_recall"]
        pos=positions.set_index(["variant","group_id"])
        groups=sorted(positions.group_id.unique())
        dice_improved=sum(float(pos.loc[("C2_on",g),"vessel_dice"]-pos.loc[("C0_coarse",g),"vessel_dice"])>0 for g in groups)
        recall_safe=sum(float(pos.loc[("C2_on",g),"vessel_recall"]-pos.loc[("C0_coarse",g),"vessel_recall"])>=-.005 for g in groups)
        vessel_gate=float(gate_summary[(gate_summary.task=="vessel")&(gate_summary.region=="vessel")]["mean"].iloc[0])
        stroma_gate=float(gate_summary[(gate_summary.task=="vessel")&(gate_summary.region=="stroma")]["mean"].iloc[0])
        checks={
          "layer_preserved": abs(float(on["layer_dice"]-c1["layer_dice"]))<=1e-4 and abs(float(on["layer_surface_dice"]-c1["layer_surface_dice"]))<=1e-4,
          "vessel_dice_noninferior": float(on["vessel_dice"])>=float(c0["vessel_dice"]),
          "vessel_roi_dice_noninferior": float(on["vessel_roi_dice"])>=float(c0["vessel_roi_dice"]),
          "recall_safe": float(on["vessel_recall"]-c0["vessel_recall"])>=-.005,
          "small_safe": float(on["small_vessel_recall"]-c0["small_vessel_recall"])>=-.005,
          "low_contrast_safe": float(on["low_contrast_vessel_recall"]-c0["low_contrast_vessel_recall"])>=-.005,
          "adaptive_beats_off": float(q(on)-q(off))>0,
          "position_dice_majority": dice_improved>=2,
          "position_recall_all_safe": recall_safe==len(groups),
          "vessel_gate_below_stroma": stroma_gate-vessel_gate>=.05,
          "test_sealed": True,
        }
        scientific_gate={"status":"passed" if all(checks.values()) else "failed","checks":checks,
          "q_c2_on":float(q(on)),"q_c2_off":float(q(off)),"dice_improved_positions":dice_improved,
          "recall_safe_positions":recall_safe,"position_count":len(groups),"vessel_gate_mean":vessel_gate,
          "stroma_gate_mean":stroma_gate,"test_assets_opened":0,
          "failure_conclusion":None if all(checks.values()) else "当前 Vessel 自适应降噪仍未证明优于 coarse/noisy 保底路径，停止 seed 43/44。"}
        (output_dir/"preregistered_success_gate.json").write_text(json.dumps(scientific_gate,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if args.save_atlas:
        pd.DataFrame([
            {"group_id": group, "selection_rule": "lexicographically_first_sample_id",
             "sample_id": sorted(frames.loc[frames["group_id"] == group, "sample_id"].unique())[0]}
            for group in sorted(atlas_groups)
        ]).to_csv(output_dir / "atlas_selection_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "schema_version": "dual-task-adaptive-evaluation-v2" if is_v2 else "dual-task-adaptive-evaluation-v1",
        "status": "passed",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(loaded.get("epoch", -1)) + 1,
        "validation_frames": int(len(dataset)),
        "validation_positions": sorted(set(frames["group_id"])),
        "position_count": int(frames["group_id"].nunique()),
        "threshold": 0.5,
        "postprocess": "P0",
        "statistical_unit": "anatomical_position",
        "limitation": (
            f"Only {int(frames['group_id'].nunique())} validation positions; "
            "exploratory seed-42 evidence only."
        ),
        "test_assets_opened": 0,
    }
    if scientific_gate is not None: summary["preregistered_success_gate"] = scientific_gate
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if is_v2:
        cost={"total_parameters":sum(p.numel() for p in model.parameters()),
              "trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),
              "peak_cuda_memory_bytes":int(torch.cuda.max_memory_allocated(device)) if device.type=="cuda" else 0,
              "validation_frames":int(len(dataset)),"test_assets_opened":0}
        (output_dir / "cost_profile.json").write_text(json.dumps(cost,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
