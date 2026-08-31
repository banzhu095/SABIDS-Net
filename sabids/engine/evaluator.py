from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data.io import write_gray, write_rgb
from ..metrics import (
    automatic_cnr,
    binary_metrics,
    edge_preservation_index,
    reference_edge_mae,
    layer_boundary_mae,
    layer_shape_metrics,
    psnr,
    reconstruction_snr,
    region_cnr,
    rmse,
    ssim,
    surface_distances,
    vessel_area_fraction,
    vessel_diagnostic_metrics,
)
from ..postprocessing import clean_layer_mask, hard_contain_vessel, regularize_lower_boundary
from ..utils import mean_dict, write_json


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Optional[str | Path] = None,
    threshold: float = 0.5,
    layer_threshold: Optional[float] = None,
    vessel_threshold: Optional[float] = None,
    axial_spacing: float = 1.0,
    lateral_spacing: float = 1.0,
    save_predictions: bool = False,
    stage: Optional[str] = None,
    input_normalization: Optional[str] = None,
    component_size_thresholds: Optional[tuple[int, int]] = None,
    boundary_band_width: float = 3.0,
    tasks: Optional[tuple[str, ...]] = None,
    postprocess_modes: tuple[str, ...] = ("p0",),
    restore_original_geometry: bool = False,
    layer_surface_tolerance: float = 3.0,
    p1_minimum_main_fraction: float = 0.5,
    p2_smoothness: float = 2.0,
    p2_max_displacement: int = 8,
) -> Dict[str, object]:
    model.eval()
    layer_threshold = threshold if layer_threshold is None else layer_threshold
    vessel_threshold = threshold if vessel_threshold is None else vessel_threshold
    requested = set(tasks) if tasks is not None else None
    evaluate_denoising = "denoise" in requested if requested is not None else stage not in {"segment", "private_seg"}
    evaluate_layer = "layer" in requested if requested is not None else stage != "denoise"
    evaluate_vessel = "vessel" in requested if requested is not None else stage != "denoise"
    evaluate_segmentation = evaluate_layer or evaluate_vessel
    modes = tuple(dict.fromkeys(mode.lower() for mode in postprocess_modes))
    invalid_modes = set(modes) - {"p0", "p1", "p2", "p3"}
    if invalid_modes:
        raise ValueError(f"Unknown postprocess modes: {sorted(invalid_modes)}")
    rows = []
    repeat_outputs = []
    qualitative_crops = []
    output_path = Path(output_dir) if output_dir else None
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        image = batch["image"].to(device, non_blocking=True)
        output = model(
            image, return_features=False, return_auxiliary=False
        )
        denoised = output["denoised"].cpu().numpy()
        layer_probability = output["layer_prob"].cpu().numpy()
        vessel_probability = output["vessel_prob"].cpu().numpy()
        batch_size = image.shape[0]
        for index in range(batch_size):
            target = None
            vessel_true = None
            layer_true = None
            error_outside_both = None
            error_outside_gt_inside_pred = None
            gt_vessel_outside_pred_layer = None
            vessel_oracle_gt_layer = None
            vessel_pred_layer_soft_gate = None
            row: Dict[str, object] = {
                "sample_id": batch["sample_id"][index],
                "group_id": batch["group_id"][index],
                "patient_id": batch["patient_id"][index],
                "dataset": batch["dataset"][index],
                "scan_protocol": batch["scan_protocol"][index],
                "original_path": batch["original_path"][index],
                "clean_path": batch["clean_path"][index],
                "layer_mask_path": batch["layer_mask_path"][index],
                "vessel_mask_path": batch["vessel_mask_path"][index],
                "label_valid_mask_path": (
                    batch["label_valid_mask_path"][index]
                    if "label_valid_mask_path" in batch else ""
                ),
                "original_height": int(batch["original_height"][index]),
                "original_width": int(batch["original_width"][index]),
                "model_input_height": int(image.shape[-2]),
                "model_input_width": int(image.shape[-1]),
                "manifest_group_frames": int(
                    batch["manifest_group_frames"][index]
                ),
            }
            valid = batch["valid_mask"][index, 0].numpy() > 0.5
            coordinates = np.argwhere(valid)
            if coordinates.size:
                y0, x0 = coordinates.min(axis=0)
                y1, x1 = coordinates.max(axis=0) + 1
            else:
                y0, x0, y1, x1 = 0, 0, valid.shape[0], valid.shape[1]
            crop = np.s_[y0:y1, x0:x1]
            layer_prob_eval = layer_probability[index, 0][crop]
            vessel_prob_eval = vessel_probability[index, 0][crop]
            noisy_eval = batch["image"][index, 0].numpy()[crop]
            denoised_eval = denoised[index, 0][crop]
            valid_eval = valid[crop]
            original_size = (row["original_width"], row["original_height"])

            def restored(array: np.ndarray, is_mask: bool = False) -> np.ndarray:
                if not restore_original_geometry or array.shape == original_size[::-1]:
                    return array
                interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
                resized = cv2.resize(array.astype(np.float32), original_size, interpolation=interpolation)
                return resized > 0.5 if is_mask else resized

            layer_prob_eval = restored(layer_prob_eval)
            vessel_prob_eval = restored(vessel_prob_eval)
            noisy_eval = restored(noisy_eval)
            denoised_eval = restored(denoised_eval)
            valid_eval = restored(valid_eval, True)
            layer_valid_eval = (
                restored(batch["label_valid_mask"][index, 0].numpy()[crop], True)
                if "label_valid_mask" in batch else valid_eval.copy()
            ) & valid_eval
            layer_pred = (layer_prob_eval >= layer_threshold) & valid_eval
            vessel_pred = (vessel_prob_eval >= vessel_threshold) & valid_eval
            row["evaluation_height"], row["evaluation_width"] = layer_pred.shape
            vessel_tp = None
            vessel_fp = None
            vessel_fn = None
            vessel_roi_tp = None
            vessel_roi_fp = None
            vessel_roi_fn = None
            if evaluate_segmentation:
                predicted_vessel_pixels = float(vessel_pred.sum())
                predicted_layer_pixels = float(layer_pred.sum())
                intersection = float(np.logical_and(vessel_pred, layer_pred).sum())
                row["pred_layer_vessel_dice"] = (
                    2.0 * intersection + 1e-6
                ) / (predicted_vessel_pixels + predicted_layer_pixels + 1e-6)
                row["pred_vessel_fraction_of_layer"] = intersection / max(
                    predicted_layer_pixels, 1.0
                )
                row["pred_vessel_outside_layer_fraction"] = (
                    predicted_vessel_pixels - intersection
                ) / max(predicted_vessel_pixels, 1.0)
            if evaluate_denoising and bool(batch["has_clean"][index]):
                target = restored(batch["clean"][index, 0].numpy()[crop])
                denoised_crop, noisy_crop = denoised_eval, noisy_eval
                row["psnr"] = psnr(denoised_crop[valid_eval], target[valid_eval])
                row["psnr_noisy"] = psnr(noisy_crop[valid_eval], target[valid_eval])
                row["psnr_gain_db"] = row["psnr"] - row["psnr_noisy"]
                row["ssim"] = ssim(denoised_crop, target)
                row["ssim_noisy"] = ssim(noisy_crop, target)
                row["ssim_gain"] = row["ssim"] - row["ssim_noisy"]
                row["rmse"] = rmse(denoised_crop, target)
                row["rmse_noisy"] = rmse(noisy_crop, target)
                row["rmse_reduction"] = row["rmse_noisy"] - row["rmse"]
                row["mse"] = float(np.mean((denoised_crop[valid_eval].astype(np.float64) - target[valid_eval]) ** 2))
                row["mse_noisy"] = float(np.mean((noisy_crop[valid_eval].astype(np.float64) - target[valid_eval]) ** 2))
                row["mse_reduction"] = row["mse_noisy"] - row["mse"]
                row["mae"] = float(np.mean(np.abs(denoised_crop[valid_eval].astype(np.float64) - target[valid_eval])))
                row["mae_noisy"] = float(np.mean(np.abs(noisy_crop[valid_eval].astype(np.float64) - target[valid_eval])))
                row["mae_reduction"] = row["mae_noisy"] - row["mae"]
                row["epi"] = edge_preservation_index(denoised_crop, target)
                row["epi_noisy"] = edge_preservation_index(noisy_crop, target)
                row["epi_gain"] = row["epi"] - row["epi_noisy"]
                row["reference_edge_mae"] = reference_edge_mae(denoised_crop, target)
                row["reference_edge_mae_noisy"] = reference_edge_mae(noisy_crop, target)
                row["reference_edge_mae_reduction"] = (
                    row["reference_edge_mae_noisy"] - row["reference_edge_mae"]
                )
                row["snr_noisy_db"] = reconstruction_snr(noisy_crop, target)
                row["snr_denoised_db"] = reconstruction_snr(denoised_crop, target)
                row["snr_gain_db"] = row["snr_denoised_db"] - row["snr_noisy_db"]
                row["cnr_noisy_auto"] = automatic_cnr(noisy_crop, target)
                row["cnr_denoised_auto"] = automatic_cnr(denoised_crop, target)
                row["cnr_clean_auto"] = automatic_cnr(target, target)
                row["cnr_error_auto"] = abs(
                    row["cnr_denoised_auto"] - row["cnr_clean_auto"]
                )
            if (evaluate_segmentation or evaluate_denoising) and bool(batch["has_layer"][index]):
                layer_true = restored(batch["layer_mask"][index, 0].numpy()[crop], True) & layer_valid_eval
                if evaluate_denoising and bool(batch["has_clean"][index]):
                    roi = layer_true & valid_eval
                    if roi.any():
                        row["layer_roi_mse"] = float(np.mean((denoised_eval[roi] - target[roi]) ** 2))
                        row["layer_roi_mse_noisy"] = float(np.mean((noisy_eval[roi] - target[roi]) ** 2))
                        row["layer_roi_psnr"] = psnr(denoised_eval[roi], target[roi])
                        row["layer_roi_psnr_noisy"] = psnr(noisy_eval[roi], target[roi])
                if evaluate_layer:
                    layer_pred_metric = layer_pred & layer_valid_eval
                    for key, value in binary_metrics(layer_pred[layer_valid_eval], layer_true[layer_valid_eval]).items():
                        row[f"layer_{key}"] = value
                    hd95, assd = surface_distances(layer_pred_metric, layer_true, (axial_spacing, lateral_spacing))
                    upper, lower, thickness = layer_boundary_mae(layer_pred_metric, layer_true, axial_spacing)
                    row.update({"layer_hd95": hd95, "layer_assd": assd,
                                "upper_boundary_mae": upper, "lower_boundary_mae": lower,
                                "thickness_mae": thickness})
                    row.update(layer_shape_metrics(
                        layer_pred_metric, layer_true, axial_spacing, layer_surface_tolerance,
                        (axial_spacing, lateral_spacing),
                    ))
            else:
                layer_true = layer_pred
            if bool(batch["has_vessel"][index]):
                vessel_true = restored(batch["vessel_mask"][index, 0].numpy()[crop], True)
                vessel_valid = restored(batch["vessel_valid_mask"][index, 0].numpy()[crop], True) & valid_eval
                if evaluate_denoising and bool(batch["has_clean"][index]) and bool(batch["has_layer"][index]):
                    stroma = layer_true & ~vessel_true & vessel_valid
                    vessel_roi = vessel_true & vessel_valid
                    for name, image_eval in (("noisy", noisy_eval), ("denoised", denoised_eval), ("clean", target)):
                        row[f"vessel_stroma_cnr_{name}"] = region_cnr(image_eval, vessel_roi, stroma)
                    row["vessel_stroma_cnr_abs_error"] = abs(
                        row["vessel_stroma_cnr_denoised"] - row["vessel_stroma_cnr_clean"]
                    )
                    row["vessel_stroma_cnr_noisy_abs_error"] = abs(
                        row["vessel_stroma_cnr_noisy"] - row["vessel_stroma_cnr_clean"]
                    )
            if evaluate_vessel and bool(batch["has_vessel"][index]):
                vessel_tp = vessel_pred & vessel_true & vessel_valid
                vessel_fp = vessel_pred & ~vessel_true & vessel_valid
                vessel_fn = ~vessel_pred & vessel_true & vessel_valid
                if bool(batch["has_layer"][index]):
                    row.update(
                        vessel_diagnostic_metrics(
                            vessel_prob_eval,
                            layer_prob_eval,
                            vessel_true,
                            layer_true,
                            vessel_valid,
                            vessel_threshold=vessel_threshold,
                            layer_threshold=layer_threshold,
                            component_size_thresholds=component_size_thresholds,
                            boundary_band_width=boundary_band_width,
                        )
                    )
                    error_outside_both = (
                        vessel_pred & ~layer_true & ~layer_pred & vessel_valid
                    )
                    error_outside_gt_inside_pred = (
                        vessel_pred & ~layer_true & layer_pred & vessel_valid
                    )
                    gt_vessel_outside_pred_layer = (
                        vessel_true & ~layer_pred & vessel_valid
                    )
                    vessel_oracle_gt_layer = vessel_pred & layer_true
                    vessel_pred_layer_soft_gate = (
                        vessel_prob_eval
                        * layer_prob_eval
                        >= vessel_threshold
                    )
                    gt_roi = layer_true & vessel_valid
                    vessel_roi_tp = vessel_tp & gt_roi
                    vessel_roi_fp = vessel_fp & gt_roi
                    vessel_roi_fn = vessel_fn & gt_roi
                else:
                    for key, value in binary_metrics(
                        vessel_pred[vessel_valid], vessel_true[vessel_valid]
                    ).items():
                        row[f"vessel_{key}"] = value
                if bool(vessel_valid.all()):
                    hd95, assd = surface_distances(
                        vessel_pred, vessel_true, (axial_spacing, lateral_spacing)
                    )
                else:
                    hd95, assd = float("nan"), float("nan")
                row["vessel_hd95"] = hd95
                row["vessel_assd"] = assd
                predicted_fraction = vessel_area_fraction(vessel_pred, layer_true)
                true_fraction = vessel_area_fraction(vessel_true, layer_true)
                row["vessel_area_fraction_pred"] = predicted_fraction
                row["vessel_area_fraction_true"] = true_fraction
                row["vessel_area_fraction_mae"] = abs(predicted_fraction - true_fraction)

            # P0 is always the immutable raw threshold result. P1/P2 affect only
            # the layer; P3 strictly clips the raw vessel prediction to P2/P1.
            layer_p1, p1_stats = clean_layer_mask(
                layer_pred, layer_valid_eval, p1_minimum_main_fraction
            )
            layer_p2, p2_stats = regularize_lower_boundary(
                layer_p1, layer_valid_eval, p2_smoothness, p2_max_displacement
            )
            row.update({key: value for key, value in p1_stats.items() if "p1" in modes})
            row.update({key: value for key, value in p2_stats.items() if "p2" in modes})
            if evaluate_layer and bool(batch["has_layer"][index]):
                for mode, prediction in (("p0", layer_pred), ("p1", layer_p1), ("p2", layer_p2)):
                    if mode not in modes:
                        continue
                    for key, value in binary_metrics(prediction[layer_valid_eval], layer_true[layer_valid_eval]).items():
                        row[f"{mode}_layer_{key}"] = value
                    upper, lower, thickness = layer_boundary_mae(prediction, layer_true, axial_spacing)
                    row[f"{mode}_upper_boundary_mae"] = upper
                    row[f"{mode}_lower_boundary_mae"] = lower
                    row[f"{mode}_thickness_mae"] = thickness
            vessel_p3 = None
            if evaluate_vessel and bool(batch["has_vessel"][index]):
                final_layer = layer_p2
                vessel_p3, p3_stats = hard_contain_vessel(
                    vessel_pred, final_layer, vessel_valid, vessel_true
                )
                if "p0" in modes:
                    for key, value in binary_metrics(vessel_pred[vessel_valid], vessel_true[vessel_valid]).items():
                        row[f"p0_vessel_{key}"] = value
                if "p3" in modes:
                    row.update(p3_stats)
                    for key, value in binary_metrics(vessel_p3[vessel_valid], vessel_true[vessel_valid]).items():
                        row[f"p3_vessel_{key}"] = value
            rows.append(row)
            repeat_outputs.append(
                {
                    "group_id": str(batch["group_id"][index]),
                    "dataset": str(batch["dataset"][index]),
                    "denoised": denoised_eval.astype(np.float32),
                    "layer": layer_pred.astype(bool),
                    "vessel": vessel_pred.astype(bool),
                    "valid": valid_eval.astype(bool),
                }
            )

            if output_path and save_predictions:
                sample_dir = output_path / "predictions" / str(batch["dataset"][index])
                sample_id = str(batch["sample_id"][index])
                write_gray(
                    sample_dir / f"{sample_id}_noisy.png",
                    noisy_eval,
                )
                write_gray(
                    sample_dir / f"{sample_id}_denoised.png",
                    denoised_eval,
                )
                if evaluate_segmentation:
                    write_gray(
                        sample_dir / f"{sample_id}_layer_prob.png",
                        layer_prob_eval,
                    )
                    write_gray(
                        sample_dir / f"{sample_id}_vessel_prob.png",
                        vessel_prob_eval,
                    )
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    np.save(sample_dir / f"{sample_id}_layer_prob_float32.npy", layer_prob_eval.astype(np.float32), allow_pickle=False)
                    np.save(sample_dir / f"{sample_id}_vessel_prob_float32.npy", vessel_prob_eval.astype(np.float32), allow_pickle=False)
                    write_gray(
                        sample_dir / f"{sample_id}_layer_mask.png",
                        layer_pred.astype(np.float32),
                    )
                    write_gray(
                        sample_dir / f"{sample_id}_vessel_mask.png",
                        vessel_pred.astype(np.float32),
                    )
                    if "p1" in modes:
                        write_gray(sample_dir / f"{sample_id}_p1_layer_mask.png", layer_p1.astype(np.float32))
                    if "p2" in modes:
                        write_gray(sample_dir / f"{sample_id}_p2_layer_mask.png", layer_p2.astype(np.float32))
                    if "p3" in modes and vessel_p3 is not None:
                        write_gray(sample_dir / f"{sample_id}_p3_vessel_mask.png", vessel_p3.astype(np.float32))
                    if bool(batch["has_layer"][index]):
                        write_gray(
                            sample_dir / f"{sample_id}_layer_gt.png",
                            layer_true.astype(np.float32),
                        )
                        layer_tp = layer_pred & layer_true & valid_eval
                        layer_fp = layer_pred & ~layer_true & valid_eval
                        layer_fn = ~layer_pred & layer_true & valid_eval
                        layer_error = np.repeat(noisy_eval[..., None], 3, axis=2)
                        layer_error[layer_tp] = (0.0, 1.0, 0.0)
                        layer_error[layer_fp] = (1.0, 0.0, 0.0)
                        layer_error[layer_fn] = (0.0, 0.35, 1.0)
                        write_rgb(sample_dir / f"{sample_id}_layer_error_tp_fp_fn.png", layer_error)
                        layer_overlay = np.repeat(noisy_eval[..., None], 3, axis=2)
                        layer_overlay[layer_pred] = 0.55 * layer_overlay[layer_pred] + 0.45 * np.array((0.0, 1.0, 0.0))
                        write_rgb(sample_dir / f"{sample_id}_layer_overlay.png", layer_overlay)
                    if bool(batch["has_vessel"][index]):
                        write_gray(
                            sample_dir / f"{sample_id}_vessel_gt.png",
                            vessel_true.astype(np.float32),
                        )
                        vessel_error = np.repeat(noisy_eval[..., None], 3, axis=2)
                        vessel_error[vessel_tp] = (0.0, 1.0, 0.0)
                        vessel_error[vessel_fp] = (1.0, 0.0, 0.0)
                        vessel_error[vessel_fn] = (0.0, 0.35, 1.0)
                        write_rgb(sample_dir / f"{sample_id}_vessel_error_tp_fp_fn.png", vessel_error)
                        vessel_overlay = np.repeat(noisy_eval[..., None], 3, axis=2)
                        vessel_overlay[vessel_pred] = 0.55 * vessel_overlay[vessel_pred] + 0.45 * np.array((1.0, 0.55, 0.0))
                        write_rgb(sample_dir / f"{sample_id}_vessel_overlay.png", vessel_overlay)
                    diagnostic_maps = {
                        "vessel_tp": vessel_tp,
                        "vessel_fp": vessel_fp,
                        "vessel_fn": vessel_fn,
                        "vessel_gt_layer_roi_tp": vessel_roi_tp,
                        "vessel_gt_layer_roi_fp": vessel_roi_fp,
                        "vessel_gt_layer_roi_fn": vessel_roi_fn,
                        "error_outside_gt_and_pred_layer": error_outside_both,
                        "error_outside_gt_inside_pred_layer": error_outside_gt_inside_pred,
                        "gt_vessel_outside_pred_layer": gt_vessel_outside_pred_layer,
                        "vessel_oracle_gt_layer": vessel_oracle_gt_layer,
                        "vessel_pred_layer_soft_gate": vessel_pred_layer_soft_gate,
                    }
                    for suffix, diagnostic in diagnostic_maps.items():
                        if diagnostic is not None:
                            write_gray(
                                sample_dir / f"{sample_id}_{suffix}.png",
                                diagnostic.astype(np.float32),
                            )
                    if error_outside_both is not None:
                        base = noisy_eval
                        overlay = np.repeat(base[..., None], 3, axis=2)
                        # Red: vessel outside both anatomical masks. Orange:
                        # vessel outside GT but admitted by predicted layer.
                        # Blue: GT vessel that predicted-layer clipping loses.
                        overlay[error_outside_both] = (1.0, 0.0, 0.0)
                        overlay[error_outside_gt_inside_pred] = (1.0, 0.55, 0.0)
                        overlay[gt_vessel_outside_pred_layer] = (0.0, 0.45, 1.0)
                        write_rgb(
                            sample_dir / f"{sample_id}_error_overlay.png",
                            overlay,
                        )
                    if bool(batch["has_layer"][index]):
                        boundary_overlay = np.repeat(noisy_eval[..., None], 3, axis=2)
                        kernel = np.ones((3, 3), np.uint8)
                        gt_boundary = layer_true & ~cv2.erode(layer_true.astype(np.uint8), kernel).astype(bool)
                        pred_boundary = layer_pred & ~cv2.erode(layer_pred.astype(np.uint8), kernel).astype(bool)
                        boundary_overlay[gt_boundary] = (0.0, 1.0, 0.0)
                        boundary_overlay[pred_boundary] = (1.0, 0.0, 1.0)
                        write_rgb(sample_dir / f"{sample_id}_boundary_overlay.png", boundary_overlay)
                if target is not None:
                    write_gray(sample_dir / f"{sample_id}_clean.png", target)
                    write_gray(sample_dir / f"{sample_id}_reference_abs_error.png", np.abs(denoised_eval - target))
                crop_mask = vessel_true if vessel_true is not None and vessel_true.any() else (
                    layer_true if layer_true is not None and layer_true.any() else None
                )
                if crop_mask is not None:
                    ys, xs = np.where(crop_mask)
                    center_y, center_x = int(np.median(ys)), int(np.median(xs))
                else:
                    center_y, center_x = noisy_eval.shape[0] // 2, noisy_eval.shape[1] // 2
                crop_h, crop_w = min(128, noisy_eval.shape[0]), min(128, noisy_eval.shape[1])
                y0 = min(max(center_y - crop_h // 2, 0), noisy_eval.shape[0] - crop_h)
                x0 = min(max(center_x - crop_w // 2, 0), noisy_eval.shape[1] - crop_w)
                qualitative_crops.append({
                    "dataset": str(batch["dataset"][index]), "sample_id": sample_id,
                    "group_id": str(batch["group_id"][index]),
                    "x": int(x0), "y": int(y0), "width": int(crop_w), "height": int(crop_h),
                    "selection": "GT-vessel median, else GT-layer median, else image center",
                })

    frame_table = pd.DataFrame(rows)
    numeric_columns = frame_table.select_dtypes(include=[np.number]).columns.tolist()
    group_columns = ["group_id", "dataset"]
    group_table = frame_table.groupby(group_columns, as_index=False)[numeric_columns].mean()
    evaluated_frames = frame_table.groupby(group_columns).size().rename(
        "n_evaluated_frames"
    )
    group_table = group_table.merge(
        evaluated_frames.reset_index(), on=group_columns, how="left"
    )
    repeat_rows = []
    repeat_index = pd.DataFrame(
        [{"group_id": item["group_id"], "dataset": item["dataset"], "index": index}
         for index, item in enumerate(repeat_outputs)]
    )
    repeat_groups = repeat_index.groupby(["group_id", "dataset"]) if not repeat_index.empty else []
    for (group_id, dataset), items in repeat_groups:
        indices = items["index"].tolist()
        denoise_mae, layer_dice, vessel_dice = [], [], []
        for left_pos, left_index in enumerate(indices):
            for right_index in indices[left_pos + 1:]:
                left, right = repeat_outputs[left_index], repeat_outputs[right_index]
                if left["denoised"].shape != right["denoised"].shape:
                    continue
                valid_pair = left["valid"] & right["valid"]
                if valid_pair.any():
                    denoise_mae.append(float(np.mean(np.abs(
                        left["denoised"][valid_pair] - right["denoised"][valid_pair]
                    ))))
                    layer_dice.append(binary_metrics(left["layer"][valid_pair], right["layer"][valid_pair])["dice"])
                    vessel_dice.append(binary_metrics(left["vessel"][valid_pair], right["vessel"][valid_pair])["dice"])
        repeat_rows.append({
            "group_id": group_id, "dataset": dataset,
            "repeat_pair_count": len(denoise_mae),
            "repeat_denoised_mae": float(np.mean(denoise_mae)) if denoise_mae else float("nan"),
            "repeat_layer_dice": float(np.mean(layer_dice)) if layer_dice else float("nan"),
            "repeat_vessel_dice": float(np.mean(vessel_dice)) if vessel_dice else float("nan"),
        })
    if repeat_rows:
        group_table = group_table.merge(pd.DataFrame(repeat_rows), on=group_columns, how="left")
        numeric_columns = group_table.select_dtypes(include=[np.number]).columns.tolist()
    summary = mean_dict(group_table[numeric_columns].to_dict("records"))
    summary["n_frames"] = int(len(frame_table))
    summary["n_groups"] = int(frame_table["group_id"].nunique())
    summary["metric_group_counts"] = {
        column: int(
            group_table.loc[group_table[column].notna(), "group_id"].nunique()
        )
        for column in numeric_columns
    }
    summary["layer_threshold"] = float(layer_threshold)
    summary["vessel_threshold"] = float(vessel_threshold)
    summary["input_normalization"] = input_normalization or "unspecified"
    summary["component_size_thresholds"] = (
        list(component_size_thresholds)
        if component_size_thresholds is not None
        else None
    )
    summary["boundary_band_width_pixels"] = float(boundary_band_width)
    summary["postprocess_modes"] = list(modes)
    summary["restored_original_geometry"] = bool(restore_original_geometry)
    summary["layer_surface_tolerance"] = float(layer_surface_tolerance)
    summary["evaluated_tasks"] = [
        name
        for name, enabled in (
            ("denoise", evaluate_denoising),
            ("layer", evaluate_layer),
            ("vessel", evaluate_vessel),
        )
        if enabled
    ]
    summary["by_dataset"] = {
        str(dataset): {
            **mean_dict(part[numeric_columns].to_dict("records")),
            "n_groups": int(part["group_id"].nunique()),
        }
        for dataset, part in group_table.groupby("dataset")
    }
    if output_path:
        frame_table.to_csv(output_path / "frame_metrics.csv", index=False, encoding="utf-8-sig")
        group_table.to_csv(output_path / "group_metrics.csv", index=False, encoding="utf-8-sig")
        if qualitative_crops:
            pd.DataFrame(qualitative_crops).drop_duplicates().to_csv(
                output_path / "qualitative_crops.csv", index=False, encoding="utf-8-sig"
            )
        write_json(summary, output_path / "summary.json")
        write_json(
            {
                "vessel_outside_gt_layer_fraction": (
                    "predicted vessel pixels outside GT layer / all predicted "
                    "vessel pixels, within vessel-valid pixels"
                ),
                "vessel_error_outside_gt_and_pred_layer_fraction": (
                    "predicted vessel outside both GT and predicted layer / all "
                    "predicted vessel pixels"
                ),
                "vessel_error_outside_gt_inside_pred_layer_fraction": (
                    "predicted vessel outside GT but inside predicted layer / all "
                    "predicted vessel pixels"
                ),
                "gt_vessel_outside_pred_layer_fraction": (
                    "GT vessel outside predicted layer / all GT vessel pixels"
                ),
                "oracle_warning": (
                    "GT-layer-restricted metrics are diagnostic only and are not "
                    "deployable model results"
                ),
                "vessel_pred_layer_soft_gate": (
                    "p_gate = p_vessel * p_layer; its global threshold must be "
                    "calibrated on validation separately from raw p_vessel"
                ),
                "vessel_gt_component_size": (
                    "Connected-component pixel area in resized/padded model "
                    "coordinates; thresholds must be derived from training labels"
                ),
            },
            output_path / "metric_definitions.json",
        )
    return summary
