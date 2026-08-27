from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from ..data.io import write_gray, write_rgb
from ..metrics import (
    automatic_cnr,
    binary_metrics,
    edge_preservation_index,
    layer_boundary_mae,
    psnr,
    reconstruction_snr,
    rmse,
    ssim,
    surface_distances,
    vessel_area_fraction,
    vessel_diagnostic_metrics,
)
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
) -> Dict[str, object]:
    model.eval()
    layer_threshold = threshold if layer_threshold is None else layer_threshold
    vessel_threshold = threshold if vessel_threshold is None else vessel_threshold
    evaluate_denoising = stage not in {"segment", "private_seg"}
    evaluate_segmentation = stage != "denoise"
    rows = []
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
            }
            valid = batch["valid_mask"][index, 0].numpy() > 0.5
            coordinates = np.argwhere(valid)
            if coordinates.size:
                y0, x0 = coordinates.min(axis=0)
                y1, x1 = coordinates.max(axis=0) + 1
            else:
                y0, x0, y1, x1 = 0, 0, valid.shape[0], valid.shape[1]
            crop = np.s_[y0:y1, x0:x1]
            layer_pred = layer_probability[index, 0][crop] >= layer_threshold
            vessel_pred = vessel_probability[index, 0][crop] >= vessel_threshold
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
                target = batch["clean"][index, 0].numpy()[crop]
                denoised_crop = denoised[index, 0][crop]
                noisy_crop = batch["image"][index, 0].numpy()[crop]
                row["psnr"] = psnr(denoised_crop, target)
                row["psnr_noisy"] = psnr(noisy_crop, target)
                row["psnr_gain_db"] = row["psnr"] - row["psnr_noisy"]
                row["ssim"] = ssim(denoised_crop, target)
                row["ssim_noisy"] = ssim(noisy_crop, target)
                row["ssim_gain"] = row["ssim"] - row["ssim_noisy"]
                row["rmse"] = rmse(denoised_crop, target)
                row["rmse_noisy"] = rmse(noisy_crop, target)
                row["rmse_reduction"] = row["rmse_noisy"] - row["rmse"]
                row["epi"] = edge_preservation_index(denoised_crop, target)
                row["snr_noisy_db"] = reconstruction_snr(noisy_crop, target)
                row["snr_denoised_db"] = reconstruction_snr(denoised_crop, target)
                row["snr_gain_db"] = row["snr_denoised_db"] - row["snr_noisy_db"]
                row["cnr_noisy_auto"] = automatic_cnr(noisy_crop, target)
                row["cnr_denoised_auto"] = automatic_cnr(denoised_crop, target)
                row["cnr_clean_auto"] = automatic_cnr(target, target)
                row["cnr_error_auto"] = abs(
                    row["cnr_denoised_auto"] - row["cnr_clean_auto"]
                )
            if evaluate_segmentation and bool(batch["has_layer"][index]):
                layer_true = batch["layer_mask"][index, 0].numpy()[crop] > 0.5
                for key, value in binary_metrics(layer_pred, layer_true).items():
                    row[f"layer_{key}"] = value
                hd95, assd = surface_distances(
                    layer_pred, layer_true, (axial_spacing, lateral_spacing)
                )
                upper, lower, thickness = layer_boundary_mae(
                    layer_pred, layer_true, axial_spacing
                )
                row.update(
                    {
                        "layer_hd95": hd95,
                        "layer_assd": assd,
                        "upper_boundary_mae": upper,
                        "lower_boundary_mae": lower,
                        "thickness_mae": thickness,
                    }
                )
            else:
                layer_true = layer_pred
            if evaluate_segmentation and bool(batch["has_vessel"][index]):
                vessel_true = batch["vessel_mask"][index, 0].numpy()[crop] > 0.5
                vessel_valid = (
                    batch["vessel_valid_mask"][index, 0].numpy()[crop] > 0.5
                )
                if bool(batch["has_layer"][index]):
                    row.update(
                        vessel_diagnostic_metrics(
                            vessel_probability[index, 0][crop],
                            layer_probability[index, 0][crop],
                            vessel_true,
                            layer_true,
                            vessel_valid,
                            vessel_threshold=vessel_threshold,
                            layer_threshold=layer_threshold,
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
                        vessel_probability[index, 0][crop]
                        * layer_probability[index, 0][crop]
                        >= vessel_threshold
                    )
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
            rows.append(row)

            if output_path and save_predictions:
                sample_dir = output_path / "predictions" / str(batch["dataset"][index])
                sample_id = str(batch["sample_id"][index])
                write_gray(
                    sample_dir / f"{sample_id}_noisy.png",
                    batch["image"][index, 0].numpy()[crop],
                )
                write_gray(
                    sample_dir / f"{sample_id}_denoised.png",
                    denoised[index, 0][crop],
                )
                if evaluate_segmentation:
                    write_gray(
                        sample_dir / f"{sample_id}_layer_prob.png",
                        layer_probability[index, 0][crop],
                    )
                    write_gray(
                        sample_dir / f"{sample_id}_vessel_prob.png",
                        vessel_probability[index, 0][crop],
                    )
                    write_gray(
                        sample_dir / f"{sample_id}_layer_mask.png",
                        layer_pred.astype(np.float32),
                    )
                    write_gray(
                        sample_dir / f"{sample_id}_vessel_mask.png",
                        vessel_pred.astype(np.float32),
                    )
                    if bool(batch["has_layer"][index]):
                        write_gray(
                            sample_dir / f"{sample_id}_layer_gt.png",
                            layer_true.astype(np.float32),
                        )
                    if bool(batch["has_vessel"][index]):
                        write_gray(
                            sample_dir / f"{sample_id}_vessel_gt.png",
                            vessel_true.astype(np.float32),
                        )
                    diagnostic_maps = {
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
                        base = batch["image"][index, 0].numpy()[crop]
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

    frame_table = pd.DataFrame(rows)
    numeric_columns = frame_table.select_dtypes(include=[np.number]).columns.tolist()
    group_columns = ["group_id", "dataset"]
    group_table = frame_table.groupby(group_columns, as_index=False)[numeric_columns].mean()
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
    summary["evaluated_tasks"] = [
        name
        for name, enabled in (
            ("denoise", evaluate_denoising),
            ("segment", evaluate_segmentation),
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
            },
            output_path / "metric_definitions.json",
        )
    return summary
