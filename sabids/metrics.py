from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    binary_erosion,
    distance_transform_edt,
    label as connected_components,
    uniform_filter,
)


def psnr(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(data_range * data_range / mse))


def ssim(prediction: np.ndarray, target: np.ndarray, window: int = 7) -> float:
    x = prediction.astype(np.float64)
    y = target.astype(np.float64)
    mu_x = uniform_filter(x, size=window, mode="reflect")
    mu_y = uniform_filter(y, size=window, mode="reflect")
    var_x = uniform_filter(x * x, size=window, mode="reflect") - mu_x * mu_x
    var_y = uniform_filter(y * y, size=window, mode="reflect") - mu_y * mu_y
    covariance = uniform_filter(x * y, size=window, mode="reflect") - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * covariance + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = float(np.logical_and(prediction, target).sum())
    fp = float(np.logical_and(prediction, ~target).sum())
    fn = float(np.logical_and(~prediction, target).sum())
    tn = float(np.logical_and(~prediction, ~target).sum())
    union = tp + fp + fn
    background_union = tn + fp + fn
    foreground_iou = (tp + 1e-6) / (union + 1e-6)
    background_iou = (tn + 1e-6) / (background_union + 1e-6)
    return {
        "dice": (2.0 * tp + 1e-6) / (2.0 * tp + fp + fn + 1e-6),
        "iou": foreground_iou,
        "background_iou": background_iou,
        "miou": 0.5 * (foreground_iou + background_iou),
        "precision": (tp + 1e-6) / (tp + fp + 1e-6),
        "recall": (tp + 1e-6) / (tp + fn + 1e-6),
        "specificity": (tn + 1e-6) / (tn + fp + 1e-6),
        "accuracy": (tp + tn + 1e-6) / (tp + tn + fp + fn + 1e-6),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "valid_pixels": tp + fp + fn + tn,
    }


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction.astype(np.float64) - target) ** 2)))


def reconstruction_snr(prediction: np.ndarray, target: np.ndarray) -> float:
    signal_power = float(np.mean(target.astype(np.float64) ** 2))
    noise_power = float(np.mean((prediction.astype(np.float64) - target) ** 2))
    if noise_power <= 1e-12:
        return 99.0
    return float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))


def edge_preservation_index(prediction: np.ndarray, target: np.ndarray) -> float:
    def magnitude(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float64)
        gx = np.zeros_like(image)
        gy = np.zeros_like(image)
        gx[:, :-1] = image[:, 1:] - image[:, :-1]
        gy[:-1, :] = image[1:, :] - image[:-1, :]
        return np.sqrt(gx * gx + gy * gy)

    predicted_edge = magnitude(prediction).ravel()
    target_edge = magnitude(target).ravel()
    pred_std = float(predicted_edge.std())
    target_std = float(target_edge.std())
    if pred_std <= 1e-12 or target_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(predicted_edge, target_edge)[0, 1])


def reference_edge_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean absolute gradient-magnitude error relative to the clean reference."""
    def magnitude(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float64)
        gx = np.zeros_like(image)
        gy = np.zeros_like(image)
        gx[:, :-1] = image[:, 1:] - image[:, :-1]
        gy[:-1, :] = image[1:, :] - image[:-1, :]
        return np.sqrt(gx * gx + gy * gy)

    return float(np.mean(np.abs(magnitude(prediction) - magnitude(target))))


def otsu_threshold(image: np.ndarray, bins: int = 256) -> float:
    values = np.clip(image.astype(np.float64), 0.0, 1.0)
    histogram, edges = np.histogram(values.ravel(), bins=bins, range=(0.0, 1.0))
    probability = histogram.astype(np.float64)
    probability /= max(probability.sum(), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cumulative_probability = np.cumsum(probability)
    cumulative_mean = np.cumsum(probability * centers)
    global_mean = cumulative_mean[-1]
    denominator = cumulative_probability * (1.0 - cumulative_probability)
    between = (global_mean * cumulative_probability - cumulative_mean) ** 2
    between /= np.maximum(denominator, 1e-12)
    return float(centers[int(np.argmax(between))])


def automatic_cnr(image: np.ndarray, reference: np.ndarray) -> float:
    """Estimate CNR using foreground/background ROIs obtained from clean reference."""
    threshold = otsu_threshold(reference)
    foreground = reference > threshold
    background = ~foreground
    if foreground.sum() < 16 or background.sum() < 16:
        return float("nan")
    foreground_values = image[foreground].astype(np.float64)
    background_values = image[background].astype(np.float64)
    denominator = np.sqrt(foreground_values.var() + background_values.var())
    if denominator <= 1e-12:
        return float("nan")
    return float(abs(foreground_values.mean() - background_values.mean()) / denominator)


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return np.logical_xor(mask, binary_erosion(mask))


def surface_distances(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> Tuple[float, float]:
    pred_surface = _surface(prediction)
    target_surface = _surface(target)
    if not pred_surface.any() and not target_surface.any():
        return 0.0, 0.0
    if not pred_surface.any() or not target_surface.any():
        diagonal = float(np.sqrt(sum((s * n) ** 2 for s, n in zip(spacing, prediction.shape))))
        return diagonal, diagonal
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate(
        [target_distance[pred_surface], pred_distance[target_surface]]
    )
    return float(np.percentile(distances, 95)), float(np.mean(distances))


def layer_boundary_mae(
    prediction: np.ndarray,
    target: np.ndarray,
    axial_spacing: float = 1.0,
) -> Tuple[float, float, float]:
    pred = prediction.astype(bool)
    true = target.astype(bool)
    upper_errors, lower_errors, thickness_errors = [], [], []
    for column in range(pred.shape[1]):
        pred_idx = np.flatnonzero(pred[:, column])
        true_idx = np.flatnonzero(true[:, column])
        if pred_idx.size == 0 or true_idx.size == 0:
            continue
        pred_upper, pred_lower = pred_idx[0], pred_idx[-1]
        true_upper, true_lower = true_idx[0], true_idx[-1]
        upper_errors.append(abs(pred_upper - true_upper) * axial_spacing)
        lower_errors.append(abs(pred_lower - true_lower) * axial_spacing)
        thickness_errors.append(
            abs((pred_lower - pred_upper) - (true_lower - true_upper)) * axial_spacing
        )
    if not upper_errors:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(upper_errors)),
        float(np.mean(lower_errors)),
        float(np.mean(thickness_errors)),
    )


def surface_dice(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: float = 3.0,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> float:
    pred_surface, target_surface = _surface(prediction), _surface(target)
    if not pred_surface.any() and not target_surface.any():
        return 1.0
    if not pred_surface.any() or not target_surface.any():
        return 0.0
    target_distance = distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = distance_transform_edt(~pred_surface, sampling=spacing)
    matched = float((target_distance[pred_surface] <= tolerance).sum())
    matched += float((pred_distance[target_surface] <= tolerance).sum())
    return matched / float(pred_surface.sum() + target_surface.sum())


def layer_shape_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    axial_spacing: float = 1.0,
    surface_tolerance: float = 3.0,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> Dict[str, float]:
    """Boundary bias and anatomical-shape diagnostics for a layer mask."""
    pred, true = prediction.astype(bool), target.astype(bool)
    labelled, count = connected_components(pred)
    areas = np.bincount(labelled.ravel())[1:] if count else np.array([])
    extra_area = float(areas.sum() - areas.max()) if areas.size else 0.0
    holes = binary_fill_holes(pred) & ~pred
    upper_bias, lower_bias, thickness_bias = [], [], []
    pred_lower_series, true_lower_series = [], []
    for column in range(pred.shape[1]):
        pred_idx, true_idx = np.flatnonzero(pred[:, column]), np.flatnonzero(true[:, column])
        if pred_idx.size == 0 or true_idx.size == 0:
            continue
        pu, pl, tu, tl = pred_idx[0], pred_idx[-1], true_idx[0], true_idx[-1]
        upper_bias.append((pu - tu) * axial_spacing)
        lower_bias.append((pl - tl) * axial_spacing)
        thickness_bias.append(((pl - pu) - (tl - tu)) * axial_spacing)
        pred_lower_series.append(pl * axial_spacing)
        true_lower_series.append(tl * axial_spacing)

    def roughness(values: list[float]) -> float:
        return float(np.mean(np.abs(np.diff(values, n=2)))) if len(values) >= 3 else float("nan")

    return {
        "layer_surface_dice": surface_dice(pred, true, surface_tolerance, spacing),
        "upper_boundary_signed_bias": float(np.mean(upper_bias)) if upper_bias else float("nan"),
        "lower_boundary_signed_bias": float(np.mean(lower_bias)) if lower_bias else float("nan"),
        "thickness_signed_bias": float(np.mean(thickness_bias)) if thickness_bias else float("nan"),
        "layer_component_count": float(count),
        "layer_extra_component_area_ratio": extra_area / max(float(pred.sum()), 1.0),
        "layer_hole_pixels": float(holes.sum()),
        "layer_hole_area_ratio": float(holes.sum()) / max(float(pred.sum() + holes.sum()), 1.0),
        "lower_boundary_roughness_pred": roughness(pred_lower_series),
        "lower_boundary_roughness_true": roughness(true_lower_series),
        "layer_valid_column_fraction": float(len(upper_bias)) / max(float(pred.shape[1]), 1.0),
    }


def region_cnr(image: np.ndarray, first_roi: np.ndarray, second_roi: np.ndarray) -> float:
    first = image[first_roi.astype(bool)].astype(np.float64)
    second = image[second_roi.astype(bool)].astype(np.float64)
    if first.size < 2 or second.size < 2:
        return float("nan")
    denominator = np.sqrt(first.var() + second.var())
    if denominator <= 1e-12:
        return float("nan")
    return float(abs(first.mean() - second.mean()) / denominator)


def vessel_area_fraction(mask: np.ndarray, layer: np.ndarray) -> float:
    denominator = float((layer > 0).sum())
    if denominator <= 0:
        return float("nan")
    return float(np.logical_and(mask > 0, layer > 0).sum() / denominator)


def soft_dice_score(
    probability: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    eps: float = 1e-6,
) -> float:
    probability = probability[valid].astype(np.float64)
    target = target[valid].astype(np.float64)
    return float(
        (2.0 * (probability * target).sum() + eps)
        / (probability.sum() + target.sum() + eps)
    )


def vessel_diagnostic_metrics(
    vessel_probability: np.ndarray,
    layer_probability: np.ndarray,
    vessel_target: np.ndarray,
    layer_target: np.ndarray,
    valid: np.ndarray,
    vessel_threshold: float = 0.5,
    layer_threshold: float = 0.5,
    component_size_thresholds: Tuple[int, int] | None = None,
    boundary_band_width: float = 3.0,
) -> Dict[str, float]:
    """Consistent full-image, GT-layer ROI, outside, and baseline metrics."""
    valid = valid.astype(bool)
    vessel_target = vessel_target.astype(bool)
    layer_target = layer_target.astype(bool)
    vessel_prediction = vessel_probability >= vessel_threshold
    layer_prediction = layer_probability >= layer_threshold
    result = {
        f"vessel_{key}": value
        for key, value in binary_metrics(
            vessel_prediction[valid], vessel_target[valid]
        ).items()
    }
    result["vessel_soft_dice"] = soft_dice_score(
        vessel_probability, vessel_target, valid
    )

    roi = valid & layer_target
    if roi.any():
        result.update(
            {
                f"vessel_roi_{key}": value
                for key, value in binary_metrics(
                    vessel_prediction[roi], vessel_target[roi]
                ).items()
            }
        )
        roi_pixels = float(roi.sum())
        predicted_fraction = float(
            np.logical_and(vessel_prediction, roi).sum() / roi_pixels
        )
        true_fraction = float(np.logical_and(vessel_target, roi).sum() / roi_pixels)
        result["vessel_area_fraction_pred"] = predicted_fraction
        result["vessel_area_fraction_true"] = true_fraction
        result["vessel_area_fraction_mae"] = abs(
            predicted_fraction - true_fraction
        )

    predicted_pixels = float(np.logical_and(vessel_prediction, valid).sum())
    outside_pixels = float(
        np.logical_and(vessel_prediction, valid & ~layer_target).sum()
    )
    result["vessel_outside_gt_layer_fraction"] = outside_pixels / max(
        predicted_pixels, 1.0
    )
    outside_gt = valid & ~layer_target
    outside_both = vessel_prediction & outside_gt & ~layer_prediction
    outside_gt_inside_pred = vessel_prediction & outside_gt & layer_prediction
    missed_by_predicted_layer = vessel_target & valid & ~layer_prediction
    true_vessel_pixels = float(np.logical_and(vessel_target, valid).sum())
    result.update(
        {
            # Fractions use predicted-vessel pixels for the two FP partitions,
            # and true-vessel pixels for the clipping-risk partition.
            "vessel_error_outside_gt_and_pred_layer_pixels": float(
                outside_both.sum()
            ),
            "vessel_error_outside_gt_and_pred_layer_fraction": float(
                outside_both.sum()
            )
            / max(predicted_pixels, 1.0),
            "vessel_error_outside_gt_inside_pred_layer_pixels": float(
                outside_gt_inside_pred.sum()
            ),
            "vessel_error_outside_gt_inside_pred_layer_fraction": float(
                outside_gt_inside_pred.sum()
            )
            / max(predicted_pixels, 1.0),
            "gt_vessel_outside_pred_layer_pixels": float(
                missed_by_predicted_layer.sum()
            ),
            "gt_vessel_outside_pred_layer_fraction": float(
                missed_by_predicted_layer.sum()
            )
            / max(true_vessel_pixels, 1.0),
        }
    )

    oracle_prediction = vessel_prediction & layer_target
    result.update(
        {
            f"vessel_oracle_gt_layer_{key}": value
            for key, value in binary_metrics(
                oracle_prediction[valid], vessel_target[valid]
            ).items()
        }
    )
    soft_gated_probability = vessel_probability * layer_probability
    soft_gated_prediction = soft_gated_probability >= vessel_threshold
    result.update(
        {
            f"vessel_pred_layer_soft_gate_{key}": value
            for key, value in binary_metrics(
                soft_gated_prediction[valid], vessel_target[valid]
            ).items()
        }
    )
    result["vessel_pred_layer_soft_gate_soft_dice"] = soft_dice_score(
        soft_gated_probability, vessel_target, valid
    )
    if component_size_thresholds is not None:
        small_max, medium_max = component_size_thresholds
        labelled, count = connected_components(vessel_target & valid)
        bins = {
            "small": lambda area: area <= small_max,
            "medium": lambda area: small_max < area <= medium_max,
            "large": lambda area: area > medium_max,
        }
        for name, selector in bins.items():
            component_ids = [
                component_id
                for component_id in range(1, count + 1)
                if selector(int((labelled == component_id).sum()))
            ]
            component_mask = np.isin(labelled, component_ids)
            pixels = float(component_mask.sum())
            result[f"vessel_gt_component_{name}_count"] = float(
                len(component_ids)
            )
            result[f"vessel_gt_component_{name}_pixels"] = pixels
            result[f"vessel_gt_component_{name}_pixel_recall"] = (
                float(np.logical_and(vessel_prediction, component_mask).sum())
                / max(pixels, 1.0)
            )
            detected = sum(
                bool(np.logical_and(vessel_prediction, labelled == component_id).any())
                for component_id in component_ids
            )
            result[f"vessel_gt_component_{name}_detection_recall"] = (
                float(detected) / max(float(len(component_ids)), 1.0)
            )

    target_surface = _surface(vessel_target & valid)
    if target_surface.any() and boundary_band_width > 0:
        distance = distance_transform_edt(~target_surface)
        boundary_band = valid & (distance <= float(boundary_band_width))
        result.update(
            {
                f"vessel_boundary_band_{key}": value
                for key, value in binary_metrics(
                    vessel_prediction[boundary_band],
                    vessel_target[boundary_band],
                ).items()
            }
        )
        result["vessel_boundary_band_fp_pixels"] = float(
            (vessel_prediction & ~vessel_target & boundary_band).sum()
        )
        result["vessel_boundary_band_fn_pixels"] = float(
            (~vessel_prediction & vessel_target & boundary_band).sum()
        )
    overlap = float(
        np.logical_and(vessel_prediction, layer_prediction & valid).sum()
    )
    result["pred_layer_vessel_dice"] = (2.0 * overlap + 1e-6) / (
        float(np.logical_and(vessel_prediction, valid).sum())
        + float(np.logical_and(layer_prediction, valid).sum())
        + 1e-6
    )
    result["whole_layer_baseline_vessel_dice"] = binary_metrics(
        layer_target[valid], vessel_target[valid]
    )["dice"]
    result["empty_baseline_vessel_dice"] = binary_metrics(
        np.zeros_like(vessel_target[valid]), vessel_target[valid]
    )["dice"]
    return result
