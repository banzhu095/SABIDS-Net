from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, uniform_filter


def psnr(prediction: np.ndarray, target: np.ndarray, data_range: float = 1.0) -> float:
    mse = float(np.mean((prediction - target) ** 2))
    if mse <= 1e-12:
        return 99.0
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


def vessel_area_fraction(mask: np.ndarray, layer: np.ndarray) -> float:
    denominator = float((layer > 0).sum())
    if denominator <= 0:
        return float("nan")
    return float(np.logical_and(mask > 0, layer > 0).sum() / denominator)
