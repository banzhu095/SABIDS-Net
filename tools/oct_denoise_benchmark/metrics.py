from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    minimum = min(a.shape)
    win = min(7, minimum if minimum % 2 else minimum - 1)
    return float(structural_similarity(a, b, data_range=1.0, win_size=max(win, 3)))


def ms_ssim(output: np.ndarray, reference: np.ndarray, levels: int = 5) -> float:
    """Wang et al. multi-scale SSIM with Gaussian local statistics."""
    weights = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=np.float64)[:levels]
    left, right = output.astype(np.float64), reference.astype(np.float64)
    contrast_structure: list[float] = []
    scale_ssim: list[float] = []
    c1, c2 = 0.01**2, 0.03**2
    for level in range(levels):
        mu_left = gaussian_filter(left, 1.5, mode="reflect", truncate=3.5)
        mu_right = gaussian_filter(right, 1.5, mode="reflect", truncate=3.5)
        sigma_left = gaussian_filter(left * left, 1.5, mode="reflect", truncate=3.5) - mu_left * mu_left
        sigma_right = gaussian_filter(right * right, 1.5, mode="reflect", truncate=3.5) - mu_right * mu_right
        covariance = gaussian_filter(left * right, 1.5, mode="reflect", truncate=3.5) - mu_left * mu_right
        cs_map = (2 * covariance + c2) / (sigma_left + sigma_right + c2)
        ssim_map = ((2 * mu_left * mu_right + c1) / (mu_left * mu_left + mu_right * mu_right + c1)) * cs_map
        contrast_structure.append(max(float(cs_map.mean()), 1e-8))
        scale_ssim.append(max(float(ssim_map.mean()), 1e-8))
        if level == levels - 1 or min(left.shape) < 16:
            break
        left = gaussian_filter(left, 1.0, mode="reflect")[::2, ::2]
        right = gaussian_filter(right, 1.0, mode="reflect")[::2, ::2]
    used = len(scale_ssim)
    local_weights = weights[:used]; local_weights /= local_weights.sum()
    if used == 1:
        return scale_ssim[0]
    return float(np.prod(np.power(contrast_structure[:-1], local_weights[:-1])) * scale_ssim[-1] ** local_weights[-1])


def _gradient(image: np.ndarray) -> np.ndarray:
    gx, gy = sobel(image, axis=1, mode="reflect"), sobel(image, axis=0, mode="reflect")
    return np.hypot(gx, gy)


def _energy_ratio(output: np.ndarray, reference: np.ndarray, operator: str) -> tuple[float, float]:
    if operator == "hf":
        out = output - gaussian_filter(output, 1.0, mode="reflect")
        ref = reference - gaussian_filter(reference, 1.0, mode="reflect")
    else:
        out, ref = laplace(output, mode="reflect"), laplace(reference, mode="reflect")
    ratio = float(np.mean(out.astype(np.float64) ** 2) / max(np.mean(ref.astype(np.float64) ** 2), 1e-12))
    return ratio, abs(math.log(max(ratio, 1e-12)))


def compute_metrics(noisy: np.ndarray, reference: np.ndarray, output: np.ndarray, algorithm_seconds: float = 0.0, io_seconds: float = 0.0) -> dict[str, float]:
    noisy, reference, output = (np.asarray(x, dtype=np.float32) for x in (noisy, reference, output))
    if noisy.shape != reference.shape or output.shape != reference.shape:
        raise ValueError("metric arrays must share shape")
    hf_ratio, hf_abs_log = _energy_ratio(output, reference, "hf")
    lap_ratio, lap_abs_log = _energy_ratio(output, reference, "lap")
    grad_out, grad_ref = _gradient(output), _gradient(reference)
    edge_mask = grad_ref >= np.quantile(grad_ref, 0.90)
    residual = noisy.astype(np.float64) - output.astype(np.float64)
    centered_out, centered_ref = grad_out - grad_out.mean(), grad_ref - grad_ref.mean()
    epi = float(np.sum(centered_out * centered_ref) / max(np.sqrt(np.sum(centered_out**2) * np.sum(centered_ref**2)), 1e-12))
    error = output.astype(np.float64) - reference.astype(np.float64)
    return {
        "psnr": float(peak_signal_noise_ratio(reference, output, data_range=1.0)),
        "ssim": _ssim(reference, output),
        "ms_ssim": ms_ssim(output, reference),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "epi": epi,
        "reference_edge_mae": float(np.mean(np.abs(error)[edge_mask])) if edge_mask.any() else float("nan"),
        "gradient_magnitude_mae": float(np.mean(np.abs(grad_out - grad_ref))),
        "hf_energy_ratio_to_reference": hf_ratio,
        "abs_log_hf_energy_ratio_to_reference": hf_abs_log,
        "laplacian_energy_ratio": lap_ratio,
        "abs_log_laplacian_energy_ratio": lap_abs_log,
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "algorithm_seconds": float(algorithm_seconds),
        "io_seconds": float(io_seconds),
        "inference_seconds": float(algorithm_seconds + io_seconds),
    }


METRIC_COLUMNS = [
    "psnr", "ssim", "ms_ssim", "rmse", "mae", "epi", "reference_edge_mae",
    "gradient_magnitude_mae", "hf_energy_ratio_to_reference",
    "abs_log_hf_energy_ratio_to_reference", "laplacian_energy_ratio",
    "abs_log_laplacian_energy_ratio", "residual_mean", "residual_std",
    "algorithm_seconds", "io_seconds", "inference_seconds",
]
