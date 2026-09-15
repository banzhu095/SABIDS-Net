from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, laplace, sobel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from tools.oct_denoise_benchmark.metrics import compute_metrics


def _win_size(shape: tuple[int, ...]) -> int:
    minimum = min(shape)
    win = min(7, minimum if minimum % 2 else minimum - 1)
    if win < 3: raise ValueError("ROI is too small for SSIM")
    return int(win)


def compute_roi_metrics(noisy: np.ndarray, reference: np.ndarray, denoised: np.ndarray, tissue: str,
                        bright_outlier_threshold: float = 0.10) -> dict[str, float | int]:
    if noisy.shape != reference.shape or denoised.shape != reference.shape:
        raise ValueError("all ROI crops must have identical shape")
    values = [np.asarray(x, dtype=np.float32) for x in (noisy, reference, denoised)]
    if any(not np.isfinite(x).all() or x.min() < 0 or x.max() > 1 for x in values): raise ValueError("ROI values must be finite float32 [0,1]")
    noisy, reference, denoised = values
    metrics = compute_metrics(noisy, reference, denoised)
    metrics.pop("ms_ssim", None); metrics.pop("algorithm_seconds", None); metrics.pop("io_seconds", None); metrics.pop("inference_seconds", None)
    win = _win_size(reference.shape)
    metrics["ssim"] = float(structural_similarity(reference, denoised, data_range=1.0, win_size=win))
    metrics["ssim_win_size"] = win
    error = denoised.astype(np.float64) - reference.astype(np.float64)
    residual = noisy.astype(np.float64) - denoised.astype(np.float64)
    metrics.update({"roi_mean": float(denoised.mean()), "roi_std": float(denoised.std()), "reference_mean": float(reference.mean()), "reference_std": float(reference.std()), "background_bias": float(error.mean()), "error_std": float(error.std()), "bright_outlier_threshold": float(bright_outlier_threshold), "bright_outlier_fraction": float(np.mean(denoised > bright_outlier_threshold)), "local_contrast": float(denoised.std())})
    hf_residual = residual - gaussian_filter(residual, 1.0, mode="reflect")
    metrics["hf_residual_energy"] = float(np.mean(hf_residual**2))
    return metrics


def vessel_stroma_pair_metrics(vessel: np.ndarray, stroma: np.ndarray, reference_vessel: np.ndarray,
                                reference_stroma: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    def cnr(left: np.ndarray, right: np.ndarray) -> float:
        return float(abs(float(left.mean()) - float(right.mean())) / math.sqrt(0.5 * (float(left.var()) + float(right.var())) + eps))
    observed_cnr, reference_cnr = cnr(vessel, stroma), cnr(reference_vessel, reference_stroma)
    contrast = float(stroma.mean() - vessel.mean())
    reference_contrast = float(reference_stroma.mean() - reference_vessel.mean())
    return {"vessel_stroma_cnr": observed_cnr, "reference_vessel_stroma_cnr": reference_cnr, "cnr_error_to_reference": abs(observed_cnr - reference_cnr), "vessel_stroma_contrast": contrast, "reference_vessel_stroma_contrast": reference_contrast, "contrast_error_to_reference": abs(contrast - reference_contrast), "expected_polarity": "vessel_darker_than_stroma", "polarity_preserved": bool(contrast > 0)}
