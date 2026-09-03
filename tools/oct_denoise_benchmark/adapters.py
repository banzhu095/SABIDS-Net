from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping

import cv2
import numpy as np
import pywt
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.restoration import denoise_tv_chambolle


class AdapterUnavailable(RuntimeError):
    """Raised when the requested published implementation cannot run fairly."""


@dataclass(frozen=True)
class AdapterSpec:
    method_id: str
    function: Callable[[np.ndarray, Mapping[str, Any], Mapping[str, Any]], np.ndarray]
    deterministic: bool = True


def _identity(image: np.ndarray, config: Mapping[str, Any], context: Mapping[str, Any]) -> np.ndarray:
    return image.copy()


def _gaussian(image: np.ndarray, config: Mapping[str, Any], context: Mapping[str, Any]) -> np.ndarray:
    sigma = float(config.get("sigma", 1.0))
    return gaussian_filter(image, sigma=sigma, mode="reflect")


def _tv(image: np.ndarray, config: Mapping[str, Any], context: Mapping[str, Any]) -> np.ndarray:
    weight = float(config.get("weight", 0.05))
    iterations = int(config.get("max_num_iter", 200))
    return denoise_tv_chambolle(image, weight=weight, max_num_iter=iterations, channel_axis=None)


def _wavelet(image: np.ndarray, config: Mapping[str, Any], context: Mapping[str, Any]) -> np.ndarray:
    # The repository's four-tap "Daubechies D4" analysis filter corresponds
    # to PyWavelets db2.  Unlike the original demo, this implementation never
    # adds synthetic noise and never contrast-stretches the reconstruction.
    wavelet = str(config.get("wavelet", "db2"))
    level = int(config.get("level", 1))
    threshold_scale = float(config.get("threshold_scale", 0.8))
    threshold_mode = str(config.get("threshold_mode", "soft"))
    multiple = 2 ** level
    pad_h = (-image.shape[0]) % multiple
    pad_w = (-image.shape[1]) % multiple
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    coeffs = pywt.wavedec2(padded, wavelet=wavelet, level=level, mode="symmetric")
    finest_hh = coeffs[-1][2]
    sigma = float(np.median(np.abs(finest_hh)) / 0.6745)
    threshold = threshold_scale * sigma
    thresholded = [coeffs[0]]
    for detail in coeffs[1:]:
        thresholded.append(tuple(pywt.threshold(c, threshold, mode=threshold_mode) for c in detail))
    restored = pywt.waverec2(thresholded, wavelet=wavelet, mode="symmetric")
    return restored[: image.shape[0], : image.shape[1]]


def _nlm_speckle(image: np.ndarray, config: Mapping[str, Any], context: Mapping[str, Any]) -> np.ndarray:
    """Pearson-distance NLM adapted from the bundled BNLPD implementation.

    The original MATLAB script downsamples the image and never restores its
    geometry.  This vectorized adapter retains its multiplicative-speckle
    distance, exp(-d/h^2) weighting, local patches and search window while
    operating at native resolution and without display normalization.
    """
    patch_size = int(config.get("patch_size", 3))
    search_radius = int(config.get("search_radius", 3))
    h = float(config.get("h", 0.15))
    gamma = float(config.get("gamma", 0.5))
    eps = float(config.get("epsilon", 1e-6))
    if patch_size % 2 != 1 or patch_size < 1:
        raise ValueError("patch_size must be a positive odd integer")
    pad = search_radius
    padded = np.pad(image, pad, mode="reflect")
    numerator = np.zeros_like(image, dtype=np.float64)
    denominator = np.zeros_like(image, dtype=np.float64)
    h2 = max(h * h, eps)
    height, width = image.shape
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            shifted = padded[pad + dy : pad + dy + height, pad + dx : pad + dx + width]
            pearson_term = (image - shifted) ** 2 / np.maximum(shifted, eps) ** (2.0 * gamma)
            distance = uniform_filter(pearson_term, size=patch_size, mode="reflect")
            weight = np.exp(-distance / h2)
            numerator += weight * shifted
            denominator += weight
    return numerator / np.maximum(denominator, eps)


def _bm3d(image: np.ndarray, config: Mapping[str, Any], context: Mapping[str, Any]) -> np.ndarray:
    try:
        import bm3d as bm3d_package
    except Exception as exc:  # pragma: no cover - environment dependent
        raise AdapterUnavailable(f"bm3d package unavailable: {exc}") from exc
    sigma_psd = float(config.get("sigma_psd", 0.08))
    stage_name = str(config.get("stage", "all")).lower()
    stage = (
        bm3d_package.BM3DStages.HARD_THRESHOLDING
        if stage_name in {"hard", "hard_thresholding"}
        else bm3d_package.BM3DStages.ALL_STAGES
    )
    profile = bm3d_package.BM3DProfile()
    # Parallel aggregation can change the last floating-point bits between
    # runs.  One thread makes the scientific output exactly reproducible.
    profile.num_threads = 1
    return bm3d_package.bm3d(image, sigma_psd=sigma_psd, profile=profile, stage_arg=stage)


def _msbtd_unavailable(image: np.ndarray, config: Mapping[str, Any], context: Mapping[str, Any]) -> np.ndarray:
    raise AdapterUnavailable(
        "Bundled MSBTD is protected MATLAB P-code. Its one-argument function expects a private struct "
        "containing TrainIm/high-SNR dictionary data, noise estimates and multiscale parameters; a "
        "validated non-GUI batch contract is unavailable, so substitution is forbidden."
    )


def _ascibp_unavailable(image: np.ndarray, config: Mapping[str, Any], context: Mapping[str, Any]) -> np.ndarray:
    raise AdapterUnavailable(
        "Bundled ASCIBP/WNNM is protected MATLAB P-code and its published Demo calls "
        "WNNM_DeNoising(noisy, clean, params). The clean argument affects iteration diagnostics and "
        "cannot be proven algorithmically inert, so the fair adapter is blocked rather than leaking reference data."
    )


ADAPTERS: Dict[str, AdapterSpec] = {
    "noisy_identity": AdapterSpec("noisy_identity", _identity),
    "gaussian": AdapterSpec("gaussian", _gaussian),
    "tv": AdapterSpec("tv", _tv),
    "wavelet": AdapterSpec("wavelet", _wavelet),
    "nlm_speckle": AdapterSpec("nlm_speckle", _nlm_speckle),
    "bm3d": AdapterSpec("bm3d", _bm3d),
    "msbtd": AdapterSpec("msbtd", _msbtd_unavailable),
    "ascibp": AdapterSpec("ascibp", _ascibp_unavailable),
}


def denoise(
    image_float32_01: np.ndarray,
    method_config: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
) -> np.ndarray:
    image = np.asarray(image_float32_01)
    if image.ndim != 2 or image.dtype != np.float32:
        raise TypeError(f"adapter input must be 2-D float32, got {image.shape} {image.dtype}")
    if not np.isfinite(image).all() or float(image.min()) < 0.0 or float(image.max()) > 1.0:
        raise ValueError("adapter input must be finite and within [0,1]")
    method_id = str(method_config.get("method_id", ""))
    if method_id not in ADAPTERS:
        raise KeyError(f"unknown method_id={method_id!r}")
    output = np.asarray(ADAPTERS[method_id].function(image, method_config, context or {}))
    if output.shape != image.shape:
        raise ValueError(f"{method_id} changed shape from {image.shape} to {output.shape}")
    if not np.isfinite(output).all():
        raise ValueError(f"{method_id} returned NaN/Inf")
    # This is the only quantitative post-processing allowed by the protocol.
    return np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
