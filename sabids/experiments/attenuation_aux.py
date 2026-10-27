"""Attenuation-inspired OCT enhancement with explicit numerical contracts.

The implementation deliberately separates a MATLAB-formula diagnostic from the
finite, bounded view used by experiments.  The latter is an image enhancement;
display-gray input is not sufficient to estimate a physical attenuation
coefficient.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VERSION = "attenuation_aux_v1"
CACHE_VERSION = "attenuation_aux_v1.ac_r2"


@dataclass(frozen=True)
class StableACConfig:
    power: float = 2.0
    depth_spacing: float = 1.0
    denominator_floor: float = 0.01
    signal_floor: float = 0.01
    output_log_low: float = 0.0
    output_log_high: float = 1.0
    gain_cap: float = 6.0
    reliability_floor: float = 0.20
    beta: float = 0.50
    output_window: float = 25.0
    algorithm_revision: str = CACHE_VERSION

    def validate(self) -> None:
        if self.power <= 0 or self.depth_spacing <= 0:
            raise ValueError("power and depth_spacing must be positive")
        if self.denominator_floor <= 0 or self.signal_floor <= 0:
            raise ValueError("stable floors must be positive signal-scale values")
        if self.gain_cap < 1:
            raise ValueError("gain_cap must be at least one")
        if not 0 <= self.reliability_floor < 1:
            raise ValueError("reliability_floor must be in [0,1)")
        if not 0 <= self.beta <= 1:
            raise ValueError("beta must be in [0,1]")
        if self.output_log_high <= self.output_log_low:
            raise ValueError("output log mapping must have positive width")
        if self.output_window <= 0 or self.algorithm_revision != CACHE_VERSION:
            raise ValueError("Invalid output window or stale AC algorithm revision")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_image(image: np.ndarray, valid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(image, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError(f"AC expects a 2-D image, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("AC input contains non-finite values")
    value = np.clip(value, 0.0, 1.0)
    mask = np.ones_like(value, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if mask.shape != value.shape:
        raise ValueError("validity mask shape differs from image")
    return value, mask


def reverse_cumtrapz(values: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Reverse cumulative trapezoidal integral along image depth (axis 0).

    Row ``i`` contains the integral from row ``i`` to the last row.  In exact
    MATLAB ``flipud(cumtrapz(flipud(x)))`` form the final row is zero.
    """
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 2:
        raise ValueError("reverse_cumtrapz expects HxW")
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    out = np.zeros_like(value, dtype=np.float64)
    if value.shape[0] > 1:
        trapezoids = 0.5 * (value[:-1] + value[1:]) * float(spacing)
        out[:-1] = np.flip(np.cumsum(np.flip(trapezoids, axis=0), axis=0), axis=0)
    return out


def matlab_compatible(
    image: np.ndarray,
    *,
    power: float = 2.0,
    depth_spacing: float = 1.0,
    valid: np.ndarray | None = None,
    depth_count: float | None = None,
    epsilon_guard: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Diagnostic reproduction of ``x**p / reverse_cumtrapz(x**p)``.

    No reliability floor or clipping is applied.  Non-finite terminal values
    are retained in the returned array so the diagnostic cannot be mistaken
    for the stable training view.
    """
    value, mask = _validate_image(image, valid)
    powered = np.where(mask, value.astype(np.float64) ** float(power), 0.0)
    n = float(value.shape[0] if depth_count is None else depth_count)
    denominator = reverse_cumtrapz(powered / n**2, depth_spacing)
    if epsilon_guard:
        denominator = np.maximum(denominator, np.finfo(np.float64).eps)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        output = (powered / denominator) / n
    output[~mask] = np.nan
    finite = np.isfinite(output) & mask
    stats = {
        "version": VERSION,
        "mode": "matlab_compatible",
        "input_domain": "display_gray_[0,1]",
        "power": float(power),
        "depth_axis": 0,
        "integration": "reverse cumulative trapezoid",
        "depth_spacing": float(depth_spacing),
        "denominator_protection": "float64 eps on scaled denominator" if epsilon_guard else "none",
        "depth_count": n,
        "formula": "x**p / reverse_cumtrapz(x**p / N**2) / N",
        "nonfinite_count": int((mask & ~np.isfinite(output)).sum()),
        "valid_count": int(mask.sum()),
        "terminal_nonfinite_count": int((mask[-1] & ~np.isfinite(output[-1])).sum()),
        "finite_min": float(output[finite].min()) if finite.any() else None,
        "finite_max": float(output[finite].max()) if finite.any() else None,
    }
    return output, stats


def fit_legacy_train_statistics(
    images: Iterable[np.ndarray],
    *,
    power: float = 2.0,
    depth_spacing: float = 1.0,
    source_records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit robust signal-scale floors from development-train images only."""
    intensity_samples: list[np.ndarray] = []
    integral_samples: list[np.ndarray] = []
    raw_log_samples: list[np.ndarray] = []
    count = 0
    for image in images:
        value, mask = _validate_image(image)
        count += 1
        positive = value[mask & (value > 0)]
        if positive.size:
            intensity_samples.append(positive[:: max(1, positive.size // 20000)])
        powered = np.where(mask, value.astype(np.float64) ** float(power), 0.0)
        denominator = reverse_cumtrapz(powered, depth_spacing)
        usable = denominator[mask & (denominator > 0)]
        if usable.size:
            integral_samples.append(usable[:: max(1, usable.size // 20000)])
        raw = powered / np.maximum(denominator, 1e-12)
        raw_usable = np.log1p(raw[mask & np.isfinite(raw) & (raw >= 0)])
        if raw_usable.size:
            raw_log_samples.append(raw_usable[:: max(1, raw_usable.size // 20000)])
    if not count or not intensity_samples or not integral_samples or not raw_log_samples:
        raise ValueError("Cannot fit AC statistics without positive train signal")
    intensities = np.concatenate(intensity_samples)
    integrals = np.concatenate(integral_samples)
    raw_logs = np.concatenate(raw_log_samples)
    records = sorted(list(source_records or []), key=lambda row: json.dumps(row, sort_keys=True))
    result = {
        "version": VERSION,
        "source_split": "train_only",
        "image_count": count,
        "input_domain": "display_gray_[0,1]",
        "power": float(power),
        "depth_spacing": float(depth_spacing),
        "signal_floor_q01": float(max(np.quantile(intensities, 0.01), 1.0 / 65535.0)),
        "denominator_floor_q005": float(max(np.quantile(integrals, 0.005), 1e-8)),
        "output_log_low_q005": float(np.quantile(raw_logs, 0.005)),
        "output_log_high_q995": float(np.quantile(raw_logs, 0.995)),
        "input_quantiles": {
            str(q): float(np.quantile(intensities, q)) for q in (0.005, 0.01, 0.5, 0.99, 0.995)
        },
        "denominator_quantiles": {
            str(q): float(np.quantile(integrals, q)) for q in (0.005, 0.01, 0.5, 0.99)
        },
        "sources": records,
    }
    result["statistics_sha256"] = stable_json_sha256(result)
    return result


def config_from_legacy_statistics(
    statistics: dict[str, Any], *, beta: float = 0.5, gain_cap: float = 6.0,
    reliability_floor: float = 0.20,
) -> StableACConfig:
    claimed = statistics.get("statistics_sha256")
    unsigned = dict(statistics)
    unsigned.pop("statistics_sha256", None)
    if claimed != stable_json_sha256(unsigned):
        raise ValueError("AC train statistics SHA mismatch")
    if statistics.get("source_split") != "train_only":
        raise ValueError("Stable AC statistics must be train-only")
    return StableACConfig(
        power=float(statistics["power"]),
        depth_spacing=float(statistics["depth_spacing"]),
        denominator_floor=float(statistics["denominator_floor_q005"]),
        signal_floor=float(statistics["signal_floor_q01"]),
        output_log_low=float(statistics["output_log_low_q005"]),
        output_log_high=float(statistics["output_log_high_q995"]),
        gain_cap=float(gain_cap),
        reliability_floor=float(reliability_floor),
        beta=float(beta),
    )


def legacy_stable_ac_v0(
    image: np.ndarray,
    config: StableACConfig,
    *,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Return a bounded enhancement and auditable reliability diagnostics."""
    # Retained solely to reproduce v3 failures; never used to prepare r2 caches.
    value, mask = _validate_image(image, valid)
    value = value.astype(np.float32)
    powered = np.where(mask, value.astype(np.float64) ** config.power, 0.0)
    denominator = reverse_cumtrapz(powered, config.depth_spacing)
    safe_denominator = np.maximum(denominator, config.denominator_floor)
    raw = powered / safe_denominator
    mapped = np.clip(
        (np.log1p(raw) - config.output_log_low)
        / (config.output_log_high - config.output_log_low),
        0.0,
        1.0,
    )
    # AC is an enhancement view, not a replacement radiometric calibration.
    # It may brighten attenuated signal, but cannot exceed a fixed multiplicative
    # cap relative to the original signal.
    enhanced = np.maximum(value, np.minimum(mapped, value * config.gain_cap))
    gain = enhanced / np.maximum(value, config.signal_floor)
    reliability = denominator / (denominator + config.denominator_floor)
    reliable_weight = np.clip(
        (reliability - config.reliability_floor) / (1.0 - config.reliability_floor),
        0.0,
        1.0,
    )
    safe = reliable_weight * enhanced + (1.0 - reliable_weight) * value
    safe = np.where(mask, safe, value).astype(np.float32)
    blended = ((1.0 - config.beta) * value + config.beta * safe).astype(np.float32)
    blended = np.clip(np.nan_to_num(blended, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    diagnostics = {
        "gain": np.where(mask, gain, 1.0).astype(np.float32),
        "reliability": np.where(mask, reliability, 0.0).astype(np.float32),
        "low_reliability": (mask & (reliable_weight <= 0)).astype(np.uint8),
        "saturated": (mask & (safe >= 1.0 - 1e-6)).astype(np.uint8),
        "difference": (blended - value).astype(np.float32),
        "safe_output": safe,
        "denominator": denominator.astype(np.float32),
    }
    stats = {
        "version": VERSION,
        "mode": "stable_ac_v1",
        "claim": "attenuation-inspired enhancement",
        "config": asdict(config),
        "input_domain": "display_gray_[0,1]",
        "finite": bool(np.isfinite(blended).all()),
        "valid_fraction": float(mask.mean()),
        "low_reliability_fraction": float(diagnostics["low_reliability"][mask].mean()) if mask.any() else 0.0,
        "saturated_fraction": float(diagnostics["saturated"][mask].mean()) if mask.any() else 0.0,
        "gain_clipped_fraction": float((gain[mask] >= config.gain_cap - 1e-9).mean()) if mask.any() else 0.0,
        "output_sha256": array_sha256(blended),
    }
    return blended, diagnostics, stats


def source_formula(image: np.ndarray, n: float = 1536.0, *, epsilon_guard: bool = False,
                   depth_spacing: float = 1.0) -> np.ndarray:
    """Source-exact float64 formula, accepting uint8-scale or normalized input.

    No rotation is applied to TIFFs. LoadFileArb flips the proprietary .img
    storage into displayed coordinates; applying those flips to TIFFs twice
    would reverse the anatomy. This entry has no display/clip/mix operation.
    """
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2 or n <= 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite HxW input and positive N")
    denominator = reverse_cumtrapz(x**2 / n**2, depth_spacing)
    if epsilon_guard:
        denominator = np.maximum(denominator, np.finfo(np.float64).eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (x**2 / denominator) / n


def _valid_integral(powered: np.ndarray, mask: np.ndarray, spacing: float) -> np.ndarray:
    # Invalid padding and unknown depth do not contribute; no trapezoid bridges
    # an invalid gap. No spatial resize or crop is performed here.
    area = .5 * (powered[:-1] + powered[1:]) * spacing
    area *= mask[:-1] & mask[1:]
    out = np.zeros_like(powered, dtype=np.float64)
    out[:-1] = np.cumsum(area[::-1], axis=0)[::-1]
    return out


def fit_train_statistics(images: Iterable[np.ndarray], *, power: float = 2.0,
                         depth_spacing: float = 1.0,
                         source_records: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Freeze r2 scales with no labels or validation/test pixels.

    Denominator regularizer = two depth samples of the train median powered
    signal (units gray**p * depth px). Mapping uses the q99.5 of N*x**p/I
    in the first 95% of depth, excluding the terminal singularity by a fixed
    geometry rule. It is linear, global, and never fitted per display image.
    The low-gray q01 is diagnostic only, never a pixel rejection criterion.
    """
    records = sorted(list(source_records or []), key=lambda row: json.dumps(row, sort_keys=True))
    if any(row.get("split", "train") != "train" for row in records):
        raise ValueError("Statistics sources must be train-only")
    gray_samples, ratio_samples, depths = [], [], []
    medians, count = [], 0
    for image in images:
        x, valid = _validate_image(image)
        count += 1
        depths.append(x.shape[0])
        powered = x**power
        medians.append(float(np.median(powered)))
        integral = _valid_integral(powered, valid, depth_spacing)
        eligible = np.zeros_like(valid)
        eligible[:max(1, int(np.floor(.95*x.shape[0])))] = True
        eligible &= integral > 0
        ratio = np.divide(x.shape[0]*powered, integral,
                          out=np.zeros_like(powered), where=integral > 0)
        usable = ratio[eligible]
        if usable.size:
            ratio_samples.append(usable[np.linspace(0, usable.size-1, min(20000, usable.size), dtype=int)])
        flat = x.ravel()
        gray_samples.append(flat[np.linspace(0, flat.size-1, min(20000, flat.size), dtype=int)])
    if not count or not ratio_samples or max(medians) <= 0:
        raise ValueError("Cannot fit AC statistics without positive train signal")
    gray = np.concatenate(gray_samples)
    result = {
        "version": CACHE_VERSION, "source_split": "train_only", "image_count": count,
        "input_domain": "display_gray_[0,1]", "power": float(power),
        "depth_spacing": float(depth_spacing), "depths": sorted(set(depths)),
        "denominator_floor": float(2 * np.median(medians) * depth_spacing),
        "denominator_unit": "unscaled reverse integral of gray**p, in gray**p*depth_px",
        "denominator_rule": "two virtual depth samples of train median powered gray",
        "signal_floor_q01": float(max(np.quantile(gray, .01), 1/65535)),
        "signal_floor_role": "diagnostic only; no dark-lumen fallback",
        "output_window": float(max(np.quantile(np.concatenate(ratio_samples), .995), 1e-8)),
        "mapping_rule": "clip(N*x**p/(I+regularizer)/train_q995,0,1)",
        "mapping_fit_region": "first 95 percent of native depth; positive I; no GT",
        "sampling_rule": "up to 20000 linspace pixels per image, no terminal row",
        "sources": records,
    }
    result["statistics_sha256"] = stable_json_sha256(result)
    return result


def config_from_statistics(statistics: dict[str, Any], *, beta: float = .5,
                           gain_cap: float = 6.0, reliability_floor: float = .2) -> StableACConfig:
    unsigned = dict(statistics)
    claimed = unsigned.pop("statistics_sha256", None)
    if claimed != stable_json_sha256(unsigned):
        raise ValueError("AC train statistics SHA mismatch")
    if statistics.get("source_split") != "train_only":
        raise ValueError("Stable AC statistics must be train-only")
    if statistics.get("version") != CACHE_VERSION:
        raise ValueError("Stale AC statistics/cache revision; refit train statistics")
    return StableACConfig(power=statistics["power"], depth_spacing=statistics["depth_spacing"],
                          denominator_floor=statistics["denominator_floor"],
                          signal_floor=statistics["signal_floor_q01"],
                          output_window=statistics["output_window"], beta=beta,
                          gain_cap=gain_cap, reliability_floor=reliability_floor)


def stable_ac_v1(image: np.ndarray, config: StableACConfig, *, valid: np.ndarray | None = None
                 ) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """r2 stable depth-adapted view, with auditable intermediate branches."""
    config.validate()
    x, mask = _validate_image(image, valid)
    energy = np.where(mask, x**config.power, 0)
    integral = _valid_integral(energy, mask, config.depth_spacing)
    n = mask.sum(axis=0, keepdims=True).astype(np.float64)
    regularized = n*energy/(integral+config.denominator_floor)
    reliability = integral/(integral+config.denominator_floor)
    weight = np.clip((reliability-config.reliability_floor)/(1-config.reliability_floor), 0, 1)
    unmapped = regularized/config.output_window
    mapped = np.clip(unmapped, 0, 1)
    # Do not force mapped >= original: that condition suppressed all choroid
    # changes in v3. Dark lumens retain their quadratic numerator.
    capped = np.minimum(mapped, x*config.gain_cap)
    safe = np.where(mask, weight*capped+(1-weight)*x, x)
    blended = np.clip((1-config.beta)*x+config.beta*safe, 0, 1).astype(np.float32)
    if config.beta == 0:
        # Preserve exact identity for float64 reference inputs as well as
        # float32 training caches, without an avoidable downcast.
        input_dtype = np.asarray(image).dtype
        blended = x.astype(input_dtype if np.issubdtype(input_dtype, np.floating) else np.float32)
    diagnostics = {
        "denominator": integral, "regularized_raw": regularized,
        "unmapped_unit_output": unmapped, "mapped": mapped, "capped": capped,
        "safe_output": safe.astype(np.float32), "reliability": reliability,
        "mix_weight": np.where(mask, weight, 0),
        "gain": np.divide(capped, x, out=np.ones_like(x), where=x>0),
        "denominator_floor_active": (integral < config.denominator_floor) & mask,
        "low_signal": (x < config.signal_floor) & mask,
        "low_reliability": (weight<=0) & mask,
        "gain_cap_active": (mapped > x*config.gain_cap) & mask,
        "output_low_clip": (unmapped <= 0) & mask,
        "output_high_clip": (unmapped >= 1) & mask,
        "saturated": (safe >= 1-1e-6) & mask,
        "difference": blended-x,
    }
    stats = {"version": CACHE_VERSION, "mode": "stable_ac_v1",
             "claim": "attenuation-inspired enhancement", "config": asdict(config),
             "input_domain": "display_gray_[0,1]", "finite": bool(np.isfinite(blended).all()),
             "valid_fraction": float(mask.mean()), "output_sha256": array_sha256(blended)}
    for name in ("denominator_floor_active", "low_signal", "low_reliability", "gain_cap_active",
                 "output_low_clip", "output_high_clip", "saturated"):
        stats[name+"_fraction"] = float(diagnostics[name][mask].mean()) if mask.any() else 0.0
    stats["gain_clipped_fraction"] = stats["gain_cap_active_fraction"]
    return blended, diagnostics, stats


def cache_identity(
    *, source_path: str | Path, source_sha256: str, config: StableACConfig,
    statistics_sha256: str, sample_id: str, position_id: str, split: str,
    geometry: str, denoiser_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    result = {
        "version": CACHE_VERSION,
        "sample_id": str(sample_id),
        "position_id": str(position_id),
        "split": str(split),
        "source_path": str(Path(source_path).resolve()),
        "source_sha256": str(source_sha256),
        "input_domain": "display_gray_[0,1]",
        "geometry": str(geometry),
        "config": asdict(config),
        "statistics_sha256": str(statistics_sha256),
        "denoiser_checkpoint_sha256": denoiser_checkpoint_sha256,
        "code_sha256": file_sha256(Path(__file__)),
        "output_mapping_version": "linear_depth_scaled_train_window_r2",
    }
    result["identity_sha256"] = stable_json_sha256(result)
    return result
