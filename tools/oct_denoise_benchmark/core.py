from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, laplace

from sabids.data.io import read_gray
from sabids.metrics import (
    edge_preservation_index,
    psnr,
    reference_edge_mae,
    rmse,
    ssim,
)


METRIC_COLUMNS = [
    "psnr", "ssim", "rmse", "mae", "epi", "reference_edge_mae",
    "hf_energy_clean", "hf_energy_noisy", "hf_energy_denoised",
    "hf_energy_ratio_to_clean", "laplacian_energy_clean", "laplacian_energy_noisy",
    "laplacian_energy_denoised", "laplacian_energy_ratio_to_clean",
    "gradient_magnitude_mae", "residual_mean", "residual_std", "inference_seconds",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_original(path: Path) -> Tuple[np.ndarray, np.dtype]:
    data = np.fromfile(str(path), dtype=np.uint8)
    array = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if array is None:
        raise RuntimeError(f"OpenCV failed to decode {path}")
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY if array.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    return array, array.dtype


def image_metadata(path: Path) -> Dict[str, Any]:
    array, dtype = read_original(path)
    return {
        "height": int(array.shape[0]),
        "width": int(array.shape[1]),
        "dtype": str(dtype),
        "bit_depth": int(np.iinfo(dtype).bits) if np.issubdtype(dtype, np.integer) else int(dtype.itemsize * 8),
        "sha256": sha256_file(path),
    }


def save_like_source(path: Path, image: np.ndarray, source_dtype: np.dtype) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0.0, 1.0)
    if np.issubdtype(source_dtype, np.integer):
        maximum = float(np.iinfo(source_dtype).max)
        output = np.round(clipped * maximum).astype(source_dtype)
    else:
        output = clipped.astype(source_dtype)
    suffix = path.suffix.lower()
    if suffix not in {".tif", ".tiff", ".png"}:
        suffix = ".png"
        path = path.with_suffix(suffix)
    ok, encoded = cv2.imencode(suffix, output)
    if not ok:
        raise RuntimeError(f"OpenCV failed to encode {path}")
    encoded.tofile(str(path))


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float64)
    gy, gx = np.gradient(image)
    return np.sqrt(gx * gx + gy * gy)


def _hf_energy(image: np.ndarray) -> float:
    high = image.astype(np.float64) - gaussian_filter(image.astype(np.float64), sigma=1.0, mode="reflect")
    return float(np.mean(high * high))


def _laplacian_energy(image: np.ndarray) -> float:
    response = laplace(image.astype(np.float64), mode="reflect")
    return float(np.mean(response * response))


def metric_row(noisy: np.ndarray, clean: np.ndarray, output: np.ndarray, inference_seconds: float) -> Dict[str, float]:
    hf_clean, hf_noisy, hf_output = _hf_energy(clean), _hf_energy(noisy), _hf_energy(output)
    lap_clean = _laplacian_energy(clean)
    lap_noisy = _laplacian_energy(noisy)
    lap_output = _laplacian_energy(output)
    residual = noisy.astype(np.float64) - output.astype(np.float64)
    row = {
        "psnr": psnr(output, clean),
        "ssim": ssim(output, clean),
        "rmse": rmse(output, clean),
        "mae": float(np.mean(np.abs(output.astype(np.float64) - clean.astype(np.float64)))),
        "epi": edge_preservation_index(output, clean),
        "reference_edge_mae": reference_edge_mae(output, clean),
        "hf_energy_clean": hf_clean,
        "hf_energy_noisy": hf_noisy,
        "hf_energy_denoised": hf_output,
        "hf_energy_ratio_to_clean": hf_output / max(hf_clean, 1e-12),
        "laplacian_energy_clean": lap_clean,
        "laplacian_energy_noisy": lap_noisy,
        "laplacian_energy_denoised": lap_output,
        "laplacian_energy_ratio_to_clean": lap_output / max(lap_clean, 1e-12),
        "gradient_magnitude_mae": float(np.mean(np.abs(_gradient_magnitude(output) - _gradient_magnitude(clean)))),
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "inference_seconds": float(inference_seconds),
    }
    return {key: float(value) for key, value in row.items()}


def resolve_manifest_paths(table: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    result = table.copy()
    for column in ("image_path", "clean_path"):
        result[column] = result[column].map(
            lambda value: str((project_root / str(value)).resolve()) if str(value).strip() else ""
        )
    return result


def select_validation_subset(table: pd.DataFrame, positions_per_dataset: int = 2, frames_per_position: int = 1) -> pd.DataFrame:
    selected: List[pd.DataFrame] = []
    val = table[table["split"].astype(str) == "val"].copy()
    for dataset, dataset_rows in val.groupby("dataset", sort=True):
        positions = sorted(dataset_rows["group_id"].astype(str).unique())[:positions_per_dataset]
        for position in positions:
            position_rows = dataset_rows[dataset_rows["group_id"].astype(str) == position].sort_values(
                ["frame_index", "sample_id"], kind="stable"
            )
            if len(position_rows) <= frames_per_position:
                selected.append(position_rows)
            else:
                indices = np.linspace(0, len(position_rows) - 1, frames_per_position, dtype=int)
                selected.append(position_rows.iloc[indices])
    if not selected:
        raise ValueError("No validation rows available")
    return pd.concat(selected, ignore_index=True)


def aggregate_metrics(per_image: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in METRIC_COLUMNS if column in per_image.columns]
    positions = (
        per_image.groupby(["dataset", "position_id", "method_id"], as_index=False)[numeric]
        .mean(numeric_only=True)
    )
    positions["frame_count"] = per_image.groupby(
        ["dataset", "position_id", "method_id"]
    ).size().to_numpy()
    datasets = positions.groupby(["dataset", "method_id"], as_index=False)[numeric].mean(numeric_only=True)
    datasets["position_count"] = positions.groupby(["dataset", "method_id"]).size().to_numpy()
    rows: List[Dict[str, Any]] = []
    for method, frame_rows in per_image.groupby("method_id", sort=True):
        position_rows = positions[positions["method_id"] == method]
        dataset_rows = datasets[datasets["method_id"] == method]
        for aggregation, source in (
            ("frame_micro", frame_rows),
            ("position_macro", position_rows),
            ("dataset_macro", dataset_rows),
        ):
            row: Dict[str, Any] = {"method_id": method, "aggregation": aggregation, "n": int(len(source))}
            row.update({column: float(source[column].mean()) for column in numeric})
            rows.append(row)
    return positions, datasets, pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, iterations: int = 10_000) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value, value
    means = np.empty(iterations, dtype=np.float64)
    batch = 500
    for start in range(0, iterations, batch):
        count = min(batch, iterations - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def infer_frame_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("frame_index", "")).strip()
    return value if value else str(row.get("sample_id", ""))


def finite_or_blank(value: Any) -> Any:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return numeric if math.isfinite(numeric) else ""
