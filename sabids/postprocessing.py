from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_fill_holes, label


def clean_layer_mask(
    mask: np.ndarray,
    valid: np.ndarray,
    minimum_main_fraction: float = 0.5,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """P1: retain the main layer component and fill only its enclosed holes."""
    source = mask.astype(bool) & valid.astype(bool)
    labelled, count = label(source)
    if count == 0:
        return source, {
            "p1_empty_input": 1.0, "p1_component_count": 0.0,
            "p1_removed_pixels": 0.0, "p1_filled_pixels": 0.0,
            "p1_main_fraction": float("nan"), "p1_cleanup_failed": 1.0,
        }
    areas = np.bincount(labelled.ravel())[1:]
    main_id = int(np.argmax(areas)) + 1
    main = labelled == main_id
    filled = binary_fill_holes(main) & valid.astype(bool)
    source_pixels = float(source.sum())
    main_fraction = float(main.sum()) / max(source_pixels, 1.0)
    stats = {
        "p1_empty_input": 0.0, "p1_component_count": float(count),
        "p1_removed_pixels": float((source & ~main).sum()),
        "p1_filled_pixels": float((filled & ~main).sum()),
        "p1_main_fraction": main_fraction,
        "p1_cleanup_failed": float(main_fraction < minimum_main_fraction),
    }
    return filled, stats


def _runs(columns: np.ndarray) -> list[np.ndarray]:
    if columns.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(columns) > 1) + 1
    return list(np.split(columns, cuts))


def regularize_lower_boundary(
    mask: np.ndarray,
    valid: np.ndarray,
    smoothness: float = 2.0,
    max_displacement: int = 8,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """P2: smooth lower boundary with quadratic data fidelity; keep upper fixed."""
    source = mask.astype(bool) & valid.astype(bool)
    height, width = source.shape
    upper = np.full(width, -1, dtype=int)
    lower = np.full(width, -1, dtype=int)
    for column in range(width):
        indices = np.flatnonzero(source[:, column])
        if indices.size:
            upper[column], lower[column] = int(indices[0]), int(indices[-1])
    present = np.flatnonzero(lower >= 0)
    if not present.size:
        return source, {
            "p2_empty_input": 1.0, "p2_changed_columns": 0.0,
            "p2_mean_abs_displacement": float("nan"), "p2_max_abs_displacement": float("nan"),
            "p2_regularization_failed": 1.0,
        }
    smoothed = lower.copy()
    for columns in _runs(present):
        if columns.size < 3 or smoothness <= 0:
            continue
        values = lower[columns].astype(np.float64)
        second = np.diff(np.eye(columns.size), n=2, axis=0)
        fitted = np.linalg.solve(
            np.eye(columns.size) + float(smoothness) * second.T @ second,
            values,
        )
        fitted = np.rint(fitted).astype(int)
        fitted = np.clip(fitted, values.astype(int) - max_displacement,
                         values.astype(int) + max_displacement)
        for offset, column in enumerate(columns):
            valid_rows = np.flatnonzero(valid[:, column])
            maximum = int(valid_rows[-1]) if valid_rows.size else lower[column]
            smoothed[column] = int(np.clip(fitted[offset], upper[column], min(maximum, height - 1)))
    result = np.zeros_like(source)
    for column in present:
        result[upper[column]:smoothed[column] + 1, column] = True
    result &= valid.astype(bool)
    displacement = np.abs(smoothed[present] - lower[present])
    return result, {
        "p2_empty_input": 0.0,
        "p2_changed_columns": float((displacement > 0).sum()),
        "p2_mean_abs_displacement": float(displacement.mean()),
        "p2_max_abs_displacement": float(displacement.max()),
        "p2_regularization_failed": 0.0,
    }


def hard_contain_vessel(
    vessel: np.ndarray,
    layer: np.ndarray,
    vessel_valid: np.ndarray,
    target: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """P3: strict deployable containment. Never expands vessel or layer masks."""
    raw = vessel.astype(bool) & vessel_valid.astype(bool)
    final = raw & layer.astype(bool)
    removed = raw & ~final
    stats = {
        "p3_raw_pixels": float(raw.sum()), "p3_final_pixels": float(final.sum()),
        "p3_removed_pixels": float(removed.sum()),
        "p3_removed_fraction": float(removed.sum()) / max(float(raw.sum()), 1.0),
        "p3_empty_raw": float(not raw.any()), "p3_empty_final": float(not final.any()),
        "p3_outside_final_layer_pixels": float((final & ~layer.astype(bool)).sum()),
    }
    if target is not None:
        truth = target.astype(bool) & vessel_valid.astype(bool)
        stats["p3_removed_tp"] = float((removed & truth).sum())
        stats["p3_removed_fp"] = float((removed & ~truth).sum())
    return final, stats
