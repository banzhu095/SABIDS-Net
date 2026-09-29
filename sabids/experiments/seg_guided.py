"""Fail-closed primitives for segmentation-guided adaptive denoising v1.

All functions operate on development data supplied by the caller.  They never
discover or open sealed-test assets.  Oracle helpers are explicitly
non-deployable and must remain behind an oracle gate.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd

from .dose_response import stable_sha


VERSION = "seg-guided-adaptive-v1"
INTERVENTIONS = ("T0", "T1", "T2", "T3", "T4a", "T4b", "T5", "T6")
CV_ARMS = ("B0", "B1", "B3", "B6", "B3R", "B6R")
ORACLE_ARMS = ("O0", "O1", "O2", "O3", "O4", "O5", "O6", "O7")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def signed_residual(noisy: np.ndarray, mild: np.ndarray) -> np.ndarray:
    if noisy.shape != mild.shape:
        raise ValueError("noisy and mild geometry differ")
    return np.ascontiguousarray(noisy, dtype=np.float32) - np.ascontiguousarray(
        mild, dtype=np.float32
    )


def audit_residual_identity(noisy: np.ndarray, mild: np.ndarray, residual: np.ndarray,
                            tolerance: float = 1e-7) -> float:
    expected = signed_residual(noisy, mild)
    if residual.shape != expected.shape:
        raise ValueError("residual geometry differs")
    error = float(np.max(np.abs(residual.astype(np.float32) - expected)))
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(f"Residual identity failed: max_abs={error} > {tolerance}")
    return error


def shift_without_wrap(value: np.ndarray, pixels: int) -> np.ndarray:
    if pixels < 0:
        raise ValueError("shift must be non-negative")
    output = np.zeros_like(value)
    if pixels == 0:
        output[...] = value
    elif pixels < value.shape[1]:
        output[:, pixels:] = value[:, :-pixels]
    return output


def block_shuffle(value: np.ndarray, block: int, seed: int, token: str) -> np.ndarray:
    if block <= 0:
        raise ValueError("block must be positive")
    h, w = value.shape
    tiles = []
    shapes = []
    for y in range(0, h, block):
        for x in range(0, w, block):
            tile = value[y:min(y + block, h), x:min(x + block, w)]
            tiles.append(tile.copy())
            shapes.append(tile.shape)
    # Shuffle only within equal-shape classes so non-square/remainder geometry
    # is preserved exactly.
    output = np.empty_like(value)
    rng_seed = int.from_bytes(
        hashlib.sha256(f"{VERSION}|{seed}|{token}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(rng_seed)
    by_shape: dict[tuple[int, int], list[int]] = {}
    for index, shape in enumerate(shapes):
        by_shape.setdefault(shape, []).append(index)
    source_for = list(range(len(tiles)))
    for indices in by_shape.values():
        permuted = rng.permutation(indices)
        for destination, source in zip(indices, permuted):
            source_for[destination] = int(source)
    index = 0
    for y in range(0, h, block):
        for x in range(0, w, block):
            tile = tiles[source_for[index]]
            output[y:y + tile.shape[0], x:x + tile.shape[1]] = tile
            index += 1
    return output


def destroy_spatial_structure(value: np.ndarray, seed: int, token: str) -> np.ndarray:
    rng_seed = int.from_bytes(
        hashlib.sha256(f"{VERSION}|hist|{seed}|{token}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(rng_seed)
    flat = np.asarray(value).reshape(-1)
    return np.ascontiguousarray(flat[rng.permutation(flat.size)].reshape(value.shape))


def deterministic_wrong_guides(rows: list[dict], seed: int, repetitions: int) -> list[dict]:
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    results = []
    ordered = sorted(rows, key=lambda row: (str(row["split"]), str(row["group_id"]), str(row["sample_id"])))
    for row in ordered:
        candidates = [candidate for candidate in ordered
                      if str(candidate["split"]) == str(row["split"])
                      and str(candidate["group_id"]) != str(row["group_id"])]
        if not candidates:
            raise ValueError(f"No wrong-position guide for {row['sample_id']}")
        for repetition in range(repetitions):
            token = f"{VERSION}|T3|{seed}|{repetition}|{row['sample_id']}"
            index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % len(candidates)
            target = candidates[index]
            results.append({
                "sample_id": str(row["sample_id"]), "group_id": str(row["group_id"]),
                "split": str(row["split"]), "repetition": repetition,
                "auxiliary_sample_id": str(target["sample_id"]),
                "auxiliary_group_id": str(target["group_id"]),
            })
    return results


def grouped_four_fold_assignment(table: pd.DataFrame, sealed_groups: Iterable[str],
                                 seed: int = 20260929, expected_groups: int = 16) -> dict:
    required = {"group_id", "split"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(table.columns))}")
    sealed = set(map(str, sealed_groups))
    source = table.copy()
    source["group_id"] = source["group_id"].astype(str)
    if source["group_id"].isin(sealed).any():
        raise ValueError("Sealed test group appears in development manifest")
    if not source["split"].astype(str).isin({"train", "val"}).all():
        raise ValueError("Development manifest contains a non-development split")
    groups = sorted(source["group_id"].unique())
    if len(groups) != expected_groups:
        raise ValueError(f"Expected {expected_groups} development positions, found {len(groups)}")
    features = source.groupby("group_id").agg(
        n_frames=("sample_id", "size"),
        labelled_frames=("vessel_mask_path", lambda values: sum(bool(str(v)) for v in values)),
    ).reset_index()
    # Stable seeded tie breaker followed by greedy load balancing.  Only
    # pre-training row counts/label availability are used.
    features["tie"] = features["group_id"].map(
        lambda value: hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()
    )
    features = features.sort_values(
        ["labelled_frames", "n_frames", "tie"], ascending=[False, False, True], kind="stable"
    )
    fold_groups: list[list[str]] = [[] for _ in range(4)]
    fold_frames = [0] * 4
    fold_labels = [0] * 4
    lookup = features.set_index("group_id")
    for group in features["group_id"]:
        fold = min(range(4), key=lambda i: (len(fold_groups[i]), fold_labels[i], fold_frames[i], i))
        fold_groups[fold].append(str(group))
        fold_frames[fold] += int(lookup.loc[group, "n_frames"])
        fold_labels[fold] += int(lookup.loc[group, "labelled_frames"])
    if sorted(map(len, fold_groups)) != [4, 4, 4, 4]:
        raise AssertionError("Four-fold assignment is not 4/4/4/4")
    validation_counts = Counter(group for fold in fold_groups for group in fold)
    if set(validation_counts) != set(groups) or set(validation_counts.values()) != {1}:
        raise AssertionError("Each development position must validate exactly once")
    assignment = {group: fold for fold, members in enumerate(fold_groups) for group in members}
    result = {
        "version": VERSION, "seed": int(seed), "n_folds": 4,
        "development_group_count": len(groups), "assignment": assignment,
        "folds": {str(i): {"val_groups": sorted(members),
                            "train_groups": sorted(set(groups) - set(members))}
                  for i, members in enumerate(fold_groups)},
        "sealed_test_groups_metadata_only": sorted(sealed),
        "test_assets_opened": 0,
    }
    result["assignment_sha256"] = stable_sha(result)
    return result


def boundary_band(vessel: np.ndarray, radius: int) -> np.ndarray:
    target = vessel.astype(bool)
    if radius <= 0:
        return np.zeros_like(target)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    dilated = cv2.dilate(target.astype(np.uint8), kernel).astype(bool)
    eroded = cv2.erode(target.astype(np.uint8), kernel).astype(bool)
    return dilated ^ eroded


def spatial_alpha_map(layer: np.ndarray, vessel: np.ndarray, valid: np.ndarray,
                      alpha_vessel: float, alpha_boundary: float,
                      alpha_stroma: float, alpha_outside: float,
                      boundary_radius: int = 2) -> tuple[np.ndarray, dict]:
    values = [alpha_vessel, alpha_boundary, alpha_stroma, alpha_outside]
    if any(not 0 <= value <= 1 for value in values):
        raise ValueError("alpha values must lie in [0,1]")
    if alpha_vessel > alpha_stroma or alpha_boundary > alpha_stroma:
        raise ValueError("vessel/boundary alpha cannot exceed stroma alpha")
    layer_b, vessel_b, valid_b = layer.astype(bool), vessel.astype(bool), valid.astype(bool)
    boundary = boundary_band(vessel_b, boundary_radius) & valid_b
    alpha = np.full(layer.shape, np.float32(alpha_outside), dtype=np.float32)
    alpha[layer_b & valid_b] = np.float32(alpha_stroma)
    alpha[boundary] = np.float32(alpha_boundary)
    alpha[vessel_b & valid_b] = np.float32(alpha_vessel)
    alpha[~valid_b] = 0.0
    regions = {
        "vessel_pixels": int((vessel_b & valid_b).sum()),
        "boundary_pixels": int((boundary & ~vessel_b).sum()),
        "stroma_pixels": int((layer_b & ~vessel_b & ~boundary & valid_b).sum()),
        "outside_pixels": int((~layer_b & valid_b).sum()),
    }
    return alpha, regions


def random_histogram_matched_map(alpha: np.ndarray, valid: np.ndarray, seed: int,
                                 token: str) -> np.ndarray:
    valid_b = valid.astype(bool)
    output = np.zeros_like(alpha, dtype=np.float32)
    values = np.asarray(alpha[valid_b], dtype=np.float32)
    rng_seed = int.from_bytes(
        hashlib.sha256(f"{VERSION}|oracle|{seed}|{token}".encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(rng_seed)
    output[valid_b] = values[rng.permutation(values.size)]
    if not np.array_equal(np.sort(output[valid_b]), np.sort(values)):
        raise AssertionError("Random oracle control changed the alpha histogram")
    return output


def apply_spatial_dose(noisy: np.ndarray, strong: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    if noisy.shape != strong.shape or noisy.shape != alpha.shape:
        raise ValueError("Spatial dose geometry differs")
    if np.nanmin(alpha) < 0 or np.nanmax(alpha) > 1:
        raise ValueError("Spatial alpha outside [0,1]")
    residual = noisy.astype(np.float32) - strong.astype(np.float32)
    return np.clip(noisy.astype(np.float32) - alpha.astype(np.float32) * residual, 0, 1)


def oracle_gate(position_results: pd.DataFrame) -> dict:
    required = {"group_id", "arm", "vessel_dice", "vessel_recall",
                "vessel_boundary_band_dice", "difficult_component_recall"}
    missing = required - set(position_results.columns)
    if missing:
        raise ValueError(f"Oracle gate missing columns: {sorted(missing)}")
    pivot = position_results.pivot_table(index="group_id", columns="arm", values=list(required - {"group_id", "arm"}))
    if not {"O1", "O4", "O5", "O6"}.issubset(position_results["arm"].unique()):
        raise ValueError("Oracle gate requires O1/O4/O5/O6")
    dice_o4 = pivot["vessel_dice"]["O4"]
    comparisons = {
        "o4_gt_o1": float((dice_o4 - pivot["vessel_dice"]["O1"]).mean()),
        "o4_gt_o5": float((dice_o4 - pivot["vessel_dice"]["O5"]).mean()),
        "o4_gt_o6": float((dice_o4 - pivot["vessel_dice"]["O6"]).mean()),
        "recall_noninferiority": float((pivot["vessel_recall"]["O4"] - pivot["vessel_recall"]["O1"]).mean()),
        "boundary_noninferiority": float((pivot["vessel_boundary_band_dice"]["O4"] - pivot["vessel_boundary_band_dice"]["O1"]).mean()),
        "difficult_component_gain": float((pivot["difficult_component_recall"]["O4"] - pivot["difficult_component_recall"]["O1"]).mean()),
        "improved_positions": int((dice_o4 > pivot["vessel_dice"]["O1"]).sum()),
        "position_count": int(len(dice_o4)),
    }
    checks = {
        "o4_gt_o1": comparisons["o4_gt_o1"] > 0,
        "o4_gt_o5": comparisons["o4_gt_o5"] > 0,
        "o4_gt_o6": comparisons["o4_gt_o6"] > 0,
        "recall_noninferior": comparisons["recall_noninferiority"] >= -0.01,
        "boundary_noninferior": comparisons["boundary_noninferiority"] >= -0.01,
        "difficult_component_improved": comparisons["difficult_component_gain"] > 0,
        "majority_positions": comparisons["improved_positions"] > comparisons["position_count"] / 2,
    }
    passed = all(checks.values())
    return {
        "version": VERSION, "status": "passed" if passed else "failed",
        "adaptive_training_allowed": passed, "checks": checks,
        "comparisons": comparisons,
        "blocked_message": None if passed else "BLOCKED: NO SPATIAL DENOISING ORACLE UPPER BOUND",
        "test_assets_opened": 0,
    }
