from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import AdapterContext


SVD_UPDATE_CALLS = 0


def _dct_dictionary(patch_size: int, atoms: int) -> np.ndarray:
    n = patch_size
    coords = np.arange(n, dtype=np.float64)
    basis = []
    for u in range(n):
        one = np.cos(np.pi * (2 * coords + 1) * u / (2 * n))
        one *= np.sqrt(1 / n) if u == 0 else np.sqrt(2 / n)
        basis.append(one)
    two_d = [np.outer(basis[u], basis[v]).ravel() for u in range(n) for v in range(n)]
    dictionary = np.stack(two_d, axis=1)
    if atoms > dictionary.shape[1]:
        rng = np.random.default_rng(0)
        extra = rng.standard_normal((n * n, atoms - dictionary.shape[1]))
        dictionary = np.concatenate([dictionary, extra], axis=1)
    return _normalize(dictionary[:, :atoms])


def _normalize(dictionary: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(dictionary, axis=0, keepdims=True)
    return dictionary / np.maximum(norms, 1e-12)


def omp(dictionary: np.ndarray, signals: np.ndarray, max_nonzero: int, residual_threshold: float = 0.0) -> np.ndarray:
    """Deterministic orthogonal matching pursuit; columns are signals."""
    codes = np.zeros((dictionary.shape[1], signals.shape[1]), dtype=np.float64)
    for column in range(signals.shape[1]):
        y = signals[:, column]
        residual = y.copy()
        selected: list[int] = []
        for _ in range(max_nonzero):
            correlations = np.abs(dictionary.T @ residual)
            if selected:
                correlations[selected] = -1
            atom = int(np.argmax(correlations))
            selected.append(atom)
            sub_dictionary = dictionary[:, selected]
            coef, *_ = np.linalg.lstsq(sub_dictionary, y, rcond=None)
            residual = y - sub_dictionary @ coef
            if residual_threshold > 0 and float(residual @ residual) <= residual_threshold**2:
                break
        codes[selected, column] = coef
    return codes


def _ksvd(signals: np.ndarray, dictionary: np.ndarray, iterations: int, sparsity: int, residual_threshold: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    global SVD_UPDATE_CALLS
    dictionary = _normalize(dictionary.astype(np.float64, copy=True))
    codes = np.zeros((dictionary.shape[1], signals.shape[1]), dtype=np.float64)
    for _ in range(iterations):
        codes = omp(dictionary, signals, sparsity, residual_threshold)
        for atom in range(dictionary.shape[1]):
            used = np.flatnonzero(np.abs(codes[atom]) > 1e-12)
            if used.size == 0:
                residual_norms = np.linalg.norm(signals - dictionary @ codes, axis=0)
                replacement = int(np.argmax(residual_norms))
                vector = signals[:, replacement]
                if np.linalg.norm(vector) < 1e-12:
                    vector = rng.standard_normal(signals.shape[0])
                dictionary[:, atom] = vector / max(np.linalg.norm(vector), 1e-12)
                continue
            residual = signals[:, used] - dictionary @ codes[:, used] + np.outer(dictionary[:, atom], codes[atom, used])
            u, singular, vh = np.linalg.svd(residual, full_matrices=False)
            SVD_UPDATE_CALLS += 1
            dictionary[:, atom] = u[:, 0]
            codes[atom, used] = singular[0] * vh[0]
        dictionary = _normalize(dictionary)
    return dictionary, codes


def _positions(length: int, patch: int, stride: int) -> list[int]:
    positions = list(range(0, max(length - patch + 1, 1), stride))
    last = max(length - patch, 0)
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def ksvd_adapter(image: np.ndarray, config: Mapping[str, Any], context: AdapterContext) -> np.ndarray:
    """Single-image test-time K-SVD. The context deliberately has no reference field."""
    patch = int(config.get("patch_size", 7))
    atoms = int(config.get("dictionary_atoms", 64))
    stride = int(config.get("stride", 3))
    iterations = int(config.get("iterations", 5))
    sparsity = int(config.get("omp_max_nonzero", 4))
    threshold = float(config.get("omp_residual_threshold", 0.0))
    noise_weight = float(config.get("noise_weight", 1.0))
    aggregation_weight = float(config.get("aggregation_weight", 1.0))
    max_train = int(config.get("max_training_patches", 4000))
    if patch < 2 or patch > min(image.shape) or stride < 1 or atoms < 1 or sparsity < 1:
        raise ValueError("invalid K-SVD configuration")
    ys, xs = _positions(image.shape[0], patch, stride), _positions(image.shape[1], patch, stride)
    patches = np.stack([image[y:y + patch, x:x + patch].ravel() for y in ys for x in xs], axis=1).astype(np.float64)
    means = patches.mean(axis=0, keepdims=True)
    centered = patches - means
    rng = np.random.default_rng(context.seed)
    if centered.shape[1] > max_train:
        train_indices = np.sort(rng.choice(centered.shape[1], max_train, replace=False))
    else:
        train_indices = np.arange(centered.shape[1])
    dictionary = _dct_dictionary(patch, atoms)
    dictionary, _ = _ksvd(centered[:, train_indices], dictionary, iterations, sparsity, threshold, rng)
    codes = omp(dictionary, centered, sparsity, threshold)
    sparse_centered = dictionary @ codes
    # ``noise_weight`` is the registered denoising strength.  Zero reproduces
    # the noisy patches and one uses the complete sparse reconstruction.
    reconstructed = centered + noise_weight * (sparse_centered - centered) + means
    patch_reliability = 1.0 / (
        1.0 + aggregation_weight * np.mean((patches - reconstructed) ** 2, axis=0)
    )
    output = np.zeros_like(image, dtype=np.float64)
    weights = np.zeros_like(image, dtype=np.float64)
    index = 0
    for y in ys:
        for x in xs:
            weight = float(patch_reliability[index])
            output[y:y + patch, x:x + patch] += weight * reconstructed[:, index].reshape(patch, patch)
            weights[y:y + patch, x:x + patch] += weight
            index += 1
    if np.any(weights <= 0):
        raise RuntimeError("K-SVD patch reconstruction left uncovered pixels")
    return output / weights
