from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from sabids.data.io import read_gray, write_gray


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build synthetic pairs from unpaired 51-line noise and HD OCT images"
    )
    parser.add_argument("--noisy-dir", required=True)
    parser.add_argument("--clean-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--noise-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-name", default="Private2Synthetic")
    return parser.parse_args()


def list_images(folder: Path):
    return [path for path in sorted(folder.rglob("*")) if path.suffix.lower() in EXTENSIONS]


def resize_like(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return cv2.resize(image, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_LINEAR)


def estimate_noise(image: np.ndarray) -> np.ndarray:
    """Estimate a conservative high-frequency/self-noise component."""
    low_frequency = cv2.bilateralFilter(image.astype(np.float32), 7, 0.08, 5.0)
    residual = image - low_frequency
    residual -= np.median(residual)
    scale = np.percentile(np.abs(residual), 99.0)
    if scale > 1e-6:
        residual = np.clip(residual / scale, -1.0, 1.0)
    return residual.astype(np.float32)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    noisy_files = list_images(Path(args.noisy_dir))
    clean_files = list_images(Path(args.clean_dir))
    if not noisy_files or not clean_files:
        raise RuntimeError("Both noisy-dir and clean-dir must contain images")
    output = Path(args.output_dir)
    noisy_output = output / "synthetic_noisy"
    clean_output = output / "clean_targets"
    rows = []
    for index in range(args.samples):
        noise_source_path = random.choice(noisy_files)
        clean_source_path = random.choice(clean_files)
        noise_source = read_gray(noise_source_path)
        clean = read_gray(clean_source_path)
        residual = resize_like(estimate_noise(noise_source), clean)
        local_std = float(np.std(noise_source - cv2.GaussianBlur(noise_source, (5, 5), 0)))
        amplitude = args.noise_strength * max(local_std, 0.01)
        synthetic = np.clip(clean + amplitude * residual, 0.0, 1.0)
        sample_id = f"syn_{index:05d}"
        noisy_path = noisy_output / f"{sample_id}.png"
        clean_path = clean_output / f"{sample_id}.png"
        write_gray(noisy_path, synthetic)
        write_gray(clean_path, clean)
        rows.append(
            {
                "sample_id": sample_id,
                "group_id": sample_id,
                "patient_id": sample_id,
                "dataset": args.dataset_name,
                "domain": "private_synthetic",
                "scan_protocol": "51-line-noise-to-HD",
                "frame_index": 0,
                "split": "train",
                "image_path": str(noisy_path.resolve()),
                "clean_path": str(clean_path.resolve()),
                "layer_mask_path": "",
                "vessel_mask_path": "",
                "is_clean": 0,
                "noise_source": str(noise_source_path.resolve()),
                "clean_source": str(clean_source_path.resolve()),
            }
        )
        if (index + 1) % 50 == 0:
            print(f"Generated {index + 1}/{args.samples}")
    manifest = output / "manifest_synthetic_pairs.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False, encoding="utf-8-sig")
    print(f"Saved: {manifest}")


if __name__ == "__main__":
    main()

