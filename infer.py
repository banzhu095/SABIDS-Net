from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from sabids.config import load_config
from sabids.data.io import read_gray, write_gray
from sabids.engine.trainer import build_model
from sabids.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SABIDS-Net inference")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--threshold", type=float, default=None, help="Fallback for both heads"
    )
    parser.add_argument("--layer-threshold", type=float, default=None)
    parser.add_argument("--vessel-threshold", type=float, default=None)
    parser.add_argument("--use-ema", action="store_true")
    return parser.parse_args()


def iter_images(path: Path) -> Iterable[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    if path.is_file():
        yield path
    else:
        for candidate in sorted(path.rglob("*")):
            if candidate.suffix.lower() in extensions:
                yield candidate


def normalize(image: np.ndarray, config: dict) -> np.ndarray:
    mode = config["data"].get("normalization", "fixed")
    if mode == "fixed":
        return np.clip(image, 0.0, 1.0)
    if mode == "percentile":
        low = float(config["data"].get("percentile_low", 0.5))
        high = float(config["data"].get("percentile_high", 99.5))
        lo, hi = np.percentile(image, [low, high])
        return np.clip((image - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    mean, std = float(image.mean()), float(image.std()) + 1e-6
    return np.clip(((image - mean) / std + 3.0) / 6.0, 0.0, 1.0)


def save_overlay(path: Path, image: np.ndarray, layer: np.ndarray, vessel: np.ndarray) -> None:
    base = np.round(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
    color = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    layer_edge = cv2.morphologyEx(layer.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    color[layer_edge > 0] = (0, 255, 0)
    color[vessel > 0] = (0, 0, 255)
    success, encoded = cv2.imencode(".png", color)
    if not success:
        raise RuntimeError(f"Unable to encode overlay: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(str(path))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("ema") if args.use_ema and checkpoint.get("ema") else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    model.eval()
    evaluation = config.get("evaluation", {})
    default_threshold = (
        args.threshold
        if args.threshold is not None
        else float(evaluation.get("threshold", 0.5))
    )
    layer_threshold = (
        args.layer_threshold
        if args.layer_threshold is not None
        else float(evaluation.get("layer_threshold", default_threshold))
    )
    vessel_threshold = (
        args.vessel_threshold
        if args.vessel_threshold is not None
        else float(evaluation.get("vessel_threshold", default_threshold))
    )
    input_path = Path(args.input)
    output_dir = Path(args.output)
    divisor = 2 ** (len(config["model"].get("channels", [32, 64, 128, 256])) - 1)

    with torch.no_grad():
        for image_path in iter_images(input_path):
            image = normalize(read_gray(image_path), config)
            height, width = image.shape
            tensor = torch.from_numpy(image[None, None]).float().to(device)
            pad_h = (divisor - height % divisor) % divisor
            pad_w = (divisor - width % divisor) % divisor
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
            output = model(
                tensor, return_features=False, return_auxiliary=False
            )
            denoised = output["denoised"][0, 0, :height, :width].cpu().numpy()
            layer_prob = output["layer_prob"][0, 0, :height, :width].cpu().numpy()
            vessel_prob = output["vessel_prob"][0, 0, :height, :width].cpu().numpy()
            layer = layer_prob >= layer_threshold
            vessel = vessel_prob >= vessel_threshold
            stem = image_path.stem
            case_dir = output_dir / stem
            write_gray(case_dir / f"{stem}_denoised.png", denoised)
            write_gray(case_dir / f"{stem}_layer_probability.png", layer_prob)
            write_gray(case_dir / f"{stem}_vessel_probability.png", vessel_prob)
            write_gray(case_dir / f"{stem}_layer_mask.png", layer.astype(np.float32))
            write_gray(case_dir / f"{stem}_vessel_mask.png", vessel.astype(np.float32))
            save_overlay(case_dir / f"{stem}_overlay.png", denoised, layer, vessel)
            print(f"Processed: {image_path}")


if __name__ == "__main__":
    main()
