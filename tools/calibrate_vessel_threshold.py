from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.engine.trainer import _make_transform, build_model
from sabids.metrics import binary_metrics, vessel_area_fraction
from sabids.utils import get_device, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the vessel probability threshold on a validation split. "
            "Never use the test split for threshold selection."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum", type=float, default=0.30)
    parser.add_argument("--maximum", type=float, default=0.85)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument(
        "--prediction-mode", choices=["raw", "soft_gate"], default="raw"
    )
    parser.add_argument("--one-frame-per-group", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    prediction_mode: str = "raw",
) -> List[Dict[str, object]]:
    model.eval()
    samples: List[Dict[str, object]] = []
    for batch in tqdm(loader, desc="Collect validation probabilities"):
        output = model(
            batch["image"].to(device, non_blocking=True),
            return_features=False,
            return_auxiliary=False,
        )
        probability_tensor = output["vessel_prob"]
        if prediction_mode == "soft_gate":
            probability_tensor = probability_tensor * output["layer_prob"]
        probability = probability_tensor.cpu().numpy()
        for index in range(probability.shape[0]):
            if not bool(batch["has_vessel"][index]):
                continue
            samples.append(
                {
                    "group_id": str(batch["group_id"][index]),
                    "probability": probability[index, 0],
                    "target": batch["vessel_mask"][index, 0].numpy() > 0.5,
                    "layer": batch["layer_mask"][index, 0].numpy() > 0.5,
                    "valid": (
                        batch["valid_mask"][index, 0].numpy() > 0.5
                    ) & (
                        batch["vessel_valid_mask"][index, 0].numpy() > 0.5
                    ),
                }
            )
    if not samples:
        raise RuntimeError("The selected split contains no manual vessel labels")
    return samples


def score_threshold(
    samples: List[Dict[str, object]], threshold: float
) -> Dict[str, float]:
    frame_rows = []
    for sample in samples:
        valid = sample["valid"]
        prediction = sample["probability"][valid] >= threshold
        target = sample["target"][valid]
        layer = sample["layer"][valid]
        metrics = binary_metrics(prediction, target)
        predicted_fraction = vessel_area_fraction(prediction, layer)
        true_fraction = vessel_area_fraction(target, layer)
        frame_rows.append(
            {
                "group_id": sample["group_id"],
                "dice": metrics["dice"],
                "iou": metrics["iou"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "area_fraction_mae": abs(predicted_fraction - true_fraction),
            }
        )
    table = pd.DataFrame(frame_rows)
    group_table = table.groupby("group_id", as_index=False).mean(numeric_only=True)
    return {
        "threshold": float(threshold),
        **{
            key: float(group_table[key].mean())
            for key in group_table.columns
            if key != "group_id"
        },
        "n_groups": int(group_table["group_id"].nunique()),
        "n_frames": int(len(table)),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 < args.minimum < args.maximum < 1.0:
        raise ValueError("Require 0 < minimum < maximum < 1")
    if args.step <= 0:
        raise ValueError("--step must be positive")

    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = (
        checkpoint.get("ema")
        if args.use_ema and checkpoint.get("ema") is not None
        else checkpoint["model"]
    )
    model.load_state_dict(state, strict=True)

    dataset = OCTManifestDataset(
        config["data"]["manifest"],
        split=args.split,
        transform=_make_transform(config, training=False),
        sample_repeat=False,
        root=config["data"].get("root"),
        datasets=config["data"].get(f"{args.split}_datasets"),
        groups=config["data"].get(f"{args.split}_groups"),
    )
    if args.one_frame_per_group:
        indices = [dataset.groups[group_id][0] for group_id in sorted(dataset.groups)]
        dataset = Subset(dataset, indices)
    workers = (
        args.num_workers
        if args.num_workers is not None
        else int(config.get("evaluation", {}).get("num_workers", 2))
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )
    samples = collect_predictions(
        model, loader, device, prediction_mode=args.prediction_mode
    )
    thresholds = np.arange(
        args.minimum, args.maximum + args.step * 0.5, args.step
    )
    rows = [score_threshold(samples, float(value)) for value in thresholds]
    # Dice is primary; area error and precision break practically relevant ties.
    best = max(
        rows,
        key=lambda row: (
            row["dice"],
            -row["area_fraction_mae"],
            row["precision"],
        ),
    )
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        output_dir / f"vessel_threshold_sweep_{args.prediction_mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    result = {
        "selection_split": args.split,
        "prediction_mode": args.prediction_mode,
        "one_frame_per_group": bool(args.one_frame_per_group),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpoint_sha256": file_sha256(
            Path(args.checkpoint).expanduser().resolve()
        ),
        "manifest_sha256": file_sha256(
            Path(config["data"]["manifest"]).expanduser().resolve()
        ),
        "best": best,
        "rule": "maximum group-mean Dice; ties use lower area MAE then higher precision",
    }
    write_json(
        result,
        output_dir / f"best_vessel_threshold_{args.prediction_mode}.json",
    )
    print(f"Best validation vessel threshold: {best['threshold']:.3f}")
    print(
        "Group Dice={dice:.5f} | precision={precision:.5f} | "
        "recall={recall:.5f} | area MAE={area_fraction_mae:.5f}".format(**best)
    )
    print("Use this value only for the untouched test split.")


if __name__ == "__main__":
    main()
