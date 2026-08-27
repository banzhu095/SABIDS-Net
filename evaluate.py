from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.engine.evaluator import evaluate_model
from sabids.engine.trainer import _make_transform, build_model
from sabids.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-level SABIDS-Net evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--one-frame-per-group",
        action="store_true",
        help="Use the first manifest row of every group for deterministic diagnostics.",
    )
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--layer-threshold", type=float, default=None)
    parser.add_argument("--vessel-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state = checkpoint.get("ema") if args.use_ema and checkpoint.get("ema") else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    dataset = OCTManifestDataset(
        config["data"]["manifest"],
        split=args.split,
        transform=_make_transform(config, False),
        sample_repeat=False,
        root=config["data"].get("root"),
        datasets=config["data"].get(f"{args.split}_datasets"),
        groups=config["data"].get(f"{args.split}_groups"),
    )
    if args.one_frame_per_group:
        indices = [dataset.groups[group_id][0] for group_id in sorted(dataset.groups)]
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("evaluation", {}).get("num_workers", 2)),
        pin_memory=True,
    )
    evaluation = config.get("evaluation", {})
    default_threshold = float(evaluation.get("threshold", 0.5))
    summary = evaluate_model(
        model,
        loader,
        device,
        output_dir=Path(args.output),
        threshold=default_threshold,
        layer_threshold=(
            args.layer_threshold
            if args.layer_threshold is not None
            else float(evaluation.get("layer_threshold", default_threshold))
        ),
        vessel_threshold=(
            args.vessel_threshold
            if args.vessel_threshold is not None
            else float(evaluation.get("vessel_threshold", default_threshold))
        ),
        axial_spacing=float(evaluation.get("axial_spacing", 1.0)),
        lateral_spacing=float(evaluation.get("lateral_spacing", 1.0)),
        save_predictions=args.save_predictions,
        stage=str(config.get("train", {}).get("stage", "joint")),
    )
    print(summary)


if __name__ == "__main__":
    main()
