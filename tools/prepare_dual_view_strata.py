#!/usr/bin/env python
"""Freeze train-derived thresholds and validation component membership."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.engine.trainer import _make_transform
from sabids.experiments.dual_view import build_fixed_component_inventory
from sabids.experiments.dose_response import write_strict_json_exclusive


def _samples(config: dict, split: str) -> list[dict]:
    data = config["data"]
    dataset = OCTManifestDataset(
        data["manifest"], split=split, transform=_make_transform(config, False),
        sample_repeat=False, root=data.get("root"),
        datasets=data.get(f"{split}_datasets"), groups=data.get(f"{split}_groups"),
        image_column=data.get("input_column", "primary_path"),
        load_segmentation_labels=True,
        pretransformed_model_grid=bool(data.get("pretransformed_model_grid", False)),
    )
    result = []
    for index in range(len(dataset)):
        item = dataset[index]
        if not bool(item["has_layer"]) or not bool(item["has_vessel"]):
            continue
        valid = (
            (item["valid_mask"][0].numpy() > 0.5)
            & (item["label_valid_mask"][0].numpy() > 0.5)
            & (item["vessel_valid_mask"][0].numpy() > 0.5)
        )
        result.append({
            "sample_id": item["sample_id"], "group_id": item["group_id"], "split": split,
            "vessel": item["vessel_mask"][0].numpy() > 0.5,
            "layer": item["layer_mask"][0].numpy() > 0.5,
            "valid": valid,
            "noisy": item["image"][0].numpy(),
            "clean": item["clean"][0].numpy() if bool(item["has_clean"]) else None,
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True, help="Prepared B0 config.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ring-width", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    config = load_config(config_path)
    inventory = build_fixed_component_inventory(
        _samples(config, config["data"].get("train_split", "train")),
        _samples(config, config["data"].get("val_split", "val")),
        args.ring_width,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = (root / output).resolve()
    write_strict_json_exclusive(output, inventory)
    print(json.dumps(inventory, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
