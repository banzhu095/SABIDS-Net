from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label as connected_components

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.engine.trainer import _make_transform
from sabids.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Derive small/medium/large vessel-component area bins from one "
            "deterministic labelled frame per training group."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data = config["data"]
    dataset = OCTManifestDataset(
        data["manifest"],
        split=data.get("train_split", "train"),
        transform=_make_transform(config, training=False),
        sample_repeat=False,
        root=data.get("root"),
        datasets=data.get("train_datasets"),
        groups=data.get("train_groups"),
    )
    areas = []
    sampled_groups = []
    for group_id in sorted(dataset.groups):
        sample = dataset[dataset.groups[group_id][0]]
        if not bool(sample["has_vessel"]):
            continue
        valid = (
            sample["valid_mask"][0].numpy() > 0.5
        ) & (sample["vessel_valid_mask"][0].numpy() > 0.5)
        target = (sample["vessel_mask"][0].numpy() > 0.5) & valid
        labelled, count = connected_components(target)
        areas.extend(
            int((labelled == component_id).sum())
            for component_id in range(1, count + 1)
        )
        sampled_groups.append(str(group_id))
    if not areas:
        raise RuntimeError("No labelled vessel components found in training groups")
    values = np.asarray(areas, dtype=np.int64)
    small_max, medium_max = [
        max(1, int(round(value)))
        for value in np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
    ]
    medium_max = max(medium_max, small_max + 1)
    result = {
        "definition": (
            "Connected-component pixel area after the resolved training resize/pad, "
            "using the first manifest frame per labelled training group."
        ),
        "component_size_thresholds": [small_max, medium_max],
        "n_components": int(len(values)),
        "n_groups": int(len(sampled_groups)),
        "group_ids": sampled_groups,
        "area_min": int(values.min()),
        "area_median": float(np.median(values)),
        "area_max": int(values.max()),
        "target_size": data.get("target_size"),
    }
    write_json(result, Path(args.output))
    print(result)


if __name__ == "__main__":
    main()
