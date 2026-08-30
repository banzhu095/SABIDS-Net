from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label as connected_components

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.data.io import read_mask
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
    manifest = Path(data["manifest"]).expanduser().resolve()
    root = Path(data.get("root") or manifest.parent).expanduser().resolve()
    table = pd.read_csv(manifest, dtype=str).fillna("")
    table = table[table["split"] == str(data.get("train_split", "train"))].copy()
    if data.get("train_datasets"):
        table = table[table["dataset"].isin(data["train_datasets"])]
    if data.get("train_groups"):
        table = table[table["group_id"].isin([str(value) for value in data["train_groups"]])]

    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else (root / path).resolve()

    areas = []
    sampled_groups = []
    if "vessel_mask_path" not in table.columns:
        raise ValueError("Manifest has no vessel_mask_path column")
    for group_id, group in table.groupby("group_id", sort=True):
        labelled_rows = group[group["vessel_mask_path"].astype(str) != ""]
        if labelled_rows.empty:
            continue
        row = labelled_rows.sort_values("sample_id").iloc[0]
        target = read_mask(resolve(str(row["vessel_mask_path"]))) > 0.5
        valid_path = str(row.get("vessel_valid_mask_path", "")).strip()
        valid = read_mask(resolve(valid_path)) > 0.5 if valid_path else np.ones_like(target, dtype=bool)
        if target.shape != valid.shape:
            raise ValueError(f"Vessel/valid shape mismatch for training group {group_id}")
        target &= valid
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
            "Connected-component pixel area in the original stored label grid, "
            "using the first manifest frame per labelled training group."
        ),
        "component_size_thresholds": [small_max, medium_max],
        "n_components": int(len(values)),
        "n_groups": int(len(sampled_groups)),
        "group_ids": sampled_groups,
        "area_min": int(values.min()),
        "area_median": float(np.median(values)),
        "area_max": int(values.max()),
        "coordinate_system": "original_label_pixels",
        "target_size_not_used": data.get("target_size"),
    }
    write_json(result, Path(args.output))
    print(result)


if __name__ == "__main__":
    main()
