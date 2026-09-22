#!/usr/bin/env python
"""Freeze vessel size/contrast strata from development-train labels only."""
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
from sabids.experiments.d2 import derive_vessel_strata
from sabids.experiments.dose_response import stable_sha, write_strict_json_exclusive
from sabids.experiments.protocol_lock import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fixed-area-thresholds", nargs=2, type=int)
    parser.add_argument("--ring-width", type=int, default=3)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    config = load_config(config_path)
    data = config["data"]
    manifest = Path(data["manifest"])
    if not manifest.is_absolute():
        manifest = (root / manifest).resolve()
    data_root = Path(data.get("root") or root)
    if not data_root.is_absolute():
        data_root = (root / data_root).resolve()
    dataset = OCTManifestDataset(
        manifest,
        split=data.get("train_split", "train"),
        transform=_make_transform(config, False),
        sample_repeat=False,
        root=data_root,
        datasets=data.get("train_datasets"),
        groups=data.get("train_groups"),
        load_segmentation_labels=True,
    )
    samples = []
    for index in range(len(dataset)):
        item = dataset[index]
        if not bool(item["has_layer"]) or not bool(item["has_vessel"]):
            continue
        samples.append({
            "split": "train",
            "vessel": item["vessel_mask"][0].numpy() > 0.5,
            "layer": item["layer_mask"][0].numpy() > 0.5,
            "valid": (
                (item["valid_mask"][0].numpy() > 0.5)
                & (item["label_valid_mask"][0].numpy() > 0.5)
                & (item["vessel_valid_mask"][0].numpy() > 0.5)
            ),
            "noisy": item["image"][0].numpy(),
            "clean": item["clean"][0].numpy() if bool(item["has_clean"]) else None,
        })
    definition = derive_vessel_strata(
        samples,
        tuple(args.fixed_area_thresholds) if args.fixed_area_thresholds else None,
        args.ring_width,
    )
    label_records = []
    for row in dataset.table.sort_values("sample_id", kind="stable").to_dict("records"):
        for column in ("layer_mask_path", "vessel_mask_path"):
            value = str(row.get(column, "")).strip()
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (data_root / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing train label asset: {path}")
            label_records.append({
                "sample_id": str(row["sample_id"]), "split": "train",
                "column": column, "sha256": sha256_file(path),
            })
    definition.update({
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "selected_split": "train",
        "selected_group_count": int(dataset.table["group_id"].nunique()),
        "selected_frame_count": int(len(dataset)),
        "training_label_asset_records": label_records,
        "training_label_asset_sha256": stable_sha(label_records),
        "validation_assets_opened": 0,
        "test_assets_opened": 0,
    })
    # Recompute after adding provenance so the SHA covers the full frozen definition.
    definition.pop("definition_sha256", None)
    definition["definition_sha256"] = stable_sha(definition)
    output = Path(args.output)
    if not output.is_absolute():
        output = (root / output).resolve()
    write_strict_json_exclusive(output, definition)
    print(json.dumps(definition, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
