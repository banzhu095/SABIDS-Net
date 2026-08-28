from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


PATH_COLUMNS = [
    "image_path", "clean_path", "layer_mask_path", "vessel_mask_path",
    "label_valid_mask_path", "vessel_valid_mask_path", "multiclass_label_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a SABIDS CSV manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--check-files", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest).resolve()
    root = Path(args.root).resolve() if args.root else manifest.parent
    table = pd.read_csv(manifest, dtype=str).fillna("")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    required = {"sample_id", "group_id", "dataset", "split", "image_path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    duplicates = table["sample_id"].duplicated().sum()
    if duplicates:
        raise ValueError(f"Found {duplicates} duplicated sample_id values")
    split_counts = table.groupby("split")["group_id"].nunique()
    group_splits = table.groupby("group_id")["split"].nunique()
    leaked = group_splits[group_splits > 1]
    if not leaked.empty:
        examples = leaked.index.tolist()[:10]
        raise ValueError(f"Group leakage across splits: {examples}")
    if "patient_id" in table.columns:
        patient_splits = table[table["patient_id"] != ""].groupby("patient_id")["split"].nunique()
        patient_leak = patient_splits[patient_splits > 1]
        if not patient_leak.empty:
            raise ValueError(f"Patient leakage across splits: {patient_leak.index.tolist()[:10]}")
    if args.check_files:
        missing_files = []
        for _, row in table.iterrows():
            for column in PATH_COLUMNS:
                value = row.get(column, "")
                if not value:
                    continue
                path = Path(value)
                if not path.is_absolute():
                    path = root / path
                if not path.is_file():
                    missing_files.append((row["sample_id"], column, str(path)))
        if missing_files:
            preview = "\n".join(map(str, missing_files[:20]))
            raise FileNotFoundError(f"Missing {len(missing_files)} files:\n{preview}")
    print(f"Rows: {len(table)}")
    print(f"Manifest SHA256: {digest}")
    print(f"Groups: {table['group_id'].nunique()}")
    print(f"Datasets: {table['dataset'].value_counts().to_dict()}")
    print(f"Groups per split: {split_counts.to_dict()}")
    print(f"Rows per split: {table['split'].value_counts().to_dict()}")
    for split, part in table.groupby("split"):
        group_ids = sorted(part["group_id"].astype(str).unique().tolist())
        vessel_groups = (
            sorted(
                part.loc[part.get("vessel_mask_path", "") != "", "group_id"]
                .astype(str)
                .unique()
                .tolist()
            )
            if "vessel_mask_path" in part.columns
            else []
        )
        print(
            f"{split} group IDs ({len(group_ids)}): {group_ids}; "
            f"vessel-labelled groups ({len(vessel_groups)}): {vessel_groups}"
        )
    if "patient_id" in table.columns:
        patients = table[table["patient_id"] != ""].groupby("split")[
            "patient_id"
        ].nunique()
        print(f"Patients per split: {patients.to_dict()}")
    for column, label_name in (
        ("layer_mask_path", "Layer-labelled rows"),
        ("vessel_mask_path", "Vessel-labelled rows"),
    ):
        if column in table.columns:
            labelled = table[table[column] != ""].groupby("split").size()
            print(f"{label_name} per split: {labelled.to_dict()}")
    if "disease" in table.columns:
        disease_counts = (
            table.groupby(["split", "disease"]).size().unstack(fill_value=0)
        )
        print("Disease rows per split:")
        print(disease_counts.to_string())
    print("Manifest validation passed.")


if __name__ == "__main__":
    main()
