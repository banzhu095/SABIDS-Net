from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_PROJECT_ROOT = r"E:\1-脉络膜\OCT降噪\SABIDS-Net"
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
MANIFEST_COLUMNS = [
    "sample_id",
    "group_id",
    "patient_id",
    "dataset",
    "domain",
    "scan_protocol",
    "frame_index",
    "split",
    "image_path",
    "clean_path",
    "layer_mask_path",
    "vessel_mask_path",
    "multiclass_label_path",
    "has_manual_label",
    "is_clean",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Duke17, Duke28 and PKU37 manifests for SABIDS-Net"
    )
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", default="Manifests")
    parser.add_argument("--layer-class", type=int, default=1)
    parser.add_argument("--vessel-class", type=int, default=2)
    parser.add_argument("--ignore-class", type=int, default=255)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-masks", action="store_true")
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--allow-other-classes", action="store_true")
    return parser.parse_args()


def image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    return [
        path
        for path in sorted(folder.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def clean_index(clean_dir: Path) -> Dict[str, Path]:
    return {path.stem.casefold(): path for path in image_files(clean_dir)}


def make_row(
    *,
    sample_id: str,
    group_id: str,
    dataset: str,
    protocol: str,
    frame_index: int,
    noisy_path: Path,
    clean_path: Path,
    root: Path,
) -> Dict[str, object]:
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "patient_id": group_id,
        "dataset": dataset,
        "domain": "public",
        "scan_protocol": protocol,
        "frame_index": frame_index,
        "split": "",
        "image_path": relative(noisy_path, root),
        "clean_path": relative(clean_path, root),
        "layer_mask_path": "",
        "vessel_mask_path": "",
        "multiclass_label_path": "",
        "has_manual_label": 0,
        "is_clean": 0,
    }


def scan_duke17(root: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    dataset_dir = root / "Data" / "Duke17"
    noisy_dir = dataset_dir / "noisy"
    clean_dir = dataset_dir / "clean"
    clean = clean_index(clean_dir)
    rows, unmatched = [], []
    used_clean = set()
    for noisy in image_files(noisy_dir):
        match = re.fullmatch(r"(.+?)_Raw Image", noisy.stem, flags=re.IGNORECASE)
        if not match:
            unmatched.append(f"Duke17 noisy filename does not match '*_Raw Image': {noisy.name}")
            continue
        identifier = match.group(1)
        clean_stem = f"{identifier}_Averaged Image".casefold()
        target = clean.get(clean_stem)
        if target is None:
            unmatched.append(f"Duke17 clean pair missing for {noisy.name}")
            continue
        used_clean.add(target.resolve())
        safe_id = re.sub(r"[^0-9A-Za-z]+", "_", identifier).strip("_").lower()
        rows.append(
            make_row(
                sample_id=f"duke17_{safe_id}",
                group_id=f"duke17_{safe_id}",
                dataset="Duke17",
                protocol="SD-OCT",
                frame_index=0,
                noisy_path=noisy,
                clean_path=target,
                root=root,
            )
        )
    for target in clean.values():
        if target.resolve() not in used_clean:
            unmatched.append(f"Duke17 clean image has no noisy pair: {target.name}")
    return rows, unmatched


def duke28_clean_stem(noisy_stem: str) -> str:
    lower = noisy_stem.casefold()
    if lower.startswith("ll"):
        return "HH" + noisy_stem[2:]
    if lower.startswith("sl"):
        return "sh" + noisy_stem[2:]
    return noisy_stem


def scan_duke28(root: Path) -> Tuple[List[Dict[str, object]], List[str]]:
    dataset_dir = root / "Data" / "Duke28"
    noisy_dir = dataset_dir / "noisy"
    clean_dir = dataset_dir / "clean"
    clean = clean_index(clean_dir)
    rows, unmatched = [], []
    used_clean = set()
    for index, noisy in enumerate(image_files(noisy_dir)):
        expected = duke28_clean_stem(noisy.stem)
        target = clean.get(expected.casefold())
        if target is None:
            unmatched.append(
                f"Duke28 clean pair missing: noisy={noisy.name}, expected={expected}{noisy.suffix}"
            )
            continue
        used_clean.add(target.resolve())
        safe_id = re.sub(r"[^0-9A-Za-z]+", "_", noisy.stem).strip("_").lower()
        rows.append(
            make_row(
                sample_id=f"duke28_{safe_id}",
                group_id=f"duke28_{safe_id}",
                dataset="Duke28",
                protocol="SD-OCT",
                frame_index=index,
                noisy_path=noisy,
                clean_path=target,
                root=root,
            )
        )
    for target in clean.values():
        if target.resolve() not in used_clean:
            unmatched.append(f"Duke28 clean image has no noisy pair: {target.name}")
    return rows, unmatched


def split_multiclass_labels(
    root: Path,
    layer_class: int,
    vessel_class: int,
    ignore_class: int,
    overwrite: bool,
    allow_other_classes: bool,
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, object]]:
    source_dir = root / "Label" / "voc_seg"
    reference_dir = root / "Label" / "voc_jpg"
    layer_dir = root / "Label" / "layer_binary"
    vessel_dir = root / "Label" / "vessel_binary"
    layer_dir.mkdir(parents=True, exist_ok=True)
    vessel_dir.mkdir(parents=True, exist_ok=True)
    labels: Dict[str, Dict[str, str]] = {}
    value_report: Dict[str, List[int]] = {}
    area_report: Dict[str, Dict[str, float]] = {}
    warnings: List[str] = []
    allowed = {0, layer_class, vessel_class, ignore_class}

    for label_path in image_files(source_dir):
        position_id = label_path.stem
        if not re.fullmatch(r"\d{4}", position_id):
            warnings.append(f"Skipped non-PKU label filename: {label_path.name}")
            continue
        with Image.open(label_path) as image:
            label = np.asarray(image)
        if label.ndim != 2:
            raise ValueError(
                f"Label must be an indexed/grayscale PNG, but {label_path} has shape {label.shape}. "
                "Please export class-index masks rather than RGB visualization masks."
            )
        unique = sorted(int(value) for value in np.unique(label))
        value_report[position_id] = unique
        unknown = set(unique) - allowed
        if unknown and not allow_other_classes:
            raise ValueError(
                f"Unexpected label values in {label_path.name}: {sorted(unknown)}; "
                f"expected background=0, layer={layer_class}, vessel={vessel_class}, ignore={ignore_class}."
            )
        layer_mask = np.logical_or(label == layer_class, label == vessel_class)
        vessel_mask = label == vessel_class
        layer_pixels = int(layer_mask.sum())
        vessel_pixels = int(vessel_mask.sum())
        area_report[position_id] = {
            "layer_pixels": layer_pixels,
            "vessel_pixels": vessel_pixels,
            "vessel_fraction_of_layer": (
                float(vessel_pixels / layer_pixels) if layer_pixels else float("nan")
            ),
        }
        layer_path = layer_dir / f"{position_id}.png"
        vessel_path = vessel_dir / f"{position_id}.png"
        if overwrite or not layer_path.exists():
            Image.fromarray((layer_mask * 255).astype(np.uint8), mode="L").save(layer_path)
        if overwrite or not vessel_path.exists():
            Image.fromarray((vessel_mask * 255).astype(np.uint8), mode="L").save(vessel_path)

        reference_path = reference_dir / f"{position_id}.jpg"
        if not reference_path.exists():
            warnings.append(f"Label has no matching voc_jpg reference: {position_id}")
        labels[position_id] = {
            "layer_mask_path": relative(layer_path, root),
            "vessel_mask_path": relative(vessel_path, root),
            "multiclass_label_path": relative(label_path, root),
        }
    report = {
        "label_positions": sorted(labels),
        "n_label_positions": len(labels),
        "unique_values_by_label": value_report,
        "area_statistics_by_label": area_report,
        "warnings": warnings,
    }
    return labels, report


def scan_pku37(
    root: Path,
    labels: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, object]], List[str], Dict[str, int]]:
    dataset_dir = root / "Data" / "PKU37_OCT_Denoising"
    clean_dir = dataset_dir / "clean"
    noisy_dir = dataset_dir / "noisy"
    clean = {
        path.stem: path
        for path in image_files(clean_dir)
        if re.fullmatch(r"\d{4}", path.stem)
    }
    rows, unmatched = [], []
    frames_per_position: Counter[str] = Counter()
    validated_label_sizes = set()
    for noisy in image_files(noisy_dir):
        match = re.fullmatch(r"(\d{4})(\d{2})", noisy.stem)
        if not match:
            unmatched.append(f"PKU noisy filename is not six digits: {noisy.name}")
            continue
        position_id, frame_id = match.groups()
        target = clean.get(position_id)
        if target is None:
            unmatched.append(f"PKU clean image missing for noisy {noisy.name}")
            continue
        group_id = f"pku_{position_id}"
        row = make_row(
            sample_id=f"{group_id}_f{frame_id}",
            group_id=group_id,
            dataset="PKU37",
            protocol="SD-OCT-repeat",
            frame_index=int(frame_id),
            noisy_path=noisy,
            clean_path=target,
            root=root,
        )
        if position_id in labels:
            row.update(labels[position_id])
            row["has_manual_label"] = 1
            if position_id not in validated_label_sizes:
                with Image.open(target) as clean_image:
                    clean_size = clean_image.size
                with Image.open(root / labels[position_id]["layer_mask_path"]) as layer_image:
                    layer_size = layer_image.size
                with Image.open(root / labels[position_id]["vessel_mask_path"]) as vessel_image:
                    vessel_size = vessel_image.size
                if clean_size != layer_size or clean_size != vessel_size:
                    raise ValueError(
                        f"Size mismatch for PKU position {position_id}: "
                        f"clean={clean_size}, layer={layer_size}, vessel={vessel_size}"
                    )
                validated_label_sizes.add(position_id)
        rows.append(row)
        frames_per_position[position_id] += 1

    for position_id in clean:
        if frames_per_position[position_id] == 0:
            unmatched.append(f"PKU clean image has no noisy frames: {position_id}.tif")
    for position_id in labels:
        if position_id not in clean:
            unmatched.append(f"Manual label has no PKU clean image: {position_id}")
    return rows, unmatched, dict(sorted(frames_per_position.items()))


def group_split(
    group_ids: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, str]:
    groups = np.array(sorted(set(group_ids)), dtype=object)
    if len(groups) < 3:
        raise ValueError("At least three groups are required for train/val/test splitting")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_total = len(groups)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_test = max(1, int(round(n_total * (1.0 - train_ratio - val_ratio))))
    if n_val + n_test >= n_total:
        n_val, n_test = 1, 1
    n_train = n_total - n_val - n_test
    assignment = {}
    for group in groups[:n_train]:
        assignment[str(group)] = "train"
    for group in groups[n_train : n_train + n_val]:
        assignment[str(group)] = "val"
    for group in groups[n_train + n_val :]:
        assignment[str(group)] = "test"
    return assignment


def segmentation_fold_assignments(
    labelled_groups: Sequence[str], folds: int, seed: int
) -> List[Dict[str, str]]:
    groups = np.array(sorted(set(labelled_groups)), dtype=object)
    if len(groups) < folds:
        raise ValueError(f"Only {len(groups)} labelled groups are available for {folds}-fold CV")
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    chunks = [list(chunk) for chunk in np.array_split(groups, folds)]
    assignments = []
    for fold in range(folds):
        test = set(chunks[fold])
        remaining = [group for group in groups if group not in test]
        fold_rng = np.random.default_rng(seed + 1000 + fold)
        fold_rng.shuffle(remaining)
        n_val = max(1, int(round(0.2 * len(remaining))))
        val = set(remaining[:n_val])
        current = {}
        for group in groups:
            if group in test:
                current[str(group)] = "test"
            elif group in val:
                current[str(group)] = "val"
            else:
                current[str(group)] = "train"
        assignments.append(current)
    return assignments


def assign_splits(rows: List[Dict[str, object]], mapping: Dict[str, str]) -> pd.DataFrame:
    table = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    table["split"] = table["group_id"].map(mapping)
    if table["split"].isna().any():
        missing = table.loc[table["split"].isna(), "group_id"].unique().tolist()
        raise RuntimeError(f"Split mapping is missing groups: {missing[:10]}")
    return table


def validate_manifest(table: pd.DataFrame, name: str) -> Dict[str, object]:
    if table["sample_id"].duplicated().any():
        duplicated = table.loc[table["sample_id"].duplicated(), "sample_id"].tolist()
        raise RuntimeError(f"Duplicated sample_id in {name}: {duplicated[:10]}")
    leakage = table.groupby("group_id")["split"].nunique()
    leakage = leakage[leakage > 1]
    if not leakage.empty:
        raise RuntimeError(f"Group leakage in {name}: {leakage.index.tolist()[:10]}")
    return {
        "rows": int(len(table)),
        "groups": int(table["group_id"].nunique()),
        "labelled_groups": int(
            table.loc[table["has_manual_label"].astype(int) == 1, "group_id"].nunique()
        ),
        "rows_by_dataset": table["dataset"].value_counts().to_dict(),
        "groups_by_dataset": table.groupby("dataset")["group_id"].nunique().to_dict(),
        "groups_by_split": table.groupby("split")["group_id"].nunique().to_dict(),
    }


def save_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, label_report = split_multiclass_labels(
        root,
        layer_class=args.layer_class,
        vessel_class=args.vessel_class,
        ignore_class=args.ignore_class,
        overwrite=args.overwrite_masks,
        allow_other_classes=args.allow_other_classes,
    )
    duke17, unmatched17 = scan_duke17(root)
    duke28, unmatched28 = scan_duke28(root)
    pku37, unmatched_pku, frames_per_position = scan_pku37(root, labels)
    unmatched = unmatched17 + unmatched28 + unmatched_pku
    if unmatched and not args.allow_unmatched:
        preview = "\n".join(f"- {item}" for item in unmatched[:30])
        raise RuntimeError(
            f"Found {len(unmatched)} unmatched files. Fix them or pass --allow-unmatched:\n{preview}"
        )

    all_rows = duke17 + duke28 + pku37
    dataset_groups: Dict[str, List[str]] = defaultdict(list)
    for row in all_rows:
        dataset_groups[str(row["dataset"])].append(str(row["group_id"]))
    denoise_mapping: Dict[str, str] = {}
    for offset, (dataset, groups) in enumerate(sorted(dataset_groups.items())):
        current = group_split(
            groups,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed + offset * 101,
        )
        denoise_mapping.update(current)

    denoise_table = assign_splits(all_rows, denoise_mapping)
    save_table(denoise_table, output_dir / "manifest_denoise.csv")
    save_table(denoise_table, output_dir / "manifest_all.csv")

    labelled_groups = sorted(
        denoise_table.loc[
            (denoise_table["dataset"] == "PKU37")
            & (denoise_table["has_manual_label"].astype(int) == 1),
            "group_id",
        ].unique()
    )
    fold_assignments = segmentation_fold_assignments(
        labelled_groups, folds=args.folds, seed=args.seed
    )
    reports: Dict[str, object] = {
        "manifest_denoise.csv": validate_manifest(denoise_table, "manifest_denoise.csv")
    }
    segmentation_source = denoise_table[
        (denoise_table["dataset"] == "PKU37")
        & (denoise_table["has_manual_label"].astype(int) == 1)
    ].copy()
    for fold, segmentation_mapping in enumerate(fold_assignments):
        seg_table = segmentation_source.copy()
        seg_table["split"] = seg_table["group_id"].map(segmentation_mapping)
        seg_name = f"segmentation_folds/manifest_seg_fold{fold}.csv"
        save_table(seg_table, output_dir / seg_name)
        reports[seg_name] = validate_manifest(seg_table, seg_name)

        joint_table = denoise_table.copy()
        labelled_mask = joint_table["group_id"].isin(segmentation_mapping)
        joint_table.loc[labelled_mask, "split"] = joint_table.loc[
            labelled_mask, "group_id"
        ].map(segmentation_mapping)
        joint_name = f"joint_folds/manifest_joint_fold{fold}.csv"
        save_table(joint_table, output_dir / joint_name)
        reports[joint_name] = validate_manifest(joint_table, joint_name)

    report = {
        "project_root": str(root),
        "pairing_rules": {
            "Duke17": "*_Raw Image.tif -> *_Averaged Image.tif",
            "Duke28": "LL* -> HH*; sl* -> sh* (case-insensitive)",
            "PKU37": "six-digit noisy ID: first four digits=clean/group, final two=frame",
        },
        "label_classes": {
            "background": 0,
            "layer": args.layer_class,
            "vessel": args.vessel_class,
            "ignore": args.ignore_class,
            "layer_binary_rule": f"class in {{{args.layer_class}, {args.vessel_class}}}",
            "vessel_binary_rule": f"class == {args.vessel_class}",
        },
        "label_report": label_report,
        "frames_per_pku_position": frames_per_position,
        "unmatched": unmatched,
        "manifests": reports,
    }
    with (output_dir / "dataset_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print("Dataset preparation completed.")
    print(f"Duke17 pairs: {len(duke17)}")
    print(f"Duke28 pairs: {len(duke28)}")
    print(f"PKU37 noisy-clean rows: {len(pku37)}")
    print(f"PKU37 positions: {len(frames_per_position)}")
    print(f"Labelled PKU37 positions: {len(labelled_groups)}")
    print(f"Output: {output_dir}")
    if unmatched:
        print(f"Warning: {len(unmatched)} unmatched items were recorded in dataset_report.json")


if __name__ == "__main__":
    main()
