from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_PROJECT_ROOT = r"E:\1-脉络膜\OCT降噪\SABIDS-Net"
DEFAULT_DATA_ROOT = r"E:\1-脉络膜\12x12choroid"
IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
PATIENT_PATTERN = re.compile(r"DYN\d+", flags=re.IGNORECASE)
SCAN_PATTERN = re.compile(r"sn\d+", flags=re.IGNORECASE)
EYE_PATTERN = re.compile(r"(?:OD|OS)\d*", flags=re.IGNORECASE)
FRAME_PATTERN = re.compile(r"_(\d{4,})(?=_|$)")

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
    "is_clean",
    "disease",
    "eye",
    "scan_id",
    "source_frame_index",
    "original_height",
    "original_width",
    "original_filename",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a leakage-safe manifest for the private 12x12 mm SS-OCT "
            "choroid layer/vessel segmentation dataset"
        )
    )
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument(
        "--image-dir", default=str(Path(DEFAULT_DATA_ROOT) / "voc_jpg")
    )
    parser.add_argument(
        "--layer-dir", default=str(Path(DEFAULT_DATA_ROOT) / "voc_seg")
    )
    parser.add_argument(
        "--vessel-dir", default=str(Path(DEFAULT_DATA_ROOT) / "voc_vessel_seg")
    )
    parser.add_argument(
        "--output-manifest", default="Manifests/manifest_private_seg.csv"
    )
    parser.add_argument(
        "--prepared-mask-dir", default="PreparedLabels/private12x12"
    )
    parser.add_argument(
        "--split-unit",
        choices=["patient", "scan"],
        default="patient",
        help="Patient is recommended to prevent both-eye and repeated-scan leakage.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-candidates", type=int, default=2048)
    parser.add_argument("--overwrite-masks", action="store_true")
    parser.add_argument("--allow-missing-layer", action="store_true")
    parser.add_argument("--allow-unparsed", action="store_true")
    return parser.parse_args()


def image_files(folder: Path) -> List[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing directory: {folder}")
    return [
        path
        for path in sorted(folder.rglob("*"), key=lambda item: str(item).casefold())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def index_by_stem(folder: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = defaultdict(list)
    for path in image_files(folder):
        key = path.stem.casefold()
        if key in index:
            duplicates[key].extend([str(index[key]), str(path)])
        else:
            index[key] = path
    if duplicates:
        preview = "\n".join(
            f"{stem}: {sorted(set(paths))}" for stem, paths in list(duplicates.items())[:10]
        )
        raise ValueError(f"Duplicate label stems were found in {folder}:\n{preview}")
    return index


def parse_private_filename(stem: str) -> Dict[str, object]:
    patient_match = PATIENT_PATTERN.search(stem)
    scan_match = SCAN_PATTERN.search(stem)
    if patient_match is None or scan_match is None:
        raise ValueError(
            f"Unable to parse patient/scan from {stem!r}; expected DYNxxxxx and snxxxx"
        )
    if scan_match.start() <= patient_match.end():
        raise ValueError(f"Unexpected patient/scan order in filename: {stem!r}")

    disease = stem[: patient_match.start()].strip(" _-") or "unknown"
    patient_id = patient_match.group(0).upper()
    scan_id = scan_match.group(0).lower()
    patient_scan_text = stem[patient_match.start() : scan_match.start()]
    eye_match = EYE_PATTERN.search(patient_scan_text)
    eye = eye_match.group(0).upper() if eye_match else "UNK"
    frame_matches = FRAME_PATTERN.findall(stem)
    source_frame_index = int(frame_matches[0]) if frame_matches else -1
    frame_index = int(frame_matches[-1]) if frame_matches else -1
    digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    sample_id = (
        f"p12_{patient_id.lower()}_{scan_id}_f{frame_index:04d}_{digest}"
        if frame_index >= 0
        else f"p12_{patient_id.lower()}_{scan_id}_{digest}"
    )
    return {
        "sample_id": sample_id,
        # Each B-scan is a different anatomical position. Do not let the dataset
        # treat adjacent slices as same-position repeated noisy observations.
        "group_id": sample_id,
        "patient_id": patient_id,
        "disease": disease,
        "eye": eye,
        "scan_id": scan_id,
        "source_frame_index": source_frame_index,
        "frame_index": frame_index,
    }


def path_for_manifest(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _mask_plane(path: Path) -> Tuple[np.ndarray, List[int]]:
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.ndim == 3:
        channels = raw[..., :3]
        if np.all(channels == channels[..., :1]):
            raw = channels[..., 0]
        else:
            raw = channels.max(axis=2)
    if raw.ndim != 2:
        raise ValueError(f"Mask must be 2-D or RGB, but {path} has shape={raw.shape}")
    unique = sorted(int(value) for value in np.unique(raw))
    return raw, unique


def binarize_mask(path: Path) -> Tuple[np.ndarray, List[int], str]:
    raw, unique = _mask_plane(path)
    maximum = float(np.max(raw)) if raw.size else 0.0
    if maximum <= 16.0:
        binary = raw > 0
        rule = "value>0 (class-index mask)"
    else:
        binary = raw > maximum * 0.5
        rule = f"value>{maximum * 0.5:g} (intensity mask)"
    return binary.astype(np.uint8), unique, rule


def prepare_binary_mask(
    source: Path,
    destination: Path,
    expected_size: Tuple[int, int],
    overwrite: bool,
) -> Dict[str, object]:
    binary, unique, rule = binarize_mask(source)
    height, width = binary.shape
    if (width, height) != expected_size:
        raise ValueError(
            f"Size mismatch: image={expected_size}, mask={(width, height)}, mask_path={source}"
        )
    foreground_ratio = float(binary.mean())
    if overwrite or not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary * 255, mode="L").save(destination)
    return {
        "source": str(source),
        "unique_values": unique,
        "binarization_rule": rule,
        "foreground_ratio": foreground_ratio,
    }


def _split_sizes(n_units: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    test_ratio = 1.0 - train_ratio - val_ratio
    if not 0.0 < train_ratio < 1.0 or not 0.0 < val_ratio < 1.0 or test_ratio <= 0:
        raise ValueError("Ratios must satisfy train>0, val>0 and train+val<1")
    if n_units < 3:
        raise ValueError("At least three split units are required")
    n_val = max(1, int(round(n_units * val_ratio)))
    n_test = max(1, int(round(n_units * test_ratio)))
    if n_val + n_test >= n_units:
        n_val, n_test = 1, 1
    return n_units - n_val - n_test, n_val, n_test


def _candidate_score(
    rows: Sequence[Dict[str, object]],
    assignment: Dict[str, str],
    split_key: str,
    ratios: Dict[str, float],
) -> float:
    total_rows = max(len(rows), 1)
    diseases = Counter(str(row["disease"]) for row in rows)
    global_distribution = {
        disease: count / total_rows for disease, count in diseases.items()
    }
    split_rows: Dict[str, List[Dict[str, object]]] = {
        split: [row for row in rows if assignment[str(row[split_key])] == split]
        for split in ("train", "val", "test")
    }
    score = 0.0
    for split, current in split_rows.items():
        score += 10.0 * abs(len(current) / total_rows - ratios[split])
        current_diseases = Counter(str(row["disease"]) for row in current)
        denominator = max(len(current), 1)
        score += sum(
            abs(current_diseases[disease] / denominator - proportion)
            for disease, proportion in global_distribution.items()
        )

    vessel_units = {
        str(row[split_key]) for row in rows if bool(row.get("vessel_mask_path"))
    }
    vessel_splits = {assignment[unit] for unit in vessel_units}
    if len(vessel_units) >= 3:
        score += 1000.0 * len({"train", "val", "test"} - vessel_splits)
    elif len(vessel_units) == 2:
        score += 1000.0 if "train" not in vessel_splits else 0.0
        score += 500.0 if not ({"val", "test"} & vessel_splits) else 0.0
    elif len(vessel_units) == 1:
        score += 1000.0 if "train" not in vessel_splits else 0.0
    return score


def split_rows(
    rows: Sequence[Dict[str, object]],
    split_unit: str,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    candidates: int,
) -> Dict[str, str]:
    split_key = "patient_id" if split_unit == "patient" else "scan_id"
    units = sorted({str(row[split_key]) for row in rows})
    n_train, n_val, _ = _split_sizes(len(units), train_ratio, val_ratio)
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": 1.0 - train_ratio - val_ratio,
    }
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_assignment: Dict[str, str] = {}
    for _ in range(max(1, candidates)):
        shuffled = list(rng.permutation(units))
        assignment = {
            unit: (
                "train"
                if index < n_train
                else "val"
                if index < n_train + n_val
                else "test"
            )
            for index, unit in enumerate(shuffled)
        }
        score = _candidate_score(rows, assignment, split_key, ratios)
        if score < best_score:
            best_score = score
            best_assignment = assignment
    return best_assignment


def _resolve_output(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _summary(rows: Sequence[Dict[str, object]], warnings: Sequence[str]) -> Dict[str, object]:
    result: Dict[str, object] = {
        "rows": len(rows),
        "patients": len({str(row["patient_id"]) for row in rows}),
        "scans": len({str(row["scan_id"]) for row in rows}),
        "layer_labels": sum(bool(row["layer_mask_path"]) for row in rows),
        "vessel_labels": sum(bool(row["vessel_mask_path"]) for row in rows),
        "image_sizes": dict(
            sorted(
                Counter(
                    f"{row['original_height']}x{row['original_width']}" for row in rows
                ).items()
            )
        ),
        "warnings": list(warnings),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        current = [row for row in rows if row["split"] == split]
        result["splits"][split] = {
            "rows": len(current),
            "patients": len({str(row["patient_id"]) for row in current}),
            "scans": len({str(row["scan_id"]) for row in current}),
            "layer_labels": sum(bool(row["layer_mask_path"]) for row in current),
            "vessel_labels": sum(bool(row["vessel_mask_path"]) for row in current),
            "diseases": dict(sorted(Counter(str(row["disease"]) for row in current).items())),
        }
    return result


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    image_dir = Path(args.image_dir).expanduser().resolve()
    layer_dir = Path(args.layer_dir).expanduser().resolve()
    vessel_dir = Path(args.vessel_dir).expanduser().resolve()
    output_manifest = _resolve_output(args.output_manifest, project_root)
    prepared_root = _resolve_output(args.prepared_mask_dir, project_root)

    images = image_files(image_dir)
    layer_index = index_by_stem(layer_dir)
    vessel_index = index_by_stem(vessel_dir)
    rows: List[Dict[str, object]] = []
    warnings: List[str] = []
    matched_layer_stems = set()
    matched_vessel_stems = set()
    mask_audit: Dict[str, Dict[str, object]] = {}

    for image_index, image_path in enumerate(images):
        stem_key = image_path.stem.casefold()
        try:
            parsed = parse_private_filename(image_path.stem)
        except ValueError as exc:
            if not args.allow_unparsed:
                raise
            digest = hashlib.sha1(image_path.stem.encode("utf-8")).hexdigest()[:10]
            parsed = {
                "sample_id": f"p12_unparsed_{digest}",
                "group_id": f"p12_unparsed_{digest}",
                "patient_id": f"unparsed_{digest}",
                "disease": "unknown",
                "eye": "UNK",
                "scan_id": f"unparsed_{digest}",
                "source_frame_index": -1,
                "frame_index": image_index,
            }
            warnings.append(str(exc))

        layer_source = layer_index.get(stem_key)
        vessel_source = vessel_index.get(stem_key)
        if layer_source is None and not args.allow_missing_layer:
            raise FileNotFoundError(f"Layer label is missing for image: {image_path.name}")

        with Image.open(image_path) as image:
            image_size = image.size
        layer_output = ""
        vessel_output = ""
        if layer_source is not None:
            matched_layer_stems.add(stem_key)
            layer_destination = prepared_root / "layer_binary" / f"{image_path.stem}.png"
            mask_audit[f"layer::{image_path.stem}"] = prepare_binary_mask(
                layer_source, layer_destination, image_size, args.overwrite_masks
            )
            layer_output = path_for_manifest(layer_destination, project_root)
        if vessel_source is not None:
            matched_vessel_stems.add(stem_key)
            vessel_destination = prepared_root / "vessel_binary" / f"{image_path.stem}.png"
            mask_audit[f"vessel::{image_path.stem}"] = prepare_binary_mask(
                vessel_source, vessel_destination, image_size, args.overwrite_masks
            )
            vessel_output = path_for_manifest(vessel_destination, project_root)

        rows.append(
            {
                **parsed,
                "dataset": "Private12x12",
                "domain": "private",
                "scan_protocol": "SS-OCT-12x12",
                "split": "",
                "image_path": path_for_manifest(image_path, project_root),
                "clean_path": "",
                "layer_mask_path": layer_output,
                "vessel_mask_path": vessel_output,
                "is_clean": 0,
                "original_height": image_size[1],
                "original_width": image_size[0],
                "original_filename": image_path.name,
            }
        )

    extra_layers = sorted(set(layer_index) - matched_layer_stems)
    extra_vessels = sorted(set(vessel_index) - matched_vessel_stems)
    if extra_layers:
        warnings.append(f"{len(extra_layers)} layer labels have no image match: {extra_layers[:10]}")
    if extra_vessels:
        warnings.append(f"{len(extra_vessels)} vessel labels have no image match: {extra_vessels[:10]}")

    assignment = split_rows(
        rows,
        split_unit=args.split_unit,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        candidates=args.split_candidates,
    )
    split_key = "patient_id" if args.split_unit == "patient" else "scan_id"
    for row in rows:
        row["split"] = assignment[str(row[split_key])]

    summary = _summary(rows, warnings)
    for split in ("train", "val", "test"):
        if summary["splits"][split]["vessel_labels"] == 0:
            message = f"No manual vessel labels were assigned to split={split}."
            summary["warnings"].append(message)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    table.to_csv(output_manifest, index=False, encoding="utf-8-sig")
    patient_split = (
        table.groupby([split_key, "split"], as_index=False)
        .agg(
            rows=("sample_id", "count"),
            layer_labels=("layer_mask_path", lambda values: sum(bool(value) for value in values)),
            vessel_labels=("vessel_mask_path", lambda values: sum(bool(value) for value in values)),
        )
        .sort_values(["split", split_key])
    )
    patient_split.to_csv(
        output_manifest.with_name("private12x12_split_units.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    with output_manifest.with_name("private12x12_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    with output_manifest.with_name("private12x12_mask_audit.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(mask_audit, handle, ensure_ascii=False, indent=2)

    print("Private 12x12 manifest preparation completed.")
    print(f"Images: {summary['rows']}")
    print(f"Patients: {summary['patients']} | scans: {summary['scans']}")
    print(f"Image sizes (H x W): {summary['image_sizes']}")
    print(
        f"Layer labels: {summary['layer_labels']} | "
        f"vessel labels: {summary['vessel_labels']}"
    )
    for split, values in summary["splits"].items():
        print(
            f"{split}: rows={values['rows']}, patients={values['patients']}, "
            f"scans={values['scans']}, vessel_labels={values['vessel_labels']}"
        )
    for warning in summary["warnings"]:
        print(f"Warning: {warning}")
    print(f"Manifest: {output_manifest}")


if __name__ == "__main__":
    main()
