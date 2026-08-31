from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import binary_fill_holes, label as connected_components

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.data.io import read_gray, read_mask
from sabids.metrics import edge_preservation_index, psnr, reference_edge_mae, region_cnr, rmse, ssim
from sabids.utils import write_json


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def shape_of(path: Path | None) -> str:
    if path is None or not path.is_file():
        return "MISSING"
    image = read_gray(path)
    return f"{image.shape[0]}x{image.shape[1]}"


def write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Could not encode plot: {path}")
    path.write_bytes(encoded.tobytes())


def boundary_statistics(layer: np.ndarray) -> dict[str, float]:
    upper, lower, thickness = [], [], []
    invalid = 0
    for column in range(layer.shape[1]):
        ys = np.flatnonzero(layer[:, column])
        if not len(ys):
            invalid += 1
            continue
        upper.append(float(ys[0])); lower.append(float(ys[-1])); thickness.append(float(ys[-1] - ys[0] + 1))
    values = np.asarray(thickness, dtype=float)
    if not len(values):
        return {"invalid_thickness_columns": float(layer.shape[1]), "thickness_median": np.nan,
                "thickness_iqr": np.nan, "thickness_outlier_fraction": np.nan,
                "upper_boundary_roughness": np.nan, "lower_boundary_roughness": np.nan,
                "boundary_crossing_columns": 0.0, "valid_column_fraction": 0.0}
    q1, q3 = np.quantile(values, [0.25, 0.75]); iqr = q3 - q1
    outliers = (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return {
        "invalid_thickness_columns": float(invalid), "thickness_median": float(np.median(values)),
        "thickness_iqr": float(iqr), "thickness_outlier_fraction": float(outliers.mean()),
        "upper_boundary_roughness": float(np.mean(np.abs(np.diff(upper)))) if len(upper) > 1 else np.nan,
        "lower_boundary_roughness": float(np.mean(np.abs(np.diff(lower)))) if len(lower) > 1 else np.nan,
        "boundary_crossing_columns": 0.0, "valid_column_fraction": float(len(values) / layer.shape[1]),
    }


def label_statistics(layer: np.ndarray, vessel: np.ndarray, component_bins: tuple[int, int] = (64, 256)) -> dict[str, Any]:
    layer = layer.astype(bool); vessel = vessel.astype(bool)
    layer_cc, layer_count = connected_components(layer)
    vessel_cc, vessel_count = connected_components(vessel)
    largest = max((int((layer_cc == i).sum()) for i in range(1, layer_count + 1)), default=0)
    hole_pixels = int((binary_fill_holes(layer) & ~layer).sum())
    vessel_areas = sorted(int((vessel_cc == i).sum()) for i in range(1, vessel_count + 1))
    small_max, medium_max = component_bins
    small = [area for area in vessel_areas if area <= small_max]
    medium = [area for area in vessel_areas if small_max < area <= medium_max]
    large = [area for area in vessel_areas if area > medium_max]
    outside = int((vessel & ~layer).sum())
    overlap = int(vessel.sum())
    whole_layer_dice = (2.0 * overlap + 1e-6) / (float(layer.sum()) + float(vessel.sum()) + 1e-6)
    result: dict[str, Any] = {
        "layer_component_count": int(layer_count),
        "layer_fragment_fraction": float((layer.sum() - largest) / max(float(layer.sum()), 1.0)),
        "layer_hole_fraction": float(hole_pixels / max(float(layer.sum()), 1.0)),
        "vessel_outside_layer_fraction": float(outside / max(float(vessel.sum()), 1.0)),
        "vessel_fraction_of_layer": float((vessel & layer).sum() / max(float(layer.sum()), 1.0)),
        "vessel_component_count": int(vessel_count),
        "vessel_component_min": min(vessel_areas, default=0),
        "vessel_component_median": float(np.median(vessel_areas)) if vessel_areas else 0.0,
        "vessel_component_max": max(vessel_areas, default=0),
        "small_component_count": len(small), "small_component_pixels": int(sum(small)),
        "medium_component_count": len(medium), "medium_component_pixels": int(sum(medium)),
        "large_component_count": len(large), "large_component_pixels": int(sum(large)),
        "component_small_max_original_pixels": int(small_max),
        "component_medium_max_original_pixels": int(medium_max),
        "whole_layer_vessel_baseline_dice": float(whole_layer_dice),
        "vessel_fully_inside_layer": outside == 0,
    }
    result.update(boundary_statistics(layer))
    return result


def fixed_rows(group: pd.DataFrame) -> pd.DataFrame:
    ordered = group.sort_values("sample_id")
    indices = sorted(set((0, len(ordered) // 2, len(ordered) - 1)))
    return ordered.iloc[indices]


def kmeans(values: np.ndarray, count: int = 4, iterations: int = 50) -> np.ndarray:
    count = max(1, min(count, len(values)))
    order = np.argsort(values[:, 0])
    centers = values[order[np.linspace(0, len(values) - 1, count).round().astype(int)]].copy()
    labels = np.zeros(len(values), dtype=int)
    for _ in range(iterations):
        new = np.argmin(((values[:, None] - centers[None]) ** 2).sum(axis=2), axis=1)
        if np.array_equal(new, labels) and _ > 0:
            break
        labels = new
        for index in range(count):
            if np.any(labels == index): centers[index] = values[labels == index].mean(axis=0)
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sealed-test-safe PKU37 position and label audit")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol", default="configs/current/input_oracle_cv/protocol.yaml")
    parser.add_argument("--output", default="runs/input_oracle_cv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); root = Path(args.project_root).expanduser().resolve()
    protocol_path = resolve(root, args.protocol)
    if protocol_path is None or not protocol_path.is_file(): raise FileNotFoundError(protocol_path)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    output = resolve(root, args.output); assert output is not None
    audit_dir, label_dir = output / "audit", output / "label_audit"
    if output.exists() and any(output.rglob("*")) and not args.resume:
        raise FileExistsError(f"Output exists; use --resume only to complete missing audit files: {output}")
    audit_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
    source = resolve(root, protocol["source_manifest"]); all_path = resolve(root, protocol["all_manifest"])
    if source is None or all_path is None: raise RuntimeError("Protocol manifest path is empty")
    seg = pd.read_csv(source, dtype=str).fillna(""); all_table = pd.read_csv(all_path, dtype=str).fillna("")
    seg = seg[seg["dataset"].str.lower() == "pku37"].copy()
    pku = all_table[all_table["dataset"].str.lower() == "pku37"].copy()
    configured_test = {str(v) for v in protocol["sealed_test_groups"]}
    manifest_test = set(seg.loc[seg["split"] == "test", "group_id"].astype(str))
    sealed = configured_test | manifest_test
    labelled = set(seg.loc[(seg["layer_mask_path"] != "") & (seg["vessel_mask_path"] != ""), "group_id"].astype(str))
    development = labelled - sealed
    split_crossing = {
        str(group): sorted(part["split"].unique().tolist())
        for group, part in seg.groupby("group_id") if part["split"].nunique() != 1
    }
    blockers = []
    if configured_test != manifest_test:
        blockers.append({"code": "SEALED_TEST_MISMATCH", "configured": sorted(configured_test), "manifest": sorted(manifest_test)})
    if len(labelled) != int(protocol["expected_labelled_groups"]):
        blockers.append({"code": "LABELLED_GROUP_COUNT_MISMATCH", "expected": int(protocol["expected_labelled_groups"]), "actual": len(labelled)})
    if len(development) != int(protocol["expected_development_labelled_groups"]):
        blockers.append({"code": "DEVELOPMENT_GROUP_COUNT_MISMATCH", "expected": int(protocol["expected_development_labelled_groups"]), "actual": len(development)})
    if split_crossing: blockers.append({"code": "GROUP_SPLIT_CROSSING", "groups": split_crossing})

    position_rows, label_rows, asset_rows, quality_rows, registration_rows, flags = [], [], [], [], [], []
    characteristics = []
    for group_id, group in pku.groupby("group_id", sort=True):
        group_id = str(group_id); is_sealed = group_id in sealed
        seg_group = seg[seg["group_id"] == group_id]
        split = str(seg_group.iloc[0]["split"]) if len(seg_group) else "unlabelled"
        first = group.sort_values("sample_id").iloc[0]
        layer_path = resolve(root, first.get("layer_mask_path", "")); vessel_path = resolve(root, first.get("vessel_mask_path", ""))
        clean_paths = sorted({str(v) for v in group["clean_path"] if str(v).strip()})
        row = {
            "group_id": group_id, "manifest_status": split, "noisy_frame_count": int(len(group)),
            "clean_reference_count": len(clean_paths), "layer_label_exists_metadata": bool(str(first.get("layer_mask_path", ""))),
            "vessel_label_exists_metadata": bool(str(first.get("vessel_mask_path", ""))),
            "label_scope": "position_level" if len(group) > 1 else "frame_level_or_single",
            "label_reference_image": str(first.get("clean_path", "")), "is_sealed_test": is_sealed,
            "allowed_in_development_cv": bool(group_id in development and not blockers),
            "cv_exclusion_reason": "sealed_test" if is_sealed else ("missing_both_labels" if group_id not in labelled else ("global_protocol_blocked" if blockers else "")),
            "asset_access": "METADATA_ONLY_NOT_OPENED" if is_sealed else "DEVELOPMENT_ASSETS_AUDITED",
        }
        if not is_sealed:
            for column in ("image_path", "clean_path", "layer_mask_path", "vessel_mask_path", "multiclass_label_path"):
                for value in sorted({str(v) for v in group[column] if str(v).strip()}) if column in group else []:
                    path = resolve(root, value); exists = bool(path and path.is_file())
                    asset_rows.append({"group_id": group_id, "column": column, "path": value, "exists": exists,
                                       "bytes": path.stat().st_size if exists else None, "sha256": sha256_file(path) if exists else None,
                                       "shape": shape_of(path) if exists else "MISSING"})
            if layer_path and vessel_path and layer_path.is_file() and vessel_path.is_file():
                layer, vessel = read_mask(layer_path) > .5, read_mask(vessel_path) > .5
                stats = label_statistics(layer, vessel, tuple(map(int, protocol.get("component_area_bins_original_pixels", [64, 256]))))
                label_rows.append({"group_id": group_id, "layer_path": str(first["layer_mask_path"]), "vessel_path": str(first["vessel_mask_path"]),
                                   "layer_shape": f"{layer.shape[0]}x{layer.shape[1]}", "vessel_shape": f"{vessel.shape[0]}x{vessel.shape[1]}",
                                   "layer_valid_pixel_fraction": float(layer.mean()), "vessel_valid_pixel_fraction": float(vessel.mean()), **stats})
                quality = {"group_id": group_id, **stats}
                clean_path = resolve(root, first.get("clean_path", "")); clean = read_gray(clean_path) if clean_path and clean_path.is_file() else None
                if clean is not None and clean.shape == layer.shape:
                    quality["clean_vessel_stroma_cnr"] = region_cnr(clean, vessel, layer & ~vessel)
                quality_rows.append(quality)
                for key, value in stats.items():
                    if ("fraction" in key and isinstance(value, (float, int)) and np.isfinite(value) and (value > .25 if "outside" in key or "fragment" in key or "hole" in key else False)):
                        flags.append({"group_id": group_id, "flag": key, "value": value, "automatic_only": True})
            selected = fixed_rows(group)
            clean_path = resolve(root, first.get("clean_path", "")); clean = read_gray(clean_path) if clean_path and clean_path.is_file() else None
            metric_rows = []
            for _, sample in selected.iterrows():
                noisy_path = resolve(root, sample["image_path"]); noisy = read_gray(noisy_path) if noisy_path and noisy_path.is_file() else None
                if noisy is None or clean is None or noisy.shape != clean.shape: continue
                shift, response = cv2.phaseCorrelate(noisy.astype(np.float32), clean.astype(np.float32))
                registration_rows.append({"group_id": group_id, "sample_id": sample["sample_id"], "selection": "first_middle_last_by_sample_id",
                                          "noisy_to_clean_shift_x": shift[0], "noisy_to_clean_shift_y": shift[1], "phase_correlation_response": response,
                                          "label_reference": str(sample.get("clean_path", "")), "label_closer_to_clean_possible": True})
                metric_rows.append({"psnr_noisy": psnr(noisy, clean), "ssim_noisy": ssim(noisy, clean), "rmse_noisy": rmse(noisy, clean),
                                    "epi_noisy": edge_preservation_index(noisy, clean), "reference_edge_mae_noisy": reference_edge_mae(noisy, clean),
                                    "noisy_mean": float(noisy.mean()), "noisy_std": float(noisy.std()), "clean_mean": float(clean.mean()), "clean_std": float(clean.std())})
            if metric_rows:
                averaged = pd.DataFrame(metric_rows).mean(numeric_only=True).to_dict(); averaged["group_id"] = group_id
                if quality_rows and quality_rows[-1].get("group_id") == group_id:
                    for key in ("vessel_fraction_of_layer", "thickness_median", "vessel_component_count", "small_component_count", "medium_component_count", "large_component_count"):
                        averaged[key] = quality_rows[-1].get(key)
                characteristics.append(averaged)
        position_rows.append(row)

    pd.DataFrame(position_rows).to_csv(audit_dir / "pku37_position_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(audit_dir / "pku37_label_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(asset_rows).to_csv(audit_dir / "pku37_asset_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality_rows).to_csv(label_dir / "label_quality_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(registration_rows).to_csv(label_dir / "registration_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flags, columns=["group_id", "flag", "value", "automatic_only"]).to_csv(label_dir / "automatic_flags.csv", index=False, encoding="utf-8-sig")
    review = pd.DataFrame([{"group_id": g, "review_status": "PENDING", "layer_quality": "PENDING", "vessel_quality": "PENDING",
                            "alignment_quality": "PENDING", "small_vessel_confidence": "PENDING", "boundary_confidence": "PENDING",
                            "suspected_missing_vessel": "PENDING", "suspected_false_vessel": "PENDING", "reviewer_1": "", "reviewer_2": "",
                            "adjudication": "", "notes": ""} for g in sorted(development)])
    review.to_csv(label_dir / "label_review_form.csv", index=False, encoding="utf-8-sig")
    chars = pd.DataFrame(characteristics)
    chars.to_csv(output / "position_characteristics.csv", index=False, encoding="utf-8-sig")
    if len(chars) >= 2:
        numeric = [c for c in chars.select_dtypes(include=[np.number]).columns if chars[c].notna().sum() >= 2]
        x = chars[numeric].copy().fillna(chars[numeric].median()).to_numpy(float)
        scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0); z = (x - x.mean(axis=0)) / scale
        u, s, _ = np.linalg.svd(z, full_matrices=False); scores = u[:, :2] * s[:2]
        typicality = np.sqrt((z * z).mean(axis=1)); percentiles = pd.Series(typicality).rank(pct=True).to_numpy()
        typicality_table = pd.DataFrame({"group_id": chars["group_id"], "typicality_distance": typicality, "atypicality_percentile": percentiles,
                      "pc1": scores[:, 0], "pc2": scores[:, 1] if scores.shape[1] > 1 else 0.0})
        for column in numeric:
            typicality_table[f"{column}_percentile"] = chars[column].rank(pct=True).to_numpy()
        typicality_table.to_csv(output / "position_typicality.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"group_id": chars["group_id"], "cluster": kmeans(z, min(4, len(chars))),
                      "clustering_features": ";".join(numeric), "segmentation_results_used": False}).to_csv(output / "position_clusters.csv", index=False, encoding="utf-8-sig")
        # Dependency-light descriptive plots. They are never used to select a
        # sample or a fold and contain development positions only.
        heat = np.clip((z - np.nanmin(z)) / max(float(np.nanmax(z) - np.nanmin(z)), 1e-6) * 255, 0, 255).astype(np.uint8)
        heat = cv2.applyColorMap(cv2.resize(heat, (max(480, 40 * len(numeric)), max(240, 28 * len(chars))), interpolation=cv2.INTER_NEAREST), cv2.COLORMAP_VIRIDIS)
        write_png(output / "position_feature_heatmap.png", heat)
        plot = np.full((640, 640, 3), 255, np.uint8); px, py = scores[:, 0], scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))
        px = (px - px.min()) / max(float(px.max() - px.min()), 1e-6); py = (py - py.min()) / max(float(py.max() - py.min()), 1e-6)
        for index, group_id in enumerate(chars["group_id"].astype(str)):
            point = (int(45 + 550 * px[index]), int(595 - 550 * py[index])); cv2.circle(plot, point, 5, (150, 70, 30), -1); cv2.putText(plot, group_id.replace("pku_", ""), (point[0] + 4, point[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, .35, (0, 0, 0), 1)
        write_png(output / "position_pca.png", plot)
    else:
        pd.DataFrame(columns=["group_id", "typicality_distance", "atypicality_percentile", "pc1", "pc2"]).to_csv(output / "position_typicality.csv", index=False)
        pd.DataFrame(columns=["group_id", "cluster"]).to_csv(output / "position_clusters.csv", index=False)
    sealed_report = {"configured_test_groups": sorted(configured_test), "manifest_test_groups": sorted(manifest_test),
                     "effective_sealed_union": sorted(sealed), "asset_files_opened": 0, "images_opened": 0, "labels_opened": 0,
                     "metadata_source": str(source), "test_used_for_folds_or_features": False}
    write_json(sealed_report, audit_dir / "sealed_test_inventory_metadata_only.json")
    summary = {"status": "blocked" if blockers else "passed", "protocol_version": protocol["protocol_version"],
               "source_manifest": str(source), "source_manifest_sha256": sha256_file(source), "labelled_groups": sorted(labelled),
               "n_labelled_groups": len(labelled), "development_groups": sorted(development), "n_development_groups": len(development),
               "sealed_test_groups": sorted(sealed), "blockers": blockers, "test_assets_opened": 0,
               "training_authorized": not blockers, "dry_run": args.dry_run, "smoke_test": args.smoke_test}
    write_json(summary, audit_dir / "audit_summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if blockers: raise SystemExit(2)


if __name__ == "__main__":
    main()
