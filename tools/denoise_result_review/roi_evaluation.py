from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import METHOD_ORDER
from .image_io import read_float01
from .roi_metrics import compute_roi_metrics, vessel_stroma_pair_metrics
from .roi_registry import require_locked
from .roi_statistics import aggregate_roi_metrics, method_differences, position_bootstrap
from .validation import find_image_manifest


def _float_range(row: pd.Series) -> float | None:
    value = row.get("data_range")
    return float(value) if value is not None and str(value).strip() and str(value).lower() != "nan" else None


def evaluate_rois(input_root: str | Path, registry_path: str | Path, output_root: str | Path,
                  require_locked_rois: bool = True, primary_only: bool = True,
                  resume: bool = False, bright_outlier_threshold: float = 0.10) -> dict[str, Any]:
    root, output = Path(input_root).resolve(), Path(output_root).resolve()
    registry = require_locked(registry_path) if require_locked_rois else pd.read_csv(registry_path, dtype={"sample_id": str, "position_id": str})
    assets = pd.read_csv(find_image_manifest(root), dtype={"sample_id": str, "position_id": str}, low_memory=False)
    if primary_only and "is_primary_seed" in assets: assets = assets[assets.is_primary_seed.astype(str).str.lower().isin({"true", "1"}) | assets.asset_role.isin(["noisy", "reference"])]
    metric_dir = output / "metrics"; metric_dir.mkdir(parents=True, exist_ok=True)
    result_path = metric_dir / "per_roi_metrics.csv"
    existing = pd.read_csv(result_path, dtype={"sample_id": str, "position_id": str}) if resume and result_path.is_file() else pd.DataFrame()
    done = set(zip(existing.get("roi_id", []), existing.get("method_id", []), existing.get("seed", [])))
    records, failures, crop_cache = [], [], {}
    try:
        from tqdm import tqdm
        iterator = tqdm(list(registry.itertuples(index=False)), desc="ROI evaluation")
    except ImportError:
        iterator = registry.itertuples(index=False)
    for roi in iterator:
        sample_assets = assets[assets.sample_id.astype(str) == str(roi.sample_id)]
        noisy_rows, reference_rows = sample_assets[sample_assets.asset_role == "noisy"], sample_assets[sample_assets.asset_role == "reference"]
        if len(noisy_rows) != 1 or len(reference_rows) != 1:
            failures.append({"roi_id": roi.roi_id, "sample_id": roi.sample_id, "method_id": "*", "failure": "noisy/reference is missing or ambiguous"}); continue
        try:
            noisy_row, ref_row = noisy_rows.iloc[0], reference_rows.iloc[0]
            noisy, noisy_meta = read_float01(root / noisy_row.packaged_path, _float_range(noisy_row))
            reference, ref_meta = read_float01(root / ref_row.packaged_path, _float_range(ref_row))
            if noisy.shape != reference.shape: raise ValueError("noisy/reference shape mismatch")
            if noisy_meta["sha256"] != str(roi.noisy_sha256) or ref_meta["sha256"] != str(roi.reference_sha256): raise ValueError("ROI registry source hash mismatch")
            x0, y0, x1, y1 = map(int, (roi.x0, roi.y0, roi.x1, roi.y1))
            if x0 < 0 or y0 < 0 or x1 > noisy.shape[1] or y1 > noisy.shape[0] or x1 - x0 != y1 - y0: raise ValueError("invalid ROI bounds")
            noisy_crop, reference_crop = noisy[y0:y1, x0:x1], reference[y0:y1, x0:x1]
        except Exception as exc:
            failures.append({"roi_id": roi.roi_id, "sample_id": roi.sample_id, "method_id": "*", "failure": f"{type(exc).__name__}: {exc}"}); continue
        method_rows = [noisy_row] + [row for _, row in sample_assets[sample_assets.asset_role == "method"].iterrows()]
        present = {"noisy_identity" if row.asset_role == "noisy" else str(row.method_id) for row in method_rows}
        for missing in set(METHOD_ORDER) - present:
            failures.append({"roi_id": roi.roi_id, "sample_id": roi.sample_id, "method_id": missing, "failure": "MISSING"})
        for method_row in method_rows:
            method = "noisy_identity" if method_row.asset_role == "noisy" else str(method_row.method_id)
            seed = int(method_row.get("seed", 0))
            already_done = (roi.roi_id, method, seed) in done
            try:
                if method == "noisy_identity": denoised, meta = noisy, noisy_meta
                else: denoised, meta = read_float01(root / method_row.packaged_path, _float_range(method_row))
                if denoised.shape != reference.shape: raise ValueError(f"shape mismatch {denoised.shape} != {reference.shape}; resize is forbidden")
                crop = denoised[y0:y1, x0:x1]
                crop_cache[(str(roi.sample_id), str(roi.tissue), method, seed, str(roi.roi_id))] = (crop, reference_crop)
                if already_done: continue
                values = compute_roi_metrics(noisy_crop, reference_crop, crop, str(roi.tissue), bright_outlier_threshold)
                records.append({"roi_id": roi.roi_id, "dataset": roi.dataset, "split": roi.split, "position_id": roi.position_id, "sample_id": roi.sample_id, "tissue": roi.tissue, "roi_size": int(roi.roi_size), "x0": x0, "y0": y0, "x1": x1, "y1": y1, "method_id": method, "seed": seed, "image_sha256": meta["sha256"], "analysis_scope": "fixed_roi", **values})
            except Exception as exc:
                failures.append({"roi_id": roi.roi_id, "sample_id": roi.sample_id, "method_id": method, "seed": seed, "failure": f"{type(exc).__name__}: {exc}"})
    combined = pd.concat([existing, pd.DataFrame(records)], ignore_index=True, sort=False).drop_duplicates(["roi_id", "method_id", "seed"], keep="last")
    combined.to_csv(result_path, index=False)
    sample, position = aggregate_roi_metrics(combined)
    sample.to_csv(metric_dir / "per_sample_tissue_metrics.csv", index=False)
    position.to_csv(metric_dir / "per_position_tissue_metrics.csv", index=False)
    differences = method_differences(position); differences.to_csv(metric_dir / "roi_method_differences.csv", index=False)
    total_positions = assets.position_id.nunique(); selected_positions = registry.position_id.nunique()
    formal_test = (
        not registry.empty
        and registry.dataset.astype(str).eq("PKU37").all()
        and registry.split.astype(str).eq("test").all()
    )
    # The locked PKU37 protocol contains six disjoint test positions. A downloaded
    # subset must never become "confirmatory" merely because every downloaded
    # position received an ROI.
    complete_formal_positions = formal_test and total_positions == 6 and selected_positions == total_positions
    bootstrap = position_bootstrap(position, complete_position_count=total_positions if complete_formal_positions else None)
    bootstrap.to_csv(metric_dir / "roi_bootstrap_confidence_intervals.csv", index=False)
    pd.DataFrame(failures).to_csv(metric_dir / "roi_failures.csv", index=False)
    cnr_rows = []
    for (sample_id, method, seed), group in combined[combined.tissue.isin(["choroid_vessel", "choroid_stroma"])].groupby(["sample_id", "method_id", "seed"]):
        vessel_ids = group[group.tissue == "choroid_vessel"].roi_id.astype(str)
        stroma_ids = group[group.tissue == "choroid_stroma"].roi_id.astype(str)
        vessel = [crop_cache[(sample_id, "choroid_vessel", method, int(seed), roi_id)] for roi_id in vessel_ids if (sample_id, "choroid_vessel", method, int(seed), roi_id) in crop_cache]
        stroma = [crop_cache[(sample_id, "choroid_stroma", method, int(seed), roi_id)] for roi_id in stroma_ids if (sample_id, "choroid_stroma", method, int(seed), roi_id) in crop_cache]
        if vessel and stroma:
            vc, vr = np.concatenate([x[0].ravel() for x in vessel]), np.concatenate([x[1].ravel() for x in vessel])
            sc, sr = np.concatenate([x[0].ravel() for x in stroma]), np.concatenate([x[1].ravel() for x in stroma])
            cnr_rows.append({"sample_id": sample_id, "method_id": method, "seed": seed, **vessel_stroma_pair_metrics(vc, sc, vr, sr)})
    pd.DataFrame(cnr_rows).to_csv(metric_dir / "vessel_stroma_cnr.csv", index=False)
    scope = "fixed_roi_confirmatory_full_test_positions" if complete_formal_positions else "exploratory_descriptive"
    return {"status": "completed_with_failures" if failures else "completed", "analysis_scope": scope, "rois": registry.roi_id.nunique(), "methods": combined.method_id.nunique(), "records": len(combined), "failures": len(failures), "selected_positions": selected_positions, "package_positions": total_positions, "per_roi_metrics": str(result_path)}
