#!/usr/bin/env python
"""Analyze signed noisy-minus-mild residual leakage on development positions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.dose_response import stable_sha
from sabids.experiments.protocol_lock import sha256_file
from sabids.experiments.seg_guided import audit_residual_identity, boundary_band, signed_residual


def _read(path: str, root: Path, mask: bool = False) -> np.ndarray:
    value = Path(path)
    value = value if value.is_absolute() else root / value
    if value.suffix.lower() == ".npy":
        array = np.load(value, allow_pickle=False)
    else:
        array = cv2.imread(str(value), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise FileNotFoundError(value)
        if array.ndim == 3:
            array = array[..., 0]
        if not mask and np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / np.iinfo(array.dtype).max
    return array > 0 if mask else array.astype(np.float32)


def _mean(value: np.ndarray, region: np.ndarray) -> float:
    return float(np.mean(value[region])) if region.any() else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--fixed-component-inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--intervention-results")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    resolve = lambda value: Path(value).resolve() if Path(value).is_absolute() else (root / value).resolve()
    manifest, inventory_path, output = map(resolve, (
        args.manifest, args.fixed_component_inventory, args.output,
    ))
    if output.exists():
        raise FileExistsError(f"Refusing existing residual report: {output}")
    table = pd.read_csv(manifest, dtype=str).fillna("")
    if not table["split"].isin(["train", "val"]).all():
        raise RuntimeError("BLOCKED: residual audit manifest contains sealed/non-development split")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8-sig"))
    members_by_sample: dict[str, list[dict]] = {}
    for member in inventory["members"]:
        members_by_sample.setdefault(str(member["sample_id"]), []).append(member)
    rows, component_rows = [], []
    for row in table[table.split.eq("val")].to_dict("records"):
        noisy = _read(row["noisy_path"], root)
        mild = _read(row["mild_path"], root)
        vessel = _read(row["vessel_mask_path"], root, True)
        layer_mask = _read(row["layer_mask_path"], root, True)
        valid = _read(row["spatial_valid_mask_path"], root, True)
        residual = signed_residual(noisy, mild)
        identity_error = audit_residual_identity(noisy, mild, residual)
        boundary = boundary_band(vessel, 2) & valid
        stroma = layer_mask & ~vessel & ~boundary & valid
        outside = ~layer_mask & valid
        abs_residual = np.abs(residual)
        gy, gx = np.gradient(noisy.astype(np.float32))
        gradient = np.sqrt(gx * gx + gy * gy)
        finite = valid & np.isfinite(abs_residual) & np.isfinite(gradient)
        correlation = float(np.corrcoef(abs_residual[finite], gradient[finite])[0, 1]) if finite.sum() > 2 else float("nan")
        rows.append({
            "sample_id": row["sample_id"], "group_id": row["group_id"],
            "residual_identity_max_abs": identity_error,
            "residual_signed_mean": _mean(residual, valid),
            "residual_abs_mean": _mean(abs_residual, valid),
            "vessel_abs_residual": _mean(abs_residual, vessel & valid),
            "boundary_abs_residual": _mean(abs_residual, boundary),
            "stroma_abs_residual": _mean(abs_residual, stroma),
            "outside_abs_residual": _mean(abs_residual, outside),
            "residual_gradient_correlation": correlation,
            "removed_vessel_fraction": float(((residual > 0) & vessel & valid).sum() / max(1, (vessel & valid).sum())),
        })
        labels, count = label(vessel & valid, structure=np.ones((3, 3), np.uint8))
        expected = {int(item["component_id"]): item for item in members_by_sample.get(str(row["sample_id"]), [])}
        if set(expected) != set(range(1, count + 1)):
            raise ValueError(f"Fixed component membership mismatch: {row['sample_id']}")
        for component_id, member in sorted(expected.items()):
            component = labels == component_id
            component_rows.append({
                "sample_id": row["sample_id"], "group_id": row["group_id"],
                "component_id": component_id, "size_bin": member["size_bin"],
                "contrast_bin": member["contrast_bin"],
                "residual_abs_mean": _mean(abs_residual, component),
                "residual_positive_fraction": float((residual[component] > 0).mean()),
                "residual_energy": float(np.square(residual[component]).mean()),
            })
    output.mkdir(parents=True)
    frames = pd.DataFrame(rows)
    components = pd.DataFrame(component_rows)
    frames.to_csv(output / "RESIDUAL_ANALYSIS.csv", index=False)
    components.to_csv(output / "RESIDUAL_COMPONENTS.csv", index=False)
    position = frames.groupby("group_id", as_index=False).mean(numeric_only=True)
    position.to_csv(output / "RESIDUAL_BY_POSITION.csv", index=False)
    correlations = []
    if args.intervention_results:
        intervention = pd.read_csv(resolve(args.intervention_results))
        last = intervention[intervention.selection.eq("last")]
        pivot = last.pivot_table(index="group_id", columns="condition", values="vessel_dice")
        joined = position.set_index("group_id").join(pivot)
        for delta_name, left, right in (("B1_minus_B0", "T2", "T1"), ("B3_minus_B0", "T2", "T0")):
            if {left, right}.issubset(joined.columns):
                delta = joined[left] - joined[right]
                correlations.append({
                    "comparison": delta_name,
                    "residual_leakage_metric": "vessel_abs_residual",
                    "pearson": float(joined["vessel_abs_residual"].corr(delta)),
                    "position_count": int(delta.notna().sum()),
                })
    pd.DataFrame(correlations).to_csv(output / "RESIDUAL_OUTCOME_CORRELATIONS.csv", index=False)
    result = {
        "status": "passed", "version": "seg-guided-adaptive-v1",
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "fixed_component_inventory_sha256": sha256_file(inventory_path),
        "frame_count": len(frames), "position_count": int(frames.group_id.nunique()),
        "max_residual_identity_error": float(frames.residual_identity_max_abs.max()),
        "records_sha256": stable_sha(rows), "test_assets_opened": 0,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
