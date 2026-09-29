#!/usr/bin/env python
"""Build development-only spatial-dose oracle inputs and optionally evaluate B3.

Every output is labelled non-deployable.  The tool rejects non-development
rows before opening any image and never discovers sealed assets.
"""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.experiments.dose_response import stable_sha, write_strict_json_exclusive
from sabids.experiments.protocol_lock import sha256_file
from sabids.experiments.seg_guided import (
    apply_spatial_dose, deterministic_wrong_guides, oracle_gate,
    random_histogram_matched_map, spatial_alpha_map,
)


WARNING = "ORACLE / USES VALIDATION GT / NOT A DEPLOYABLE PERFORMANCE ESTIMATE"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read(root: Path, value: str, mask: bool = False) -> np.ndarray:
    path = _resolve(root, value)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if array is None:
            raise FileNotFoundError(path)
        if array.ndim == 3:
            array = array[..., 0]
        if not mask and np.issubdtype(array.dtype, np.integer):
            array = array.astype(np.float32) / np.iinfo(array.dtype).max
    return array > 0 if mask else array.astype(np.float32)


def _save_once(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), value):
            raise FileExistsError(f"Existing oracle cache differs: {path}")
        return
    with path.open("xb") as handle:
        np.save(handle, value.astype(np.float32), allow_pickle=False)


def _write_once(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != text:
            raise FileExistsError(f"Existing oracle artifact differs: {path}")
        return
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--b3-run", required=True)
    parser.add_argument("--fixed-component-inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    manifest = _resolve(root, args.manifest)
    b3_run = _resolve(root, args.b3_run)
    inventory = _resolve(root, args.fixed_component_inventory)
    output = _resolve(root, args.output)
    if output.exists():
        raise FileExistsError(f"Refusing existing oracle report: {output}")
    table = pd.read_csv(manifest, dtype=str).fillna("")
    required = {"sample_id", "group_id", "split", "noisy_path", "strong_path",
                "layer_mask_path", "vessel_mask_path", "spatial_valid_mask_path"}
    if missing := required - set(table.columns):
        raise ValueError(f"Oracle manifest missing columns: {sorted(missing)}")
    if not table["split"].isin(("train", "val")).all():
        raise RuntimeError("BLOCKED: oracle source contains sealed/non-development rows")
    # Stage D1 is a same-checkpoint validation intervention.  Do not materialize
    # the full grid for train rows (roughly 80 GiB for the current protocol).
    # A later D2 oracle-aware training protocol must generate fold-train inputs
    # on demand after locking a train-selected combination.
    rows = table[table.split.eq("val")].to_dict("records")
    if not rows:
        raise ValueError("Oracle intervention has no validation rows")
    wrong = deterministic_wrong_guides(rows, args.seed, 1)
    wrong_by_id = {item["sample_id"]: item["auxiliary_sample_id"] for item in wrong}
    by_id = {str(row["sample_id"]): row for row in rows}
    grid = list(itertools.product((0.0, 0.1), (0.0, 0.1), (0.25, 0.5), (0.25, 0.5)))
    primary = (0.0, 0.0, 0.25, 0.5)
    condition_tables: dict[str, list[dict]] = {}
    registry_rows = []
    for row in rows:
        sample = str(row["sample_id"])
        noisy = _read(root, row["noisy_path"])
        strong = _read(root, row["strong_path"])
        layer = _read(root, row["layer_mask_path"], True)
        vessel = _read(root, row["vessel_mask_path"], True)
        valid = _read(root, row["spatial_valid_mask_path"], True)
        other = by_id[wrong_by_id[sample]]
        other_layer = _read(root, other["layer_mask_path"], True)
        other_vessel = _read(root, other["vessel_mask_path"], True)
        conditions: list[tuple[str, np.ndarray, dict]] = [
            ("O0", np.zeros_like(noisy, np.float32), {}),
            ("O1", np.full_like(noisy, 0.25, np.float32) * valid, {}),
        ]
        for av, ab, ast, ao in grid:
            combo = f"av{av:.2f}_ab{ab:.2f}_as{ast:.2f}_ao{ao:.2f}".replace(".", "p")
            full, regions = spatial_alpha_map(layer, vessel, valid, av, ab, ast, ao, 2)
            layer_only, _ = spatial_alpha_map(layer, np.zeros_like(vessel), valid, ast, ast, ast, ao, 2)
            vessel_only, _ = spatial_alpha_map(np.ones_like(layer), vessel, valid, av, ab, ast, ast, 2)
            conditions.extend(((f"O2_{combo}", layer_only, regions),
                               (f"O3_{combo}", vessel_only, regions),
                               (f"O4_{combo}", full, regions),
                               (f"O5_{combo}", random_histogram_matched_map(full, valid, args.seed, sample + combo), regions)))
            wrong_map, _ = spatial_alpha_map(other_layer, other_vessel, valid, av, ab, ast, ao, 2)
            conditions.append((f"O6_{combo}", wrong_map, regions))
            for radius in (1, 4):
                sensitivity, _ = spatial_alpha_map(layer, vessel, valid, av, ab, ast, ao, radius)
                conditions.append((f"O7_r{radius}_{combo}", sensitivity, regions))
        for condition, alpha, regions in conditions:
            image = apply_spatial_dose(noisy, strong, alpha)
            cache = output / "cache" / condition / str(row["split"]) / f"{sample}.npy"
            _save_once(cache, image)
            item = dict(row)
            item.update(primary_path=row["noisy_path"], auxiliary_path=str(cache), oracle_condition=condition)
            condition_tables.setdefault(condition, []).append(item)
            registry_rows.append({"sample_id": sample, "group_id": row["group_id"],
                                  "split": row["split"], "condition": condition,
                                  "alpha_min": float(alpha.min()), "alpha_max": float(alpha.max()),
                                  "alpha_mean": float(alpha[valid].mean()), **regions})
    manifests = {}
    output.mkdir(parents=True, exist_ok=True)
    for condition, prepared in condition_tables.items():
        path = output / "manifests" / f"{condition.lower()}.csv"
        _write_once(path, pd.DataFrame(prepared).to_csv(index=False, lineterminator="\n"))
        manifests[condition] = path
    registry = {"status": "prepared", "warning": WARNING, "manifest": str(manifest),
                "manifest_sha256": sha256_file(manifest), "grid": grid,
                "primary_combo": primary, "scope": "D1_same_checkpoint_validation_only",
                "oracle_aware_retraining": "NOT_IMPLEMENTED_USE_STREAMED_FOLD_TRAIN_PROTOCOL",
                "records_sha256": stable_sha(registry_rows),
                "records": registry_rows, "test_assets_opened": 0}
    write_strict_json_exclusive(output / "oracle_registry.json", registry)
    if not args.execute:
        print(json.dumps({key: value for key, value in registry.items() if key != "records"}, indent=2))
        return
    config_path = b3_run / "resolved_config.yaml"
    base = load_config(config_path)
    metrics = []
    for selection in ("last", "best"):
        for condition, condition_manifest in manifests.items():
            config = load_config(config_path)
            config.pop("runtime", None)
            config["data"].update(manifest=str(condition_manifest), root=str(root),
                                  auxiliary_mode="column", auxiliary_input_column="auxiliary_path")
            config.setdefault("oracle", {}).update(warning=WARNING, condition=condition)
            cfg = output / "configs" / f"{selection}_{condition.lower()}.yaml"
            _write_once(cfg, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            target = output / "evaluations" / selection / condition
            subprocess.run([sys.executable, str(root / "evaluate.py"), "--config", str(cfg),
                            "--checkpoint", str(b3_run / f"{selection}.pth"), "--split", "val",
                            "--output", str(target), "--tasks", "layer", "vessel",
                            "--postprocess-modes", "p0", "--layer-threshold", "0.5",
                            "--vessel-threshold", "0.5", "--no-restore-original-geometry",
                            "--fixed-component-inventory", str(inventory), "--capture-dual-diagnostics"],
                           cwd=root, check=True)
            part = pd.read_csv(target / "group_metrics.csv")
            part.insert(0, "condition", condition)
            part.insert(1, "arm", condition.split("_")[0])
            part.insert(2, "selection", selection)
            metrics.append(part)
    results = pd.concat(metrics, ignore_index=True)
    results.to_csv(output / "ORACLE_RESULTS.csv", index=False)
    primary_suffix = "av0p00_ab0p00_as0p25_ao0p50"
    primary_rows = results[(results.selection == "last") &
                           results.condition.isin(["O1", *(f"{arm}_{primary_suffix}" for arm in ("O4", "O5", "O6"))])].copy()
    primary_rows["arm"] = primary_rows["condition"].str.split("_").str[0]
    # Reuse an existing difficult-component aggregate when present; never invent values.
    candidates = [column for column in primary_rows if "small_low" in column and "recall" in column]
    if not candidates:
        candidates = [column for column in primary_rows if "component" in column and "recall" in column]
    if not candidates:
        raise RuntimeError("Oracle gate requires a fixed difficult-component recall metric")
    primary_rows["difficult_component_recall"] = primary_rows[candidates[0]]
    gate = oracle_gate(primary_rows)
    gate["warning"] = WARNING
    (output / "oracle_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
