#!/usr/bin/env python
"""Prepare and optionally run same-checkpoint B3 auxiliary-input interventions."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

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
    INTERVENTIONS, array_sha256, block_shuffle, destroy_spatial_structure,
    deterministic_wrong_guides, shift_without_wrap,
)


def _write_once(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != payload:
            raise FileExistsError(f"Existing intervention artifact differs: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def _save_array(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not np.array_equal(np.load(path, allow_pickle=False), value):
            raise FileExistsError(f"Existing intervention cache differs: {path}")
        return
    with path.open("xb") as handle:
        np.save(handle, np.asarray(value, dtype=np.float32), allow_pickle=False)


def _resolve_asset(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--b3-run", required=True)
    parser.add_argument("--fixed-component-inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--wrong-repetitions", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    resolve = lambda value: Path(value).resolve() if Path(value).is_absolute() else (root / value).resolve()
    run, inventory_path, output = map(resolve, (
        args.b3_run, args.fixed_component_inventory, args.output,
    ))
    if output.exists():
        raise FileExistsError(f"Refusing existing intervention report: {output}")
    config_path = run / "resolved_config.yaml"
    config = load_config(config_path)
    if config.get("dual_view", {}).get("arm") != "B3":
        raise ValueError("Interventions require a trained B3 run")
    source_path = resolve(config["data"]["manifest"])
    source = pd.read_csv(source_path, dtype=str).fillna("")
    if not source["split"].isin(["train", "val"]).all():
        raise RuntimeError("BLOCKED: intervention source contains non-development split")
    if not {"noisy_path", "mild_path", "group_id", "sample_id"}.issubset(source.columns):
        raise ValueError("B3 manifest lacks noisy/mild identity columns")
    rows = source.to_dict("records")
    wrong = deterministic_wrong_guides(rows, args.seed, args.wrong_repetitions)
    wrong_lookup = {(row["sample_id"], row["repetition"]): row for row in wrong}
    by_id = {str(row["sample_id"]): row for row in rows}
    output.mkdir(parents=True)
    manifest_root = output / "manifests"
    cache_root = output / "cache"
    registry_rows = []
    manifests: dict[str, Path] = {}
    conditions = ["T0", "T1", "T2", *(f"T3_r{i}" for i in range(args.wrong_repetitions)),
                  "T4a", "T4b", "T5", "T6"]
    for condition in conditions:
        prepared = []
        for row in rows:
            value = dict(row)
            value["primary_path"] = row["noisy_path"]
            source_aux_id = str(row["sample_id"])
            transform = "identity"
            if condition == "T0":
                value["auxiliary_path"] = row["mild_path"]
                transform = "disabled"
            elif condition == "T1":
                value["auxiliary_path"] = row["noisy_path"]
                transform = "same_noisy"
            elif condition == "T2":
                value["auxiliary_path"] = row["mild_path"]
                transform = "correct_mild"
            elif condition.startswith("T3_r"):
                repetition = int(condition.split("r")[-1])
                mapping = wrong_lookup[(str(row["sample_id"]), repetition)]
                source_aux_id = mapping["auxiliary_sample_id"]
                value["auxiliary_path"] = by_id[source_aux_id]["mild_path"]
                transform = "same_split_different_group"
            else:
                mild = np.load(_resolve_asset(root, row["mild_path"]), allow_pickle=False).astype(np.float32)
                if condition == "T4a":
                    auxiliary, transform = shift_without_wrap(mild, 16), "shift_right_16px"
                elif condition == "T4b":
                    auxiliary, transform = shift_without_wrap(mild, 32), "shift_right_32px"
                elif condition == "T5":
                    auxiliary, transform = block_shuffle(mild, 32, args.seed, str(row["sample_id"])), "block_shuffle_32px"
                elif condition == "T6":
                    auxiliary, transform = destroy_spatial_structure(mild, args.seed, str(row["sample_id"])), "histogram_matched_spatial_destroy"
                else:
                    raise AssertionError(condition)
                path = cache_root / condition / str(row["split"]) / f"{row['sample_id']}.npy"
                _save_array(path, auxiliary)
                value["auxiliary_path"] = str(path)
            value["intervention"] = condition
            value["auxiliary_sample_id"] = source_aux_id
            prepared.append(value)
            if str(row["split"]) == "val":
                registry_rows.append({
                    "condition": condition, "sample_id": str(row["sample_id"]),
                    "group_id": str(row["group_id"]), "split": "val",
                    "auxiliary_sample_id": source_aux_id, "transform": transform,
                    "auxiliary_path": value["auxiliary_path"],
                    "auxiliary_sha256": array_sha256(
                        np.load(_resolve_asset(root, value["auxiliary_path"]), allow_pickle=False)
                    ),
                })
        path = manifest_root / f"{condition.lower()}.csv"
        _write_once(path, pd.DataFrame(prepared).to_csv(index=False, lineterminator="\n"))
        manifests[condition] = path
    registry = {
        "version": "seg-guided-adaptive-v1", "status": "prepared",
        "b3_run": str(run), "source_manifest": str(source_path),
        "source_manifest_sha256": sha256_file(source_path), "seed": args.seed,
        "wrong_repetitions": args.wrong_repetitions,
        "conditions": conditions, "records": registry_rows,
        "records_sha256": stable_sha(registry_rows),
        "checkpoint_sha256": {
            selection: sha256_file(run / f"{selection}.pth") for selection in ("last", "best")
        },
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output / "intervention_registry.json", registry)
    if not args.execute:
        print(json.dumps(registry, ensure_ascii=False, indent=2))
        return
    metric_tables = []
    for selection in ("last", "best"):
        for condition in conditions:
            condition_config = load_config(config_path)
            condition_config.pop("runtime", None)
            condition_config["data"].update(
                manifest=str(manifests[condition]), root=str(root),
                auxiliary_mode="column", auxiliary_input_column="auxiliary_path",
            )
            condition_config["dual_view"].update(
                intervention=condition, intervention_registry=str(output / "intervention_registry.json")
            )
            generated_config = output / "configs" / f"{selection}_{condition.lower()}.yaml"
            _write_once(generated_config, yaml.safe_dump(condition_config, sort_keys=False, allow_unicode=True))
            evaluation = output / "evaluations" / selection / condition
            command = [
                sys.executable, str(root / "evaluate.py"), "--config", str(generated_config),
                "--checkpoint", str(run / f"{selection}.pth"), "--split", "val",
                "--output", str(evaluation), "--tasks", "layer", "vessel",
                "--postprocess-modes", "p0", "--layer-threshold", "0.5",
                "--vessel-threshold", "0.5", "--no-restore-original-geometry",
                "--fixed-component-inventory", str(inventory_path),
                "--capture-dual-diagnostics", "--save-predictions",
            ]
            if condition == "T0":
                command.append("--disable-dual-view-auxiliary")
            subprocess.run(command, cwd=root, check=True)
            table = pd.read_csv(evaluation / "group_metrics.csv")
            table.insert(0, "condition", condition.split("_r")[0])
            table.insert(1, "repetition", int(condition.split("_r")[-1]) if "_r" in condition else 0)
            table.insert(2, "selection", selection)
            metric_tables.append(table)
    metrics = pd.concat(metric_tables, ignore_index=True)
    numeric = metrics.select_dtypes(include=[np.number]).columns.difference(["repetition"])
    # Wrong-guide repetitions are averaged inside each anatomical position.
    grouped = metrics.groupby(["selection", "condition", "group_id"], as_index=False)[list(numeric)].mean()
    grouped.to_csv(output / "INTERVENTION_RESULTS.csv", index=False)
    primary = grouped[grouped.selection.eq("last")]
    pivot = primary.pivot_table(index="group_id", columns="condition", values="vessel_dice")
    comparisons = {}
    checks = {}
    for control in ("T0", "T1", "T3", "T4a", "T4b", "T5"):
        delta = pivot["T2"] - pivot[control]
        comparisons[f"T2_minus_{control}"] = {
            "mean": float(delta.mean()), "improved_positions": int((delta > 0).sum()),
            "position_count": int(delta.notna().sum()),
        }
        if control in {"T0", "T1", "T3", "T4a", "T4b"}:
            checks[f"T2_gt_{control}"] = float(delta.mean()) > 0
    passed = all(checks.values())
    result = {
        **registry, "status": "passed" if passed else "failed",
        "content_attribution_supported": passed, "checks": checks,
        "comparisons": comparisons, "test_assets_opened": 0,
    }
    (output / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
