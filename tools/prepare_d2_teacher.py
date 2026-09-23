#!/usr/bin/env python
"""Preflight and prepare a fresh protocol-native D2 segmentation teacher."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config, save_config
from sabids.experiments.d2_teacher import (
    filtered_teacher_manifest,
    validate_teacher_cohort,
)
from sabids.experiments.dose_response import (
    effective_split_sha,
    formal_preflight,
    resolve,
    write_strict_json_exclusive,
)
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=("preflight", "overfit", "formal"), required=True)
    parser.add_argument(
        "--template", default="configs/adaptive_denoising/d2_teacher_pku37_v3_seed42.yaml"
    )
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--d1-checkpoint", required=True)
    parser.add_argument("--d1-training-asset-inventory", required=True)
    parser.add_argument("--d1-checkpoint-binding", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True, help="Preflight JSON or generated YAML")
    parser.add_argument("--run-dir", help="Required for overfit/formal config generation")
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    root = Path(args.project_root).expanduser().resolve()
    lock_path = resolve(root, args.protocol_lock)
    split_path = resolve(root, args.split_contract)
    d1_checkpoint = resolve(root, args.d1_checkpoint)
    output = resolve(root, args.output)
    lock = load_protocol_lock(lock_path)
    d1_report = formal_preflight(
        root,
        str(d1_checkpoint),
        str(lock_path),
        str(split_path),
        "best_validation_psnr",
        str(resolve(root, args.d1_training_asset_inventory)),
        str(resolve(root, args.d1_checkpoint_binding)),
    )
    if d1_report.get("status") != "passed":
        raise RuntimeError("D1-best formal preflight failed: " + "; ".join(d1_report["issues"]))

    config = load_config(resolve(root, args.template))
    config["device"] = args.device
    config["protocol_id"] = lock["protocol_id"]
    for key in ("manifest_root", "data_plan_sha256", "label_inventory_sha256"):
        config[key] = lock[key]
    config["data"]["manifest"] = str(
        (resolve(root, lock["manifest_root"]) / "train_segment.csv").resolve()
    )
    config["data"]["root"] = str(root)
    config["train"]["pretrained"] = str(d1_checkpoint)
    config["train"]["resume"] = None
    config["training_asset_evidence"].update({
        "project_root": str(root),
        "protocol_lock": str(lock_path),
    })
    config["formal_d2_teacher"].update({
        "template_only": False,
        "run_mode": args.mode,
        "split_contract": str(split_path),
    })
    table = pd.read_csv(config["data"]["manifest"], dtype=str).fillna("")
    filtered = filtered_teacher_manifest(config, table)
    train_groups = sorted(filtered.loc[filtered["split"].eq("train"), "group_id"].unique())
    val_groups = sorted(filtered.loc[filtered["split"].eq("val"), "group_id"].unique())
    cohort = validate_teacher_cohort(train_groups, val_groups, lock)
    config["formal_d2_teacher"].update({
        "expected_train_positions": train_groups,
        "expected_validation_positions": val_groups,
    })
    runtime = config.setdefault("runtime", {})
    runtime.update({
        "active_protocol_lock": lock,
        "active_protocol_lock_path": str(lock_path),
        "manifest_sha256": sha256_file(Path(config["data"]["manifest"])),
        "effective_split_sha256": effective_split_sha(filtered),
        "formal_d2_teacher_split_contract": str(split_path),
        "formal_d2_teacher_split_contract_sha256": sha256_file(split_path),
        "d1_checkpoint_binding": str(resolve(root, args.d1_checkpoint_binding)),
        "d1_checkpoint_binding_sha256": sha256_file(
            resolve(root, args.d1_checkpoint_binding)
        ),
        "test_assets_opened": 0,
    })
    report = {
        "status": "passed",
        "mode": args.mode,
        "protocol_id": lock["protocol_id"],
        "protocol_lock_sha256": sha256_file(lock_path),
        "split_contract_sha256": sha256_file(split_path),
        "d1_checkpoint_sha256": sha256_file(d1_checkpoint),
        "d1_preflight_status": d1_report["status"],
        "teacher_manifest_sha256": runtime["manifest_sha256"],
        "teacher_effective_split_sha256": runtime["effective_split_sha256"],
        "train_positions": train_groups,
        "validation_positions": val_groups,
        "protocol_train_positions": cohort["protocol_train_positions"],
        "teacher_train_subset_of_protocol": cohort["teacher_train_subset_of_protocol"],
        "selection_rule": "best_validation_vessel_soft_dice",
        "test_assets_opened": 0,
    }
    if args.mode == "preflight":
        write_strict_json_exclusive(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if not args.run_dir:
        raise ValueError("--run-dir is required for overfit/formal preparation")
    run_dir = resolve(root, args.run_dir)
    if output.exists() or run_dir.exists():
        raise FileExistsError(f"Refusing existing teacher config/run: {output} / {run_dir}")
    config["train"]["output_dir"] = str(run_dir)
    if args.mode == "overfit":
        config["train"].update({
            "epochs": 12,
            "early_stopping_patience": 13,
            "batch_size": 1,
            "gradient_accumulation_steps": 1,
            "num_workers": 0,
            "train_eval_every": 0,
        })
        config["data"].update({
            "train_groups": train_groups[:2],
            "val_groups": val_groups[:1],
            "samples_per_epoch": 8,
        })
    else:
        config["data"].pop("train_groups", None)
        config["data"].pop("val_groups", None)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_config(config, output)
    report.update({
        "generated_config": str(output),
        "run_dir": str(run_dir),
        "command": f"python train.py --config {output}",
    })
    report_path = output.with_suffix(output.suffix + ".preflight.json")
    write_strict_json_exclusive(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
