#!/usr/bin/env python
"""Create fail-closed provenance evidence for an explicit frozen D2 teacher."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from sabids.config import load_config
from sabids.experiments.d2_teacher import audit_teacher_binding
from sabids.engine.trainer import build_model
from sabids.experiments.dose_response import resolve, stable_sha, write_strict_json_exclusive
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file, validate_checkpoint_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--selection-rule", help="Optional expected derived selection rule")
    parser.add_argument("--training-data", help="Optional expected derived training-cohort identity")
    parser.add_argument("--checkpoint-binding")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    checkpoint = resolve(root, args.checkpoint)
    config_path = resolve(root, args.resolved_config)
    lock_path = resolve(root, args.protocol_lock)
    lock = load_protocol_lock(lock_path)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not raw.get("model") or not raw.get("config"):
        raise ValueError("Teacher checkpoint lacks model/config")
    binding = None
    evidence_type = "unverified"
    if args.checkpoint_binding:
        binding_path = resolve(root, args.checkpoint_binding)
        binding = audit_teacher_binding(binding_path, checkpoint)
        evidence_type = binding["evidence_type"]
        if binding.get("protocol_id") != lock["protocol_id"]:
            raise ValueError("Teacher binding protocol differs from active lock")
        for key in ("data_plan_sha256", "label_inventory_sha256",
                    "dataset_inventory_sha256", "split_contract_sha256"):
            if binding.get("protocol_values", {}).get(key) != lock[key]:
                raise ValueError(f"Teacher binding protocol lock mismatch: {key}")
    if evidence_type == "native_protocol_binding" or binding is None:
        validate_checkpoint_config(raw, lock, "D2 teacher")
    config = load_config(config_path)
    if config.get("formal_d2_teacher", {}).get("enabled") is True and binding is None:
        raise ValueError("Formal D2 teacher requires an immutable checkpoint binding")
    for section in (
        "model", "data", "train", "loss", "seed", "protocol_id", "label_type",
        "formal_d2_teacher", "training_asset_evidence",
    ):
        if raw["config"].get(section) != config.get(section):
            raise ValueError(f"Teacher checkpoint/resolved config mismatch: {section}")
    with torch.random.fork_rng(devices=[]):
        teacher = build_model(config)
    teacher.load_state_dict(raw["model"], strict=True)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    requires_grad = sum(parameter.numel() for parameter in teacher.parameters()
                        if parameter.requires_grad)
    if requires_grad:
        raise RuntimeError("Frozen teacher audit found trainable parameters")
    if config.get("train", {}).get("stage") not in {
        "segment", "interaction", "input_segment", "joint"
    }:
        raise ValueError("Teacher was not trained by a segmentation-capable stage")
    manifest = resolve(root, config["data"]["manifest"])
    table = pd.read_csv(manifest, dtype=str).fillna("")
    if not table["split"].isin(["train", "val"]).all():
        raise ValueError("Teacher manifest contains a non-development split")
    train = table[table["split"].eq(config["data"].get("train_split", "train"))]
    val = table[table["split"].eq(config["data"].get("val_split", "val"))]
    if config["data"].get("train_groups"):
        train = train[train["group_id"].isin(map(str, config["data"]["train_groups"]))]
    if config["data"].get("val_groups"):
        val = val[val["group_id"].isin(map(str, config["data"]["val_groups"]))]
    if (set(train["group_id"]) != set(lock["train_positions"])
            or set(val["group_id"]) != set(lock["validation_positions"])):
        raise ValueError("Teacher effective train/validation groups differ from protocol")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint.name == "best.pth":
        metadata_path = checkpoint.parent / "run_metadata.json"
        if not metadata_path.is_file():
            raise ValueError("Teacher best.pth lacks run_metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata.get("best_checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Teacher run metadata does not bind best.pth")
        if int(metadata.get("best_epoch", -1)) != int(raw.get("epoch", -1)) + 1:
            raise ValueError("Teacher run metadata best epoch mismatch")
        monitor = str(metadata.get("monitor", ""))
        if not monitor or monitor != str(config.get("train", {}).get("monitor", "")):
            raise ValueError("Teacher run metadata monitor mismatch")
        recorded_checkpoint = Path(str(metadata.get("best_checkpoint", ""))).expanduser()
        if not recorded_checkpoint.is_absolute():
            recorded_checkpoint = (root / recorded_checkpoint).resolve()
        else:
            recorded_checkpoint = recorded_checkpoint.resolve()
        if recorded_checkpoint != checkpoint:
            raise ValueError("Teacher run metadata points to a different best checkpoint")
        source_selection = f"best_validation_{monitor}"
    elif checkpoint.name == "last.pth":
        source_selection = "fixed_final"
        if int(raw.get("epoch", -1)) + 1 != int(config["train"]["epochs"]):
            raise ValueError("Teacher last.pth is not the configured final epoch")
    else:
        raise ValueError("Teacher checkpoint must be best.pth or last.pth")
    cohort = {
        "protocol_id": lock["protocol_id"],
        "manifest_sha256": sha256_file(manifest),
        "train_groups": sorted(set(train["group_id"])),
        "validation_groups": sorted(set(val["group_id"])),
    }
    training_data = f"teacher-development-cohort:{stable_sha(cohort)}"
    if args.selection_rule and args.selection_rule != source_selection:
        raise ValueError(
            f"Teacher selection rule mismatch: expected {args.selection_rule}, derived {source_selection}"
        )
    if args.training_data and args.training_data != training_data:
        raise ValueError("Teacher training-cohort identity mismatch")
    result = {
        "schema_version": "d2-frozen-teacher-evidence-v2", "status": "passed",
        "evidence_type": evidence_type,
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(raw.get("epoch", -1)) + 1,
        "resolved_config_path": str(config_path), "resolved_config_sha256": sha256_file(config_path),
        "manifest_path": str(manifest), "manifest_sha256": sha256_file(manifest),
        "protocol_id": lock["protocol_id"], "protocol_lock_sha256": sha256_file(lock_path),
        "selection_rule": source_selection, "source_monitor_or_rule": source_selection,
        "training_data": training_data, "training_data_definition": cohort,
        "train_groups": sorted(set(train["group_id"])),
        "validation_groups": sorted(set(val["group_id"])), "split": "development_train_val",
        "parameter_count": sum(parameter.numel() for parameter in teacher.parameters()),
        "requires_grad_parameter_count": requires_grad,
        "changed_parameter_count": 0,
        "checkpoint_binding": str(binding_path) if binding is not None else None,
        "checkpoint_binding_sha256": sha256_file(binding_path) if binding is not None else None,
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(resolve(root, args.output), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
