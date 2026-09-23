"""Fail-closed provenance for the frozen segmentation teacher used by D2.

Native bindings are produced by a fresh, explicitly registered teacher run.
Legacy bindings are accepted only when a separate training-time protocol record
already contains every locked identity; a protocol ID added after training is
not evidence.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from sabids.config import load_config
from sabids.experiments.dose_response import (
    asset_inventory,
    effective_split_sha,
    resolve,
    stable_sha,
    write_strict_json_exclusive,
)
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file


TEACHER_BINDING_SCHEMA = "d2-teacher-checkpoint-binding-v2"
TEACHER_EVIDENCE_TYPES = {"native_protocol_binding", "derived_legacy_binding"}
LOCK_KEYS = (
    "protocol_id",
    "data_plan_sha256",
    "label_inventory_sha256",
    "dataset_inventory_sha256",
    "split_contract_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict:
    _require(path.is_file(), f"Missing evidence source: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _normal_positions(values: Any) -> list[str]:
    if values is None:
        return []
    return sorted({str(value) for value in values})


def audit_teacher_history(path: Path, configured_epochs: int) -> dict:
    """Strictly audit a rectangular, complete teacher selection history."""
    _require(path.is_file(), f"Missing teacher history: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle, strict=True))
    _require(bool(rows), "Teacher history is empty")
    header, data = rows[0], rows[1:]
    widths = Counter(len(row) for row in data)
    _require(len(widths) == 1 and next(iter(widths), 0) == len(header),
             "Teacher history schema drift prevents trusted best-epoch selection")
    _require(len(data) == int(configured_epochs),
             "Teacher history is shorter than the configured fixed budget")
    for name in ("epoch", "val_vessel_soft_dice"):
        _require(header.count(name) == 1, f"Teacher history requires one {name} column")
    epoch_index = header.index("epoch")
    metric_index = header.index("val_vessel_soft_dice")
    epochs: list[int] = []
    metrics: list[float] = []
    for number, row in enumerate(data, start=2):
        try:
            epoch_float = float(row[epoch_index])
            metric = float(row[metric_index])
        except (TypeError, ValueError, IndexError) as error:
            raise ValueError(f"Teacher history has an invalid value at CSV row {number}") from error
        _require(np.isfinite(epoch_float) and epoch_float.is_integer(),
                 f"Teacher history epoch is invalid at CSV row {number}")
        _require(np.isfinite(metric),
                 f"Teacher history vessel soft Dice is non-finite at CSV row {number}")
        epochs.append(int(epoch_float)); metrics.append(metric)
    _require(epochs == list(range(1, int(configured_epochs) + 1)),
             "Teacher history epochs must be complete, unique, and ordered 1...configured_epochs")
    maximum = max(metrics)
    best_index = metrics.index(maximum)
    return {
        "history_sha256": sha256_file(path),
        "configured_epochs": int(configured_epochs),
        "history_row_count": len(data),
        "history_header_width": len(header),
        "history_row_width_distribution": {
            str(width): int(count) for width, count in sorted(widths.items())
        },
        "history_schema_drift_detected": False,
        "selection_rule": "best_validation_vessel_soft_dice",
        "selection_metric": "val_vessel_soft_dice",
        "best_epoch": epochs[best_index],
        "best_value": maximum,
        "first_maximum_tie_policy": True,
    }


def filtered_teacher_manifest(config: dict, table: pd.DataFrame) -> pd.DataFrame:
    _require("split" in table and "group_id" in table,
             "Teacher manifest lacks split/group_id")
    _require(table["split"].astype(str).isin(["train", "val"]).all(),
             "Teacher manifest references a non-development split")
    parts = []
    for role in ("train", "val"):
        split = str(config["data"].get(f"{role}_split", role))
        part = table[table["split"].astype(str).eq(split)].copy()
        datasets = config["data"].get(f"{role}_datasets")
        groups = config["data"].get(f"{role}_groups")
        if datasets:
            part = part[part["dataset"].astype(str).isin(map(str, datasets))]
        if groups:
            part = part[part["group_id"].astype(str).isin(map(str, groups))]
        part["split"] = role
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    _require(set(result["split"].astype(str)) == {"train", "val"},
             "Teacher effective cohort needs train and validation rows")
    return result


def validate_teacher_cohort(
    train: Any,
    validation: Any,
    lock: dict,
    *,
    expected_train: Any | None = None,
    expected_validation: Any | None = None,
) -> dict:
    """Validate the label-eligible teacher cohort inside the locked data split.

    ``lock.train_positions`` is the complete denoising/development pool.  A
    segmentation teacher may use only the label-eligible subset represented by
    ``train_segment.csv``; it must never be expanded to all denoising groups.
    """
    train_positions = _normal_positions(train)
    validation_positions = _normal_positions(validation)
    locked_train = _normal_positions(lock["train_positions"])
    locked_validation = _normal_positions(lock["validation_positions"])
    sealed_test = set(_normal_positions(lock["sealed_test_positions"]))
    _require(bool(train_positions), "Teacher training cohort is empty")
    _require(set(train_positions) <= set(locked_train),
             "Teacher train positions exceed the active-lock train cohort")
    _require(validation_positions == locked_validation,
             "Teacher validation positions differ from active lock")
    _require(not (set(train_positions) | set(validation_positions)) & sealed_test,
             "Teacher development cohort overlaps sealed test")
    if expected_train is not None:
        _require(train_positions == _normal_positions(expected_train),
                 "Teacher train positions differ from the registered label-eligible cohort")
    if expected_validation is not None:
        _require(validation_positions == _normal_positions(expected_validation),
                 "Teacher validation positions differ from the registered cohort")
    return {
        "train_positions": train_positions,
        "validation_positions": validation_positions,
        "protocol_train_positions": locked_train,
        "teacher_train_subset_of_protocol": True,
    }


def locked_teacher_cohort(root: Path, config: dict, lock: dict) -> dict:
    """Read the active protocol's segmentation manifest without opening assets."""
    manifest = resolve(root, lock["manifest_root"]) / "train_segment.csv"
    _require(manifest.is_file(), f"Missing locked teacher manifest: {manifest}")
    table = pd.read_csv(manifest, dtype=str).fillna("")
    # Historical run-specific group filters are evidence about that run, not
    # the definition of the current locked label-eligible cohort.
    cohort_config = {**config, "data": dict(config["data"])}
    cohort_config["data"].pop("train_groups", None)
    cohort_config["data"].pop("val_groups", None)
    filtered = filtered_teacher_manifest(cohort_config, table)
    cohort = validate_teacher_cohort(
        filtered.loc[filtered["split"].eq("train"), "group_id"],
        filtered.loc[filtered["split"].eq("val"), "group_id"],
        lock,
    )
    return {
        **cohort,
        "manifest": manifest,
        "manifest_sha256": sha256_file(manifest),
        "effective_split_sha256": effective_split_sha(filtered),
    }


def audit_teacher_protocol(
    root: Path, config: dict, protocol_lock: Path, split_contract: Path
) -> dict:
    lock = load_protocol_lock(protocol_lock)
    _require(config.get("protocol_id") == lock["protocol_id"],
             "Teacher protocol_id differs from active lock")
    embedded = config.get("runtime", {}).get("active_protocol_lock", {})
    for key in LOCK_KEYS[1:]:
        _require(embedded.get(key, config.get(key)) == lock[key],
                 f"Teacher protocol lock mismatch: {key}")
    _require(sha256_file(split_contract) == lock["split_contract_sha256"],
             "Teacher split-contract SHA mismatch")
    contract = yaml.safe_load(split_contract.read_text(encoding="utf-8-sig")) or {}
    _require(contract.get("protocol_id") == lock["protocol_id"],
             "Teacher split-contract protocol mismatch")
    manifest = resolve(root, config["data"]["manifest"])
    table = pd.read_csv(manifest, dtype=str).fillna("")
    filtered = filtered_teacher_manifest(config, table)
    train = filtered.loc[filtered["split"].eq("train"), "group_id"]
    val = filtered.loc[filtered["split"].eq("val"), "group_id"]
    teacher_protocol = config.get("formal_d2_teacher", {})
    cohort = validate_teacher_cohort(
        train,
        val,
        lock,
        expected_train=teacher_protocol.get("expected_train_positions"),
        expected_validation=teacher_protocol.get("expected_validation_positions"),
    )
    _require(teacher_protocol.get("expected_train_positions") is not None,
             "Formal teacher lacks its registered label-eligible train cohort")
    _require(teacher_protocol.get("expected_validation_positions") is not None,
             "Formal teacher lacks its registered validation cohort")
    runtime = config.get("runtime", {})
    _require(runtime.get("manifest_sha256") == sha256_file(manifest),
             "Teacher manifest SHA missing or changed")
    split_sha = effective_split_sha(filtered)
    _require(runtime.get("effective_split_sha256") == split_sha,
             "Teacher effective split SHA missing or changed")
    return {
        "lock": lock,
        "manifest": manifest,
        "filtered": filtered,
        **cohort,
        "effective_split_sha256": split_sha,
    }


def _checkpoint_optimizer_step(raw: dict) -> int:
    explicit = raw.get("global_optimizer_step")
    if explicit is not None:
        return int(explicit)
    steps = []
    for state in (raw.get("optimizer") or {}).get("state", {}).values():
        if isinstance(state, dict) and state.get("step") is not None:
            value = state["step"]
            steps.append(int(value.item() if torch.is_tensor(value) else value))
    return max(steps, default=0)


def _audit_checkpoint_and_selection(
    root: Path,
    checkpoint: Path,
    history: Path,
    resolved_config: Path,
    run_metadata: Path,
) -> tuple[dict, dict, dict, dict]:
    _require(checkpoint.name == "best.pth" and checkpoint.is_file(),
             "Teacher binding requires an existing best.pth")
    config = load_config(resolved_config)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(isinstance(raw, dict) and raw.get("model") and raw.get("config"),
             "Teacher checkpoint lacks model/config provenance")
    for section in (
        "model", "data", "train", "loss", "seed", "protocol_id", "label_type",
        "formal_d2_teacher", "training_asset_evidence",
    ):
        _require(raw["config"].get(section) == config.get(section),
                 f"Teacher checkpoint/resolved config mismatch: {section}")
    _require(config.get("train", {}).get("stage") == "segment",
             "Teacher stage is not segment")
    _require(config["train"].get("monitor") == "vessel_soft_dice",
             "Teacher monitor is not vessel_soft_dice")
    _require(config["train"].get("checkpoint_selection_rule")
             == "best_validation_vessel_soft_dice",
             "Teacher selection rule is not best_validation_vessel_soft_dice")
    history_audit = audit_teacher_history(history, int(config["train"]["epochs"]))
    checkpoint_epoch = int(raw.get("epoch", -1)) + 1
    _require(checkpoint_epoch == int(history_audit["best_epoch"]),
             "Teacher checkpoint epoch differs from history best epoch")
    _require(np.isfinite(float(raw.get("best_metric", float("nan"))))
             and abs(float(raw["best_metric"]) - float(history_audit["best_value"])) <= 1e-8,
             "Teacher checkpoint best metric differs from history")
    step = _checkpoint_optimizer_step(raw)
    _require(step > 0, "Teacher checkpoint lacks a positive optimizer step")
    metadata = _json(run_metadata)
    checkpoint_sha = sha256_file(checkpoint)
    _require(metadata.get("best_checkpoint_sha256") == checkpoint_sha,
             "Teacher run metadata checkpoint SHA mismatch")
    _require(int(metadata.get("best_epoch", -1)) == checkpoint_epoch,
             "Teacher run metadata best epoch mismatch")
    _require(metadata.get("monitor") == "vessel_soft_dice"
             and metadata.get("selection_rule") == "best_validation_vessel_soft_dice",
             "Teacher run metadata selection metric/rule mismatch")
    _require(abs(float(metadata.get("best_metric", float("nan")))
                 - float(history_audit["best_value"])) <= 1e-8,
             "Teacher run metadata best value mismatch")
    _require(str(metadata.get("run_id")) == checkpoint.parent.name,
             "Teacher run ID mismatch")
    recorded = Path(str(metadata.get("best_checkpoint", ""))).expanduser()
    recorded = recorded.resolve() if recorded.is_absolute() else (root / recorded).resolve()
    _require(recorded == checkpoint.resolve(),
             "Teacher run metadata points to a different checkpoint")
    return config, raw, metadata, {**history_audit, "global_optimizer_step": step}


def bind_native_teacher(
    root: Path,
    checkpoint: Path,
    history: Path,
    resolved_config: Path,
    run_metadata: Path,
    protocol_lock: Path,
    split_contract: Path,
    initial_inventory: Path,
    initial_checkpoint: Path,
    initialization_audit: Path,
    parameter_audit: Path,
    output: Path,
) -> dict:
    """Bind a freshly retrained formal teacher to native protocol evidence."""
    sources = {
        "checkpoint": checkpoint, "history": history,
        "resolved_config": resolved_config, "run_metadata": run_metadata,
        "protocol_lock": protocol_lock, "split_contract": split_contract,
        "initial_inventory": initial_inventory, "initial_checkpoint": initial_checkpoint,
        "initialization_audit": initialization_audit, "parameter_audit": parameter_audit,
    }
    for path in sources.values():
        _require(path.is_file(), f"Missing native teacher source: {path}")
    config, raw, metadata, history_audit = _audit_checkpoint_and_selection(
        root, checkpoint, history, resolved_config, run_metadata
    )
    _require(config.get("formal_d2_teacher", {}).get("enabled") is True,
             "Checkpoint is not an opt-in formal D2 teacher")
    protocol = audit_teacher_protocol(root, config, protocol_lock, split_contract)
    initial = _json(initial_inventory)
    _require(initial.get("recorded_at_training") is True
             and initial.get("recorded_before_optimizer_step") is True,
             "Teacher initial asset inventory is not training-time evidence")
    _require(initial.get("manifest_sha256") == sha256_file(protocol["manifest"])
             and initial.get("effective_split_sha256") == protocol["effective_split_sha256"],
             "Teacher initial inventory manifest/split mismatch")
    data_root = resolve(root, config["data"].get("root") or root)
    records = asset_inventory(data_root, protocol["filtered"], include_labels=True)
    records_sha = stable_sha(records)
    _require(initial.get("records") == records
             and initial.get("records_sha256") == records_sha,
             "Teacher train/validation pixel inventory changed")
    initial_raw = torch.load(initial_checkpoint, map_location="cpu", weights_only=False)
    _require(int(initial_raw.get("epoch", -2)) == -1
             and _checkpoint_optimizer_step(initial_raw) == 0,
             "Teacher initial checkpoint is not pre-optimisation")
    _require(initial_raw.get("config", {}).get("formal_d2_teacher", {}).get("enabled") is True,
             "Teacher initial checkpoint lacks formal protocol")
    initialization = _json(initialization_audit)
    _require(initialization.get("model_state_sha256"),
             "Teacher initialization audit lacks model state SHA")
    parameters = _json(parameter_audit)
    _require(parameters.get("status") == "passed"
             and int(parameters.get("changed_trainable_parameter_count", 0)) > 0
             and int(parameters.get("changed_frozen_parameter_count", -1)) == 0
             and int(parameters.get("optimizer_steps", 0)) > 0,
             "Teacher parameter/optimizer audit failed")
    lock = protocol["lock"]
    result = {
        "schema_version": TEACHER_BINDING_SCHEMA,
        "status": "passed",
        "evidence_type": "native_protocol_binding",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": checkpoint.parent.name,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(raw["epoch"]) + 1,
        "checkpoint_global_optimizer_step": int(history_audit["global_optimizer_step"]),
        "selection_rule": "best_validation_vessel_soft_dice",
        "selection_metric": "val_vessel_soft_dice",
        "selection_value": float(history_audit["best_value"]),
        "protocol_id": lock["protocol_id"],
        "protocol_values": {key: lock[key] for key in LOCK_KEYS},
        "manifest_sha256": sha256_file(protocol["manifest"]),
        "effective_split_sha256": protocol["effective_split_sha256"],
        "train_positions": protocol["train_positions"],
        "validation_positions": protocol["validation_positions"],
        "input_resolution": list(config["data"]["target_size"]),
        "normalization": config["data"]["normalization"],
        "label_semantics": config.get("label_type", "binary"),
        "history_audit": history_audit,
        "source_files": {name: str(path.resolve()) for name, path in sources.items()},
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output, result)
    return result


def bind_derived_legacy_teacher(
    root: Path,
    checkpoint: Path,
    history: Path,
    resolved_config: Path,
    run_metadata: Path,
    protocol_lock: Path,
    split_contract: Path,
    historical_protocol_evidence: Path,
    parameter_audit: Path,
    output: Path,
) -> dict:
    """Bind legacy teacher only from independent immutable training-time facts."""
    sources = {
        "checkpoint": checkpoint, "history": history,
        "resolved_config": resolved_config, "run_metadata": run_metadata,
        "protocol_lock": protocol_lock, "split_contract": split_contract,
        "historical_protocol_evidence": historical_protocol_evidence,
        "parameter_audit": parameter_audit,
    }
    for path in sources.values():
        _require(path.is_file(), f"Missing derived teacher source: {path}")
    config, raw, metadata, history_audit = _audit_checkpoint_and_selection(
        root, checkpoint, history, resolved_config, run_metadata
    )
    historical = _json(historical_protocol_evidence)
    _require(historical.get("recorded_at_training") is True
             and historical.get("recorded_before_optimizer_step") is True
             and historical.get("evidence_origin") == "immutable_training_record",
             "Legacy protocol evidence was not recorded at training time")
    lock = load_protocol_lock(protocol_lock)
    for key in LOCK_KEYS:
        _require(historical.get(key) == lock[key],
                 f"Legacy protocol evidence mismatch: {key}")
    _require(sha256_file(split_contract) == historical["split_contract_sha256"],
             "Legacy split-contract SHA mismatch")
    registered = locked_teacher_cohort(root, config, lock)
    _require(_normal_positions(historical.get("train_positions"))
             == registered["train_positions"],
             "Legacy train positions differ from the locked label-eligible cohort")
    _require(_normal_positions(historical.get("validation_positions"))
             == registered["validation_positions"],
             "Legacy validation positions differ from the locked teacher cohort")
    _require(int(historical.get("test_assets_opened", -1)) == 0
             and not historical.get("test_asset_paths"),
             "Legacy evidence references sealed test assets")
    _require(historical.get("run_id") == checkpoint.parent.name,
             "Legacy evidence run ID mismatch")
    _require(historical.get("checkpoint_sha256") == sha256_file(checkpoint),
             "Legacy evidence checkpoint SHA mismatch")
    _require(int(historical.get("checkpoint_epoch", -1)) == int(raw["epoch"]) + 1,
             "Legacy evidence checkpoint epoch mismatch")
    _require(historical.get("selection_rule") == "best_validation_vessel_soft_dice"
             and historical.get("selection_metric") == "val_vessel_soft_dice",
             "Legacy evidence selection metric/rule mismatch")
    _require(historical.get("manifest_sha256")
             == config.get("runtime", {}).get("manifest_sha256"),
             "Legacy manifest SHA mismatch")
    _require(historical.get("effective_split_sha256")
             == config.get("runtime", {}).get("effective_split_sha256"),
             "Legacy effective split SHA mismatch")
    _require(historical.get("input_resolution") == list(config["data"]["target_size"])
             and historical.get("normalization") == config["data"]["normalization"],
             "Legacy input geometry/normalization mismatch")
    _require(historical.get("label_semantics") == "binary",
             "Legacy label semantics are not the locked binary definition")
    parameters = _json(parameter_audit)
    _require(parameters.get("optimizer_parameter_count", 0) > 0,
             "Legacy parameter audit lacks optimizer evidence")
    result = {
        "schema_version": TEACHER_BINDING_SCHEMA,
        "status": "passed",
        "evidence_type": "derived_legacy_binding",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": checkpoint.parent.name,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(raw["epoch"]) + 1,
        "checkpoint_global_optimizer_step": int(history_audit["global_optimizer_step"]),
        "selection_rule": "best_validation_vessel_soft_dice",
        "selection_metric": "val_vessel_soft_dice",
        "selection_value": float(history_audit["best_value"]),
        "protocol_id": lock["protocol_id"],
        "protocol_values": {key: lock[key] for key in LOCK_KEYS},
        "manifest_sha256": historical["manifest_sha256"],
        "effective_split_sha256": historical["effective_split_sha256"],
        "train_positions": _normal_positions(historical["train_positions"]),
        "validation_positions": _normal_positions(historical["validation_positions"]),
        "input_resolution": historical["input_resolution"],
        "normalization": historical["normalization"],
        "label_semantics": historical["label_semantics"],
        "history_audit": history_audit,
        "source_files": {name: str(path.resolve()) for name, path in sources.items()},
        "source_sha256": {name: sha256_file(path) for name, path in sources.items()},
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output, result)
    return result


def audit_teacher_binding(path: Path, checkpoint: Path) -> dict:
    binding = _json(path)
    _require(binding.get("schema_version") == TEACHER_BINDING_SCHEMA
             and binding.get("status") == "passed",
             "Invalid D2 teacher binding schema/status")
    _require(binding.get("evidence_type") in TEACHER_EVIDENCE_TYPES,
             "Unverified D2 teacher evidence type")
    _require(binding.get("checkpoint_sha256") == sha256_file(checkpoint),
             "D2 teacher binding checkpoint SHA mismatch")
    _require(binding.get("selection_rule") == "best_validation_vessel_soft_dice"
             and binding.get("selection_metric") == "val_vessel_soft_dice",
             "D2 teacher binding selection mismatch")
    _require(int(binding.get("test_assets_opened", -1)) == 0,
             "D2 teacher binding does not keep test sealed")
    for name, value in binding.get("source_files", {}).items():
        source = Path(str(value)).expanduser().resolve()
        _require(source.is_file()
                 and binding.get("source_sha256", {}).get(name) == sha256_file(source),
                 f"D2 teacher binding source changed: {name}")
    return binding
