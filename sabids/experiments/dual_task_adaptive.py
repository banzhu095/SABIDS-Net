from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import yaml


SCHEMA_VERSION = "dual-task-adaptive-v1"
SEALED_TEST_GROUPS = {"pku_0024", "pku_0031", "pku_0037", "pku_0039"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _normalise_group(value: object) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text.startswith("pku") and not text.startswith("pku_"):
        text = "pku_" + text[3:]
    return text


def audit_adaptive_inputs(config: Dict[str, Any], project_root: str | Path) -> Dict[str, Any]:
    """Metadata-only, fail-closed audit for the deployable seed-42 experiment."""
    root = Path(project_root).expanduser().resolve()
    cfg = config.get("dual_task_adaptive", {})
    anchors = cfg.get("anchors", {})
    evidence = cfg.get("evidence", {})
    issues: list[str] = []
    checks: Dict[str, bool] = {}

    def resolve(value: object) -> Path:
        path = Path(str(value)).expanduser()
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    required_files = {
        "d2_checkpoint": anchors.get("d2_checkpoint"),
        "coarse_checkpoint": anchors.get("coarse_checkpoint"),
        "d2_binding": evidence.get("d2_binding"),
        "coarse_binding": evidence.get("coarse_binding"),
        "d2_inventory": evidence.get("d2_inventory"),
        "protocol_lock": evidence.get("protocol_lock"),
        "split_contract": evidence.get("split_contract"),
        "manifest": config.get("data", {}).get("manifest"),
    }
    paths: Dict[str, Path] = {}
    for name, value in required_files.items():
        if not value:
            issues.append(f"Missing configured path: {name}")
            checks[f"{name}_present"] = False
            continue
        path = resolve(value)
        paths[name] = path
        checks[f"{name}_present"] = path.is_file()
        if not path.is_file():
            issues.append(f"Missing {name}: {path}")

    expected_hashes = {
        "d2_checkpoint": anchors.get("d2_checkpoint_sha256"),
        "coarse_checkpoint": anchors.get("coarse_checkpoint_sha256"),
        "d2_binding": evidence.get("d2_binding_sha256"),
        "d2_inventory": evidence.get("d2_inventory_sha256"),
        "protocol_lock": evidence.get("protocol_lock_sha256"),
        "split_contract": evidence.get("split_contract_sha256"),
        "manifest": evidence.get("manifest_sha256"),
    }
    hashes: Dict[str, str] = {}
    for name, expected in expected_hashes.items():
        if name not in paths or not paths[name].is_file():
            continue
        actual = sha256_file(paths[name])
        hashes[name] = actual
        checks[f"{name}_sha256"] = bool(expected) and actual == expected
        if not checks[f"{name}_sha256"]:
            issues.append(f"SHA256 mismatch or missing frozen hash: {name}")

    if all(name in paths and paths[name].is_file() for name in ("d2_binding", "coarse_binding")):
        d2 = read_json(paths["d2_binding"])
        coarse = read_json(paths["coarse_binding"])
        checks["d2_binding_status"] = d2.get("status") == "passed"
        checks["coarse_binding_status"] = coarse.get("status") == "passed"
        checks["d2_binding_checkpoint"] = d2.get("checkpoint_sha256") == anchors.get("d2_checkpoint_sha256")
        checks["coarse_binding_checkpoint"] = coarse.get("checkpoint_sha256") == anchors.get("coarse_checkpoint_sha256")
        checks["coarse_selection_rule"] = coarse.get("selection_rule") == "best_validation_vessel_soft_dice"
        checks["coarse_completed_budget"] = int(coarse.get("completed_epochs", -1)) == 20
        checks["bindings_no_test"] = (
            int(d2.get("test_assets_opened", -1)) == 0
            and int(coarse.get("test_assets_opened", -1)) == 0
        )
        for name in (
            "d2_binding_status", "coarse_binding_status", "d2_binding_checkpoint",
            "coarse_binding_checkpoint", "coarse_selection_rule",
            "coarse_completed_budget", "bindings_no_test",
        ):
            if not checks[name]:
                issues.append(f"Failed evidence check: {name}")

    if "protocol_lock" in paths and paths["protocol_lock"].is_file():
        lock = read_json(paths["protocol_lock"])
        checks["protocol_id"] = lock.get("protocol_id") == "pku37_binary_v3"
        checks["protocol_no_test_open"] = int(lock.get("test_assets_opened", 0)) == 0
        if not checks["protocol_id"]:
            issues.append("Active protocol is not pku37_binary_v3")

    if "split_contract" in paths and paths["split_contract"].is_file():
        split = yaml.safe_load(paths["split_contract"].read_text(encoding="utf-8")) or {}
        contract_test = {_normalise_group(v) for v in split.get("test_positions", [])}
        checks["sealed_test_contract"] = contract_test == SEALED_TEST_GROUPS
        if not checks["sealed_test_contract"]:
            issues.append("Sealed test groups differ from the registered split contract")

    if "manifest" in paths and paths["manifest"].is_file():
        table = pd.read_csv(paths["manifest"])
        split_values = set(table["split"].astype(str).str.lower()) if "split" in table else set()
        group_column = next((key for key in ("group_id", "anatomical_position", "group") if key in table), None)
        groups = {_normalise_group(v) for v in table[group_column]} if group_column else set()
        checks["manifest_train_val_only"] = bool(split_values) and split_values <= {"train", "val"}
        checks["manifest_excludes_test_groups"] = not bool(groups & SEALED_TEST_GROUPS)
        checks["manifest_has_labels"] = all(
            column in table for column in ("layer_mask_path", "vessel_mask_path")
        )
        for name in ("manifest_train_val_only", "manifest_excludes_test_groups", "manifest_has_labels"):
            if not checks[name]:
                issues.append(f"Failed manifest check: {name}")

    checks["seed42_only"] = int(config.get("seed", -1)) == 42
    checks["coarse_alpha_025"] = float(cfg.get("coarse_strength", -1)) == 0.25
    checks["fixed_p0"] = all(
        float(config.get("evaluation", {}).get(key, 0.5)) == 0.5
        for key in ("threshold", "layer_threshold", "vessel_threshold")
    )
    for name in ("seed42_only", "coarse_alpha_025", "fixed_p0"):
        if not checks[name]:
            issues.append(f"Failed protocol check: {name}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not issues else "blocked",
        "blocked_message": None if not issues else "BLOCKED: DUAL-TASK ADAPTIVE INPUT EVIDENCE",
        "issues": issues,
        "checks": checks,
        "sha256": hashes,
        "test_assets_opened": 0,
    }


def gate_statistics(values: np.ndarray, masks: Dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for region, mask in masks.items():
        selected = np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]
        if selected.size == 0:
            continue
        rows.append({
            "region": region,
            "count": int(selected.size),
            "mean": float(selected.mean()),
            "std": float(selected.std()),
            "p10": float(np.quantile(selected, 0.10)),
            "p50": float(np.quantile(selected, 0.50)),
            "p90": float(np.quantile(selected, 0.90)),
        })
    return rows


def position_equal(table: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    columns = [name for name in metrics if name in table.columns]
    return table.groupby(["variant", "group_id"], as_index=False)[columns].mean(numeric_only=True)
