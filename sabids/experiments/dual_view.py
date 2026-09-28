"""Fail-closed provenance and fixed-component metrics for dual-view v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.ndimage import label as connected_components

from .d2 import component_measurements
from .dose_response import stable_sha
from .protocol_lock import sha256_file


VERSION = "noisy-mild-dual-view-v1"
BLOCKED = "BLOCKED: DUAL-VIEW INPUT EVIDENCE"
PRIMARY_ARMS = ("B0", "B1", "B3", "B6", "C1")


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def audit_formal_input_evidence(
    protocol_lock: Path,
    split_contract: Path,
    d1_checkpoint: Path,
    checkpoint_binding: Path,
    training_asset_inventory: Path,
    dose_registry: Path,
) -> dict:
    paths = {
        "protocol_lock": protocol_lock,
        "split_contract": split_contract,
        "d1_checkpoint": d1_checkpoint,
        "checkpoint_binding": checkpoint_binding,
        "training_asset_inventory": training_asset_inventory,
        "dose_registry": dose_registry,
    }
    issues = [f"missing {name}: {path}" for name, path in paths.items() if not path.is_file()]
    result = {
        "status": "blocked" if issues else "pending",
        "message": BLOCKED if issues else None,
        "issues": issues,
        "test_assets_opened": 0,
    }
    if issues:
        return result
    documents = {
        name: json.loads(path.read_text(encoding="utf-8-sig"))
        for name, path in paths.items() if path.suffix.lower() == ".json"
    }
    lock = documents["protocol_lock"]
    binding = documents["checkpoint_binding"]
    inventory = documents["training_asset_inventory"]
    registry = documents["dose_registry"]
    protocol_id = lock.get("protocol_id")
    checks = {
        "protocol_id_present": bool(protocol_id),
        "split_contract_matches_lock": lock.get("split_contract_sha256") == sha256_file(split_contract),
        "binding_checkpoint_sha": binding.get("checkpoint_sha256") == sha256_file(d1_checkpoint),
        "binding_status": binding.get("status") == "passed",
        "binding_selection_rule": binding.get("selection_rule") == "best_validation_psnr",
        "binding_inventory_chain": binding.get("source_sha256", {}).get("initial_inventory") == sha256_file(training_asset_inventory),
        "binding_protocol_chain": binding.get("source_sha256", {}).get("protocol_lock") == sha256_file(protocol_lock),
        "binding_split_chain": binding.get("source_sha256", {}).get("split_contract") == sha256_file(split_contract),
        "binding_no_test": binding.get("test_assets_opened") == 0,
        "training_time_inventory": inventory.get("recorded_at_training") is True,
        "inventory_before_optimizer": inventory.get("recorded_before_optimizer_step") is True,
        "registry_formal": registry.get("preflight", {}).get("mode") == "formal",
        "registry_passed": registry.get("preflight", {}).get("status") == "passed",
        "registry_protocol": registry.get("identity", {}).get("protocol_id") == protocol_id,
        "registry_checkpoint": registry.get("identity", {}).get("checkpoint_sha256") == sha256_file(d1_checkpoint),
        "registry_selection_rule": registry.get("preflight", {}).get("selection_rule") == "best_validation_psnr",
        "registry_d1_curve": set(registry.get("identity", {}).get("curves", [])) == {"d1"},
        "registry_required_alphas": {0.0, 0.25, 1.0}.issubset(
            {float(value) for value in registry.get("identity", {}).get("alphas", [])}
        ),
        "registry_no_test": registry.get("test_assets_opened") == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        result.update(status="blocked", message=BLOCKED, issues=failed, checks=checks)
        return result
    result.update(
        status="passed",
        message=None,
        checks=checks,
        protocol_id=protocol_id,
        sha256={name: sha256_file(path) for name, path in paths.items()},
    )
    return result


def deterministic_shuffle(rows: list[dict], seed: int) -> dict[str, str]:
    """Map every sample to a different anatomical group, deterministically."""
    ordered = sorted(rows, key=lambda row: (str(row["split"]), str(row["sample_id"])))
    mapping: dict[str, str] = {}
    for split in sorted({str(row["split"]) for row in ordered}):
        part = [row for row in ordered if str(row["split"]) == split]
        if len({str(row["group_id"]) for row in part}) < 2:
            raise ValueError("Shuffle control requires at least two groups per split")
        rng = np.random.default_rng(int(seed) + sum(split.encode("utf-8")))
        candidates = np.arange(len(part))
        for _ in range(10_000):
            rng.shuffle(candidates)
            if all(
                str(part[index]["group_id"]) != str(part[int(candidates[index])]["group_id"])
                for index in range(len(part))
            ):
                break
        else:
            raise RuntimeError("Could not construct a cross-position shuffle")
        mapping.update(
            {
                str(row["sample_id"]): str(part[int(candidates[index])]["sample_id"])
                for index, row in enumerate(part)
            }
        )
    return mapping


def build_fixed_component_inventory(
    training_samples: Iterable[dict], validation_samples: Iterable[dict], ring_width: int = 3
) -> dict:
    train_rows = []
    for sample in training_samples:
        if str(sample.get("split")) != "train":
            raise ValueError("Threshold fitting accepts train only")
        train_rows.extend(component_measurements(
            sample["vessel"], sample["layer"], sample["valid"], sample["noisy"],
            sample.get("clean"), ring_width,
        ))
    if not train_rows:
        raise ValueError("No train vessel components")
    areas = np.asarray([row["area_pixels"] for row in train_rows], dtype=np.float64)
    contrasts = np.asarray([row["noisy_local_contrast"] for row in train_rows], dtype=np.float64)
    finite = contrasts[np.isfinite(contrasts)]
    if not finite.size:
        raise ValueError("No valid train noisy contrast")
    q33, q67 = np.quantile(areas, (1 / 3, 2 / 3), method="linear")
    low_q25 = float(np.quantile(finite, 0.25, method="linear"))
    members = []
    for sample in validation_samples:
        if str(sample.get("split")) != "val":
            raise ValueError("Fixed membership accepts validation only")
        target = sample["vessel"].astype(bool) & sample["valid"].astype(bool)
        labels, count = connected_components(target, structure=np.ones((3, 3), np.uint8))
        measured = component_measurements(
            sample["vessel"], sample["layer"], sample["valid"], sample["noisy"],
            sample.get("clean"), ring_width,
        )
        if len(measured) != count:
            raise AssertionError("Component enumeration changed")
        for row in measured:
            component = labels == int(row["component_id"])
            area = int(row["area_pixels"])
            contrast = float(row["noisy_local_contrast"])
            members.append({
                "sample_id": str(sample["sample_id"]),
                "group_id": str(sample["group_id"]),
                "component_id": int(row["component_id"]),
                "component_mask_sha256": array_sha256(component.astype(np.uint8)),
                "area_pixels": area,
                "size_bin": "small" if area <= q33 else "medium" if area <= q67 else "large",
                "noisy_local_contrast": contrast if np.isfinite(contrast) else None,
                "contrast_bin": (
                    "unknown" if not np.isfinite(contrast)
                    else "low" if contrast <= low_q25 else "normal_high"
                ),
                "clean_contrast_bin": "unknown" if sample.get("clean") is None else (
                    "recorded_sensitivity_only" if row["clean_contrast_valid"] else "unknown"
                ),
            })
    result = {
        "version": VERSION,
        "threshold_source": "train_gt_and_train_noisy_only",
        "area_q33": float(q33),
        "area_q67": float(q67),
        "low_contrast_q25_noisy": low_q25,
        "ring_width_pixels": int(ring_width),
        "members": sorted(members, key=lambda row: (row["sample_id"], row["component_id"])),
        "test_assets_opened": 0,
    }
    result["inventory_sha256"] = stable_sha(result)
    return result


def evaluate_fixed_components(
    sample_id: str,
    prediction: np.ndarray,
    vessel: np.ndarray,
    valid: np.ndarray,
    inventory: dict,
) -> list[dict]:
    claimed = inventory.get("inventory_sha256")
    unsigned = dict(inventory)
    unsigned.pop("inventory_sha256", None)
    if claimed != stable_sha(unsigned):
        raise ValueError("Fixed component inventory SHA mismatch")
    target = vessel.astype(bool) & valid.astype(bool)
    labels, count = connected_components(target, structure=np.ones((3, 3), np.uint8))
    expected = {
        int(row["component_id"]): row
        for row in inventory["members"] if str(row["sample_id"]) == str(sample_id)
    }
    if set(expected) != set(range(1, count + 1)):
        raise ValueError(f"Fixed component membership mismatch for {sample_id}")
    pred = prediction.astype(bool) & valid.astype(bool)
    rows = []
    for component_id, member in sorted(expected.items()):
        component = labels == component_id
        if array_sha256(component.astype(np.uint8)) != member["component_mask_sha256"]:
            raise ValueError(f"GT component changed for {sample_id}+{component_id}")
        coverage = float(pred[component].mean())
        rows.append({
            **member,
            "area_bin_quantile": member["size_bin"],
            "coverage": coverage,
            "recall_at_025": float(coverage >= 0.25),
            "recall_at_050": float(coverage >= 0.5),
            "completely_missed": float(coverage == 0.0),
        })
    return rows
