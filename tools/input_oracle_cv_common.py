from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ARMS = ("noisy", "clean", "denoised")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_protocol(root: Path, value: str | Path) -> tuple[Path, dict[str, Any]]:
    path = resolve(root, value)
    return path, yaml.safe_load(path.read_text(encoding="utf-8"))


def require_phase0(root: Path, output: Path) -> dict[str, Any]:
    path = output / "audit/audit_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Phase 0 audit: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or not audit.get("training_authorized"):
        raise RuntimeError(f"Phase 0 status is blocked; no training is permitted: {path}")
    if int(audit.get("test_assets_opened", -1)) != 0:
        raise RuntimeError("Phase 0 reports test asset access")
    return audit


def require_split_audit(output: Path, fold: int) -> dict[str, Any]:
    path = output / "splits/split_audit.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing fold audit: {path}")
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or str(fold) not in audit.get("folds", {}):
        raise RuntimeError(f"Fold {fold} is not authorized by {path}")
    return audit


def audit_d0_checkpoint(checkpoint: Path, manifest: Path, val_groups: set[str], sealed: set[str]) -> dict[str, Any]:
    import torch

    checkpoint = checkpoint.resolve()
    manifest = manifest.resolve()
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload.get("config") or {}
    runtime = config.get("runtime", {})
    sidecar_path = checkpoint.parent / "run_metadata.json"
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else {}
    checkpoint_sha = sha256_file(checkpoint)
    provenance = runtime
    source = "checkpoint_runtime"
    if not runtime.get("effective_groups", {}).get("train") and sidecar.get("best_checkpoint_sha256") == checkpoint_sha:
        provenance, source = sidecar, "sha256_linked_run_metadata"
    train_groups = set(map(str, provenance.get("effective_groups", {}).get("train", [])))
    declared_hash = provenance.get("manifest_sha256")
    overlap_val = sorted(train_groups & val_groups)
    overlap_test = sorted(train_groups & sealed)
    passed = bool(train_groups) and declared_hash == sha256_file(manifest) and not overlap_val and not overlap_test
    return {
        "status": "passed" if passed else "blocked", "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": int(payload.get("epoch", -1)) + 1,
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "declared_manifest_sha256": declared_hash, "provenance_source": source,
        "d0_train_groups": sorted(train_groups), "fold_validation_overlap": overlap_val,
        "sealed_test_overlap": overlap_test, "test_assets_opened": 0,
    }


def read_fold_groups(output: Path, fold: int) -> tuple[set[str], set[str]]:
    train = set(pd.read_csv(output / f"splits/fold_{fold}_train.csv", dtype=str)["group_id"])
    val = set(pd.read_csv(output / f"splits/fold_{fold}_val.csv", dtype=str)["group_id"])
    return train, val
