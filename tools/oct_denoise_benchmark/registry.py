from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .io import sha256_file


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def yaml_plain_value(value: Any) -> Any:
    """Recursively remove NumPy/path objects before SafeDumper sees them."""
    if isinstance(value, np.generic):
        return yaml_plain_value(value.item())
    if isinstance(value, np.ndarray):
        return [yaml_plain_value(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {yaml_plain_value(key): yaml_plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [yaml_plain_value(item) for item in value]
    return value


def save_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(yaml_plain_value(dict(value)), sort_keys=False, allow_unicode=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def git_commit(project_root: Path) -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def lock_run(project_root: Path, run_dir: Path, test_started: bool = False) -> dict[str, Any]:
    config_dir = run_dir / "configs"
    configs = {}
    for path in sorted(config_dir.glob("*.yaml")):
        configs[path.name] = sha256_file(path)
    registry_path = config_dir / "inference_registry.yaml"
    registry = load_yaml(registry_path) if registry_path.exists() else {}
    checkpoints = {}
    for method, entry in registry.get("methods", {}).items():
        candidates = []
        if entry.get("checkpoint"):
            candidates.append({"seed": entry.get("seed"), "checkpoint": entry["checkpoint"]})
        candidates.extend(entry.get("evaluation_checkpoints", []))
        seen = set()
        inventory = []
        for candidate in candidates:
            checkpoint = candidate.get("checkpoint")
            resolved = Path(checkpoint)
            if not resolved.is_absolute():
                resolved = (project_root / resolved).resolve()
            key = (candidate.get("seed"), str(resolved))
            if key not in seen:
                inventory.append({"seed": candidate.get("seed"), "path": str(resolved), "sha256": sha256_file(resolved) if resolved.is_file() else "missing"})
                seen.add(key)
        if inventory:
            checkpoints[method] = inventory
    now = datetime.now(timezone.utc).isoformat()
    value = {
        "status": "locked",
        "git_commit": git_commit(project_root),
        "locked_at_utc": now,
        "test_started_at_utc": now if test_started else None,
        "config_sha256": configs,
        "checkpoint_sha256": checkpoints,
    }
    path = run_dir / "audit" / "config_lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return value
