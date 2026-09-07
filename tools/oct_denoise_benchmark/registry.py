from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from .io import sha256_file


def stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def save_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True), encoding="utf-8")


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
        checkpoint = entry.get("checkpoint")
        if checkpoint:
            resolved = Path(checkpoint)
            if not resolved.is_absolute():
                resolved = (project_root / resolved).resolve()
            checkpoints[method] = {"path": str(resolved), "sha256": sha256_file(resolved) if resolved.is_file() else "missing"}
    now = datetime.now(timezone.utc).isoformat()
    value = {
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
