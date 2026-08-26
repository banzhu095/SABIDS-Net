from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML config with optional ``_base_`` inheritance."""
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref is None:
        result = cfg
    else:
        base_path = (path.parent / base_ref).resolve()
        result = _deep_update(load_config(base_path), cfg)

    result.setdefault("runtime", {})
    result["runtime"]["config_path"] = str(path)
    result["runtime"]["config_dir"] = str(path.parent)
    return result


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

