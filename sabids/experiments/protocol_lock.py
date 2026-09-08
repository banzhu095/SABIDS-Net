from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sabids.config import load_config


CONSISTENCY_KEYS = (
    "protocol_id",
    "data_plan_sha256",
    "label_inventory_sha256",
    "dataset_inventory_sha256",
    "split_contract_sha256",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    return value if isinstance(value, dict) else {}


def _first(mapping: dict[str, Any], *paths: str) -> Any:
    for path in paths:
        current: Any = mapping
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                current = None
                break
            current = current[part]
        if current not in (None, "", []):
            return current
    return None


def _load_run_documents(run_dir: Path) -> dict[str, dict[str, Any]]:
    config_path = next(
        (run_dir / name for name in ("config_resolved.yaml", "resolved_config.yaml", "config.yaml") if (run_dir / name).is_file()),
        None,
    )
    config = load_config(config_path) if config_path else {}
    manifest_root = _first(config, "manifest_root")
    if manifest_root:
        protocol_root = Path(str(manifest_root))
        if not protocol_root.is_absolute():
            # run_dir is <project>/runs/current/<run>; three parents reach project.
            protocol_root = (run_dir.parents[2] / protocol_root).resolve()
    else:
        manifest = _first(config, "data.manifest")
        protocol_root = Path(str(manifest)).resolve().parent if manifest else Path()
    return {
        "config": config,
        "metadata": read_json(run_dir / "run_metadata.json"),
        "data": read_json(run_dir / "data_plan_audit.json"),
        "initialization": read_json(run_dir / "initialization_audit.json"),
        "protocol": read_json(protocol_root / "protocol_audit.json") if manifest_root or _first(config, "data.manifest") else {},
    }


def _fact(documents: dict[str, dict[str, Any]], key: str) -> Any:
    aliases = {
        "protocol_id": ("protocol_id", "runtime.active_protocol_lock.protocol_id"),
        "manifest_root": ("manifest_root", "runtime.active_protocol_lock.manifest_root"),
        "data_plan_sha256": ("data_plan_sha256", "runtime.active_protocol_lock.data_plan_sha256"),
        "label_inventory_sha256": ("label_inventory_sha256", "runtime.active_protocol_lock.label_inventory_sha256"),
        "dataset_inventory_sha256": ("dataset_inventory_sha256", "runtime.active_protocol_lock.dataset_inventory_sha256"),
        "split_contract_sha256": ("split_contract_sha256", "runtime.active_protocol_lock.split_contract_sha256"),
        "train_positions": ("train_positions", "runtime.active_protocol_lock.train_positions"),
        "validation_positions": ("validation_positions", "runtime.active_protocol_lock.validation_positions"),
        "sealed_test_positions": ("sealed_test_positions", "test_positions", "runtime.active_protocol_lock.sealed_test_positions"),
        "git_commit": ("git_commit", "git_commit_at_start", "runtime.active_protocol_lock.git_commit_at_d1_start"),
        "input_resolution": ("input_resolution", "data.target_size"),
        "normalization": ("normalization", "data.normalization"),
        "seed": ("seed",),
        "fold": ("fold",),
        "epochs": ("train.epochs", "train.fixed_epoch"),
        "loss": ("loss",),
    }
    for source in ("metadata", "data", "initialization", "protocol", "config"):
        value = _first(documents[source], *aliases[key])
        if value not in (None, "", []):
            return value
    return None


def find_d1_runs(project_root: str | Path) -> list[Path]:
    current = Path(project_root).resolve() / "runs" / "current"
    if not current.is_dir():
        return []
    result: list[Path] = []
    for run_dir in sorted(path for path in current.iterdir() if path.is_dir()):
        docs = _load_run_documents(run_dir)
        config = docs["config"]
        if (
            str(_first(config, "loss.restoration_mode") or "") == "structure_d1"
            and int(_first(config, "train.epochs") or _first(config, "train.fixed_epoch") or 0) >= 60
            and "_pilot_" not in run_dir.name
        ):
            result.append(run_dir)
    return result


def run_matches_protocol_lock(run_dir: str | Path, lock: dict[str, Any]) -> bool:
    documents = _load_run_documents(Path(run_dir))
    return all(
        _fact(documents, key) == lock.get(key)
        for key in ("protocol_id", "data_plan_sha256", "label_inventory_sha256")
    )


def _consistent(values: Iterable[Any]) -> bool:
    encoded = {json.dumps(value, sort_keys=True, default=str) for value in values if value not in (None, "", [])}
    return len(encoded) <= 1


def extract_active_protocol_lock(run_dirs: list[Path]) -> dict[str, Any]:
    if not run_dirs:
        raise RuntimeError("No D1 run with a resolved configuration was found")
    documents = [_load_run_documents(path) for path in run_dirs]
    required = ("protocol_id", "manifest_root", "data_plan_sha256", "label_inventory_sha256", "dataset_inventory_sha256", "split_contract_sha256")
    facts = {key: [_fact(doc, key) for doc in documents] for key in (
        *CONSISTENCY_KEYS, "manifest_root", "train_positions", "validation_positions",
        "sealed_test_positions", "git_commit", "input_resolution", "normalization", "loss", "epochs",
    )}
    problems = []
    for key in (*CONSISTENCY_KEYS, "validation_positions", "input_resolution", "loss", "epochs"):
        if not _consistent(facts[key]):
            problems.append(f"D1 runs disagree on {key}")
    for key in required:
        if any(value in (None, "", []) for value in facts[key]):
            problems.append(f"D1 evidence is missing {key}")
    if problems:
        raise RuntimeError("; ".join(problems))

    def one(key: str, default: Any = None) -> Any:
        return next((value for value in facts[key] if value not in (None, "", [])), default)

    return {
        "source": "active_d1_run",
        "protocol_id": one("protocol_id"),
        "manifest_root": one("manifest_root"),
        "data_plan_sha256": one("data_plan_sha256"),
        "label_inventory_sha256": one("label_inventory_sha256"),
        "dataset_inventory_sha256": one("dataset_inventory_sha256"),
        "split_contract_sha256": one("split_contract_sha256"),
        "train_positions": one("train_positions", []),
        "validation_positions": one("validation_positions", []),
        "sealed_test_positions": one("sealed_test_positions", []),
        "source_run_ids": [path.name for path in run_dirs],
        "git_commit_at_d1_start": one("git_commit"),
        "input_resolution": one("input_resolution"),
        "normalization": one("normalization"),
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "test_assets_opened": 0,
    }


def load_protocol_lock(path: str | Path) -> dict[str, Any]:
    lock = read_json(Path(path).resolve())
    missing = [key for key in ("protocol_id", "manifest_root", "data_plan_sha256", "label_inventory_sha256", "dataset_inventory_sha256", "split_contract_sha256") if not lock.get(key)]
    if missing:
        raise RuntimeError(f"Protocol lock is incomplete: {missing}")
    if int(lock.get("test_assets_opened", -1)) != 0:
        raise RuntimeError("Protocol lock does not prove zero test access")
    return lock


def validate_checkpoint_config(checkpoint: dict[str, Any], lock: dict[str, Any], checkpoint_name: str = "checkpoint") -> None:
    config = checkpoint.get("config", {})
    embedded = config.get("runtime", {}).get("active_protocol_lock", {})
    for key in ("protocol_id", "data_plan_sha256", "label_inventory_sha256"):
        if not lock.get(key):
            continue
        found = embedded.get(key, config.get(key))
        if found != lock.get(key):
            raise RuntimeError(f"{checkpoint_name} {key} mismatch: {found!r} != {lock.get(key)!r}")


def process_alive(pid: int, hostname: str | None = None) -> bool:
    if hostname and hostname != socket.gethostname():
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def find_training_pid(run_id: str) -> int | None:
    if os.name == "nt":
        return None
    process = subprocess.run(["pgrep", "-af", "train.py"], text=True, capture_output=True, check=False)
    for line in process.stdout.splitlines():
        if run_id in line:
            try: return int(line.split(maxsplit=1)[0])
            except (ValueError, IndexError): pass
    return None


def _history_summary(path: Path) -> tuple[int, int | None, bool]:
    if not path.is_file():
        return 0, None, False
    try:
        rows = list(csv.DictReader(path.open("r", encoding="utf-8-sig", newline="")))
        epochs = [int(float(row["epoch"])) for row in rows if row.get("epoch")]
        finite = all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if value not in (None, "") and key != "training_phase"
        )
        return max(epochs, default=0), None, finite
    except Exception:
        return 0, None, False


def d1_completion_rows(project_root: str | Path, lock: dict[str, Any], seeds: Iterable[int] = (42, 43, 44)) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    current = root / "runs" / "current"
    known = {
        path.name: path
        for path in (current.iterdir() if current.is_dir() else [])
        if path.is_dir()
        and str(_first(_load_run_documents(path)["config"], "train.stage") or "") == "denoise"
    }
    rows: list[dict[str, Any]] = []
    for method, prefix in (("D0", "d1_denoise_d0"), ("D1", "d1_denoise_struct")):
        for seed in seeds:
            candidates = [
                path for name, path in known.items()
                if prefix in name and f"seed{seed}" in name and "_pilot_" not in name
                and int(_first(_load_run_documents(path)["config"], "train.epochs") or 0) >= 60
            ]
            run_dir = candidates[0] if len(candidates) == 1 else None
            docs = _load_run_documents(run_dir) if run_dir else {key: {} for key in ("config", "metadata", "data", "initialization", "protocol")}
            config = docs["config"]
            expected = int(_first(config, "train.epochs") or _first(config, "train.fixed_epoch") or 0)
            current, best_epoch, finite = _history_summary(run_dir / "history.csv") if run_dir else (0, None, False)
            lock_info = read_json(run_dir / ".run.lock") if run_dir else {}
            detected_pid = find_training_pid(run_dir.name) if run_dir else None
            running = bool(
                (lock_info.get("pid") and process_alive(lock_info["pid"], lock_info.get("hostname")))
                or detected_pid
            )
            sha_ok = all(_fact(docs, key) == lock.get(key) for key in ("protocol_id", "data_plan_sha256", "label_inventory_sha256")) if run_dir else False
            completed = bool(expected and current >= expected and (run_dir / "last.pth").is_file()) if run_dir else False
            status = "running" if running else "completed" if completed else "interrupted" if run_dir and (run_dir / "last.pth").is_file() else "failed" if run_dir and not finite and current else "missing"
            rows.append({
                "run_id": run_dir.name if run_dir else f"{prefix}_{lock['protocol_id']}_fold0_seed{seed}",
                "method": method, "fold": int(_first(config, "fold") or 0), "seed": seed,
                "status": status, "pid_if_running": (lock_info.get("pid") or detected_pid or "") if running else "",
                "start_time": lock_info.get("start_time", ""),
                "last_update_time": datetime.fromtimestamp((run_dir / "history.csv").stat().st_mtime, timezone.utc).isoformat() if run_dir and (run_dir / "history.csv").is_file() else "",
                "current_epoch": current, "expected_epoch": expected, "best_epoch": _first(docs["metadata"], "best_epoch", "epoch") or best_epoch or "",
                "last_checkpoint_exists": bool(run_dir and (run_dir / "last.pth").is_file()),
                "best_checkpoint_exists": bool(run_dir and (run_dir / "best.pth").is_file()),
                "history_complete": completed, "protocol_id": _fact(docs, "protocol_id") or "",
                "data_plan_sha256": _fact(docs, "data_plan_sha256") or "",
                "label_inventory_sha256": _fact(docs, "label_inventory_sha256") or "",
                "finite": finite, "needs_resume": status == "interrupted" and sha_ok,
                "safe_to_merge": completed and finite and sha_ok,
            })
    return rows
