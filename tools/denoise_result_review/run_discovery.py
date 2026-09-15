from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import METHOD_ORDER


REQUIRED_FILES = (
    "metrics/per_image_metrics.csv", "metrics/per_position_metrics.csv",
    "metrics/per_dataset_metrics.csv", "metrics/asset_inventory.csv",
    "metrics/selected_parameters.csv", "configs/inference_registry.yaml",
    "audit/config_lock.json", "manifests/denoised_dataset_manifest.csv",
)


@dataclass
class RunCandidate:
    path: str
    required_present: int
    required_total: int
    missing_files: list[str]
    methods: list[str]
    successful_methods: list[str]
    missing_methods: list[str]
    pku_test_rows: int
    pku_test_samples: int
    pku_test_positions: int
    duplicate_keys: int
    package_ready: bool
    result_complete: bool
    modified_ns: int


def inspect_run(path: str | Path) -> RunCandidate:
    root = Path(path).resolve()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    methods: list[str] = []
    successful: list[str] = []
    rows = samples = positions = duplicates = 0
    metrics_path = root / "metrics" / "per_image_metrics.csv"
    if metrics_path.is_file():
        table = pd.read_csv(metrics_path, low_memory=False)
        methods = sorted(table.get("method_id", pd.Series(dtype=str)).dropna().astype(str).unique())
        if "status" in table:
            ok = table[table.status.astype(str).str.lower().isin({"success", "completed", "ok"})]
        else:
            ok = table
        successful = sorted(ok.get("method_id", pd.Series(dtype=str)).dropna().astype(str).unique())
        mask = (table.get("dataset", "").astype(str) == "PKU37") & (table.get("split", "").astype(str) == "test")
        test = table[mask]
        rows = len(test)
        samples = test.get("sample_id", pd.Series(dtype=str)).nunique()
        positions = test.get("position_id", pd.Series(dtype=str)).nunique()
        keys = [column for column in ("dataset", "split", "position_id", "sample_id", "method_id", "seed", "checkpoint_sha256") if column in table]
        duplicates = int(table.duplicated(keys, keep=False).sum()) if keys else len(table)
    missing_methods = sorted(set(METHOD_ORDER) - set(successful))
    package_ready = not missing and rows > 0 and not duplicates
    complete = package_ready and not missing_methods
    return RunCandidate(
        str(root), len(REQUIRED_FILES) - len(missing), len(REQUIRED_FILES), missing,
        methods, successful, missing_methods, rows, samples, positions, duplicates,
        package_ready, complete, root.stat().st_mtime_ns,
    )


def discover_runs(project_root: str | Path) -> list[RunCandidate]:
    runs = Path(project_root).resolve() / "runs"
    return sorted(
        (inspect_run(path) for path in runs.iterdir() if path.is_dir()),
        key=lambda item: (item.package_ready, item.required_present == item.required_total, item.modified_ns),
        reverse=True,
    ) if runs.is_dir() else []


def resolve_run(project_root: str | Path, run_dir: str | Path = "auto", require_complete: bool = False) -> Path:
    if str(run_dir).lower() != "auto":
        candidate = inspect_run(run_dir)
        if require_complete and not candidate.package_ready:
            raise RuntimeError(f"requested run is not package-ready: {asdict(candidate)}")
        return Path(candidate.path)
    candidates = discover_runs(project_root)
    eligible = [item for item in candidates if item.package_ready]
    if eligible:
        return Path(eligible[0].path)
    structured = [item for item in candidates if item.required_present == item.required_total]
    if require_complete:
        raise FileNotFoundError("no package-ready denoising run found; candidates=" + json.dumps([asdict(x) for x in candidates], ensure_ascii=False))
    if not structured:
        raise FileNotFoundError("no structurally complete denoising run found")
    return Path(structured[0].path)


def discovery_frame(candidates: Iterable[RunCandidate]) -> pd.DataFrame:
    rows = []
    for item in candidates:
        row = asdict(item)
        for key in ("missing_files", "methods", "successful_methods", "missing_methods"):
            row[key] = ";".join(row[key])
        rows.append(row)
    return pd.DataFrame(rows)
