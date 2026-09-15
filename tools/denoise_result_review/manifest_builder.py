from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import DEEP_METHODS, METHOD_ORDER


KEY_COLUMNS = ["dataset", "split", "position_id", "sample_id", "method_id", "seed", "checkpoint_sha256"]


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def primary_seeds(run_dir: str | Path) -> dict[str, int]:
    run = Path(run_dir)
    seeds = {method: 0 for method in METHOD_ORDER}
    registry_path = run / "configs" / "inference_registry.yaml"
    registry = _read_yaml(registry_path) if registry_path.is_file() else {}
    for method, entry in registry.get("methods", {}).items():
        if method not in seeds:
            continue
        if isinstance(entry, dict):
            seeds[method] = int(entry.get("primary_seed", entry.get("seed", 42 if method in DEEP_METHODS else 0)))
    for method in DEEP_METHODS:
        if method in registry.get("methods", {}) and seeds[method] == 0:
            seeds[method] = 42
    return seeds


def build_asset_manifest(run_dir: str | Path, dataset: str | None = None, split: str | None = None,
                         primary_only: bool = True, methods: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = Path(run_dir).resolve()
    metrics = pd.read_csv(run / "metrics" / "per_image_metrics.csv", low_memory=False)
    aliases = {"clean_path": "reference_path", "output_path": "denoised_path", "config_hash": "config_sha256"}
    metrics = metrics.rename(columns={old: new for old, new in aliases.items() if old in metrics and new not in metrics})
    if "seed" not in metrics: metrics["seed"] = 0
    if "checkpoint_sha256" not in metrics: metrics["checkpoint_sha256"] = ""
    denoised_path = run / "manifests" / "denoised_dataset_manifest.csv"
    denoised = pd.read_csv(denoised_path, low_memory=False) if denoised_path.is_file() else pd.DataFrame()
    if not denoised.empty:
        denoised = denoised.rename(columns={old: new for old, new in aliases.items() if old in denoised and new not in denoised})
        if "seed" not in denoised: denoised["seed"] = 0
        if "checkpoint_sha256" not in denoised: denoised["checkpoint_sha256"] = ""
    sources = [(metrics, "per_image_metrics.csv")]
    if not denoised.empty:
        sources.append((denoised, "denoised_dataset_manifest.csv"))
    for frame, source in sources:
        missing = set(KEY_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"{source} missing logical key columns: {sorted(missing)}")
    duplicated = metrics[metrics.duplicated(KEY_COLUMNS, keep=False)].copy()
    if not duplicated.empty:
        return pd.DataFrame(), duplicated.assign(failure="duplicate metric logical key")
    use = metrics.copy()
    if dataset is not None: use = use[use.dataset.astype(str) == dataset]
    if split is not None: use = use[use.split.astype(str) == split]
    if methods: use = use[use.method_id.astype(str).isin(methods)]
    if "status" in use:
        use = use[use.status.astype(str).str.lower().isin({"success", "completed", "ok"})]
    seeds = primary_seeds(run)
    use["is_primary_seed"] = [int(seed) == seeds.get(str(method), 0) for method, seed in zip(use.method_id, use.seed)]
    if primary_only: use = use[use.is_primary_seed]
    manifest_paths = [column for column in ("noisy_path", "reference_path", "denoised_path", "output_sha256", "bit_depth", "width", "height") if column in denoised]
    if denoised.empty:
        result = use.copy()
    else:
        supplement = denoised[KEY_COLUMNS + manifest_paths].drop_duplicates(KEY_COLUMNS)
        result = use.merge(supplement, on=KEY_COLUMNS, how="left", suffixes=("", "_manifest"), validate="one_to_one")
    for column in manifest_paths:
        other = f"{column}_manifest"
        if other in result:
            if column not in result: result[column] = result[other]
            else: result[column] = result[column].where(result[column].notna() & (result[column].astype(str) != ""), result[other])
            result.drop(columns=[other], inplace=True)
    failures = []
    for row in result.itertuples(index=False):
        path = Path(str(getattr(row, "denoised_path", "")))
        if not path.is_file(): failures.append({**dict(zip(result.columns, row)), "failure": "denoised asset missing"})
    failure_frame = pd.concat([duplicated.assign(failure="duplicate metric logical key"), pd.DataFrame(failures)], ignore_index=True, sort=False)
    return result.reset_index(drop=True), failure_frame
