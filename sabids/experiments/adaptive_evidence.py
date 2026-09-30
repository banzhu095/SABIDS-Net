"""Metadata-only evidence snapshot for adaptive-denoising follow-up studies.

This module deliberately never follows image paths recorded in manifests and
never deserializes checkpoints.  It only reads JSON/YAML/CSV provenance files
and, when a checkpoint path is present in a binding, hashes the checkpoint as
an opaque byte stream.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


VERSION = "adaptive-evidence-snapshot-v1"
METADATA_NAMES = (
    "resolved_config.yaml",
    "run_metadata.json",
    "dual_view_training_metadata.json",
    "initialization_audit.json",
    "data_plan.json",
    "history.csv",
)
REPORT_TABLES = (
    "summary.json",
    "metrics_by_position.csv",
    "metrics_by_seed_position.csv",
    "metrics_by_image.csv",
    "component_metrics.csv",
    "contrast_metrics.csv",
    "RESULTS_TABLE.csv",
    "report_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _unique(paths: list[Path]) -> list[Path]:
    return sorted({path.resolve() for path in paths if path.exists()}, key=str)


def discover_candidates(root: Path) -> dict[str, list[str]]:
    """Find only top-level evidence; do not enumerate prediction/oracle grids."""
    root = root.resolve()
    dose_reports = _unique([
        path.parent
        for path in root.glob(
            "reports/adaptive_denoising/dose_seed42_best_sensitivity_*/metrics_by_position.csv"
        )
    ])
    dual_summaries = _unique(list(root.glob(
        "reports/adaptive_denoising/dual_view_v1/*/pilot*/summary/summary.json"
    )))
    b3_runs = _unique([
        path.parent
        for path in root.glob(
            "runs/adaptive_denoising/pku37_binary_v3/dual_view_v1/*/b3_seed42/"
            "dual_view_training_metadata.json"
        )
    ])
    d2_bindings = _unique(list(root.glob(
        "runs/adaptive_denoising/pku37_binary_v3/d2_v1/**/d25_seed42/"
        "checkpoint_binding_best_task_preserving.json"
    )))
    dose_registries = _unique(list(root.glob(
        "cache/adaptive_denoising/pku37_binary_v3/dose_v1/preparations/"
        "pilot_s42_d2_task*/preparation_registry.json"
    )))
    dual_registries = _unique(list(root.glob(
        "cache/adaptive_denoising/pku37_binary_v3/dual_view_v1/*_pilot/"
        "preparation_registry.json"
    )))
    fixed = {
        "protocol_lock": [root / "Manifests/pku37_binary_v3/active_protocol_lock.json"],
        "split_contract": [root / "configs/data/pku37_binary_v3_split.yaml"],
    }
    values: dict[str, list[Path]] = {
        "dose_report": dose_reports,
        "dual_summary": dual_summaries,
        "b3_run": b3_runs,
        "d2_binding": d2_bindings,
        "dose_registry": dose_registries,
        "dual_registry": dual_registries,
        **fixed,
    }
    return {
        key: [_relative(root, path) for path in _unique(paths)]
        for key, paths in values.items()
    }


def _load_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported structured metadata: {path}")


def _walk(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, child in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk(child, name)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _binding_checkpoint(binding_path: Path, binding: dict) -> tuple[Path | None, str | None]:
    paths: list[str] = []
    hashes: list[str] = []
    for key, value in _walk(binding):
        leaf = key.rsplit(".", 1)[-1].lower()
        if leaf in {"checkpoint_path", "checkpoint"} and isinstance(value, str):
            paths.append(value)
        if leaf in {"checkpoint_sha256", "checkpoint_hash"} and isinstance(value, str):
            hashes.append(value.lower())
    if len(set(paths)) != 1 or len(set(hashes)) != 1:
        return None, None
    checkpoint = Path(paths[0]).expanduser()
    if not checkpoint.is_absolute():
        # Bindings normally store project-relative paths.  Locate the project
        # root from the first "runs" ancestor without guessing another run.
        ancestors = [p for p in binding_path.resolve().parents if p.name == "runs"]
        if not ancestors:
            return None, hashes[0]
        checkpoint = ancestors[0].parent / checkpoint
    return checkpoint.resolve(), hashes[0]


def _alpha_and_curve(table: pd.DataFrame) -> pd.DataFrame:
    result = table.copy()
    if "alpha" not in result:
        source = result.get("arm", result.get("run_id", pd.Series("", index=result.index)))
        extracted = source.astype(str).str.extract(r"(?:alpha=|_a)(\d+(?:\.\d+)?)", expand=False)
        numeric = pd.to_numeric(extracted, errors="coerce")
        # a025/a100 encodes hundredths; decimal alpha= values do not.
        coded = source.astype(str).str.contains(r"_a\d{3}(?:_|$)", regex=True)
        numeric.loc[coded] = numeric.loc[coded] / 100.0
        result["alpha"] = numeric
    if "curve_type" not in result:
        source = result.get("arm", result.get("run_id", pd.Series("", index=result.index)))
        result["curve_type"] = source.astype(str).str.extract(
            r"(d2_task|d2_pixel|d2_last|d1|oracle)", expand=False
        )
    return result


def summarize_dose(position_csv: Path) -> tuple[pd.DataFrame, dict]:
    table = _alpha_and_curve(pd.read_csv(position_csv, low_memory=False))
    if "split" in table.columns:
        observed = set(table["split"].dropna().astype(str).str.lower())
        if observed != {"val"}:
            raise ValueError(f"Dose table is not validation-only: {sorted(observed)}")
    d2 = table.loc[table["curve_type"].astype(str) == "d2_task"].copy()
    if d2.empty or d2["alpha"].isna().any():
        raise ValueError("No unambiguous d2_task alpha rows in metrics_by_position.csv")
    identifiers = {"seed", "fold", "epoch", "alpha"}
    metrics = [
        column for column in d2.select_dtypes(include="number").columns
        if column not in identifiers
    ]
    if not metrics:
        raise ValueError("Dose position table has no numeric metrics")
    summary = d2.groupby("alpha", as_index=False)[metrics].mean(numeric_only=True)
    layer = [c for c in metrics if "layer" in c.lower() and "dice" in c.lower()]
    vessel = [c for c in metrics if "vessel" in c.lower() and "dice" in c.lower()]
    audit = {
        "position_rows": int(len(d2)),
        "position_count": int(d2["group_id"].nunique()) if "group_id" in d2 else None,
        "alphas": sorted(float(value) for value in d2["alpha"].unique()),
        "layer_dice_columns": layer,
        "vessel_dice_columns": vessel,
        "selection_deferred_if_metric_ambiguous": len(layer) != 1 or len(vessel) != 1,
    }
    if len(layer) == 1:
        audit["best_layer_alpha"] = float(summary.loc[summary[layer[0]].idxmax(), "alpha"])
        audit["best_layer_metric"] = layer[0]
    if len(vessel) == 1:
        audit["best_vessel_alpha"] = float(summary.loc[summary[vessel[0]].idxmax(), "alpha"])
        audit["best_vessel_metric"] = vessel[0]
    return summary, audit


def _copy(source: Path, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return {
        "source": str(source.resolve()),
        "snapshot": target.as_posix(),
        "size_bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def export_snapshot(root: Path, output: Path, selected: dict[str, Path]) -> dict:
    root, output = root.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence snapshot: {output}")
    issues: list[str] = []
    for key, path in selected.items():
        if not path.exists():
            issues.append(f"missing {key}: {path}")
        lowered = {part.lower() for part in path.parts}
        if "test_results" in lowered or "test" in lowered:
            issues.append(f"refusing possible test path for {key}: {path}")
    output.mkdir(parents=True)
    manifest: list[dict] = []

    def copy_file(path: Path, category: str) -> None:
        if path.is_file():
            manifest.append(_copy(path, output / category / path.name))

    for key in ("protocol_lock", "split_contract", "dose_registry", "dual_registry", "d2_binding"):
        # Registries intentionally share the same basename.  Keep each source
        # in its own category so one cannot overwrite another in the snapshot.
        copy_file(selected[key], key)
    for name in METADATA_NAMES:
        copy_file(selected["b3_run"] / name, "b3_run")
        copy_file(selected["d2_binding"].parent / name, "d2_run")
    for name in REPORT_TABLES:
        copy_file(selected["dose_report"] / name, "dose_report")
    dual_summary = selected["dual_summary"]
    copy_file(dual_summary, "dual_view_report")
    for name in ("metrics_by_seed_position.csv", "metrics_by_position.csv"):
        copy_file(dual_summary.parent / name, "dual_view_report")

    binding = _load_structured(selected["d2_binding"]) if selected["d2_binding"].is_file() else {}
    checkpoint, recorded_sha = _binding_checkpoint(selected["d2_binding"], binding)
    checkpoint_audit = {
        "path": str(checkpoint) if checkpoint else None,
        "recorded_sha256": recorded_sha,
        "exists": bool(checkpoint and checkpoint.is_file()),
        "actual_sha256": None,
        "sha256_matches": False,
        "deserialized": False,
    }
    if checkpoint and checkpoint.is_file():
        checkpoint_audit["actual_sha256"] = sha256_file(checkpoint)
        checkpoint_audit["sha256_matches"] = checkpoint_audit["actual_sha256"] == recorded_sha
    if not checkpoint_audit["sha256_matches"]:
        issues.append("D2 checkpoint path/SHA is absent, ambiguous, missing, or mismatched")

    dose_csv = selected["dose_report"] / "metrics_by_position.csv"
    dose_audit: dict[str, Any] = {}
    if dose_csv.is_file():
        try:
            dose_summary, dose_audit = summarize_dose(dose_csv)
            dose_summary.to_csv(output / "dose_alpha_position_equal_summary.csv", index=False)
        except Exception as error:  # Report the evidence defect; do not silently select.
            issues.append(f"dose summary blocked: {error}")
    else:
        issues.append(f"missing dose position table: {dose_csv}")

    for path in (selected["dose_registry"], selected["dual_registry"]):
        if path.is_file():
            payload = _load_structured(path)
            test_counts = [value for key, value in _walk(payload) if key.endswith("test_assets_opened")]
            if any(value != 0 for value in test_counts):
                issues.append(f"registry reports test access: {path}")

    result = {
        "version": VERSION,
        "status": "passed" if not issues else "blocked",
        "blocked_message": None if not issues else "BLOCKED: BEST CHECKPOINT EVIDENCE",
        "issues": issues,
        "selected": {key: _relative(root, value) for key, value in selected.items()},
        "d2_checkpoint": checkpoint_audit,
        "dose_position_equal_audit": dose_audit,
        "metadata_files": manifest,
        "checkpoint_bytes_copied": False,
        "image_assets_opened": 0,
        "test_assets_opened": 0,
    }
    (output / "evidence_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def resolve_selection(
    root: Path, candidates: dict[str, list[str]], overrides: dict[str, str | None]
) -> tuple[dict[str, Path], list[str]]:
    selected: dict[str, Path] = {}
    issues: list[str] = []
    for key, values in candidates.items():
        override = overrides.get(key)
        if override:
            path = Path(override).expanduser()
            selected[key] = path.resolve() if path.is_absolute() else (root / path).resolve()
        elif len(values) == 1:
            selected[key] = (root / values[0]).resolve()
        else:
            issues.append(f"{key}: expected one candidate, found {len(values)}")
    return selected, issues
