from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .io import sha256_file


HASH_COLUMNS = ("config_sha256", "checkpoint_sha256", "output_sha256", "sha256")


def normalize_hash_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in HASH_COLUMNS:
        if column in result.columns:
            result[column] = result[column].fillna("").astype(str)
    return result


def atomic_write_csv(frame: pd.DataFrame, path: Path, run_dir: Path, backup: bool = True) -> None:
    path = Path(path); run_dir = Path(run_dir).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.is_file():
        backup_dir = run_dir / "audit" / "backups"; backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup_path = backup_dir / f"{path.name}.{stamp}.{sha256_file(path)[:12]}.bak"
        shutil.copy2(path, backup_path)
    with tempfile.NamedTemporaryFile("w", suffix=".tmp", prefix=f"{path.name}.", dir=path.parent, delete=False, encoding="utf-8", newline="") as stream:
        temporary = Path(stream.name)
        frame.to_csv(stream, index=False)
    temporary.replace(path)


def merge_records(
    path: Path,
    incoming: pd.DataFrame,
    run_dir: Path,
    identity_keys: Sequence[str],
    consistency_fields: Sequence[str],
    *,
    replace_consistent: bool = False,
) -> pd.DataFrame:
    """Merge a staged result table while rejecting provenance conflicts."""
    incoming = normalize_hash_columns(incoming)
    existing = normalize_hash_columns(pd.read_csv(path)) if Path(path).is_file() and Path(path).stat().st_size else pd.DataFrame()
    if incoming.empty:
        return existing
    missing = [column for column in identity_keys if column not in incoming.columns]
    if missing: raise ValueError(f"incoming table missing identity columns: {missing}")
    for identity, group in incoming.groupby(list(identity_keys), dropna=False, sort=False):
        conflicts = {field: group[field].astype(str).unique().tolist() for field in consistency_fields if field in group and group[field].astype(str).nunique() > 1}
        if conflicts:
            raise RuntimeError(f"incoming result provenance conflict for {identity}: {conflicts}")
    if existing.empty:
        merged = incoming.drop_duplicates(list(identity_keys), keep="last")
        atomic_write_csv(merged, path, run_dir, backup=False); return merged
    # Older development inventories may lack newly added provenance columns;
    # retain them, but only compare rows for which the complete logical key is available.
    original_existing_columns = set(existing.columns)
    for column in incoming.columns:
        if column not in existing.columns: existing[column] = ""
    for column in existing.columns:
        if column not in incoming.columns: incoming[column] = ""
    incoming = incoming[existing.columns]
    comparable = existing.copy()
    for record in incoming.to_dict("records"):
        mask = pd.Series(True, index=comparable.index)
        for key in identity_keys: mask &= comparable[key].astype(str) == str(record[key])
        matches = comparable[mask]
        if matches.empty: continue
        old = matches.iloc[-1]
        conflicts = {}
        for field in consistency_fields:
            if field not in original_existing_columns:
                continue
            old_value = "" if pd.isna(old.get(field, "")) else str(old.get(field, ""))
            new_value = "" if pd.isna(record.get(field, "")) else str(record.get(field, ""))
            if old_value != new_value: conflicts[field] = (old_value, new_value)
        if conflicts:
            identity = {key: record[key] for key in identity_keys}
            raise RuntimeError(f"result provenance conflict for {identity}: {conflicts}")
    combined = pd.concat([existing, incoming], ignore_index=True)
    merged = combined.drop_duplicates(list(identity_keys), keep="last" if replace_consistent else "first")
    atomic_write_csv(merged, path, run_dir, backup=True)
    return merged
