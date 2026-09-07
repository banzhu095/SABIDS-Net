from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_protocol_manifest(project_root: Path, manifest: Path | None = None) -> pd.DataFrame:
    root = Path(project_root).resolve()
    path = Path(manifest) if manifest else root / "Manifests" / "manifest_denoise.csv"
    table = pd.read_csv(path)
    required = {"sample_id", "group_id", "dataset", "split", "image_path", "clean_path"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    table = table.copy()
    table["original_split"] = table["split"].astype(str)
    external = table["dataset"].isin(["Duke17", "Duke28"])
    table.loc[external, "split"] = "external_test"
    table["position_id"] = table["group_id"].astype(str)
    table["frame_id"] = table.get("frame_index", table["sample_id"]).astype(str)
    for column in ("image_path", "clean_path"):
        table[column] = table[column].map(lambda value: str((root / str(value)).resolve()) if str(value).strip() else "")
    return table


def audit_protocol(table: pd.DataFrame) -> dict:
    pku = table[table["dataset"] == "PKU37"]
    leakage = pku.groupby("position_id")["split"].nunique()
    missing = []
    for row in table.itertuples():
        for column in ("image_path", "clean_path"):
            path = Path(getattr(row, column))
            if not path.is_file():
                missing.append({"sample_id": row.sample_id, "column": column, "path": str(path)})
    return {
        "rows_by_dataset_split": table.groupby(["dataset", "split"]).size().rename("rows").reset_index().to_dict("records"),
        "positions_by_dataset_split": table.groupby(["dataset", "split"])["position_id"].nunique().rename("positions").reset_index().to_dict("records"),
        "pku_positions_crossing_splits": sorted(leakage[leakage > 1].index.astype(str).tolist()),
        "duke_non_external_rows": int(((table["dataset"].isin(["Duke17", "Duke28"])) & (table["split"] != "external_test")).sum()),
        "missing_files": missing,
        "passed": not missing and not bool((leakage > 1).any()) and not bool(((table["dataset"].isin(["Duke17", "Duke28"])) & (table["split"] != "external_test")).any()),
    }


def development_rows(table: pd.DataFrame, split: str) -> pd.DataFrame:
    if split not in {"train", "val"}:
        raise ValueError("development_rows only accepts train or val")
    return table[(table["dataset"] == "PKU37") & (table["split"] == split)].copy()


def final_test_rows(table: pd.DataFrame) -> pd.DataFrame:
    return table[((table["dataset"] == "PKU37") & (table["split"] == "test")) | table["dataset"].isin(["Duke17", "Duke28"])].copy()
