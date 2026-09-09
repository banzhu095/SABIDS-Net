from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from .registry import stable_sha256
from .table_store import merge_records


def _atomic_yaml(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            yaml.safe_dump(value, stream, sort_keys=False, allow_unicode=True)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def merge_tracks(run_dir: Path, classical_dir: Path) -> None:
    run, classical = run_dir.resolve(), classical_dir.resolve()
    source_config = classical / "configs" / "locked_classical_configs.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if config.get("status") != "locked_on_pku37_validation":
        raise RuntimeError("classical track is incomplete; refusing merge")
    config["track_provenance"] = {"track": "classical", "path": str(classical), "config_sha256": stable_sha256(config.get("methods", {}))}
    destination_config = run / "configs" / "locked_classical_configs.yaml"
    if destination_config.is_file():
        existing = yaml.safe_load(destination_config.read_text(encoding="utf-8")) or {}
        if existing.get("status") not in {"unlocked", "partially_locked_on_pku37_validation", "locked_on_pku37_validation"}:
            raise RuntimeError(f"unexpected destination classical status: {existing.get('status')}")
        prior_methods = existing.get("methods", {})
        for method in set(prior_methods) & set(config["methods"]):
            if stable_sha256(prior_methods[method]) != stable_sha256(config["methods"][method]):
                raise RuntimeError(f"classical config provenance conflict for {method}")
    _atomic_yaml(destination_config, config)

    tables = {
        "parameter_search_results.csv": (["method_id", "search_phase", "search_round", "candidate_uid", "sample_id"], ["candidate_json", "source_sha256"]),
        "parameter_search_partial.csv": (["method_id", "search_phase", "search_round", "candidate_uid", "sample_id"], ["candidate_json", "source_sha256"]),
        "selected_parameters.csv": (["method_id"], ["status", "selection_rule"]),
        "runtime_records.csv": (["method_id", "seed", "dataset", "sample_id"], ["config_sha256", "checkpoint_sha256"]),
    }
    for name, (keys, consistency) in tables.items():
        source = classical / "metrics" / name
        if source.is_file() and source.stat().st_size:
            frame = pd.read_csv(source)
            frame["provenance_track"] = "classical"
            frame["provenance_root"] = str(classical)
            merge_records(run / "metrics" / name, frame, run, keys, consistency, replace_consistent=False)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--classical-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    merge_tracks(args.run_dir, args.classical_dir)


if __name__ == "__main__":
    main()
