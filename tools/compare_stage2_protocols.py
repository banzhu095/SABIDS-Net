from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.utils import write_json


FINGERPRINT_FIELDS = (
    "manifest_sha256",
    "effective_split_sha256",
    "label_assets_sha256",
    "initialization_checkpoint_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Stage 2 protocols before making ablation claims."
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        metavar="NAME=RUN_DIR_OR_HISTORY_CSV",
    )
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output", default="outputs/stage2_protocol_comparison")
    return parser.parse_args()


def _best_history(history_path: Path) -> Dict[str, Any]:
    table = pd.read_csv(history_path)
    if table.empty:
        return {}
    monitor = "val_vessel_soft_dice"
    index = table[monitor].idxmax() if monitor in table else table.index[-1]
    row = table.loc[index]
    result: Dict[str, Any] = {
        "history_rows": int(len(table)),
        "best_epoch": int(row.get("epoch", index + 1)),
    }
    for key in (
        "val_vessel_soft_dice",
        "val_vessel_dice",
        "val_vessel_precision",
        "val_vessel_recall",
        "val_vessel_roi_dice",
        "val_layer_dice",
        "val_n_groups_vessel_dice",
        "train_eval_n_groups_vessel_dice",
    ):
        if key in row and pd.notna(row[key]):
            result[key] = float(row[key])
    return result


def _load_experiment(name: str, path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"name": name, "source": str(path.resolve())}
    if path.is_file():
        result.update(_best_history(path))
        result["fingerprint_status"] = "missing: history CSV has no protocol hashes"
        return result
    metadata_path = path / "run_metadata.json"
    config_path = path / "resolved_config.yaml"
    history_path = path / "history.csv"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_path.is_file()
        else {}
    )
    result.update(metadata)
    if history_path.is_file():
        result.update(_best_history(history_path))
    if config_path.is_file():
        config = load_config(config_path)
        result.update(
            {
                "seed": config.get("seed"),
                "target_size": config.get("data", {}).get("target_size"),
                "normalization": config.get("data", {}).get("normalization"),
                "samples_per_epoch": config.get("data", {}).get(
                    "samples_per_epoch"
                ),
                "loss_definition": config.get("loss", {}).get(
                    "definition_version"
                ),
                "vessel_supervision_mode": config.get("loss", {}).get(
                    "vessel_supervision_mode", "composite"
                ),
                "vessel_outside_weight": config.get("loss", {})
                .get("weights", {})
                .get("vessel_outside", 0.0),
                "containment_weight": config.get("loss", {})
                .get("weights", {})
                .get("containment", 0.0),
                "d2s_enabled": config.get("model", {}).get(
                    "enable_denoise_to_seg"
                ),
                "epochs": config.get("train", {}).get("epochs"),
                "learning_rate": config.get("train", {}).get("learning_rate"),
                "scheduler": config.get("train", {}).get("scheduler"),
            }
        )
    missing = [field for field in FINGERPRINT_FIELDS if not result.get(field)]
    result["fingerprint_status"] = (
        "complete" if not missing else f"missing: {', '.join(missing)}"
    )
    return result


def main() -> None:
    args = parse_args()
    experiments: Dict[str, Dict[str, Any]] = {}
    for specification in args.experiments:
        if "=" not in specification:
            raise ValueError(f"Expected NAME=PATH, got {specification!r}")
        name, raw_path = specification.split("=", 1)
        experiments[name] = _load_experiment(name, Path(raw_path).expanduser())
    reference_name = args.reference or next(iter(experiments))
    if reference_name not in experiments:
        raise ValueError(f"Unknown reference {reference_name!r}")
    reference = experiments[reference_name]
    comparisons = {}
    for name, experiment in experiments.items():
        differences = {
            field: {"reference": reference.get(field), "candidate": experiment.get(field)}
            for field in FINGERPRINT_FIELDS
            if reference.get(field) != experiment.get(field)
        }
        comparisons[name] = {
            "comparable_protocol": not differences
            and reference.get("fingerprint_status") == "complete"
            and experiment.get("fingerprint_status") == "complete",
            "fingerprint_differences": differences,
            "reference_missing_fingerprints": [
                field for field in FINGERPRINT_FIELDS if not reference.get(field)
            ],
            "candidate_missing_fingerprints": [
                field for field in FINGERPRINT_FIELDS if not experiment.get(field)
            ],
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(experiments.values()).to_csv(
        output / "protocols.csv", index=False, encoding="utf-8-sig"
    )
    write_json(
        {
            "reference": reference_name,
            "experiments": experiments,
            "comparisons": comparisons,
            "warning": (
                "A missing or different fingerprint forbids a single-factor "
                "performance attribution; history CSV metrics alone are insufficient."
            ),
        },
        output / "comparison.json",
    )
    for name, comparison in comparisons.items():
        print(
            f"{name}: comparable={comparison['comparable_protocol']} "
            f"differences={list(comparison['fingerprint_differences'])}"
        )


if __name__ == "__main__":
    main()
