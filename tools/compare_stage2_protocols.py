from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.utils import write_json

HASH_SCHEMA_VERSION = "stage2-fingerprint-v2"
FINGERPRINT_FIELDS = (
    "manifest_sha256", "effective_split_sha256", "label_assets_raw_sha256",
    "label_assets_decoded_sha256", "initialization_checkpoint_sha256",
)
IGNORED_PROTOCOL_PATHS = {
    "loss.definition_version", "train.output_dir", "runtime.config_path",
    "data.manifest", "data.root", "train.pretrained", "train.resume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Stage 2 data identity and single-factor comparability.")
    parser.add_argument("--experiments", nargs="+", required=True, metavar="NAME=RUN_DIR_OR_HISTORY_CSV")
    parser.add_argument("--reference", default=None)
    parser.add_argument("--output", default="outputs/stage2_protocol_comparison")
    parser.add_argument("--allowed-differences", action="append", default=[], metavar="NAME=DOT.PATH,DOT.PATH")
    return parser.parse_args()


def _best_history(path: Path) -> Dict[str, Any]:
    table = pd.read_csv(path)
    if table.empty:
        return {}
    monitor = "val_vessel_soft_dice"
    index = table[monitor].idxmax() if monitor in table else table.index[-1]
    row = table.loc[index]
    result: Dict[str, Any] = {"history_rows": int(len(table)), "best_epoch": int(row.get("epoch", index + 1))}
    for key in (
        "val_vessel_soft_dice", "val_vessel_dice", "val_vessel_precision",
        "val_vessel_recall", "val_vessel_roi_dice", "val_layer_dice",
        "val_n_groups_vessel_dice", "train_eval_n_groups_vessel_dice",
    ):
        if key in row and pd.notna(row[key]):
            result[key] = float(row[key])
    return result


def _flatten(value: Any, prefix: str = "") -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    result: Dict[str, Any] = {}
    for key in sorted(value):
        path = f"{prefix}.{key}" if prefix else str(key)
        result.update(_flatten(value[key], path))
    return result


def _protocol(config: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in _flatten(config).items()
            if key not in IGNORED_PROTOCOL_PATHS
            and not key.startswith("runtime.")
            and not key.startswith("evaluation.output")}


def _load_inventory(run_dir: Path) -> Dict[str, Dict[str, Any]] | None:
    path = run_dir / "label_asset_inventory.json"
    if not path.is_file():
        return None
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row.get("asset_id") or f"{row.get('group_id')}|{row.get('column')}"): row
        for row in rows
    }


def _load_experiment(name: str, path: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "name": name, "source": str(path.resolve()), "run_dir": None,
        "metadata_source": None, "protocol": None, "label_inventory": None,
    }
    if path.is_file():
        result.update(_best_history(path))
        return result
    result["run_dir"] = str(path.resolve())
    metadata_path, config_path, history_path = (
        path / "run_metadata.json", path / "resolved_config.yaml", path / "history.csv"
    )
    if metadata_path.is_file():
        result.update(json.loads(metadata_path.read_text(encoding="utf-8")))
        result["metadata_source"] = str(metadata_path.resolve())
    if history_path.is_file():
        result.update(_best_history(history_path))
    if config_path.is_file():
        result["protocol"] = _protocol(load_config(config_path))
        result["resolved_config_source"] = str(config_path.resolve())
    result["label_inventory"] = _load_inventory(path)
    return result


def _field_comparison(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for field in FINGERPRINT_FIELDS:
        left, right = reference.get(field), candidate.get(field)
        status = "unknown" if left is None or right is None else ("matched" if left == right else "different")
        fields[field] = {"status": status, "reference": left, "candidate": right}
    left, right = reference.get("hash_schema_version"), candidate.get("hash_schema_version")
    fields["hash_schema_version"] = {
        "status": "matched" if left == right == HASH_SCHEMA_VERSION else "unknown",
        "reference": left, "candidate": right,
    }
    left, right = reference.get("metadata_version"), candidate.get("metadata_version")
    fields["metadata_version"] = {
        "status": "matched" if left == right == 2 else "unknown",
        "reference": left, "candidate": right,
    }
    if reference.get("missing_label_assets") or candidate.get("missing_label_assets"):
        for field in ("label_assets_raw_sha256", "label_assets_decoded_sha256"):
            fields[field]["status"] = "unknown"
            fields[field]["reason"] = "one or more referenced label assets were missing"
    statuses = {value["status"] for value in fields.values()}
    overall = "different" if "different" in statuses else ("unknown" if "unknown" in statuses else "matched")
    return {"status": overall, "fields": fields}


def _label_differences(reference: Dict[str, Dict[str, Any]] | None,
                       candidate: Dict[str, Dict[str, Any]] | None) -> Dict[str, Any]:
    if reference is None or candidate is None:
        return {"status": "unknown", "reason": "label inventory missing"}
    rows = []
    for key in sorted(set(reference) | set(candidate)):
        left, right = reference.get(key), candidate.get(key)
        if left is None or right is None:
            rows.append({"asset": key, "status": "missing", "reference": left, "candidate": right})
            continue
        raw_equal = left.get("raw_sha256") == right.get("raw_sha256")
        decoded_equal = left.get("decoded_sha256") == right.get("decoded_sha256")
        if not (raw_equal and decoded_equal):
            rows.append({
                "asset": key,
                "status": "serialization_only" if decoded_equal else "decoded_content_different",
                "reference": left, "candidate": right,
            })
    return {"status": "matched" if not rows else "different", "differences": rows}


def _protocol_comparison(reference: Dict[str, Any] | None, candidate: Dict[str, Any] | None,
                         allowed: Iterable[str]) -> Dict[str, Any]:
    if reference is None or candidate is None:
        return {"status": "unknown", "reason": "resolved config missing"}
    allowed_set = set(allowed)
    differences, unexpected = {}, {}
    for key in sorted(set(reference) | set(candidate)):
        if reference.get(key) != candidate.get(key):
            item = {"reference": reference.get(key), "candidate": candidate.get(key)}
            differences[key] = item
            if key not in allowed_set:
                unexpected[key] = item
    unused = sorted(allowed_set - set(differences))
    return {
        "status": "matched" if not unexpected and not unused else "different",
        "all_differences": differences, "unexpected_differences": unexpected,
        "unused_allowed_differences": unused,
    }


def _parse_allowed(specifications: Iterable[str]) -> Dict[str, list[str]]:
    result: Dict[str, list[str]] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Expected NAME=DOT.PATH,DOT.PATH, got {specification!r}")
        name, paths = specification.split("=", 1)
        result[name] = [path.strip() for path in paths.split(",") if path.strip()]
    return result


def main() -> None:
    args = parse_args()
    allowed = _parse_allowed(args.allowed_differences)
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
        identity = _field_comparison(reference, experiment)
        protocol = _protocol_comparison(reference.get("protocol"), experiment.get("protocol"), allowed.get(name, []))
        labels = _label_differences(reference.get("label_inventory"), experiment.get("label_inventory"))
        comparisons[name] = {
            "identity": identity, "training_protocol": protocol, "label_assets": labels,
            "causal_comparison_status": (
                "matched" if identity["status"] == protocol["status"] == "matched"
                else "different" if identity["status"] == "different" or protocol["status"] == "different"
                else "unknown"
            ),
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    public_rows = [{key: value for key, value in item.items() if key not in {"protocol", "label_inventory"}}
                   for item in experiments.values()]
    pd.DataFrame(public_rows).to_csv(output / "protocols.csv", index=False, encoding="utf-8-sig")
    write_json({
        "hash_schema_version_required": HASH_SCHEMA_VERSION, "reference": reference_name,
        "experiments": public_rows, "comparisons": comparisons,
        "warning": "Only causal_comparison_status=matched supports a declared single-factor comparison. unknown is not evidence of equality; historical hashes must not be backfilled without snapshots.",
    }, output / "comparison.json")
    for name, comparison in comparisons.items():
        write_json(comparison["label_assets"], output / f"label_differences_{name}.json")
        print(f"{name}: identity={comparison['identity']['status']} protocol={comparison['training_protocol']['status']} causal={comparison['causal_comparison_status']}")


if __name__ == "__main__":
    main()
