from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.utils import write_json


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_once(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = table.to_csv(index=False, lineterminator="\n")
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != rendered:
            raise FileExistsError(f"Existing split artifact differs; refusing overwrite: {path}")
        return
    path.write_text(rendered, encoding="utf-8")


def _balanced_assignment(features: pd.DataFrame, n_folds: int, seed: int) -> dict[str, int]:
    numeric = features.select_dtypes(include=[np.number]).copy()
    numeric = numeric.drop(columns=[c for c in numeric if c in {"n_frames"}], errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
    scale = numeric.std(ddof=0).replace(0, 1.0)
    z = (numeric - numeric.mean()) / scale
    rng = np.random.default_rng(seed)
    projection = z.to_numpy() @ rng.normal(size=z.shape[1]) if z.shape[1] else np.zeros(len(z))
    order = np.argsort(projection, kind="stable")
    fold_order: list[int] = []
    while len(fold_order) < len(order):
        block = list(range(n_folds))
        if (len(fold_order) // n_folds) % 2:
            block.reverse()
        fold_order.extend(block)
    return {str(features.iloc[index]["group_id"]): int(fold_order[rank]) for rank, index in enumerate(order)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build leakage-safe position-level PKU37 CV folds")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol", default="configs/current/input_oracle_cv/protocol.yaml")
    parser.add_argument("--output", default="runs/input_oracle_cv")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--fold", type=int, default=None, help="Only validate/print this fold; registry remains global")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    output = (root / args.output).resolve()
    protocol_path = (root / args.protocol).resolve()
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    audit_path = output / "audit/audit_summary.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Run Phase 0 audit first: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or not audit.get("training_authorized"):
        raise RuntimeError(f"Phase 0 is blocked; folds must not be generated. Inspect {audit_path}")

    dev = [str(x) for x in audit["development_groups"]]
    n_folds = int(protocol.get("preferred_folds", 4))
    if len(dev) // n_folds < 3:
        n_folds = max(2, len(dev) // 3)
    if len(dev) < n_folds * 3:
        raise RuntimeError("Cannot give every validation fold at least three independent positions")
    characteristics = pd.read_csv(output / "position_characteristics.csv")
    characteristics = characteristics[characteristics["group_id"].astype(str).isin(dev)].sort_values("group_id").reset_index(drop=True)
    if set(characteristics["group_id"].astype(str)) != set(dev):
        raise RuntimeError("Position-characteristics table does not exactly cover development groups")
    assignment = _balanced_assignment(characteristics, n_folds, args.seed)
    registry_rows = []
    source = pd.read_csv(Path(audit["source_manifest"]), dtype=str).fillna("")
    source = source[source["group_id"].isin(dev)].copy()
    joint = pd.read_csv((root / protocol["joint_manifest"]).resolve(), dtype=str).fillna("")
    sealed = set(map(str, audit["sealed_test_groups"]))
    manifests = root / "Manifests/input_oracle_cv"
    split_root = output / "splits"
    split_hashes = {}
    for fold in range(n_folds):
        val_groups = sorted(g for g, f in assignment.items() if f == fold)
        train_groups = sorted(set(dev) - set(val_groups))
        if len(val_groups) < 3 or set(train_groups) & set(val_groups):
            raise RuntimeError(f"Invalid fold {fold}")
        for group in sorted(dev):
            registry_rows.append({"group_id": group, "fold": fold, "split": "val" if group in val_groups else "train", "assignment_seed": args.seed})
        train_path = split_root / f"fold_{fold}_train.csv"
        val_path = split_root / f"fold_{fold}_val.csv"
        _write_csv_once(pd.DataFrame({"group_id": train_groups}), train_path)
        _write_csv_once(pd.DataFrame({"group_id": val_groups}), val_path)
        seg = source.copy()
        seg["split"] = seg["group_id"].map(lambda g: "val" if g in val_groups else "train")
        seg = seg.sort_values(["split", "group_id", "frame_index", "sample_id"])
        seg_path = manifests / f"fold_{fold}_seg.csv"
        _write_csv_once(seg, seg_path)

        # D0 may use Duke train/val and non-sealed PKU positions, but never this
        # fold's validation positions or any configured/legacy sealed position.
        d0 = joint[~joint["group_id"].isin(sealed | set(val_groups))].copy()
        d0 = d0[~((d0["dataset"] != "PKU37") & (d0["split"] == "test"))].copy()
        d0["split"] = np.where(d0["dataset"].eq("PKU37"), "train", d0["split"])
        d0 = d0[d0["split"].isin(["train", "val"])].sort_values(["split", "dataset", "group_id", "frame_index", "sample_id"])
        if set(d0.loc[d0["split"] == "train", "group_id"]) & (sealed | set(val_groups)):
            raise RuntimeError(f"Fold {fold} D0 leakage detected")
        d0_path = manifests / f"fold_{fold}_d0.csv"
        _write_csv_once(d0, d0_path)
        split_hashes[str(fold)] = {
            "train_groups": train_groups,
            "val_groups": val_groups,
            "train_registry_sha256": _sha(train_path),
            "val_registry_sha256": _sha(val_path),
            "seg_manifest": str(seg_path.relative_to(root)).replace("\\", "/"),
            "seg_manifest_sha256": _sha(seg_path),
            "d0_manifest": str(d0_path.relative_to(root)).replace("\\", "/"),
            "d0_manifest_sha256": _sha(d0_path),
            "d0_train_groups": sorted(set(d0.loc[d0["split"] == "train", "group_id"])),
        }
    registry = pd.DataFrame(registry_rows).sort_values(["fold", "split", "group_id"])
    _write_csv_once(registry, split_root / "fold_registry.csv")
    coverage = registry[registry["split"] == "val"].groupby("group_id").size().to_dict()
    report = {
        "status": "passed" if set(coverage) == set(dev) and set(coverage.values()) == {1} else "blocked",
        "protocol_sha256": _sha(protocol_path), "phase0_audit_sha256": _sha(audit_path),
        "assignment_seed": args.seed, "n_folds": n_folds, "n_development_groups": len(dev),
        "validation_coverage": coverage, "folds": split_hashes,
        "sealed_test_groups_metadata_only": sorted(sealed), "test_assets_opened": 0,
        "dry_run": args.dry_run, "smoke_test": args.smoke_test,
    }
    write_json(report, split_root / "split_audit.json")
    if report["status"] != "passed":
        raise RuntimeError("Generated fold coverage audit failed")
    if args.fold is not None and str(args.fold) not in split_hashes:
        raise ValueError(f"Fold {args.fold} is outside 0..{n_folds - 1}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
