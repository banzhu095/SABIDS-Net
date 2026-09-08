"""Write a fixed, within-split, cross-position guidance permutation."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.protocol_lock import load_protocol_lock


def make_mapping(table: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for split, split_table in table.groupby("split", sort=True):
        groups = sorted(split_table["group_id"].astype(str).unique())
        if len(groups) < 2:
            raise RuntimeError(f"Split {split!r} needs at least two positions for shuffle control")
        shuffled = groups[:]
        for _ in range(1000):
            rng.shuffle(shuffled)
            if all(left != right for left, right in zip(groups, shuffled)):
                break
        else:
            raise RuntimeError(f"Could not create a derangement for split {split!r}")
        target_group = dict(zip(groups, shuffled))
        by_group = {key: sorted(value["sample_id"].astype(str)) for key, value in split_table.groupby("group_id")}
        for item in split_table.sort_values("sample_id").itertuples(index=False):
            candidates = by_group[target_group[str(item.group_id)]]
            index = sum(ord(char) for char in str(item.sample_id)) % len(candidates)
            rows.append({"sample_id": str(item.sample_id), "guidance_sample_id": candidates[index], "split": split, "source_group_id": str(item.group_id), "guidance_group_id": target_group[str(item.group_id)], "seed": seed})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--expected-protocol-lock", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    lock_path = Path(args.expected_protocol_lock)
    if not lock_path.is_absolute(): lock_path = root / lock_path
    lock = load_protocol_lock(lock_path)
    manifest_root = Path(str(lock["manifest_root"]))
    if not manifest_root.is_absolute(): manifest_root = (root / manifest_root).resolve()
    table = pd.read_csv(manifest_root / "train_joint.csv", dtype=str).fillna("")
    table = table[table["split"].isin(["train", "val"])].copy()
    outputs = []
    for seed in args.seeds:
        output = manifest_root / f"interaction_shuffle_seed{seed}.csv"
        mapping = make_mapping(table, seed)
        if output.exists():
            if not args.reuse_existing: raise FileExistsError(f"Refusing to overwrite shuffle map: {output}; use --reuse-existing after audit")
            existing = pd.read_csv(output, dtype=str).fillna("")
            comparable = ["sample_id", "guidance_sample_id", "split", "source_group_id", "guidance_group_id"]
            if not existing[comparable].equals(mapping.astype(str)[comparable]):
                raise RuntimeError(f"Existing shuffle mapping differs: {output}")
            outputs.append(str(output)); continue
        if (mapping["sample_id"] == mapping["guidance_sample_id"]).any() or (mapping["source_group_id"] == mapping["guidance_group_id"]).any():
            raise AssertionError("Generated mapping contains self guidance")
        mapping.to_csv(output, index=False, encoding="utf-8-sig")
        outputs.append(str(output))
    print(json.dumps({"status": "passed", "outputs": outputs, "test_assets_opened": 0}, indent=2))


if __name__ == "__main__": main()
