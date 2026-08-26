from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-free grouped data splits")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-key", default="group_id")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--external-datasets", nargs="*", default=[])
    return parser.parse_args()


def split_train_val(table: pd.DataFrame, group_key: str, fraction: float, seed: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    train_index, val_index = next(splitter.split(table, groups=table[group_key]))
    return table.index[train_index], table.index[val_index]


def assert_no_leakage(table: pd.DataFrame, group_key: str) -> None:
    leaked = table.groupby(group_key)["split"].nunique()
    leaked = leaked[leaked > 1]
    if not leaked.empty:
        raise RuntimeError(f"Leakage detected for {group_key}: {leaked.index.tolist()[:10]}")


def main() -> None:
    args = parse_args()
    source = pd.read_csv(args.manifest, dtype=str).fillna("")
    if args.group_key not in source.columns:
        raise ValueError(f"Missing group key: {args.group_key}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.external_datasets:
        external = source["dataset"].isin(args.external_datasets)
        development = source[~external].copy()
        train_index, val_index = split_train_val(
            development, args.group_key, args.val_fraction, args.seed
        )
        result = source.copy()
        result["split"] = "test"
        result.loc[train_index, "split"] = "train"
        result.loc[val_index, "split"] = "val"
        assert_no_leakage(result, args.group_key)
        target = output_dir / "manifest_external_test.csv"
        result.to_csv(target, index=False, encoding="utf-8-sig")
        print(f"Saved {target}")
        return

    unique_groups = source[[args.group_key]].drop_duplicates().reset_index(drop=True)
    splitter = GroupKFold(n_splits=args.folds)
    for fold, (_, test_group_index) in enumerate(
        splitter.split(unique_groups, groups=unique_groups[args.group_key])
    ):
        test_groups = set(unique_groups.iloc[test_group_index][args.group_key])
        test_mask = source[args.group_key].isin(test_groups)
        development = source[~test_mask].copy()
        train_index, val_index = split_train_val(
            development, args.group_key, args.val_fraction, args.seed + fold
        )
        result = source.copy()
        result["split"] = "test"
        result.loc[train_index, "split"] = "train"
        result.loc[val_index, "split"] = "val"
        assert_no_leakage(result, args.group_key)
        target = output_dir / f"manifest_fold{fold}.csv"
        result.to_csv(target, index=False, encoding="utf-8-sig")
        print(f"Saved {target}")


if __name__ == "__main__":
    main()

