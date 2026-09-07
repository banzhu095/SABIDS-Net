from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd

from .metrics import METRIC_COLUMNS


KEYS = ["dataset", "split", "position_id", "method_id", "seed"]


def aggregate(per_image: pd.DataFrame) -> dict[str, pd.DataFrame]:
    metrics = [column for column in METRIC_COLUMNS if column in per_image]
    working = per_image.copy()
    if "seed" not in working:
        working["seed"] = 0
    position = working.groupby(KEYS, dropna=False, as_index=False)[metrics].mean(numeric_only=True)
    position["frame_count"] = working.groupby(KEYS, dropna=False).size().to_numpy()
    seed = position.groupby(["dataset", "split", "method_id", "seed"], dropna=False, as_index=False)[metrics].mean(numeric_only=True)
    seed["position_count"] = position.groupby(["dataset", "split", "method_id", "seed"], dropna=False).size().to_numpy()
    dataset = seed.groupby(["dataset", "split", "method_id"], as_index=False)[metrics].mean(numeric_only=True)
    seed_std = seed.groupby(["dataset", "split", "method_id"], as_index=False)[metrics].std(numeric_only=True).add_suffix("_seed_std")
    seed_std = seed_std.rename(columns={"dataset_seed_std": "dataset", "split_seed_std": "split", "method_id_seed_std": "method_id"})
    dataset = dataset.merge(seed_std, on=["dataset", "split", "method_id"], how="left")
    return {"per_position_metrics": position, "per_seed_metrics": seed, "per_dataset_metrics": dataset}


def paired_differences(position: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in METRIC_COLUMNS if column in position]
    rows = []
    requested = [("noisy_identity", method) for method in sorted(position.method_id.unique()) if method != "noisy_identity"]
    requested += [("bm3d_standard", method) for method in sorted(position.method_id.unique()) if method != "bm3d_standard"]
    requested += [("nafnet_paired", "dncnn_paired")]
    for dataset, dataset_rows in position.groupby("dataset"):
        collapsed = dataset_rows.groupby(["position_id", "method_id"], as_index=False)[metrics].mean(numeric_only=True)
        for baseline, method in dict.fromkeys(requested):
            left = collapsed[collapsed.method_id == baseline].set_index("position_id")
            right = collapsed[collapsed.method_id == method].set_index("position_id")
            common = left.index.intersection(right.index)
            if common.empty:
                continue
            for metric in metrics:
                values = right.loc[common, metric] - left.loc[common, metric]
                rows.append({"dataset": dataset, "baseline_method": baseline, "method_id": method, "metric": metric, "n_positions": len(common), "mean_paired_difference": float(values.mean())})
    return pd.DataFrame(rows)


def bootstrap_confidence_intervals(position: pd.DataFrame, iterations: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    metrics = [column for column in METRIC_COLUMNS if column in position]
    rows = []
    for (dataset, method), group in position.groupby(["dataset", "method_id"]):
        collapsed = group.groupby("position_id", as_index=False)[metrics].mean(numeric_only=True)
        for metric in metrics:
            values = collapsed[metric].dropna().to_numpy(float)
            if not len(values):
                continue
            if len(values) == 1:
                low = high = values[0]
            else:
                indices = rng.integers(0, len(values), size=(iterations, len(values)))
                samples = values[indices].mean(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
            rows.append({"dataset": dataset, "method_id": method, "metric": metric, "n_positions": len(values), "mean": float(values.mean()), "ci95_low": float(low), "ci95_high": float(high), "iterations": iterations, "seed": seed})
    return pd.DataFrame(rows)
