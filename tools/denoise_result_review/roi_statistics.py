from __future__ import annotations

import numpy as np
import pandas as pd


IDENTIFIERS = {"seed", "roi_size", "ssim_win_size", "bright_outlier_threshold", "center_x", "center_y", "x0", "y0", "x1", "y1"}


def aggregate_roi_metrics(per_roi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = [column for column in per_roi.select_dtypes(include="number").columns if column not in IDENTIFIERS]
    sample = per_roi.groupby(["dataset", "split", "position_id", "sample_id", "tissue", "method_id"], as_index=False)[numeric].mean()
    position = sample.groupby(["dataset", "split", "position_id", "tissue", "method_id"], as_index=False)[numeric].mean()
    return sample, position


def method_differences(position: pd.DataFrame) -> pd.DataFrame:
    metrics = [column for column in position.select_dtypes(include="number") if column not in IDENTIFIERS]
    rows = []
    keys = ["dataset", "split", "position_id", "tissue"]
    for group_key, part in position.groupby(keys):
        baseline = part[part.method_id == "noisy_identity"]
        if baseline.empty: continue
        base = baseline.iloc[0]
        for row in part.itertuples(index=False):
            if row.method_id == "noisy_identity": continue
            item = dict(zip(keys, group_key)); item.update({"method_id": row.method_id, "baseline_method": "noisy_identity"})
            for metric in metrics: item[f"delta_{metric}"] = float(getattr(row, metric) - base[metric])
            rows.append(item)
    return pd.DataFrame(rows)


def position_bootstrap(position: pd.DataFrame, iterations: int = 10_000, seed: int = 42,
                       complete_position_count: int | None = None) -> pd.DataFrame:
    observed = position.position_id.nunique()
    if complete_position_count is None or observed != complete_position_count or observed < 2:
        return pd.DataFrame(columns=["analysis_scope", "method_id", "tissue", "metric", "mean", "ci_low", "ci_high", "n_positions", "iterations", "seed"])
    numeric = [column for column in position.select_dtypes(include="number") if column not in IDENTIFIERS]
    rng, rows = np.random.default_rng(seed), []
    for (method, tissue), part in position.groupby(["method_id", "tissue"]):
        by_position = part.groupby("position_id", as_index=False)[numeric].mean()
        n = len(by_position)
        indices = rng.integers(0, n, size=(iterations, n))
        for metric in numeric:
            values = by_position[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            if len(values) != n: continue
            draws = values[indices].mean(axis=1)
            rows.append({"analysis_scope": "fixed_roi_confirmatory_full_test_positions", "method_id": method, "tissue": tissue, "metric": metric, "mean": float(values.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "n_positions": n, "iterations": iterations, "seed": seed})
    return pd.DataFrame(rows)
