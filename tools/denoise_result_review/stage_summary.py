from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import METHOD_ORDER
from .method_literature import implementation_summary, load_literature
from .workbook import write_workbook


def _optional_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.is_file() else pd.DataFrame()


def _parameters(run: Path) -> pd.DataFrame:
    selected = _optional_csv(run / "metrics" / "selected_parameters.csv")
    if selected.empty: return pd.DataFrame(columns=["method_id", "locked_parameters"])
    keys = [column for column in selected if column not in {"method_id", "dataset", "split", "status"}]
    rows = []
    for method, part in selected.groupby("method_id"):
        records = part[keys].where(pd.notna(part[keys]), None).to_dict("records")
        rows.append({"method_id": method, "locked_parameters": json.dumps(records, ensure_ascii=False, sort_keys=True)})
    return pd.DataFrame(rows)


def build_stage_summary(project_root: str | Path, run_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Path]:
    root, run = Path(project_root).resolve(), Path(run_dir).resolve()
    output = Path(output_dir).resolve() if output_dir else run / "reports"
    output.mkdir(parents=True, exist_ok=True)
    literature = load_literature()
    implementation = implementation_summary(root, run)
    implementation = implementation.merge(_parameters(run), on="method_id", how="left")
    datasets = _optional_csv(run / "metrics" / "per_dataset_metrics.csv")
    seeds = _optional_csv(run / "metrics" / "per_seed_metrics.csv")
    runtime = _optional_csv(run / "metrics" / "runtime_summary.csv")
    complexity = _optional_csv(run / "metrics" / "model_complexity.csv")
    performance = datasets[datasets.method_id.astype(str).isin(METHOD_ORDER)].copy() if not datasets.empty and "method_id" in datasets else pd.DataFrame()
    if not seeds.empty: seeds.to_csv(output / "method_seed_stability.csv", index=False)
    literature.to_csv(output / "method_literature.csv", index=False)
    implementation.to_csv(output / "method_implementation_summary.csv", index=False)
    performance.to_csv(output / "method_performance_summary.csv", index=False)
    runtime_out = runtime.merge(complexity, on="method_id", how="outer", suffixes=("_runtime", "_complexity")) if not runtime.empty and not complexity.empty else (runtime if not runtime.empty else complexity)
    runtime_out.to_csv(output / "method_runtime_summary.csv", index=False)
    merged = literature.merge(implementation, on="method_id", how="left", suffixes=("_literature", "_implementation"))
    lines = ["# OCT denoising stage summary", "", f"Source run: `{run}`", "", "Local harmonized metrics are reported separately from paper metadata; no published metric is inserted into a local performance column.", "", "## Method status", "", merged[["method_id", "display_name", "category", "supervision", "status"]].to_markdown(index=False), "", "## Harmonized performance", "", performance.to_markdown(index=False) if not performance.empty else "No harmonized performance table is available.", "", "## Current limitations", "", "Methods absent from the locked registry or successful per-image records are marked `missing/not_completed`; no image or metric is synthesized."]
    markdown = output / "denoise_stage_summary.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    workbook = output / "denoise_stage_summary.xlsx"
    write_workbook(workbook, {
        "Method Literature": literature, "Method Configuration": implementation,
        "Performance": performance, "Seed Stability": seeds, "Runtime Complexity": runtime_out,
    }, readme=[["OCT denoising stage summary", ""], ["Source run", str(run)], ["Metric scope", "Project harmonized metrics; paper bibliographic data remain separate."], ["Missing methods", ", ".join(implementation.loc[implementation.status != "completed", "method_id"].astype(str))]])
    return {"markdown": markdown, "workbook": workbook, "literature": output / "method_literature.csv", "implementation": output / "method_implementation_summary.csv", "performance": output / "method_performance_summary.csv", "runtime": output / "method_runtime_summary.csv"}
