"""Build a validation-only, evidence-traceable SABIDS-Net stage summary.

This command is intentionally records-first.  It never opens test assets, never
trains a model, and never invents a metric when a checkpoint or prediction is
missing.  The generated CSV/PNG/Markdown files are later assembled into XLSX
workbooks by ``tools/build_stage_summary_workbooks.mjs``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import yaml  # type: ignore
except ImportError:  # The desktop artifact runtime intentionally has no PyYAML.
    yaml = None


FORBIDDEN_PARTS = {"test", "test_results", "test-results", "sealed_test"}
CURRENT_ALIASES = {"E1-current", "E3-current", "E3b", "E3b-no-D2S"}
ERROR_METRICS = {
    "rmse", "mse", "mae", "hd95", "assd", "outside_gt_layer_fraction",
    "area_fraction_mae", "boundary_mae", "thickness_mae", "cnr_error",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--source-record", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-packaging", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true", help="Validate/package an already generated directory after workbook creation")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def is_forbidden(path: Path) -> bool:
    return any(part.lower() in FORBIDDEN_PARTS for part in path.parts)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str] | None = None) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0]) if rows else ["status", "reason"]
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def latest_record(root: Path) -> Path:
    candidates = []
    for child in (root / "reports" / "experiment_records").glob("20*"):
        if child.is_dir() and (child / "provenance.json").is_file():
            provenance = json.loads((child / "provenance.json").read_text(encoding="utf-8"))
            if provenance.get("test_metrics_exported") is False:
                candidates.append(child)
    if not candidates:
        raise FileNotFoundError("No validation-only experiment record was found")
    return sorted(candidates)[-1]


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        root: dict[str, Any] = {}
        stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw:
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            key, value = raw.strip().split(":", 1)
            while stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]
            value = value.strip()
            if not value:
                parent[key] = {}
                stack.append((indent, parent[key]))
                continue
            lowered = value.lower()
            if lowered in {"true", "false"}:
                parsed: Any = lowered == "true"
            elif lowered in {"null", "none", "~"}:
                parsed = None
            else:
                try:
                    parsed = float(value) if any(c in value for c in ".eE") else int(value)
                except ValueError:
                    parsed = value.strip("'\"")
            parent[key] = parsed
        return root
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def nested(cfg: dict[str, Any], dotted: str, default: Any = "") -> Any:
    value: Any = cfg
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def classify(alias: str, role: str, smoke: bool) -> tuple[str, str, bool]:
    a = alias.lower()
    if smoke:
        return "A", "engineering_smoke", False
    if alias in CURRENT_ALIASES:
        return "C", "current_stage2_ablation", True
    if any(token in a for token in ("e0", "e1-old", "e2-old", "e3-old", "history")) or "historical" in role:
        return "A", "historical_debug", False
    if "stage1" in a or "denoise" in a:
        return "B", "stage1_denoising", True
    if a in {"b0", "j00", "j10", "j01", "j11"}:
        return "E", "joint_factorial", True
    if a in {"i-noisy", "i-denoised"}:
        return "F", "input_factorial", True
    if "post" in a or "p0" in a:
        return "D", "inference_or_postprocess", False
    return "A", "unclassified_or_incomplete", False


def audit_runs(root: Path, source: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    registry = read_csv(source / "experiment_registry.csv")
    run_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for _, row in registry.iterrows():
        alias = str(row.get("experiment_alias", ""))
        run_id = str(row.get("run_id", alias))
        seen_ids.add(run_id)
        family, classification, ranked = classify(alias, str(row.get("role", "")), False)
        source_path = root / str(row.get("source_file", ""))
        run_rows.append({
            "run_id": run_id, "alias": alias, "family": family,
            "classification": classification, "ranking_eligible": ranked,
            "stage": row.get("stage", ""), "fold": row.get("fold", ""),
            "seed": row.get("seed", ""), "status": row.get("status", ""),
            "role": row.get("role", ""), "train_groups": row.get("train_groups", ""),
            "val_groups": row.get("val_groups", ""), "train_frames": row.get("train_frames", ""),
            "val_frames": row.get("val_frames", ""), "input_hw": row.get("input_hw", ""),
            "best_epoch": row.get("best_epoch", ""), "last_epoch": row.get("last_epoch", ""),
            "monitor": row.get("monitor", ""), "selection_status": row.get("selection_status", ""),
            "manifest_sha256": row.get("manifest_sha256", ""),
            "effective_split_sha256": row.get("effective_split_sha256", ""),
            "initialization_sha256": row.get("initialization_sha256", ""),
            "config_present": False, "metadata_present": False,
            "history_present": source_path.is_file(), "checkpoint_present": False,
            "source_file": row.get("source_file", ""), "evidence_scope": "validation-only archive metadata",
            "notes": "Source run files/checkpoint are absent locally unless separately inventoried",
        })
        cp_sha = str(row.get("checkpoint_sha256", ""))
        checkpoint_rows.append({
            "run_id": run_id, "alias": alias, "checkpoint_type": "best",
            "epoch": row.get("best_epoch", ""), "checkpoint_path": "MISSING_LOCAL_PATH",
            "checkpoint_sha256": cp_sha, "sha_source": "archived run metadata",
            "file_present": False, "selection_monitor": row.get("monitor", ""),
            "eligible_for_inference": False, "reason": "checkpoint binary not included in records-only archive",
        })
        if not source_path.is_file():
            missing.append({"family": family, "run_id": run_id, "asset": "history source file", "status": "MISSING", "reason": str(row.get("source_file", "")), "impact": "cannot independently replay history"})

    for history in (root / "runs" / "current").rglob("history.csv"):
        if is_forbidden(history):
            continue
        run_dir = history.parent
        run_id = safe_rel(run_dir, root).replace("/", "-")
        if run_id in seen_ids:
            continue
        config_path = run_dir / "resolved_config.yaml"
        metadata_path = run_dir / "run_metadata.json"
        cfg = load_yaml(config_path) if config_path.is_file() else {}
        history_df = read_csv(history)
        alias = run_dir.name
        smoke = "smoke" in {part.lower() for part in run_dir.parts}
        family, classification, ranked = classify(alias, "", smoke)
        run_rows.append({
            "run_id": run_id, "alias": alias, "family": family,
            "classification": classification, "ranking_eligible": ranked and not smoke,
            "stage": nested(cfg, "train.stage", nested(cfg, "stage", "")),
            "fold": nested(cfg, "data.fold", 0), "seed": nested(cfg, "train.seed", nested(cfg, "seed", "")),
            "status": "smoke_only" if smoke else "local_history",
            "role": "engineering smoke; not scientific evidence" if smoke else "local runtime",
            "train_groups": "", "val_groups": "", "train_frames": "", "val_frames": "",
            "input_hw": f"{nested(cfg, 'data.target_height', '')}x{nested(cfg, 'data.target_width', '')}",
            "best_epoch": "", "last_epoch": history_df["epoch"].max() if "epoch" in history_df else "",
            "monitor": nested(cfg, "train.monitor", ""), "selection_status": "not_ranked",
            "manifest_sha256": "", "effective_split_sha256": "", "initialization_sha256": "",
            "config_present": config_path.is_file(), "metadata_present": metadata_path.is_file(),
            "history_present": True, "checkpoint_present": any(run_dir.glob("*.pth")),
            "source_file": safe_rel(history, root), "evidence_scope": "local smoke/runtime",
            "notes": f"d2s={nested(cfg, 'model.d2s_enabled', nested(cfg, 'model.enable_denoise_to_seg', 'unknown'))}; s2d={nested(cfg, 'model.s2d_enabled', nested(cfg, 'model.enable_seg_to_denoise', 'unknown'))}; outside_bce={nested(cfg, 'loss.weights.vessel_outside', 'unknown')}",
        })
        if not config_path.is_file():
            missing.append({"family": family, "run_id": run_id, "asset": "resolved_config.yaml", "status": "MISSING", "reason": "not present", "impact": "protocol cannot be fully audited"})
        if not metadata_path.is_file():
            missing.append({"family": family, "run_id": run_id, "asset": "run_metadata.json", "status": "MISSING", "reason": "not present", "impact": "checkpoint/provenance audit incomplete"})
        for cp in sorted(run_dir.glob("*.pth")):
            checkpoint_rows.append({
                "run_id": run_id, "alias": alias, "checkpoint_type": cp.stem,
                "epoch": "", "checkpoint_path": safe_rel(cp, root), "checkpoint_sha256": sha256(cp),
                "sha_source": "computed", "file_present": True,
                "selection_monitor": nested(cfg, "train.monitor", ""), "eligible_for_inference": not smoke,
                "reason": "smoke checkpoint excluded" if smoke else "local file",
            })

    required_missing = [
        ("E", alias, "fixed-final validation result", "No formal local J factorial output; no values inferred from docs")
        for alias in ("B0", "J00", "J10", "J01", "J11")
    ] + [
        ("F", alias, "fixed-epoch-60 validation result", "No formal local input-factorial output")
        for alias in ("I-NOISY", "I-DENOISED")
    ] + [("F", "I-CLEAN", "completed experiment", "Planned only; explicitly outside current task")]
    for family, run_id, asset, reason in required_missing:
        missing.append({"family": family, "run_id": run_id, "asset": asset, "status": "MISSING", "reason": reason, "impact": "no performance claim or paired gain allowed"})

    return pd.DataFrame(run_rows), pd.DataFrame(checkpoint_rows), missing


def make_metrics(source: Path, output: Path, best: pd.DataFrame) -> None:
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        ("frame_metrics.csv", "frame_metrics_long.csv"),
        ("group_metrics.csv", "position_metrics_long.csv"),
        ("metrics_long.csv", "metrics_long.csv"),
    ):
        shutil.copy2(source / src_name, metrics_dir / dst_name)

    trajectory = read_csv(source / "metrics_long.csv")
    cols = [c for c in ["experiment_alias", "run_id", "seed", "fold", "epoch", "metric", "value", "unit", "aggregation_level", "source_file", "status"] if c in trajectory]
    trajectory[cols].to_csv(metrics_dir / "training_trajectory_summary.csv", index=False, encoding="utf-8-sig")

    comparisons = [("E3b", "E3-current"), ("E3b", "E3b-no-D2S"), ("E3b", "E1-current")]
    metric_cols = [c for c in best.columns if c.startswith("val_") and pd.api.types.is_numeric_dtype(best[c])]
    rows = []
    for left, right in comparisons:
        a = best[best["experiment_alias"] == left]
        b = best[best["experiment_alias"] == right]
        if a.empty or b.empty:
            continue
        for metric in metric_cols:
            av, bv = float(a.iloc[0][metric]), float(b.iloc[0][metric])
            if not (math.isfinite(av) and math.isfinite(bv)):
                continue
            short = metric.removeprefix("val_")
            direction = -1.0 if any(token in short for token in ERROR_METRICS) else 1.0
            rows.append({
                "experiment_family": "current_stage2_ablation", "comparison": f"{left} - {right}",
                "left": left, "right": right, "seed": 42, "fold": 0, "metric": short,
                "left_value": av, "right_value": bv, "raw_difference": av - bv,
                "improvement": direction * (av - bv), "positive_means": "improvement",
                "unit": "ratio", "pairing_level": "single-seed group-macro history best",
                "status": "supported_limited" if right != "E1-current" else "caution_multifactor",
                "limitation": "seed=42 only; best-history model-grid metrics; position pairs unavailable for all arms",
            })
    gains = pd.DataFrame(rows)
    gains.to_csv(metrics_dir / "paired_gains_by_seed.csv", index=False, encoding="utf-8-sig")
    gains.to_csv(metrics_dir / "paired_gains_summary.csv", index=False, encoding="utf-8-sig")
    write_csv(metrics_dir / "paired_gains_by_position.csv", [], ["experiment_family", "comparison", "group_id", "seed", "metric", "improvement", "status", "missing_reason"])

    placeholders = []
    for family, variants in (("joint_factorial", ["B0", "J00", "J10", "J01", "J11"]), ("input_experiment", ["I-NOISY", "I-DENOISED", "I-CLEAN(planned)"])):
        for variant in variants:
            placeholders.append({"experiment_family": family, "variant": variant, "status": "MISSING", "checkpoint_type": "last" if family == "joint_factorial" else "epoch60-last", "threshold": 0.5, "postprocess": "P0", "reason": "formal local validation output absent"})
    pd.DataFrame([r for r in placeholders if r["experiment_family"] == "joint_factorial"]).to_csv(metrics_dir / "joint_factorial.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([r for r in placeholders if r["experiment_family"] == "input_experiment"]).to_csv(metrics_dir / "input_experiment.csv", index=False, encoding="utf-8-sig")


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str = "") -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1920, 125), fill="#17365D")
    draw.text((70, 30), title, fill="white", font=_font(42, True))
    if subtitle:
        draw.text((72, 135), subtitle, fill="#666666", font=_font(24))
    return image, draw


def _axes(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], ymin: float, ymax: float, ylabel: str) -> None:
    x0, y0, x1, y1 = box
    draw.line((x0, y0, x0, y1), fill="#404040", width=3)
    draw.line((x0, y1, x1, y1), fill="#404040", width=3)
    for i in range(6):
        y = y1 - int((y1-y0)*i/5)
        value = ymin + (ymax-ymin)*i/5
        draw.line((x0, y, x1, y), fill="#E6E6E6", width=1)
        draw.text((x0-85, y-12), f"{value:.2f}", fill="#404040", font=_font(18))
    draw.text((x0-110, y0-38), ylabel, fill="#404040", font=_font(21, True))


def _legend(draw: ImageDraw.ImageDraw, items: list[tuple[str, str]], x: int, y: int) -> None:
    for i, (label, color) in enumerate(items):
        yy = y + i * 34
        draw.rectangle((x, yy, x+24, yy+18), fill=color)
        draw.text((x+34, yy-6), label, fill="#333333", font=_font(20))


def placeholder_figure(path: Path, title: str, message: str) -> None:
    image, draw = _canvas(title, "Validation-only evidence ledger")
    draw.rounded_rectangle((250, 285, 1670, 825), radius=30, fill="#F2F2F2", outline="#A5A5A5", width=3)
    draw.text((785, 395), "MISSING", fill="#C00000", font=_font(58, True))
    wrapped = []
    words = message.split()
    line = ""
    for word in words:
        if len(line) + len(word) > 90:
            wrapped.append(line); line = word
        else:
            line = f"{line} {word}".strip()
    if line: wrapped.append(line)
    for i, text in enumerate(wrapped):
        draw.text((365, 535+i*40), text, fill="#404040", font=_font(25))
    image.save(path)


def make_figures(source: Path, output: Path, best: pd.DataFrame, evidence: pd.DataFrame) -> None:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    den = read_csv(source / "denoising_val.csv")
    image, draw = _canvas("Stage 1/D0 denoising", "Fixed f01 samples only: 3 validation positions; not the full 141-frame validation")
    chart = (180, 260, 1740, 910); _axes(draw, chart, 20, 40, "PSNR (dB)")
    groups = den["group_id"].tolist(); colors = {"Noisy":"#A5A5A5", "Denoised":"#5B9BD5"}
    group_width = (chart[2]-chart[0]) / len(groups)
    for gi, (_, row) in enumerate(den.iterrows()):
        center = chart[0] + group_width*(gi+.5)
        for bi, (label, col) in enumerate((("Noisy", "psnr_noisy"), ("Denoised", "psnr"))):
            value=float(row[col]); height=(value-20)/20*(chart[3]-chart[1]); x0=center-90+bi*95
            draw.rectangle((x0, chart[3]-height, x0+70, chart[3]), fill=colors[label])
            draw.text((x0-5, chart[3]-height-30), f"{value:.2f}", fill="#333333", font=_font(18))
        draw.text((center-60, chart[3]+22), str(row["group_id"]), fill="#333333", font=_font(20))
    _legend(draw, list(colors.items()), 1510, 185); image.save(figures / "figure_01_stage1_denoising.png")

    current = best[best["experiment_alias"].isin(CURRENT_ALIASES)].copy()
    current = current.set_index("experiment_alias").reindex(["E1-current", "E3-current", "E3b", "E3b-no-D2S"]).reset_index()
    image, draw = _canvas("Current matched-protocol Stage 2", "Metadata-selected best checkpoint; seed 42 only; model-grid validation histories")
    chart=(180,260,1740,910); _axes(draw, chart, 0, 1, "Dice")
    specs=[("Layer Dice","val_layer_dice","#70AD47"),("Vessel Dice","val_vessel_dice","#5B9BD5"),("Vessel soft Dice","val_vessel_soft_dice","#ED7D31")]
    gw=(chart[2]-chart[0])/len(current)
    for gi, (_, row) in enumerate(current.iterrows()):
        center=chart[0]+gw*(gi+.5)
        for bi, (_, col, color) in enumerate(specs):
            value=float(row[col]); x0=center-105+bi*72; height=value*(chart[3]-chart[1])
            draw.rectangle((x0,chart[3]-height,x0+54,chart[3]),fill=color)
        draw.text((center-85,chart[3]+25),str(row["experiment_alias"]),fill="#333333",font=_font(19))
    _legend(draw, [(a,c) for a,_,c in specs], 1460, 170); image.save(figures / "figure_02_stage2_ablation.png")

    pair = current[current["experiment_alias"].isin(["E3-current", "E3b"])]
    image, draw = _canvas("Outside-BCE contrast", "E3b vs E3-current; single-factor configuration audit; seed 42 only")
    chart=(240,260,1640,910); _axes(draw, chart, 0, 1, "Ratio")
    specs=[("Precision","val_vessel_precision","#5B9BD5"),("Recall","val_vessel_recall","#70AD47"),("Outside fraction","val_vessel_outside_gt_layer_fraction","#C00000")]
    gw=(chart[2]-chart[0])/len(pair)
    for gi, (_, row) in enumerate(pair.iterrows()):
        center=chart[0]+gw*(gi+.5)
        for bi, (_, col, color) in enumerate(specs):
            value=float(row[col]); x0=center-150+bi*100; height=value*(chart[3]-chart[1])
            draw.rectangle((x0,chart[3]-height,x0+76,chart[3]),fill=color)
            draw.text((x0,chart[3]-height-26),f"{value:.3f}",fill="#333333",font=_font(17))
        draw.text((center-90,chart[3]+25),str(row["experiment_alias"]),fill="#333333",font=_font(22))
    _legend(draw, [(a,c) for a,_,c in specs], 1480, 170); image.save(figures / "figure_03_outside_bce_effect.png")

    groups = read_csv(source / "group_metrics.csv")
    image, draw = _canvas("P0 vs P3 postprocessing", "Inference/postprocessing only; not network training gain")
    chart=(220,260,1680,910); _axes(draw, chart, .5, .9, "Vessel Dice")
    colors={"P0 raw":"#A5A5A5","P3":"#5B9BD5"}; gw=(chart[2]-chart[0])/len(groups)
    for gi, (_, row) in enumerate(groups.iterrows()):
        center=chart[0]+gw*(gi+.5)
        for bi,(label,col) in enumerate((("P0 raw","p0_vessel_dice"),("P3","p3_vessel_dice"))):
            value=float(row[col]); height=(value-.5)/.4*(chart[3]-chart[1]); x0=center-90+bi*100
            draw.rectangle((x0,chart[3]-height,x0+72,chart[3]),fill=colors[label])
            draw.text((x0-4,chart[3]-height-28),f"{value:.3f}",fill="#333333",font=_font(18))
        draw.text((center-70,chart[3]+25),str(row["group_id"]),fill="#333333",font=_font(21))
    _legend(draw,list(colors.items()),1500,175); image.save(figures / "figure_04_postprocessing.png")

    placeholder_figure(figures / "figure_05_joint_factorial.png", "J00/J10/J01/J11 factorial", "Formal fixed-final, three-seed validation assets are not present locally. No interaction benefit is claimed.")
    placeholder_figure(figures / "figure_06_input_noisy_vs_denoised.png", "I-NOISY vs I-DENOISED", "Formal epoch-60 paired validation assets are not present locally. I-CLEAN remains planned.")

    trajectory = read_csv(source / "metrics_long.csv")
    curve = trajectory[(trajectory["aggregation_level"] == "group_macro_training_scale") & trajectory["metric"].isin(["vessel_soft_dice"])].copy()
    image, draw = _canvas("Current Stage 2 training trajectories", "Vessel soft Dice; curves are not independent validation cases")
    chart=(180,260,1680,900); _axes(draw, chart, .4, .8, "Soft Dice")
    palette=["#5B9BD5","#ED7D31","#70AD47","#7030A0","#C00000","#00B0F0"]
    legend=[]
    for idx,(alias,part) in enumerate(curve.groupby("experiment_alias")):
        part=part.sort_values("epoch"); color=palette[idx%len(palette)]; pts=[]
        for _,row in part.iterrows():
            x=chart[0]+float(row["epoch"])/60*(chart[2]-chart[0]); y=chart[3]-(float(row["value"])-.4)/.4*(chart[3]-chart[1]); pts.append((x,y))
        if len(pts)>1: draw.line(pts,fill=color,width=4)
        legend.append((str(alias),color))
    _legend(draw,legend,1460,165); image.save(figures / "figure_07_training_trajectories.png")

    counts = evidence["evidence_strength"].value_counts() if "evidence_strength" in evidence else pd.Series(dtype=int)
    image, draw = _canvas("Current evidence map", "Validation only; absent results remain missing")
    colors={"supported_limited":"#70AD47", "not_supported":"#C00000", "missing":"#A5A5A5", "planned":"#5B9BD5"}
    chart=(260,260,1650,900); _axes(draw,chart,0,max(5,float(counts.max() if len(counts) else 5)),"Claims")
    bw=(chart[2]-chart[0])/max(1,len(counts))
    for i,(label,value) in enumerate(counts.items()):
        height=float(value)/max(5,float(counts.max()))*(chart[3]-chart[1]); x0=chart[0]+bw*i+bw*.2
        draw.rectangle((x0,chart[3]-height,x0+bw*.6,chart[3]),fill=colors.get(label,"#FFC000"))
        draw.text((x0,chart[3]+25),str(label),fill="#333333",font=_font(19)); draw.text((x0+bw*.25,chart[3]-height-32),str(value),fill="#333333",font=_font(22,True))
    image.save(figures / "figure_08_current_evidence_map.png")


def build_evidence(output: Path) -> pd.DataFrame:
    rows = [
        ["D0 improves current PKU37 validation image quality", "E3b V0 frozen denoiser", "Denoised vs noisy", "PSNR/SSIM/RMSE/EPI", "positive on 3 fixed f01 samples", "three positions; one frame each", "supported_limited", "improved the three archived validation samples", "proved general denoising performance", "not full 141-frame validation"],
        ["Stage 2 segments layer and vessel", "current matched Stage2 histories", "four current arms", "Layer/Vessel Dice", "non-trivial metrics", "seed 42 only", "supported_limited", "demonstrated feasibility on the current validation protocol", "established generalization", "only E3b has three original-grid frames"],
        ["outside BCE reduces exterior vessel FP", "E3b vs E3-current", "single-factor history comparison", "outside fraction/precision", "outside fraction 0.1515 to 0.0373", "single seed", "supported_limited", "is associated with lower layer-exterior predictions", "significantly improves generalization", "best-history model-grid comparison"],
        ["P3 improves anatomical plausibility", "E3b V0", "P3 vs P0", "Dice/removed TP-FP", "mixed positive fixed-frame change", "three f01 frames", "supported_limited", "postprocessing changed fixed-frame metrics", "network training gain", "P3 uncalibrated; inference only"],
        ["D2S has stable net training benefit", "E3b vs E3b-noD2S", "best-history", "Vessel Dice", "about +0.0034", "one seed", "not_supported", "net benefit remains uncertain", "D2S significantly improves segmentation", "no repeated seeds or position pairs"],
        ["Bidirectional Joint interaction helps both tasks", "B0/J00/J10/J01/J11", "fixed-final factorial", "paired task gains", "not available", "none", "missing", "not yet demonstrated", "proved mutual promotion", "formal outputs absent locally"],
        ["Denoised input improves segmentation", "I-NOISY/I-DENOISED", "epoch60 paired", "Layer/Vessel metrics", "not available", "none", "missing", "not established from local evidence", "improves layer but hurts vessel", "formal outputs absent locally"],
        ["I-CLEAN oracle", "planned", "three-arm oracle CV", "all metrics", "not run", "none", "planned", "planned experiment", "completed result", "explicitly outside this summary"],
        ["Test performance", "sealed test", "none", "none", "not opened", "N/A", "missing", "test remains sealed", "final paper test result", "test_assets_opened=0"],
    ]
    cols = ["claim", "supporting_experiment", "comparison", "metric", "effect", "stability", "evidence_strength", "allowed_wording", "forbidden_wording", "limitation"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(output / "summaries" / "evidence_matrix.csv", index=False, encoding="utf-8-sig")
    df.to_csv(output / "evidence_matrix.csv", index=False, encoding="utf-8-sig")
    return df


def make_atlas(source: Path, output: Path) -> None:
    atlas = output / "atlas"
    images = atlas / "stage2_e3b_postprocess"
    images.mkdir(parents=True, exist_ok=True)
    rows, assets, missing = [], [], []
    for group in ("pku_0006", "pku_0012", "pku_0040"):
        rows.append({"group_id": group, "sample_id": f"{group}_f01", "selection_rule": "first frame per current validation position; fixed before method review", "selection_source": "archived E3b V0"})
        for src in sorted((source / "qualitative").glob(f"{group}_f01_*.png")):
            dst = images / src.name
            shutil.copy2(src, dst)
            assets.append({"group_id": group, "sample_id": f"{group}_f01", "asset": src.stem.split(f"{group}_f01_")[-1], "path": dst.relative_to(output).as_posix(), "status": "present", "source": src.relative_to(source).as_posix()})
    for family, panels in (("stage1", ["clean", "residual", "zoom"]), ("stage2_four_arm", ["E1", "E3", "E3b-noD2S probabilities"]), ("joint", ["J00/J10/J01/J11"]), ("input", ["I-NOISY/I-DENOISED"])):
        for panel in panels:
            missing.append({"family": family, "panel": panel, "status": "MISSING", "reason": "matched prediction/image assets not present in validation-only archive"})
    write_csv(atlas / "atlas_selection.csv", rows)
    write_csv(atlas / "atlas_asset_inventory.csv", assets)
    write_csv(atlas / "atlas_missing_assets.csv", missing)
    cards = "\n".join(f'<figure><img src="stage2_e3b_postprocess/{Path(r["path"]).name}"><figcaption>{r["group_id"]} — {r["asset"]}</figcaption></figure>' for r in assets)
    html = f"""<!doctype html><meta charset='utf-8'><title>SABIDS fixed atlas</title><style>body{{font-family:Arial;margin:24px}}.warn{{padding:12px;background:#fff2cc}}main{{display:grid;grid-template-columns:repeat(3,minmax(260px,1fr));gap:14px}}img{{max-width:100%;border:1px solid #bbb}}figure{{margin:0}}</style><h1>Fixed validation atlas</h1><p class='warn'>E3b/P0–P3 archived f01 assets only. Postprocessing is inference-only. Missing matched arms are not reconstructed.</p><main>{cards}</main>"""
    (atlas / "index.html").write_text(html, encoding="utf-8")


def write_summaries(output: Path, source: Path, run_df: pd.DataFrame, missing: list[dict[str, Any]]) -> None:
    summaries = output / "summaries"; summaries.mkdir(parents=True, exist_ok=True)
    presentation = """# SABIDS-Net 阶段性汇报摘要

本归档仅使用已有 validation 记录，不读取 test、不训练、不校准阈值。

1. Stage 1/D0 在三个固定 validation 位置的 f01 样本上均明显改善 PSNR、SSIM、RMSE 与 EPI；这不是完整 141 帧评价。
2. 当前同协议 Stage 2 已显示层与血管分割可行性。E3b 与 E3-current 的单因素历史对比支持 outside BCE 主要减少层外预测并提高 Precision，但仅有 seed 42。
3. E3b 与 E3b-noD2S 的血管 Dice 差约 0.34 个百分点，现有证据不足以声称 D→S 稳定有效。
4. P3 在三张固定图上改变 Dice/Precision/Recall，但它是推理后处理，不是网络训练收益，且未独立校准。
5. 本机缺少 B0/J00/J10/J01/J11 与 I-NOISY/I-DENOISED 的正式固定终轮 validation 资产，因此不报告双向互助或输入效应。
6. 只有三个独立 validation 位置；重复帧不能作为独立病例。test 保持封存。
"""
    technical = f"""# Technical summary

## Scope

- Source record: `{source.as_posix()}`
- Validation only; no model forward was rerun.
- Archived E3b original-grid evidence covers three positions and one frame per position.
- Current Stage 2 history comparison uses seed 42, 512×512 training coordinates, threshold 0.5, and each run's metadata-selected best checkpoint.

## Attribution boundaries

- E3b−E3-current is the cleanest available outside-BCE contrast.
- E3b−E3b-noD2S is a D→S retraining contrast, but single-seed and best-history only.
- E3b−E1-current changes multiple supervision factors and is labelled multifactor.
- P0/P1/P2/P3 are inference/postprocessing contrasts.
- Same-checkpoint D2S-off diagnostics describe dependency, not retraining gain.

## Missing formal families

Joint factorial and image-input fixed-final outputs are absent locally. Their tables and figures are intentionally marked `MISSING`.
"""
    limitations = """# Limitations

- Only three independent PKU37 validation positions are represented in original-grid V0 outputs, with one frame per position.
- Stage 2 matched comparisons have only seed 42; no seed standard deviation or clustered confidence interval is estimable.
- The archived checkpoint binaries and most resolved configs are not present locally, although their metadata SHA256 values are retained.
- Small/medium/large vessel component bins were derived at 512 but the V0 images are 640; those component-bin results are not used as valid evidence.
- P3 was not separately calibrated and is reported only as postprocessing.
- Patient identity is not independently verified because archived patient_id duplicates group_id.
- Joint and input-factorial conclusions cannot be recovered from design documents; numerical assets are required.
- test was not opened and this is not a final paper test report.
"""
    next_exp = """# Next experiments

1. Export fixed-final (`last.pth`) J00/J10/J01/J11 validation metrics for seeds 42/43/44, reduced first by group.
2. Export epoch-60 I-NOISY/I-DENOISED validation metrics and matched predictions for the same fixed atlas samples.
3. Run complete 141-frame validation inference for the formal D0/Stage2 checkpoint while keeping ROI and threshold fixed.
4. Add matched four-arm Stage2 probability maps and failure panels without sample re-selection.
5. Keep I-CLEAN and multi-position CV as a separate future oracle study; do not mix it into the current result.
"""
    guide = """# GPT analysis guide

Start with `audit/protocol_comparison.csv` and `audit/missing_assets.csv`. Keep historical debugging, current Stage2 ablation, same-checkpoint diagnostics, and postprocessing separate. Use anatomical position as the primary unit; never treat repeated frames as independent cases. Inspect Precision/Recall, FP/FN, ROI and outside-layer fractions together, plus layer boundaries/thickness. For interaction results, require scale, relative RMS and gradient diagnostics. Check whether seed 42 or pku_0040 drives an effect. Use the fixed atlas to inspect oversmoothing, whole-layer vessel false positives and hallucination. Do not use test, threshold calibration, postprocessing, or selective examples to manufacture a positive conclusion.
"""
    for name, text in (("PRESENTATION_SUMMARY.md", presentation), ("TECHNICAL_SUMMARY.md", technical), ("LIMITATIONS.md", limitations), ("NEXT_EXPERIMENTS.md", next_exp), ("ANALYSIS_GUIDE.md", guide)):
        (summaries / name).write_text(text, encoding="utf-8")
    top = [
        "summaries/PRESENTATION_SUMMARY.md", "figures/figure_02_stage2_ablation.png",
        "figures/figure_03_outside_bce_effect.png", "figures/figure_01_stage1_denoising.png",
        "figures/figure_04_postprocessing.png", "SABIDS_stage_experiment_ledger.xlsx",
        "audit/protocol_comparison.csv", "metrics/paired_gains_summary.csv",
        "summaries/evidence_matrix.csv", "audit/missing_assets.csv",
    ]
    (output / "README_FIRST.md").write_text("# Read first\n\nValidation-only stage summary. Test assets were not opened. Start with:\n\n" + "\n".join(f"{i}. `{p}`" for i, p in enumerate(top, 1)) + "\n", encoding="utf-8")


def copy_supporting(source: Path, output: Path, root: Path) -> None:
    for name in ("training_best.csv", "denoising_val.csv", "postprocess_comparison.csv", "protocol_comparison.csv", "experiment_registry.csv", "evidence_table.csv", "missing_table.csv"):
        if (source / name).is_file(): shutil.copy2(source / name, output / "metrics" / f"source_{name}")
    for history in (root / "runs" / "current" / "smoke").rglob("history.csv") if (root / "runs" / "current" / "smoke").exists() else []:
        if is_forbidden(history): continue
        rel = history.relative_to(root / "runs" / "current" / "smoke")
        dst = output / "histories" / rel
        dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(history, dst)
        cfg = history.parent / "resolved_config.yaml"
        if cfg.is_file():
            cfg_dst = output / "configs" / rel.parent / cfg.name
            cfg_dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(cfg, cfg_dst)


def make_workbook_csvs(output: Path, run_df: pd.DataFrame, checkpoint_df: pd.DataFrame, source: Path, evidence: pd.DataFrame, missing_df: pd.DataFrame) -> None:
    wb = output / "workbook_inputs"; wb.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"item": "scope", "value": "validation only; no test; no new training"},
        {"item": "primary unit", "value": "anatomical position/group_id"},
        {"item": "formal J/I status", "value": "MISSING locally"},
        {"item": "source", "value": source.as_posix()},
    ]).to_csv(wb / "README.csv", index=False, encoding="utf-8-sig")
    timeline = run_df[[c for c in ["family", "classification", "alias", "status", "best_epoch", "last_epoch", "evidence_scope", "notes"] if c in run_df]]
    timeline.to_csv(wb / "experiment_timeline.csv", index=False, encoding="utf-8-sig")
    run_df.to_csv(wb / "run_registry.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(output / "audit" / "protocol_comparison.csv", wb / "protocol_audit.csv")
    shutil.copy2(source / "denoising_val.csv", wb / "stage1_denoising.csv")
    shutil.copy2(source / "training_best.csv", wb / "stage2_ablation.csv")
    shutil.copy2(source / "postprocess_comparison.csv", wb / "postprocessing.csv")
    shutil.copy2(output / "metrics" / "joint_factorial.csv", wb / "joint_factorial.csv")
    shutil.copy2(output / "metrics" / "input_experiment.csv", wb / "input_experiment.csv")
    shutil.copy2(output / "metrics" / "metrics_long.csv", wb / "metrics_long.csv")
    shutil.copy2(output / "metrics" / "position_metrics_long.csv", wb / "position_results.csv")
    shutil.copy2(output / "metrics" / "training_trajectory_summary.csv", wb / "training_trajectory.csv")
    shutil.copy2(output / "atlas" / "atlas_asset_inventory.csv", wb / "image_inventory.csv")
    evidence.to_csv(wb / "evidence_matrix.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"limitation": [line[2:] for line in (output / "summaries" / "LIMITATIONS.md").read_text(encoding="utf-8").splitlines() if line.startswith("- ")]}).to_csv(wb / "limitations.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(wb / "missing_assets.csv", index=False, encoding="utf-8-sig")
    checkpoint_df.to_csv(wb / "checkpoint_inventory.csv", index=False, encoding="utf-8-sig")


def validate_outputs(output: Path) -> dict[str, Any]:
    csv_checks, image_checks, failures = [], [], []
    for path in output.rglob("*.csv"):
        if is_forbidden(path):
            failures.append(f"forbidden path: {path}")
            continue
        try:
            df = pd.read_csv(path)
            numeric = df.select_dtypes(include=[np.number])
            inf_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if not numeric.empty else 0
            csv_checks.append({"path": path.relative_to(output).as_posix(), "rows": len(df), "columns": len(df.columns), "duplicate_columns": bool(df.columns.duplicated().any()), "infinite_values": inf_count})
            if df.columns.duplicated().any() or inf_count: failures.append(f"invalid CSV: {path}")
        except Exception as exc:
            failures.append(f"CSV decode failed: {path}: {exc}")
    try:
        from PIL import Image
        for path in output.rglob("*.png"):
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image_checks.append({"path": path.relative_to(output).as_posix(), "width": image.width, "height": image.height})
                if image.width < 100 or image.height < 100: failures.append(f"small image: {path}")
    except Exception as exc:
        failures.append(f"image validation unavailable: {exc}")
    return {"status": "passed" if not failures else "failed", "csv_checks": csv_checks, "image_checks": image_checks, "failures": failures, "test_assets_opened": 0, "test_evaluation_performed": False}


def manifest(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or path.suffix.lower() == ".zip": continue
        rel = path.relative_to(output).as_posix()
        rows.append({"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": sha256(path), "source_run": "derived from validation record/local audit", "generated_by_reevaluation": False, "contains_image": path.suffix.lower() in {".png", ".jpg", ".jpeg"}, "gpt_eligible": path.suffix.lower() not in {".pth", ".npy", ".npz"}})
    return rows


def package(output: Path, timestamp: str) -> dict[str, Any]:
    package_dir = output / "packages"; package_dir.mkdir(exist_ok=True)
    full = package_dir / f"SABIDS_stage_results_full_{timestamp}.zip"
    gpt = package_dir / f"SABIDS_stage_results_for_GPT_{timestamp}.zip"
    excluded_ext = {".pth", ".npy", ".npz"}
    files = [p for p in output.rglob("*") if p.is_file() and p.suffix.lower() not in excluded_ext and "packages" not in p.parts and not is_forbidden(p)]
    gpt_prefix = {"README_FIRST.md", "SABIDS_stage_experiment_ledger.xlsx", "MANIFEST.json", "missing_and_failure_checklist.json"}
    gpt_files = [p for p in files if p.name in gpt_prefix or p.parts[-2] in {"summaries", "audit", "metrics", "figures"} or ("atlas" in p.parts and (p.name.endswith("contact.png") or p.name in {"index.html", "atlas_selection.csv", "atlas_missing_assets.csv"}))]
    results = {}
    for kind, path, members in (("full", full, files), ("gpt", gpt, gpt_files)):
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for member in members:
                archive.write(member, member.relative_to(output).as_posix())
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            if bad: raise RuntimeError(f"ZIP CRC failure: {bad}")
            count = len(archive.infolist())
        results[kind] = {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path), "file_count": count, "crc_status": "passed"}
    return results


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    source = (args.source_record.resolve() if args.source_record else latest_record(root))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = (args.output.resolve() if args.output else root / "runs" / f"stage_summary_{stamp}")
    if args.finalize_existing:
        if not output.is_dir():
            raise FileNotFoundError(f"Stage summary directory does not exist: {output}")
        ledger = output / "SABIDS_stage_experiment_ledger.xlsx"
        if not ledger.is_file():
            raise FileNotFoundError(f"Workbook must be generated before finalization: {ledger}")
        validation = validate_outputs(output)
        (output / "missing_and_failure_checklist.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
        entries = manifest(output)
        (output / "MANIFEST.json").write_text(json.dumps({"generated_at": datetime.now().isoformat(), "scope": "validation-only records-first; no test; no training", "files": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
        package_stamp = output.name.removeprefix("stage_summary_")
        package_result = package(output, package_stamp)
        result = {"status": "complete" if validation["status"] == "passed" else "complete_with_validation_failures", "output": str(output), "validation": validation["status"], "packages": package_result}
        (output / "build_state.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    if output.exists(): raise FileExistsError(f"Refusing to overwrite stage summary: {output}")
    for directory in ("audit", "tables", "figures", "atlas", "metrics", "configs", "histories", "summaries"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("test_metrics_exported") is not False:
        raise RuntimeError("Source record is not explicitly validation-only")
    run_df, checkpoint_df, missing = audit_runs(root, source, output)
    run_df.to_csv(output / "audit" / "run_inventory.csv", index=False, encoding="utf-8-sig")
    checkpoint_df.to_csv(output / "audit" / "checkpoint_inventory.csv", index=False, encoding="utf-8-sig")
    shutil.copy2(source / "protocol_comparison.csv", output / "audit" / "protocol_comparison.csv")
    excluded = run_df[~run_df["ranking_eligible"].astype(bool)].copy()
    excluded["exclusion_reason"] = excluded["classification"].map(lambda x: "not a current comparable scientific run: " + str(x))
    excluded.to_csv(output / "audit" / "excluded_runs.csv", index=False, encoding="utf-8-sig")

    asset_rows = []
    for path in sorted(source.rglob("*")):
        if path.is_file() and not is_forbidden(path):
            asset_rows.append({"source": "validation_record", "relative_path": path.relative_to(source).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256(path), "asset_type": path.suffix.lower(), "scope": "validation-only"})
    write_csv(output / "audit" / "asset_inventory.csv", asset_rows)
    missing_df = pd.DataFrame(missing)
    missing_df.to_csv(output / "audit" / "missing_assets.csv", index=False, encoding="utf-8-sig")
    audit_summary = {"generated_at": datetime.now().isoformat(), "source_record": str(source), "source_record_sha256": sha256(source / "provenance.json"), "run_count": len(run_df), "ranking_eligible_count": int(run_df["ranking_eligible"].astype(bool).sum()), "checkpoint_file_count": int(checkpoint_df["file_present"].astype(bool).sum()) if not checkpoint_df.empty else 0, "missing_asset_count": len(missing_df), "test_assets_opened": 0, "test_evaluation_performed": False, "threshold_calibration_performed": False, "training_performed": False}
    (output / "audit" / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    best = read_csv(source / "training_best.csv")
    make_metrics(source, output, best)
    evidence = build_evidence(output)
    make_figures(source, output, best, evidence)
    make_atlas(source, output)
    copy_supporting(source, output, root)
    write_summaries(output, source, run_df, missing)
    make_workbook_csvs(output, run_df, checkpoint_df, source, evidence, missing_df)
    validation = validate_outputs(output)
    (output / "missing_and_failure_checklist.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8")
    entries = manifest(output)
    (output / "MANIFEST.json").write_text(json.dumps({"generated_at": datetime.now().isoformat(), "scope": "validation-only records-first; no test; no training", "files": entries}, indent=2, ensure_ascii=False), encoding="utf-8")
    result = {"status": "csv_png_markdown_ready", "output": str(output), "timestamp": stamp, "source_record": str(source), "validation": validation["status"], "workbook_command": f"node tools/build_stage_summary_workbooks.mjs {output} {output / 'SABIDS_stage_experiment_ledger.xlsx'}"}
    if not args.skip_packaging:
        result["warning"] = "Run workbook builder first, then rerun with --package-only in a future revision; current command intentionally leaves packaging to finalize step."
    (output / "build_state.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
