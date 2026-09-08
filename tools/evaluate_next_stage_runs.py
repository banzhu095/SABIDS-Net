"""Evaluate fixed-final checkpoints on complete validation frames only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.engine.evaluator import evaluate_model
from sabids.engine.trainer import build_diagnostic_loader, build_model
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file


def _config(run: Path) -> Path | None:
    return next((run / name for name in ("resolved_config.yaml", "config_resolved.yaml", "config.yaml") if (run / name).is_file()), None)


def _suite(run_id: str) -> str | None:
    if run_id.startswith(("d1_denoise_d0", "d1_denoise_struct")): return "d1_structure"
    if run_id.startswith("input_"): return "input_image"
    if run_id.startswith("order_"): return "training_order"
    if "shuffle" in run_id or "self_adapter" in run_id: return "decoder_interaction_controls"
    if run_id.startswith("interaction_j"): return "decoder_interaction_confirm"
    if run_id.startswith("interaction_a"): return "decoder_interaction_strength"
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--suites", required=True, help="Comma-separated suite names")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-fixed-atlas", action="store_true")
    parser.add_argument("--atlas-groups", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    lock_path = Path(args.protocol_lock)
    if not lock_path.is_absolute(): lock_path = root / lock_path
    lock = load_protocol_lock(lock_path)
    wanted = set(args.suites.split(","))
    device = torch.device(args.device)
    records = []
    for run in sorted((root / "runs" / "current").iterdir() if (root / "runs" / "current").is_dir() else []):
        suite = _suite(run.name)
        if suite == "decoder_interaction_strength" and "_pilot_" not in run.name: continue
        if suite and suite != "decoder_interaction_strength" and "_pilot_" in run.name: continue
        config_path = _config(run)
        if suite not in wanted or config_path is None:
            continue
        cfg = load_config(config_path)
        if cfg.get("protocol_id") != lock["protocol_id"]:
            continue
        for key in ("data_plan_sha256", "label_inventory_sha256"):
            if cfg.get(key) != lock.get(key): raise RuntimeError(f"{run.name}: {key} mismatch")
        checkpoint = run / "last.pth"
        if not checkpoint.is_file():
            records.append({"run_id": run.name, "status": "missing_fixed_final_checkpoint"}); continue
        output = run / "validation_results"
        if args.resume and (output / "summary.json").is_file():
            records.append({"run_id": run.name, "status": "skipped_complete"}); continue
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = build_model(cfg).to(device)
        model.load_state_dict(raw.get("model", raw), strict=True)
        cfg["data"].pop("max_val_samples", None)
        loader = build_diagnostic_loader(cfg, cfg["data"].get("val_split", "val"))
        stage = str(cfg["train"].get("stage", "joint"))
        tasks = ("denoise",) if stage == "denoise" else ("layer", "vessel") if stage == "input_segment" else ("denoise", "layer", "vessel")
        summary = evaluate_model(model, loader, device, output, threshold=0.5, layer_threshold=0.5, vessel_threshold=0.5, save_predictions=False, stage=stage, input_normalization=cfg["data"].get("normalization"), tasks=tasks, postprocess_modes=("p0",), restore_original_geometry=True)
        if args.save_fixed_atlas:
            manifest = pd.read_csv(cfg["data"]["manifest"], dtype=str).fillna("")
            validation = manifest[manifest["split"].eq(str(cfg["data"].get("val_split", "val")))]
            groups = sorted(validation["group_id"].unique())[: args.atlas_groups]
            cfg["data"]["val_groups"] = groups
            atlas_loader = build_diagnostic_loader(cfg, cfg["data"].get("val_split", "val"))
            atlas_dir = output / "fixed_atlas"
            evaluate_model(model, atlas_loader, device, atlas_dir, threshold=0.5, layer_threshold=0.5, vessel_threshold=0.5, save_predictions=True, stage=stage, input_normalization=cfg["data"].get("normalization"), tasks=tasks, postprocess_modes=("p0",), restore_original_geometry=True)
            pd.DataFrame({"group_id": groups, "selection_rule": "lexicographic validation group ID before metrics"}).to_csv(output / "atlas_selection.csv", index=False, encoding="utf-8-sig")
        records.append({"run_id": run.name, "status": "passed", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "n_frames": summary.get("n_frames"), "n_groups": summary.get("n_groups")})
    print(json.dumps({"status": "passed", "records": records, "test_assets_opened": 0}, indent=2))


if __name__ == "__main__": main()
