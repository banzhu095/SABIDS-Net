from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config, save_config
from sabids.engine.trainer import Trainer
from sabids.utils import write_json
from tools.input_oracle_cv_common import (
    audit_d0_checkpoint, load_protocol, read_fold_groups, require_phase0,
    require_split_audit, resolve,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/audit one fold-specific leakage-safe D0")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol", default="configs/current/input_oracle_cv/protocol.yaml")
    parser.add_argument("--output", default="runs/input_oracle_cv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--mode", choices=("train", "audit"), required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = resolve(root, args.output)
    _, protocol = load_protocol(root, args.protocol)
    phase0 = require_phase0(root, output)
    split = require_split_audit(output, args.fold)
    _, val = read_fold_groups(output, args.fold)
    sealed = set(map(str, phase0["sealed_test_groups"]))
    manifest = root / split["folds"][str(args.fold)]["d0_manifest"]
    run = output / f"d0_fold{args.fold}{'_smoke' if args.smoke_test else ''}"
    checkpoint = run / "best.pth"
    if args.mode == "audit":
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing D0 checkpoint: {checkpoint}")
        report = audit_d0_checkpoint(checkpoint, manifest, val, sealed)
        write_json(report, run / "d0_leakage_audit.json")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "passed":
            raise SystemExit(2)
        return

    config = copy.deepcopy(load_config(root / "configs/current/stage1_denoise_fold0.yaml"))
    config["device"] = args.device
    config["data"]["manifest"] = str(manifest)
    config["data"]["root"] = str(root)
    config["train"]["output_dir"] = str(run)
    if args.smoke_test:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0
        config["data"]["samples_per_epoch"] = 2
        config["data"]["max_val_samples"] = 2
    if args.resume:
        last = run / "last.pth"
        if not last.is_file():
            raise FileNotFoundError(f"--resume requires {last}")
        config["train"]["resume"] = str(last)
    elif run.exists() and any(run.iterdir()):
        raise FileExistsError(f"Refusing to overwrite D0 run: {run}; use --resume only for an exact continuation")
    registry = {
        "fold": args.fold, "manifest": str(manifest), "validation_groups": sorted(val),
        "sealed_test_groups_metadata_only": sorted(sealed), "test_assets_opened": 0,
        "command_status": "dry_run" if args.dry_run else "authorized",
    }
    if args.dry_run:
        print(json.dumps(registry, ensure_ascii=False, indent=2))
        return
    trainer = Trainer(config)
    save_config(config, run / "resolved_config.yaml")
    write_json(registry, run / "preflight_registry.json")
    trainer.fit()
    if not checkpoint.is_file():
        raise RuntimeError(f"Training finished without best checkpoint: {checkpoint}")
    report = audit_d0_checkpoint(checkpoint, manifest, val, sealed)
    write_json(report, run / "d0_leakage_audit.json")
    if report["status"] != "passed":
        raise RuntimeError("Fold-specific D0 completed but provenance/leakage audit failed")


if __name__ == "__main__":
    main()
