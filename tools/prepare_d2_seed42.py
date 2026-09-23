#!/usr/bin/env python
"""Generate fresh, evidence-gated seed-42 D2 smoke/overfit/pilot configs."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sabids.config import load_config, save_config
from sabids.experiments.dose_response import formal_preflight, sha256_file, stable_sha
from sabids.experiments.protocol_lock import load_protocol_lock, validate_checkpoint_config


ARMS = {
    "D20": {},
    "D21": {"vessel_roi": 0.5, "stroma_roi": 0.1, "outside_roi": 0.05},
    "D22": {"vessel_roi": 0.5, "stroma_roi": 0.1, "outside_roi": 0.05, "boundary": 0.2},
    "D23": {"vessel_roi": 0.5, "stroma_roi": 0.1, "outside_roi": 0.05, "boundary": 0.2, "cnr": 0.1},
    "D24": {"vessel_roi": 0.5, "stroma_roi": 0.1, "outside_roi": 0.05, "boundary": 0.2, "cnr": 0.1,
            "teacher_task": 0.2, "teacher_consistency": 0.05},
    "D25": {"vessel_roi": 0.5, "stroma_roi": 0.1, "outside_roi": 0.05, "boundary": 0.2, "cnr": 0.1,
            "teacher_task": 0.2, "teacher_consistency": 0.05, "leak": 0.1, "residual_amplitude": 0.01},
}
BASE_WEIGHTS = {"charbonnier": 1.0, "ms_ssim": 0.2, "gradient": 0.1, "laplacian": 0.05}
MODE = {
    "smoke": {"epochs": 1, "device": "cpu", "max_train_samples": 2, "max_val_samples": 2},
    "overfit": {"epochs": 3, "device": "cuda", "max_train_samples": 4, "max_val_samples": 4},
    "pilot": {"epochs": 20, "device": "cuda", "max_train_samples": None, "max_val_samples": None},
    "full": {"epochs": 60, "device": "cuda", "max_train_samples": None, "max_val_samples": None},
}


def _resolved(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=tuple(MODE), required=True)
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Explicit generated-config device; defaults to the selected mode",
    )
    parser.add_argument("--seed42-gate")
    parser.add_argument("--d1-checkpoint", required=True)
    parser.add_argument("--d1-initial-inventory", required=True)
    parser.add_argument("--d1-checkpoint-binding", required=True)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--vessel-strata-definition", required=True)
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-evidence")
    parser.add_argument("--teacher-selection-rule")
    parser.add_argument("--teacher-training-data")
    parser.add_argument("--teacher-split", default="train")
    parser.add_argument("--output-dir", default="runs/adaptive_denoising/d2_v1_launch_configs")
    return parser.parse_args(argv)


def main() -> None:
    args = _arguments()
    if args.mode == "full":
        if args.seed not in {42, 43, 44} or not args.seed42_gate:
            raise SystemExit("BLOCKED: full configs require seed 42/43/44 and an explicit seed-42 gate")
        gate = json.loads(_resolved(Path(args.project_root).expanduser().resolve(), args.seed42_gate).read_text(encoding="utf-8-sig"))
        gate_source = Path(str(gate.get("source", ""))).expanduser().resolve()
        if (gate.get("schema_version") != "d2-seed42-gate-v1"
                or gate.get("status") != "passed"
                or gate.get("multi_seed_configs_authorized") is not True
                or gate.get("failures") != []
                or not gate_source.is_file()
                or sha256_file(gate_source) != gate.get("source_sha256")):
            raise SystemExit("BLOCKED: seed-42 gate does not authorize multi-seed config generation")
    elif args.seed != 42:
        raise SystemExit("BLOCKED: smoke/overfit/pilot are seed 42 only")
    teacher_required = any(arm in {"D24", "D25"} for arm in args.arms)
    if teacher_required and not all((args.teacher_checkpoint, args.teacher_evidence,
                                     args.teacher_selection_rule, args.teacher_training_data)):
        raise SystemExit("BLOCKED: D24/D25 require explicit teacher checkpoint/evidence/selection/training provenance")

    root = Path(args.project_root).expanduser().resolve()
    d1_checkpoint = _resolved(root, args.d1_checkpoint)
    preflight = formal_preflight(
        root, str(d1_checkpoint), args.protocol_lock, args.split_contract,
        "best_validation_psnr", args.d1_initial_inventory, args.d1_checkpoint_binding,
    )
    if preflight["status"] != "passed":
        print(json.dumps(preflight, ensure_ascii=False, indent=2, allow_nan=False))
        raise SystemExit("BLOCKED: BEST CHECKPOINT EVIDENCE")
    lock_path = _resolved(root, args.protocol_lock)
    lock = load_protocol_lock(lock_path)
    strata_path = _resolved(root, args.vessel_strata_definition)
    strata = json.loads(strata_path.read_text(encoding="utf-8-sig"))
    claimed = strata.get("definition_sha256")
    payload = dict(strata); payload.pop("definition_sha256", None)
    if claimed != stable_sha(payload) or strata.get("threshold_source") != "development_train_gt_only":
        raise SystemExit("BLOCKED: METRIC VALIDATION")

    teacher_path = _resolved(root, args.teacher_checkpoint) if args.teacher_checkpoint else None
    teacher_evidence_path = _resolved(root, args.teacher_evidence) if args.teacher_evidence else None
    teacher_sha = None
    if teacher_path:
        raw = torch.load(teacher_path, map_location="cpu", weights_only=False)
        validate_checkpoint_config(raw, lock, "D2 teacher")
        teacher_sha = sha256_file(teacher_path)
        evidence = json.loads(teacher_evidence_path.read_text(encoding="utf-8-sig"))
        evidence_checkpoint_sha = evidence.get("checkpoint_sha256") or evidence.get("best_checkpoint_sha256")
        if (evidence_checkpoint_sha != teacher_sha
                or evidence.get("status", "passed") not in {"passed", "completed"}
                or int(evidence.get("test_assets_opened", -1)) != 0
                or evidence.get("selection_rule") != args.teacher_selection_rule
                or evidence.get("training_data") != args.teacher_training_data
                or evidence.get("split") != args.teacher_split):
            raise SystemExit("BLOCKED: teacher evidence does not bind the selected checkpoint")

    base = load_config(root / "configs/adaptive_denoising/d2_seed42_template.yaml")
    base.pop("runtime", None)
    mode = MODE[args.mode]
    output_root = _resolved(root, args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / f"prepare_{args.mode}_{args.tag}_seed{args.seed}.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing existing preparation report: {report_path}")
    generated = []
    for arm in args.arms:
        cfg = deepcopy(base)
        cfg["seed"] = args.seed
        cfg["deterministic"] = True
        cfg["device"] = args.device or mode["device"]
        cfg.setdefault("data", {}).update({
            "load_segmentation_labels": True,
            "max_train_samples": mode["max_train_samples"],
            "max_val_samples": mode["max_val_samples"],
        })
        if args.mode == "overfit":
            cfg["data"].setdefault("augmentation", {})["horizontal_flip"] = 0.0
        cfg.setdefault("model", {}).update({
            "d2s_enabled": False, "s2d_enabled": False,
            "enable_denoise_to_seg": False, "enable_seg_to_denoise": False,
        })
        run = root / "runs/adaptive_denoising/pku37_binary_v3/d2_v1" / f"{args.mode}_{args.tag}" / f"{arm.lower()}_seed{args.seed}"
        cfg.setdefault("train", {}).update({
            "stage": "denoise", "output_dir": str(run), "epochs": mode["epochs"],
            "fixed_epoch": mode["epochs"], "early_stopping_patience": mode["epochs"] + 1,
            "checkpoint_selection_rule": "d2_hierarchical_fixed_budget_v1",
            "monitor": "psnr", "pretrained": str(d1_checkpoint), "strict_pretrained": True,
            "resume": None, "num_workers": 0 if args.mode == "smoke" else cfg["train"].get("num_workers", 4),
            "memory_safe_d2_teacher": True,
        })
        cfg["training_asset_evidence"] = {
            "enabled": True, "project_root": str(root), "protocol_lock": str(lock_path),
        }
        weights = {**BASE_WEIGHTS, **ARMS[arm]}
        cfg.setdefault("loss", {}).update({
            "definition_version": f"pku37-v3-{arm.lower()}-structure-d2-v1",
            "restoration_mode": "structure_d2",
            "d2": {"weights": weights, "boundary_width_pixels": 3, "epsilon": 1e-6,
                   "cnr_error_cap": 5.0, "structure_beta": 2.0},
        })
        cfg["loss"].setdefault("weights", {}).update({
            "reconstruction": 1.0, "residual": 0.0,
            "identity": 0.05 if arm == "D25" else 0.0,
        })
        cfg["d2"] = {
            "enabled": True, "template_only": False, "arm": arm, "run_mode": args.mode,
            "scientific_evaluation": args.mode in {"pilot", "full"},
            "notice": (
                "" if args.mode in {"pilot", "full"}
                else "NOT FOR SCIENTIFIC EVALUATION"
            ),
            "selection": {"rule": "psnr_noninferiority_then_teacher", "psnr_noninferiority_db": 0.2,
                          "tie_break": "earliest_epoch", "fixed_before_training": True},
            "vessel_strata_definition": str(strata_path),
            "vessel_strata_definition_sha256": claimed,
            "teacher": {
                "enabled": arm in {"D24", "D25"},
                "checkpoint": str(teacher_path) if arm in {"D24", "D25"} else None,
                "sha256": teacher_sha if arm in {"D24", "D25"} else None,
                "selection_rule": args.teacher_selection_rule if arm in {"D24", "D25"} else None,
                "training_data": args.teacher_training_data if arm in {"D24", "D25"} else None,
                "split": args.teacher_split if arm in {"D24", "D25"} else None,
                "evidence": str(teacher_evidence_path) if arm in {"D24", "D25"} else None,
                "evidence_sha256": sha256_file(teacher_evidence_path) if arm in {"D24", "D25"} else None,
            },
        }
        config_path = output_root / f"{arm.lower()}_{args.mode}_{args.tag}_seed{args.seed}.yaml"
        if config_path.exists() or run.exists():
            raise FileExistsError(f"Refusing existing config/run: {config_path} / {run}")
        save_config(cfg, config_path)
        generated.append({"arm": arm, "config": str(config_path), "run": str(run),
                          "command": f"python train.py --config {config_path}"})
    report = {
        "status": "prepared", "seed": args.seed, "mode": args.mode, "tag": args.tag,
        "device": args.device or mode["device"],
        "d1_checkpoint_sha256": sha256_file(d1_checkpoint),
        "d1_checkpoint_binding": str(_resolved(root, args.d1_checkpoint_binding)),
        "vessel_strata_definition_sha256": claimed, "runs": generated,
        "scientific_evaluation": args.mode in {"pilot", "full"},
        "notice": "" if args.mode in {"pilot", "full"} else "NOT FOR SCIENTIFIC EVALUATION",
        "test_assets_opened": 0,
    }
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
