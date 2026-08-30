from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config, save_config
from sabids.engine.evaluator import evaluate_model
from sabids.engine.trainer import Trainer
from sabids.utils import load_checkpoint, write_json


VARIANTS = ("noisy", "denoised")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_for(root: Path, variant: str, seed: int, args: argparse.Namespace) -> Dict:
    config = load_config(root / "configs/current" / f"input_{variant}_fold0.yaml")
    config = copy.deepcopy(config)
    config["seed"] = seed
    config["device"] = args.device
    config["train"]["epochs"] = 1 if args.smoke_test else args.epochs
    config["train"]["num_workers"] = 0 if args.smoke_test else int(config["train"].get("num_workers", 4))
    config["evaluation"]["num_workers"] = 0 if args.smoke_test else int(config["evaluation"].get("num_workers", 2))
    if args.smoke_test:
        config["data"]["samples_per_epoch"] = 2
        config["data"]["max_val_samples"] = 2
    suffix = "_smoke" if args.smoke_test else ""
    config["train"]["output_dir"] = str(
        root / "runs/current" / f"input_{variant}_fold0_seed{seed}{suffix}"
    )
    for section, key in (("data", "manifest"), ("data", "root"), ("train", "pretrained")):
        value = config[section].get(key)
        if value is None:
            continue
        path = Path(value)
        config[section][key] = str(path.resolve() if path.is_absolute() else (root / path).resolve())
    return config


def validate_pair(configs: Dict[str, Dict]) -> Dict:
    noisy, denoised = configs["noisy"], configs["denoised"]
    snapshot = Path(noisy["train"]["pretrained"])
    manifest = Path(noisy["data"]["manifest"])
    missing = [str(path) for path in (snapshot, manifest) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input-factorial prerequisite: " + ", ".join(missing))
    controlled = (
        ("seed",), ("data", "manifest"), ("data", "target_size"),
        ("data", "samples_per_epoch"), ("data", "augmentation"),
        ("train", "epochs"), ("train", "batch_size"),
        ("train", "learning_rate"), ("train", "scheduler"),
        ("train", "pretrained"), ("loss",),
    )
    for path in controlled:
        left, right = noisy, denoised
        for key in path:
            left, right = left[key], right[key]
        if left != right:
            raise ValueError(f"Paired configs differ outside input_column: {'.'.join(path)}")
    if noisy["data"]["input_column"] != "noisy_cache_path" or denoised["data"]["input_column"] != "denoised_cache_path":
        raise ValueError("Input aliases do not select the paired float cache columns")
    for config in configs.values():
        if config["train"]["stage"] != "input_segment":
            raise ValueError("I experiment must use input_segment")
        if config["model"].get("d2s_enabled") or config["model"].get("s2d_enabled"):
            raise ValueError("Interactions must be disabled")
        for key in ("reconstruction", "residual", "rmac", "pseudo", "identity"):
            if float(config["loss"]["weights"].get(key, 0.0)) != 0.0:
                raise ValueError(f"Forbidden I-experiment loss is active: {key}")
    return {
        "snapshot": str(snapshot), "snapshot_sha256": sha256_file(snapshot),
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "fixed_final_epoch": int(noisy["train"]["epochs"]),
        "threshold": 0.5, "postprocess": "P0", "test_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired I_NOISY/I_DENOISED runner")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=("audit", "train", "evaluate"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--component-size-thresholds", type=int, nargs=2, default=None,
        metavar=("SMALL_MAX", "MEDIUM_MAX"),
        help="Pre-registered vessel component areas in restored original-image pixels.",
    )
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    registry_root = root / "runs/input_factorial_registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        configs = {variant: config_for(root, variant, seed, args) for variant in VARIANTS}
        registry = validate_pair(configs)
        registry.update({"seed": seed, "mode": args.mode, "variants": list(VARIANTS)})
        digest = hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()[:12]
        registry_path = registry_root / f"{args.mode}_seed{seed}_{digest}.json"
        if not registry_path.exists():
            write_json(registry, registry_path)
        if args.mode == "audit":
            print(json.dumps(registry, ensure_ascii=False, indent=2))
            continue
        expected_state = expected_plan = expected_trainable = None
        pair_audit = {}
        for variant, config in configs.items():
            output = Path(config["train"]["output_dir"])
            if args.mode == "train":
                if output.exists() and any(output.iterdir()):
                    raise FileExistsError(f"Refusing to overwrite existing run: {output}")
                trainer = Trainer(config)
                save_config(config, output / "resolved_config.yaml")
                audit = json.loads((output / "initialization_audit.json").read_text(encoding="utf-8"))
                parameter_audit = json.loads((output / "parameter_audit.json").read_text(encoding="utf-8"))
                state, plan = audit["model_state_sha256"], audit["data_plan_sha256"]
                trainable = sorted(parameter_audit["trainable"])
                expected_state = state if expected_state is None else expected_state
                expected_plan = plan if expected_plan is None else expected_plan
                expected_trainable = trainable if expected_trainable is None else expected_trainable
                if state != expected_state or plan != expected_plan or trainable != expected_trainable:
                    raise RuntimeError(f"Paired initialization/data/trainable mismatch: {variant}, seed {seed}")
                pair_audit[variant] = {
                    "model_state_sha256": state, "data_plan_sha256": plan,
                    "trainable_parameter_names_sha256": hashlib.sha256("\n".join(trainable).encode()).hexdigest(),
                    "input_column": config["data"]["input_column"],
                }
                trainer.fit()
            else:
                checkpoint = output / "last.pth"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing fixed-final checkpoint: {checkpoint}")
                payload = torch.load(checkpoint, map_location="cpu")
                expected_epoch = int(config["train"]["epochs"])
                if int(payload.get("epoch", -1)) + 1 != expected_epoch:
                    raise RuntimeError(f"{variant} did not reach fixed epoch {expected_epoch}")
                destination = output / "final_validation"
                if destination.exists() and any(destination.iterdir()):
                    raise FileExistsError(f"Refusing to overwrite evaluation: {destination}")
                eval_config = copy.deepcopy(config)
                eval_config["train"]["output_dir"] = str(output / "evaluation_audit")
                evaluation_audit = Path(eval_config["train"]["output_dir"])
                if evaluation_audit.exists() and any(evaluation_audit.iterdir()):
                    raise FileExistsError(
                        f"Refusing to overwrite evaluation audit: {evaluation_audit}"
                    )
                trainer = Trainer(eval_config)
                load_checkpoint(checkpoint, trainer.model, strict=True, map_location=trainer.device)
                evaluate_model(
                    trainer.model, trainer.val_loader, trainer.device,
                    output_dir=destination, threshold=0.5, layer_threshold=0.5,
                    vessel_threshold=0.5, save_predictions=args.save_predictions,
                    stage="input_segment", tasks=("layer", "vessel"),
                    postprocess_modes=("p0",), restore_original_geometry=True,
                    input_normalization=str(config["data"].get("normalization", "fixed")),
                    component_size_thresholds=(
                        tuple(args.component_size_thresholds)
                        if args.component_size_thresholds else None
                    ),
                )
                write_json({
                    "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                    "fixed_final_epoch": expected_epoch, "threshold": 0.5, "postprocess": "P0",
                    "restored_original_geometry": True, "test_used": False,
                    "component_size_thresholds_original_pixels": args.component_size_thresholds,
                }, destination / "evaluation_registry.json")
                trainer.writer.close()
        if args.mode == "train":
            write_json(pair_audit, registry_root / f"paired_data_plan_audit_seed{seed}.json")


if __name__ == "__main__":
    main()
