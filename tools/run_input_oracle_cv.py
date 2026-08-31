from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config, save_config
from sabids.engine.evaluator import evaluate_model
from sabids.engine.trainer import Trainer
from sabids.utils import load_checkpoint, write_json
from tools.input_oracle_cv_common import (
    ARMS, read_fold_groups, require_phase0, require_split_audit, resolve,
    sha256_file,
)


INPUT_COLUMNS = {"noisy": "noisy_cache_path", "clean": "clean_cache_path", "denoised": "denoised_cache_path"}


def _config(root: Path, output: Path, fold: int, arm: str, seed: int, args: argparse.Namespace) -> dict:
    config = copy.deepcopy(load_config(root / "configs/input_factorial_common.yaml"))
    fold_root = output / f"fold{fold}{'_smoke' if args.smoke_test else ''}"
    run = fold_root / f"{arm}_seed{seed}"
    config.update({"seed": seed, "device": args.device})
    config["data"].update({
        "manifest": str(root / f"Manifests/input_oracle_cv/fold_{fold}_input{'_smoke' if args.smoke_test else ''}.csv"),
        "root": str(root), "input_column": INPUT_COLUMNS[arm],
    })
    config["train"].update({
        "output_dir": str(run), "epochs": 1 if args.smoke_test else args.epochs,
        "pretrained": str(fold_root / "preseg_initialization.pth"),
    })
    config.setdefault("runtime", {}).update({
        "input_oracle_arm": arm, "input_oracle_fold": fold,
        "input_oracle_common_initialization": config["train"]["pretrained"],
    })
    if args.smoke_test:
        config["train"]["num_workers"] = 0
        config["evaluation"]["num_workers"] = 0
        config["data"]["samples_per_epoch"] = 2
        config["data"]["max_val_samples"] = 2
    if args.resume:
        last = run / "last.pth"
        if not last.is_file():
            raise FileNotFoundError(f"--resume requires {last}")
        config["train"]["resume"] = str(last)
    return config


def _validate_triplet(configs: dict[str, dict], fold: int, seed: int) -> dict:
    reference = configs["noisy"]
    controlled = (
        ("seed",), ("data", "manifest"), ("data", "target_size"),
        ("data", "samples_per_epoch"), ("data", "augmentation"),
        ("train", "epochs"), ("train", "batch_size"), ("train", "learning_rate"),
        ("train", "scheduler"), ("train", "pretrained"), ("loss",),
    )
    for arm, config in configs.items():
        for keys in controlled:
            left, right = reference, config
            for key in keys:
                left, right = left[key], right[key]
            if left != right:
                raise ValueError(f"Fold {fold} seed {seed}: {arm} differs at {'.'.join(keys)}")
        if config["train"]["stage"] != "input_segment":
            raise ValueError("Three-arm experiment must use input_segment")
        if config["model"].get("d2s_enabled") or config["model"].get("s2d_enabled"):
            raise ValueError("Cross-task interactions must be off in the input-oracle experiment")
        for key in ("reconstruction", "residual", "identity", "rmac", "pseudo"):
            if float(config["loss"]["weights"].get(key, 0.0)) != 0.0:
                raise ValueError(f"Forbidden loss active: {key}")
    manifest = Path(reference["data"]["manifest"])
    snapshot = Path(reference["train"]["pretrained"])
    missing = [str(x) for x in (manifest, snapshot) if not x.is_file()]
    if missing:
        raise FileNotFoundError("Missing three-arm prerequisite: " + ", ".join(missing))
    return {
        "fold": fold, "seed": seed, "arms": list(ARMS),
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "common_initialization": str(snapshot), "common_initialization_sha256": sha256_file(snapshot),
        "fixed_final_epoch": int(reference["train"]["epochs"]), "threshold": 0.5,
        "postprocess": "P0", "test_assets_opened": 0,
    }


def _clean_once_loader(loader: DataLoader) -> DataLoader:
    dataset = loader.dataset
    if isinstance(dataset, Subset):
        base = dataset.dataset
        candidates = list(dataset.indices)
    else:
        base = dataset
        candidates = list(range(len(dataset)))
    selected, seen = [], set()
    for index in candidates:
        group = str(base.table.iloc[index]["group_id"])
        if group not in seen:
            selected.append(index)
            seen.add(group)
    return DataLoader(Subset(base, selected), batch_size=1, shuffle=False, num_workers=0, drop_last=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired NOISY/CLEAN/DENOISED segmentation CV runner")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="runs/input_oracle_cv")
    parser.add_argument("--mode", choices=("audit", "train", "evaluate"), required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Single-seed alias")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--component-size-thresholds", type=int, nargs=2, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.seed is not None and args.seeds is not None:
        parser.error("Use either --seed or --seeds, not both")
    args.seeds = args.seeds or ([args.seed] if args.seed is not None else [42])
    root = Path(args.project_root).expanduser().resolve()
    output = resolve(root, args.output)
    phase0 = require_phase0(root, output)
    split = require_split_audit(output, args.fold)
    _, val_groups = read_fold_groups(output, args.fold)
    if val_groups & set(map(str, phase0["sealed_test_groups"])):
        raise RuntimeError("Fold validation contains a sealed test group")
    registry_root = output / "registry"
    registry_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        configs = {arm: _config(root, output, args.fold, arm, seed, args) for arm in ARMS}
        registry = _validate_triplet(configs, args.fold, seed)
        registry["mode"] = args.mode
        digest = hashlib.sha256(json.dumps(registry, sort_keys=True).encode()).hexdigest()[:12]
        registry_path = registry_root / f"fold{args.fold}_{args.mode}_seed{seed}_{digest}.json"
        if not registry_path.exists():
            write_json(registry, registry_path)
        for arm, config in configs.items():
            config_path = registry_root / f"fold{args.fold}_{arm}_seed{seed}{'_smoke' if args.smoke_test else ''}.yaml"
            if not config_path.exists():
                save_config(config, config_path)
        if args.mode == "audit" or args.dry_run:
            print(json.dumps(registry, ensure_ascii=False, indent=2))
            continue
        expected_state = expected_plan = expected_trainable = None
        paired = {}
        for arm in args.arms:
            config = configs[arm]
            run = Path(config["train"]["output_dir"])
            if args.mode == "train":
                if not args.resume and run.exists() and any(run.iterdir()):
                    raise FileExistsError(f"Refusing to overwrite run: {run}")
                started = time.perf_counter()
                trainer = Trainer(config)
                if arm == "clean":
                    trainer.val_loader = _clean_once_loader(trainer.val_loader)
                save_config(config, run / "resolved_config.yaml")
                initialization = json.loads((run / "initialization_audit.json").read_text(encoding="utf-8"))
                parameters = json.loads((run / "parameter_audit.json").read_text(encoding="utf-8"))
                state, plan, trainable = initialization["model_state_sha256"], initialization["data_plan_sha256"], sorted(parameters["trainable"])
                expected_state = state if expected_state is None else expected_state
                expected_plan = plan if expected_plan is None else expected_plan
                expected_trainable = trainable if expected_trainable is None else expected_trainable
                if state != expected_state or plan != expected_plan or trainable != expected_trainable:
                    raise RuntimeError(f"Three-arm pairing failed for {arm}, fold {args.fold}, seed {seed}")
                trainer.fit()
                paired[arm] = {
                    "model_state_sha256": state, "data_plan_sha256": plan,
                    "trainable_names_sha256": hashlib.sha256("\n".join(trainable).encode()).hexdigest(),
                    "wall_seconds": time.perf_counter() - started,
                    "max_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
                    "trainable_parameters": int(parameters.get("trainable_parameters", sum(p.numel() for p in trainer.model.parameters() if p.requires_grad))),
                }
                write_json(paired[arm], run / "cost_profile.json")
                trainer.writer.close()
            else:
                checkpoint = run / "last.pth"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Missing fixed-final checkpoint: {checkpoint}")
                payload = torch.load(checkpoint, map_location="cpu")
                if int(payload.get("epoch", -1)) + 1 != int(config["train"]["epochs"]):
                    raise RuntimeError(f"{arm} checkpoint is not fixed final epoch {config['train']['epochs']}")
                destination = run / "final_validation"
                if destination.exists() and any(destination.iterdir()):
                    raise FileExistsError(f"Refusing to overwrite evaluation: {destination}")
                eval_config = copy.deepcopy(config)
                eval_config["train"]["output_dir"] = str(run / "evaluation_audit")
                trainer = Trainer(eval_config)
                load_checkpoint(checkpoint, trainer.model, strict=True, map_location=trainer.device)
                loader = _clean_once_loader(trainer.val_loader) if arm == "clean" else trainer.val_loader
                evaluate_model(
                    trainer.model, loader, trainer.device, output_dir=destination,
                    threshold=0.5, layer_threshold=0.5, vessel_threshold=0.5,
                    save_predictions=args.save_predictions, stage="input_segment", tasks=("layer", "vessel"),
                    postprocess_modes=("p0",), restore_original_geometry=True,
                    input_normalization=str(config["data"].get("normalization", "fixed")),
                    component_size_thresholds=tuple(args.component_size_thresholds) if args.component_size_thresholds else None,
                )
                write_json({
                    "arm": arm, "fold": args.fold, "seed": seed,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                    "fixed_final_epoch": int(config["train"]["epochs"]), "threshold": 0.5,
                    "postprocess": "P0", "clean_evaluated_once_per_position": arm == "clean",
                    "repeat_stability_applicable": arm != "clean", "test_assets_opened": 0,
                }, destination / "evaluation_registry.json")
                trainer.writer.close()
        if args.mode == "train":
            pair_path = output / f"fold{args.fold}{'_smoke' if args.smoke_test else ''}/paired_data_plan_audit_fold{args.fold}_seed{seed}.json"
            write_json({"fold": args.fold, "seed": seed, "arms": paired, "all_equal": True, "test_assets_opened": 0}, pair_path)


if __name__ == "__main__":
    main()
