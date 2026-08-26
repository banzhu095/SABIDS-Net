from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader, Subset

from sabids.config import load_config, save_config
from sabids.data import OCTManifestDataset
from sabids.engine import Trainer, evaluate_model
from sabids.engine.trainer import _make_transform, build_model
from sabids.utils import get_device, write_json


DEFAULT_PROJECT_ROOT = r"E:\1-脉络膜\OCT降噪\SABIDS-Net"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run public denoising/segmentation/joint training and optional "
            "private segmentation adaptation"
        )
    )
    parser.add_argument("--project-root", default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["denoise", "segment", "joint"],
        choices=["denoise", "segment", "joint", "private"],
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target-height", type=int, default=None)
    parser.add_argument("--target-width", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--private-manifest",
        default=None,
        help="Private segmentation manifest; defaults to Manifests/manifest_private_seg.csv",
    )
    parser.add_argument("--samples-per-epoch", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--epochs-denoise", type=int, default=None)
    parser.add_argument("--epochs-segment", type=int, default=None)
    parser.add_argument("--epochs-joint", type=int, default=None)
    parser.add_argument("--epochs-private", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true", help="Retrain even if best.pth exists"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume each selected stage from its last.pth checkpoint",
    )
    parser.add_argument("--layer-threshold", type=float, default=None)
    parser.add_argument("--vessel-threshold", type=float, default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Train/validate selected stages without opening the held-out test split",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny CPU-friendly end-to-end check in an isolated output directory",
    )
    return parser.parse_args()


def stage_specifications(
    root: Path, fold: int, smoke_test: bool = False
) -> Dict[str, Dict[str, Path]]:
    if smoke_test:
        run_root = root / "runs" / "current" / "smoke" / f"fold{fold}"
    else:
        run_root = root / "runs" / "current"
    return {
        "denoise": {
            "config": root / "configs" / "current" / "stage1_denoise_fold0.yaml",
            "manifest": root
            / "Manifests"
            / "joint_folds"
            / f"manifest_joint_fold{fold}.csv",
            "output": (
                run_root / "stage1_denoise"
                if smoke_test
                else run_root / f"stage1_denoise_fold{fold}"
            ),
        },
        "segment": {
            "config": root / "configs" / "current" / "stage2_segment_fold0.yaml",
            "manifest": root
            / "Manifests"
            / "segmentation_folds"
            / f"manifest_seg_fold{fold}.csv",
            "output": (
                run_root / "stage2_segment"
                if smoke_test
                else run_root / f"stage2_segment_fold{fold}"
            ),
        },
        "joint": {
            "config": root / "configs" / "current" / "stage4_joint_fold0.yaml",
            "manifest": root
            / "Manifests"
            / "joint_folds"
            / f"manifest_joint_fold{fold}.csv",
            "output": (
                run_root / "stage4_joint"
                if smoke_test
                else run_root / f"stage4_joint_fold{fold}"
            ),
        },
        "private": {
            "config": root
            / "configs"
            / "current"
            / "stage5_private_seg_fold0.yaml",
            "manifest": root / "Manifests" / "manifest_private_seg.csv",
            "output": (
                run_root / "stage5_private_seg"
                if smoke_test
                else run_root / f"stage5_private_seg_fold{fold}"
            ),
        },
    }


def prepare_config(
    stage: str,
    specification: Dict[str, Path],
    root: Path,
    args: argparse.Namespace,
) -> Dict:
    config = load_config(specification["config"])
    config["data"]["manifest"] = str(specification["manifest"])
    config["data"]["root"] = str(root)
    config["device"] = args.device
    if args.smoke_test:
        config["data"]["target_size"] = [
            args.target_height or 64,
            args.target_width or 128,
        ]
        config["model"].update(
            {
                "channels": [8, 16, 32, 64],
                "encoder_depths": [1, 1, 1, 1],
                "decoder_depth": 1,
            }
        )
        config["data"]["samples_per_epoch"] = args.samples_per_epoch or 4
        config["data"]["max_val_samples"] = args.max_val_samples or 2
        config.setdefault("evaluation", {})["max_test_samples"] = (
            args.max_test_samples or 2
        )
        config["train"]["batch_size"] = args.batch_size or 1
        config["train"]["num_workers"] = (
            args.num_workers if args.num_workers is not None else 0
        )
        config["evaluation"]["num_workers"] = (
            args.num_workers if args.num_workers is not None else 0
        )
    else:
        if args.target_height is not None or args.target_width is not None:
            current_height, current_width = config["data"].get(
                "target_size", [512, 512]
            )
            config["data"]["target_size"] = [
                args.target_height or current_height,
                args.target_width or current_width,
            ]
        if args.samples_per_epoch is not None:
            config["data"]["samples_per_epoch"] = args.samples_per_epoch
        if args.max_val_samples is not None:
            config["data"]["max_val_samples"] = args.max_val_samples
        if args.max_test_samples is not None:
            config.setdefault("evaluation", {})["max_test_samples"] = (
                args.max_test_samples
            )
    config["train"]["output_dir"] = str(specification["output"])
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.gradient_accumulation_steps is not None:
        if args.gradient_accumulation_steps < 1:
            raise ValueError("--gradient-accumulation-steps must be >= 1")
        config["train"]["gradient_accumulation_steps"] = (
            args.gradient_accumulation_steps
        )
    if args.num_workers is not None:
        config["train"]["num_workers"] = args.num_workers
        config.setdefault("evaluation", {})["num_workers"] = args.num_workers
    if args.layer_threshold is not None:
        config.setdefault("evaluation", {})["layer_threshold"] = (
            args.layer_threshold
        )
    if args.vessel_threshold is not None:
        config.setdefault("evaluation", {})["vessel_threshold"] = (
            args.vessel_threshold
        )
    epoch_override = {
        "denoise": args.epochs_denoise,
        "segment": args.epochs_segment,
        "joint": args.epochs_joint,
        "private": args.epochs_private,
    }[stage]
    if epoch_override is not None:
        config["train"]["epochs"] = epoch_override
    elif args.smoke_test:
        config["train"]["epochs"] = 1

    if args.device == "cpu":
        config["train"]["amp"] = False

    if args.smoke_test:
        smoke_root = root / "runs" / "current" / "smoke" / f"fold{args.fold}"
        stage1_best = smoke_root / "stage1_denoise" / "best.pth"
        stage2_best = smoke_root / "stage2_segment" / "best.pth"
        stage4_best = smoke_root / "stage4_joint" / "best.pth"
    else:
        stage1_best = (
            root / "runs" / "current" / f"stage1_denoise_fold{args.fold}" / "best.pth"
        )
        stage2_best = (
            root / "runs" / "current" / f"stage2_segment_fold{args.fold}" / "best.pth"
        )
        stage4_best = (
            root / "runs" / "current" / f"stage4_joint_fold{args.fold}" / "best.pth"
        )
    if stage == "segment":
        config["train"]["pretrained"] = str(stage1_best)
    elif stage == "joint":
        config["train"]["pretrained"] = str(stage2_best)
    elif stage == "private":
        config["train"]["pretrained"] = str(stage4_best)
    return config


def _training_signature(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fields that must agree before an existing checkpoint may be reused."""
    data = config.get("data", {})
    train = config.get("train", {})
    return {
        "stage": train.get("stage"),
        "model": config.get("model", {}),
        "data": {
            "manifest": str(data.get("manifest", "")),
            "target_size": data.get("target_size"),
            "normalization": data.get("normalization"),
            "train_datasets": data.get("train_datasets"),
            "val_datasets": data.get("val_datasets"),
            "vessel_oversample_fraction": data.get(
                "vessel_oversample_fraction", 0.0
            ),
            "augmentation": data.get("augmentation", {}),
        },
        "loss": config.get("loss", {}),
        "joint_runtime": {
            key: train.get(key)
            for key in (
                "detach_cross_epochs",
                "ramp_epochs",
                "memory_safe_joint",
                "stopgrad_repeat_teacher",
                "clean_teacher_no_grad",
            )
        },
    }


def _signature_differences(
    checkpoint_path: Path, config: Dict[str, Any]
) -> list[str]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        return ["checkpoint has no saved resolved config"]
    old = _training_signature(saved_config)
    new = _training_signature(config)
    differences = []
    for key in new:
        if json.dumps(old.get(key), sort_keys=True, default=str) != json.dumps(
            new.get(key), sort_keys=True, default=str
        ):
            differences.append(
                f"{key}: checkpoint={old.get(key)!r}, requested={new.get(key)!r}"
            )
    return differences


def _archive_run_outputs(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    candidates = [
        output_dir / "best.pth",
        output_dir / "last.pth",
        output_dir / "history.csv",
        output_dir / "resolved_config.yaml",
        output_dir / "tensorboard",
        output_dir / "test_results",
    ]
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = output_dir / f"archive_{stamp}"
    suffix = 1
    while archive.exists():
        archive = output_dir / f"archive_{stamp}_{suffix}"
        suffix += 1
    archive.mkdir(parents=True)
    for path in existing:
        path.rename(archive / path.name)
    print(f"Archived previous run outputs to: {archive}")


@torch.no_grad()
def evaluate_checkpoint(
    config: Dict,
    checkpoint_path: Path,
    output_dir: Path,
    save_predictions: bool,
) -> Dict:
    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    evaluation = config.get("evaluation", {})
    use_ema = (
        bool(evaluation.get("use_ema", False))
        and checkpoint.get("ema") is not None
    )
    state = checkpoint["ema"] if use_ema else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    dataset = OCTManifestDataset(
        config["data"]["manifest"],
        split="test",
        transform=_make_transform(config, training=False),
        sample_repeat=False,
        root=config["data"].get("root"),
    )
    max_test_samples = config.get("evaluation", {}).get("max_test_samples")
    if max_test_samples is not None:
        count = min(int(max_test_samples), len(dataset))
        dataset = Subset(dataset, range(count))
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("evaluation", {}).get("num_workers", 2)),
        pin_memory=True,
    )
    return evaluate_model(
        model,
        loader,
        device,
        output_dir=output_dir,
        threshold=float(evaluation.get("threshold", 0.5)),
        layer_threshold=float(
            evaluation.get("layer_threshold", evaluation.get("threshold", 0.5))
        ),
        vessel_threshold=float(
            evaluation.get("vessel_threshold", evaluation.get("threshold", 0.5))
        ),
        axial_spacing=float(evaluation.get("axial_spacing", 1.0)),
        lateral_spacing=float(evaluation.get("lateral_spacing", 1.0)),
        save_predictions=save_predictions,
        stage=str(config.get("train", {}).get("stage", "joint")),
    )


def main() -> None:
    args = parse_args()
    if args.force and args.resume:
        raise ValueError("--force and --resume are mutually exclusive")
    project_root = str(args.project_root).strip()
    root = Path(project_root or ".").expanduser().resolve()
    specifications = stage_specifications(root, args.fold, args.smoke_test)
    if args.private_manifest:
        private_manifest = Path(args.private_manifest).expanduser()
        if not private_manifest.is_absolute():
            private_manifest = root / private_manifest
        specifications["private"]["manifest"] = private_manifest.resolve()
    pipeline_summary: Dict[str, Dict] = {}
    mode = "SMOKE TEST" if args.smoke_test else "FULL TRAINING"
    print(f"Project root: {root}")
    print(f"Mode: {mode} | requested device: {args.device} | fold: {args.fold}")
    if not args.smoke_test and args.device == "cpu":
        print("Warning: full training on CPU can take many hours or days.")

    for stage in args.stages:
        specification = specifications[stage]
        if not specification["manifest"].is_file():
            hint = (
                "Copy examples/manifest_private_example.csv to that path, fill "
                "patient-level train/val/test rows, or pass --private-manifest."
                if stage == "private"
                else "Run tools/prepare_current_data.py first."
            )
            raise FileNotFoundError(
                f"Manifest does not exist: {specification['manifest']}. "
                f"{hint}"
            )
        config = prepare_config(stage, specification, root, args)
        output_dir = specification["output"]
        best_checkpoint = output_dir / "best.pth"
        if stage in {"segment", "joint", "private"}:
            pretrained = Path(config["train"]["pretrained"])
            if not pretrained.is_file():
                raise FileNotFoundError(
                    f"Required pretrained checkpoint is missing: {pretrained}"
                )
        last_checkpoint = output_dir / "last.pth"
        if args.resume:
            if not last_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Cannot resume {stage}: missing {last_checkpoint}"
                )
            differences = _signature_differences(last_checkpoint, config)
            if differences:
                details = "\n".join(f"- {item}" for item in differences)
                raise RuntimeError(
                    f"Cannot resume incompatible {stage} checkpoint:\n{details}\n"
                    "Resume requires the same model, input size and loss. "
                    "Use --force to start the stage again from its pretrained stage."
                )
            config["train"]["resume"] = str(last_checkpoint)
            config["train"]["pretrained"] = None
            trainer = Trainer(config)
            save_config(config, output_dir / "resolved_config.yaml")
            trainer.fit()
        elif args.force or not best_checkpoint.is_file():
            if args.force:
                _archive_run_outputs(output_dir)
            trainer = Trainer(config)
            save_config(config, output_dir / "resolved_config.yaml")
            trainer.fit()
        else:
            differences = _signature_differences(best_checkpoint, config)
            if differences:
                details = "\n".join(f"- {item}" for item in differences)
                raise RuntimeError(
                    f"Existing {stage} checkpoint is incompatible with the requested run:\n"
                    f"{details}\nThe checkpoint was NOT reused. Use --force to retrain "
                    "this stage, or restore the checkpoint's original options."
                )
            print(f"Reuse existing checkpoint: {best_checkpoint}")

        if args.skip_test:
            pipeline_summary[stage] = {
                "status": "trained_validation_only",
                "checkpoint": str(best_checkpoint),
            }
            print(
                f"{stage}: held-out test evaluation skipped; "
                f"best checkpoint: {best_checkpoint}"
            )
            continue

        test_output = output_dir / "test_results"
        summary = evaluate_checkpoint(
            config,
            best_checkpoint,
            test_output,
            save_predictions=args.save_predictions,
        )
        pipeline_summary[stage] = summary
        print(f"{stage} test summary: {summary}")

    if args.smoke_test:
        summary_path = (
            root
            / "runs"
            / "current"
            / "smoke"
            / f"fold{args.fold}"
            / "pipeline_summary.json"
        )
    else:
        summary_path = (
            root / "runs" / "current" / f"pipeline_summary_fold{args.fold}.json"
        )
    write_json(pipeline_summary, summary_path)
    print(f"Pipeline summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
