from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config, save_config
from sabids.engine.evaluator import evaluate_model
from sabids.engine.trainer import Trainer, build_model
from sabids.utils import load_checkpoint, write_json


VARIANTS = ("j00", "j10", "j01", "j11")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audited runner for the detached 2x2 interaction factorial."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=("audit", "b0", "train", "evaluate"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--component-size-thresholds", type=int, nargs=2, default=None,
        metavar=("SMALL_MAX", "MEDIUM_MAX"),
        help="Pre-registered component-area bins in restored original-image pixels.",
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def resolved_config(root: Path, variant: str, seed: int, args: argparse.Namespace) -> Dict:
    config = load_config(root / "configs" / "current" / f"interaction_{variant}_fold0.yaml")
    config = copy.deepcopy(config)
    config["seed"] = seed
    config["train"]["epochs"] = args.epochs
    config["train"]["output_dir"] = str(
        root / "runs" / "current" / f"interaction_{variant}_fold0_seed{seed}"
    )
    if args.device:
        config["device"] = args.device
    if args.smoke_test:
        config["train"]["epochs"] = 1
        config["train"]["num_workers"] = 0
        config["evaluation"]["num_workers"] = 0
        config["data"]["samples_per_epoch"] = 2
        config["data"]["max_val_samples"] = 2
    for key in ("manifest",):
        path = Path(config["data"][key])
        if not path.is_absolute():
            config["data"][key] = str((root / path).resolve())
    data_root = Path(config["data"].get("root", "."))
    if not data_root.is_absolute():
        config["data"]["root"] = str((root / data_root).resolve())
    checkpoint = Path(config["train"]["pretrained"])
    if not checkpoint.is_absolute():
        config["train"]["pretrained"] = str((root / checkpoint).resolve())
    return config


def validate_contract(configs: Dict[str, Dict]) -> Dict:
    first = next(iter(configs.values()))
    checkpoint = Path(first["train"]["pretrained"])
    manifest = Path(first["data"]["manifest"])
    missing = [str(path) for path in (checkpoint, manifest) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required registered input is missing; refusing historical fallback: " + ", ".join(missing)
        )
    table = pd.read_csv(manifest, dtype=str).fillna("")
    if "split" not in table or "group_id" not in table:
        raise ValueError("Manifest must contain split and group_id")
    if set(first["data"].get("train_split", "train") for _ in [0]) & {"test"}:
        raise ValueError("The training split must not be test")
    if first["data"].get("val_split", "val") == "test":
        raise ValueError("The validation split must not be test")
    invariant_keys = (
        ("data", "manifest"), ("data", "train_split"), ("data", "val_split"),
        ("data", "target_size"), ("train", "pretrained"), ("train", "epochs"),
        ("train", "batch_size"), ("train", "gradient_accumulation_steps"),
    )
    for variant, config in configs.items():
        for section, key in invariant_keys:
            if config[section].get(key) != first[section].get(key):
                raise ValueError(f"{variant} differs on controlled field {section}.{key}")
        if config["train"].get("stage") != "interaction":
            raise ValueError(f"{variant} must use train.stage=interaction")
        if not config["model"].get("freeze_shared_encoder"):
            raise ValueError(f"{variant} must freeze the shared encoder")
        if not config["model"].get("detach_d2s_source") or not config["model"].get("detach_s2d_source"):
            raise ValueError(f"{variant} must detach both interaction sources")
        for forbidden in ("rmac", "pseudo", "identity"):
            if float(config["loss"]["weights"].get(forbidden, 0.0)) != 0.0:
                raise ValueError(f"{variant} unexpectedly enables {forbidden}")
    retained = table[table["split"].isin([
        first["data"].get("train_split", "train"), first["data"].get("val_split", "val")
    ])].copy()
    patients = {}
    group_patient_map = []
    if "patient_id" in table:
        patients = {
            split: sorted(part["patient_id"].replace("", pd.NA).dropna().unique().tolist())
            for split, part in retained.groupby("split")
        }
        group_patient_map = (
            retained[["split", "group_id", "patient_id"]]
            .drop_duplicates().sort_values(["split", "patient_id", "group_id"])
            .to_dict("records")
        )
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    source_config = checkpoint_payload.get("config", {})
    source_runtime = source_config.get("runtime", {})
    source_model = source_config.get("model", {})
    source_loss = source_config.get("loss", {})
    source_d2s = source_model.get("d2s_enabled", source_model.get("enable_denoise_to_seg"))
    source_identity = {
        "stage": source_config.get("train", {}).get("stage"),
        "loss_definition_version": source_loss.get("definition_version"),
        "d2s_enabled": source_d2s,
        "manifest_sha256": source_runtime.get("manifest_sha256"),
        "effective_split_sha256": source_runtime.get("effective_split_sha256"),
        "effective_groups": source_runtime.get("effective_groups"),
        "label_assets_decoded_sha256": source_runtime.get("label_assets_decoded_sha256"),
        "stage1_initialization_checkpoint": source_runtime.get("initialization_checkpoint"),
        "stage1_initialization_checkpoint_sha256": source_runtime.get("initialization_checkpoint_sha256"),
    }
    if source_d2s is not False or source_loss.get("definition_version") != "roi-bce-dice-outside-bce-no-d2s-v1":
        raise ValueError(
            "The registered checkpoint does not identify itself as the E3b-noD2S protocol"
        )
    required_identity_fields = (
        "manifest_sha256", "effective_split_sha256", "effective_groups",
        "label_assets_decoded_sha256", "stage1_initialization_checkpoint_sha256",
    )
    unknown_identity_fields = [
        key for key in required_identity_fields if source_identity.get(key) in (None, "", {})
    ]
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "rows_by_split": retained["split"].value_counts().astype(int).to_dict(),
        "groups_by_split": {
            split: sorted(part["group_id"].unique().tolist()) for split, part in retained.groupby("split")
        },
        "patient_ids_by_split": patients,
        "group_patient_cluster_map": group_patient_map,
        "source_checkpoint_identity": source_identity,
        "source_checkpoint_identity_status": (
            "matched_protocol_with_complete_fingerprints" if not unknown_identity_fields
            else "matched_protocol_but_fingerprint_fields_unknown"
        ),
        "source_checkpoint_unknown_fields": unknown_identity_fields,
        "test_assets_opened": 0,
        "test_evaluation_performed": False,
        "fixed_threshold": 0.5,
        "primary_checkpoint_rule": "last epoch; all tasks from the same last.pth",
    }


def evaluate(
    config: Dict, checkpoint: Path, output: Path, save_predictions: bool,
    reset_interaction_scales: bool = False,
    component_size_thresholds: tuple[int, int] | None = None,
) -> None:
    trainer = Trainer(config)
    load_checkpoint(checkpoint, trainer.model, strict=True, map_location=trainer.device)
    if reset_interaction_scales:
        with torch.no_grad():
            for name, parameter in trainer.model.named_parameters():
                if name.endswith(("seg_scale", "layer_scale", "vessel_scale")):
                    parameter.zero_()
    trainer.model.eval()
    profile_batch = next(iter(trainer.val_loader))["image"][:1].to(trainer.device)
    timings = []
    with torch.no_grad():
        _ = trainer.model(profile_batch, return_features=False, return_auxiliary=False)
        for _ in range(5):
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
                torch.cuda.reset_peak_memory_stats(trainer.device)
            start = time.perf_counter()
            _ = trainer.model(profile_batch, return_features=False, return_auxiliary=False)
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            timings.append(time.perf_counter() - start)
    output.mkdir(parents=True, exist_ok=True)
    write_json({
        "batch_size": 1,
        "input_shape": list(profile_batch.shape),
        "forward_seconds_mean": float(sum(timings) / len(timings)),
        "forward_seconds_samples": timings,
        "cuda_peak_memory_bytes_last_forward": (
            int(torch.cuda.max_memory_allocated(trainer.device)) if trainer.device.type == "cuda" else None
        ),
        "total_parameters": int(sum(parameter.numel() for parameter in trainer.model.parameters())),
        "trainable_parameters_under_variant": int(sum(
            parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
        )),
    }, output / "cost_profile.json")
    write_json({
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_role": "B0 initialization" if reset_interaction_scales else "fixed final epoch",
        "interaction_scales_reset_to_zero": bool(reset_interaction_scales),
        "split": str(config["data"].get("val_split", "val")),
        "tasks": ["denoise", "layer", "vessel"],
        "postprocess_mode": "P0 only",
        "layer_threshold": 0.5,
        "vessel_threshold": 0.5,
        "restored_original_geometry": True,
        "component_size_thresholds_original_pixels": component_size_thresholds,
        "test_evaluation_performed": False,
    }, output / "evaluation_registry.json")
    evaluate_model(
        trainer.model, trainer.val_loader, trainer.device, output_dir=output,
        threshold=0.5, layer_threshold=0.5, vessel_threshold=0.5,
        save_predictions=save_predictions, stage="interaction",
        input_normalization=str(config["data"].get("normalization", "fixed")),
        tasks=("denoise", "layer", "vessel"), postprocess_modes=("p0",),
        restore_original_geometry=True,
        component_size_thresholds=component_size_thresholds,
    )
    trainer.writer.close()


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    registry_dir = root / "runs" / "interaction_factorial_registry"
    registry_dir.mkdir(parents=True, exist_ok=True)
    expected_initial_hash: Dict[int, str] = {}
    expected_data_plan_hash: Dict[int, str] = {}
    for seed in args.seeds:
        configs = {variant: resolved_config(root, variant, seed, args) for variant in args.variants}
        registry = validate_contract(configs)
        registry.update({
            "registry_schema": "interaction-factorial-invocation-v2",
            "invocation_mode": args.mode,
            "seed": seed, "variants": list(configs), "epochs": args.epochs,
            "component_size_thresholds_original_pixels": args.component_size_thresholds,
            "resolved_config_sha256_by_variant": {
                variant: hashlib.sha256(
                    json.dumps(config, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()
                for variant, config in configs.items()
            },
        })
        # Audit/B0/train/evaluate may legitimately differ in device and other
        # execution-only fields.  Keep each invocation immutable without
        # treating a later phase as an attempted overwrite of the audit record.
        registry_path = registry_dir / f"{args.mode}_seed{seed}_registry.json"
        if registry_path.is_file():
            existing = json.loads(registry_path.read_text(encoding="utf-8"))
            if existing != registry:
                raise FileExistsError(
                    f"Existing experiment registry differs; refusing overwrite: {registry_path}"
                )
        else:
            write_json(registry, registry_path)
        if args.mode == "audit":
            print(json.dumps(registry, ensure_ascii=False, indent=2))
            continue
        if registry["source_checkpoint_unknown_fields"]:
            raise RuntimeError(
                "Source checkpoint identity is incomplete; do not start an attributable run. "
                "Unknown fields: " + ", ".join(registry["source_checkpoint_unknown_fields"])
            )
        if args.mode == "b0":
            config = configs["j00"]
            output = root / "runs" / "current" / f"interaction_b0_fold0_seed{seed}"
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(f"B0 output already exists; refusing overwrite: {output}")
            config["train"]["output_dir"] = str(output / "audit")
            evaluate(
                config, Path(config["train"]["pretrained"]), output / "validation",
                args.save_predictions, reset_interaction_scales=True,
                component_size_thresholds=(tuple(args.component_size_thresholds) if args.component_size_thresholds else None),
            )
            continue
        for variant, config in configs.items():
            output = Path(config["train"]["output_dir"])
            if args.mode == "train":
                if output.exists() and any(output.iterdir()):
                    raise FileExistsError(f"Run already exists; refusing to overwrite: {output}")
                trainer = Trainer(config)
                save_config(config, output / "resolved_config.yaml")
                audit = json.loads((output / "initialization_audit.json").read_text(encoding="utf-8"))
                state_hash = audit["model_state_sha256"]
                expected_initial_hash.setdefault(seed, state_hash)
                if state_hash != expected_initial_hash[seed]:
                    raise RuntimeError(f"Common initialization mismatch at seed {seed}: {variant}")
                plan_hash = audit["data_plan_sha256"]
                expected_data_plan_hash.setdefault(seed, plan_hash)
                if plan_hash != expected_data_plan_hash[seed]:
                    raise RuntimeError(f"Data/augmentation plan mismatch at seed {seed}: {variant}")
                trainer.fit()
            else:
                checkpoint = output / "last.pth"
                if not checkpoint.is_file():
                    raise FileNotFoundError(f"Final checkpoint is missing: {checkpoint}")
                final_payload = torch.load(checkpoint, map_location="cpu")
                completed_epoch = int(final_payload.get("epoch", -1)) + 1
                expected_epoch = int(config["train"]["epochs"])
                if completed_epoch != expected_epoch:
                    raise RuntimeError(
                        f"Fixed-final checkpoint mismatch for {variant}, seed {seed}: "
                        f"found epoch {completed_epoch}, expected {expected_epoch}"
                    )
                final_output = output / "final_validation"
                audit_output = output / "evaluation_audit"
                for target in (final_output, audit_output):
                    if target.exists() and any(target.iterdir()):
                        raise FileExistsError(f"Evaluation output already exists; refusing overwrite: {target}")
                eval_config = copy.deepcopy(config)
                eval_config["train"]["output_dir"] = str(audit_output)
                evaluate(
                    eval_config, checkpoint, final_output, args.save_predictions,
                    component_size_thresholds=(tuple(args.component_size_thresholds) if args.component_size_thresholds else None),
                )


if __name__ == "__main__":
    main()
