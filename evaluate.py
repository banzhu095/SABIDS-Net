from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.engine.evaluator import evaluate_model
from sabids.engine.trainer import _make_transform, build_model
from sabids.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Group-level SABIDS-Net evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/evaluation")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument(
        "--one-frame-per-group",
        action="store_true",
        help="Use the first manifest row of every group for deterministic diagnostics.",
    )
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--layer-threshold", type=float, default=None)
    parser.add_argument("--vessel-threshold", type=float, default=None)
    parser.add_argument(
        "--component-size-thresholds",
        type=int,
        nargs=2,
        metavar=("SMALL_MAX", "MEDIUM_MAX"),
        default=None,
        help="Training-defined GT component-area thresholds in model pixels.",
    )
    parser.add_argument("--boundary-band-width", type=float, default=None)
    parser.add_argument(
        "--tasks", nargs="+", choices=("denoise", "layer", "vessel"), default=None,
        help="Explicit V0 task selection; otherwise use the stage-aware default.",
    )
    parser.add_argument(
        "--postprocess-modes", nargs="+", choices=("p0", "p1", "p2", "p3"),
        default=("p0",), help="Evaluate immutable raw P0 and selected anatomical postprocessing modes.",
    )
    parser.add_argument("--no-restore-original-geometry", action="store_true")
    parser.add_argument("--layer-surface-tolerance", type=float, default=None)
    parser.add_argument("--p1-minimum-main-fraction", type=float, default=0.5)
    parser.add_argument("--p2-smoothness", type=float, default=2.0)
    parser.add_argument("--p2-max-displacement", type=int, default=8)
    parser.add_argument(
        "--vessel-strata-definition",
        help="Frozen train-only D2 vessel-strata JSON; omitted for legacy evaluation.",
    )
    parser.add_argument(
        "--evaluate-clean-identity", action="store_true",
        help="Explicit denoising diagnostic: run the model on clean validation input.",
    )
    parser.add_argument(
        "--d2-diagnostics", action="store_true",
        help="Enable opt-in D2 vessel-ROI, residual leakage and hallucination metrics.",
    )
    parser.add_argument(
        "--d2-checkpoint-kind", choices=("d2_pixel", "d2_task", "d2_last"),
        help="Required explicit checkpoint identity for an opt-in D2 run.",
    )
    parser.add_argument(
        "--d2-checkpoint-binding",
        help="Required immutable binding JSON for an opt-in D2 checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dose = config.get("dose_response", {}).get("enabled", False)
    if dose:
        from sabids.experiments.dose_response import validate_dose_config
        validate_dose_config(config, require_fresh=False)
        if (args.split != "val" or args.one_frame_per_group or args.use_ema
                or not args.no_restore_original_geometry or tuple(args.postprocess_modes) != ("p0",)
                or args.tasks != ["layer", "vessel"]
                or args.layer_threshold not in (None, 0.5) or args.vessel_threshold not in (None, 0.5)):
            raise ValueError("Dose evaluation requires complete val, tasks layer vessel, P0 0.5, --no-restore-original-geometry; no test/EMA")
        evaluation_output = Path(args.output).resolve()
        if Path(config["train"]["output_dir"]).resolve() not in evaluation_output.parents:
            raise ValueError("Dose evaluation output must be a new child of this dose run")
        if evaluation_output.exists():
            raise FileExistsError("Dose evaluation output must be fresh")
    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, **({"weights_only": False} if dose else {}))
    if config.get("d2", {}).get("enabled", False):
        if not args.d2_checkpoint_kind or not args.d2_checkpoint_binding:
            raise ValueError("D2 evaluation requires explicit checkpoint kind and binding")
        from sabids.experiments.d2 import audit_d2_checkpoint_binding
        expected_binding_kind = {
            "d2_pixel": "best_pixel",
            "d2_task": "best_task_preserving",
            "d2_last": "last",
        }[args.d2_checkpoint_kind]
        audit_d2_checkpoint_binding(
            Path(args.d2_checkpoint_binding).expanduser().resolve(),
            Path(args.checkpoint).expanduser().resolve(),
            expected_binding_kind,
        )
    if dose:
        import json
        from sabids.experiments.dose_response import sha256_file
        expected = {k: v for k, v in config.items() if k != "runtime"}
        embedded = {k: v for k, v in checkpoint["config"].items() if k != "runtime"}
        if expected != embedded:
            raise ValueError("Dose evaluation checkpoint is not from this registered arm")
        if Path(args.checkpoint).name not in {"last.pth", "best.pth"}:
            raise ValueError("Dose evaluation expects registered final or secondary best")
        run = Path(config["train"]["output_dir"])
        completion = json.loads((run / "dose_training_metadata.json").read_text(encoding="utf-8"))
        if (completion["completed_epochs"] != config["train"]["epochs"]
                or completion["completed_optimizer_steps"] != completion["expected_optimizer_steps"]
                or completion["changed_frozen_parameter_names"] or not completion["changed_trainable_parameter_names"]):
            raise ValueError("Dose evaluation requires completed, budget-matched training with valid parameter updates")
        if Path(args.checkpoint).name == "last.pth":
            expected_sha = completion["primary_checkpoint_sha256"]
            if checkpoint["epoch"] + 1 != config["train"]["epochs"]:
                raise ValueError("Incomplete fixed-final checkpoint")
        else:
            expected_sha = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))["best_checkpoint_sha256"]
        if sha256_file(args.checkpoint) != expected_sha:
            raise ValueError("Dose checkpoint completion/selection SHA mismatch")
    state = checkpoint.get("ema") if args.use_ema and checkpoint.get("ema") else checkpoint["model"]
    model.load_state_dict(state, strict=True)
    dataset = OCTManifestDataset(
        config["data"]["manifest"],
        split=args.split,
        transform=_make_transform(config, False),
        sample_repeat=False,
        root=config["data"].get("root"),
        datasets=config["data"].get(f"{args.split}_datasets"),
        groups=config["data"].get(f"{args.split}_groups"),
        **({"image_column": config["data"]["input_column"], "pretransformed_model_grid": True} if dose else {}),
    )
    if args.one_frame_per_group:
        indices = [dataset.groups[group_id][0] for group_id in sorted(dataset.groups)]
        dataset = Subset(dataset, indices)
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("evaluation", {}).get("batch_size", 1)),
        shuffle=False,
        num_workers=int(config.get("evaluation", {}).get("num_workers", 2)),
        pin_memory=True,
    )
    evaluation = config.get("evaluation", {})
    vessel_strata_definition = None
    strata_path = args.vessel_strata_definition or evaluation.get("vessel_strata_definition")
    if strata_path:
        import json
        resolved_strata = Path(strata_path).expanduser().resolve()
        if not resolved_strata.is_file():
            raise FileNotFoundError(f"Missing vessel strata definition: {resolved_strata}")
        vessel_strata_definition = json.loads(resolved_strata.read_text(encoding="utf-8-sig"))
    default_threshold = float(evaluation.get("threshold", 0.5))
    summary = evaluate_model(
        model,
        loader,
        device,
        output_dir=Path(args.output),
        threshold=default_threshold,
        layer_threshold=(
            args.layer_threshold
            if args.layer_threshold is not None
            else float(evaluation.get("layer_threshold", default_threshold))
        ),
        vessel_threshold=(
            args.vessel_threshold
            if args.vessel_threshold is not None
            else float(evaluation.get("vessel_threshold", default_threshold))
        ),
        axial_spacing=float(evaluation.get("axial_spacing", 1.0)),
        lateral_spacing=float(evaluation.get("lateral_spacing", 1.0)),
        save_predictions=args.save_predictions,
        stage=str(config.get("train", {}).get("stage", "joint")),
        input_normalization=str(config["data"].get("normalization", "fixed")),
        component_size_thresholds=(
            tuple(args.component_size_thresholds)
            if args.component_size_thresholds is not None
            else (
                tuple(evaluation["component_size_thresholds"])
                if isinstance(
                    evaluation.get("component_size_thresholds"), (list, tuple)
                )
                else None
            )
        ),
        boundary_band_width=(
            args.boundary_band_width
            if args.boundary_band_width is not None
            else float(evaluation.get("boundary_band_width", 3.0))
        ),
        tasks=tuple(args.tasks) if args.tasks is not None else None,
        postprocess_modes=tuple(args.postprocess_modes),
        restore_original_geometry=not args.no_restore_original_geometry,
        layer_surface_tolerance=(
            args.layer_surface_tolerance if args.layer_surface_tolerance is not None
            else float(evaluation.get("layer_surface_tolerance", 3.0))
        ),
        p1_minimum_main_fraction=args.p1_minimum_main_fraction,
        p2_smoothness=args.p2_smoothness,
        p2_max_displacement=args.p2_max_displacement,
        model_grid_contract=dose,
        dose_metadata=({**config["dose_response"], "seed": config["seed"],
                        "checkpoint_sha256": sha256_file(args.checkpoint),
                        "checkpoint_epoch": checkpoint["epoch"] + 1,
                        "evaluation_checkpoint_kind": Path(args.checkpoint).name} if dose else None),
        vessel_strata_definition=vessel_strata_definition,
        evaluate_clean_identity=args.evaluate_clean_identity,
        d2_diagnostics=args.d2_diagnostics,
    )
    print(summary)


if __name__ == "__main__":
    main()
