#!/usr/bin/env python
"""Audit one completed dual-view run without opening dataset assets."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.engine.trainer import build_model
from sabids.experiments.dose_response import write_strict_json
from sabids.experiments.protocol_lock import sha256_file


def _cost(model: torch.nn.Module, size: tuple[int, int], dual: bool, device: torch.device) -> dict:
    macs = 0
    handles = []

    def hook(module, inputs, output):
        nonlocal macs
        if not isinstance(module, torch.nn.Conv2d):
            return
        batch, out_channels, height, width = output.shape
        kernel = module.kernel_size[0] * module.kernel_size[1]
        macs += int(batch * out_channels * height * width * kernel * (module.in_channels // module.groups))

    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    image = torch.zeros(1, 1, *size, device=device)
    kwargs = {"auxiliary_image": image.clone()} if dual else {"disable_auxiliary": True}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(2):
            model(image, return_features=False, return_auxiliary=False, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        repeats = 5
        for _ in range(repeats):
            model(image, return_features=False, return_auxiliary=False, **kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) / repeats
    for handle in handles:
        handle.remove()
    return {
        "macs_per_frame": macs // 7,  # 2 warm-up + 5 timed forwards were hooked.
        "inference_seconds_per_frame": elapsed,
        "peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    config = load_config(run / "resolved_config.yaml")
    history = pd.read_csv(run / "history.csv")
    metadata = json.loads((run / "dual_view_training_metadata.json").read_text(encoding="utf-8"))
    checkpoint_path = run / "last.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    configured = int(config["train"]["epochs"])
    epochs = pd.to_numeric(history["epoch"], errors="raise").astype(int).tolist()
    arm = str(config["dual_view"]["arm"])
    fusion_levels = [int(level) for level in config["dual_view"].get("fusion_levels", [])]
    active_fusion = arm in {"B3", "B6", "C1"}
    fusion_checks = {}
    for level in fusion_levels:
        for kind in ("gamma", "adapter", "gate"):
            gradient = f"train_dual_level{level}_{kind}_gradient_norm"
            update = f"train_dual_level{level}_{kind}_update_abs_mean"
            if active_fusion:
                fusion_checks[f"level{level}_{kind}_gradient_started"] = bool(
                    gradient in history and (pd.to_numeric(history[gradient], errors="coerce") > 0).any()
                )
                fusion_checks[f"level{level}_{kind}_updated"] = bool(
                    update in history and (pd.to_numeric(history[update], errors="coerce") > 0).any()
                )
            else:
                fusion_checks[f"level{level}_{kind}_remained_disabled"] = bool(
                    update in history
                    and (pd.to_numeric(history[update], errors="coerce").fillna(0) == 0).all()
                )
    checks = {
        "dual_view_opt_in": config.get("dual_view", {}).get("enabled") is True,
        "validation_only": config.get("dual_view", {}).get("validation_only") is True,
        "fixed_p0": all(float(config["evaluation"].get(key, 0.5)) == 0.5 for key in ("threshold", "layer_threshold", "vessel_threshold")),
        "complete_epoch_sequence": epochs == list(range(1, configured + 1)),
        "fixed_final_checkpoint_epoch": int(checkpoint["epoch"]) + 1 == configured,
        "checkpoint_bound": metadata.get("primary_checkpoint_sha256") == sha256_file(checkpoint_path),
        "optimizer_budget_complete": metadata.get("completed_optimizer_steps") == metadata.get("expected_optimizer_steps"),
        "frozen_parameters_unchanged": not metadata.get("changed_frozen_parameter_names"),
        "frozen_denoiser_unchanged": not metadata.get("frozen_denoiser_changed_parameter_names"),
        "d1_not_instantiated_in_training_graph": not any(
            name.startswith("d1_") for name in checkpoint["model"]
        ),
        "test_assets_opened_zero": metadata.get("test_assets_opened") == 0,
        **fusion_checks,
    }
    device = torch.device(args.device)
    model = build_model(config).to(device).eval()
    model.set_train_stage("input_segment")
    model.load_state_dict(checkpoint["model"], strict=True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    shared = sum(parameter.numel() for name, parameter in model.named_parameters() if name.startswith(("stem.", "encoder_blocks.", "downsamples.")))
    fusion = sum(parameter.numel() for name, parameter in model.named_parameters() if name.startswith("dual_fusions."))
    use_auxiliary = bool(config["dual_view"].get("use_auxiliary", True))
    result = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "run_dir": str(run),
        "arm": arm,
        "seed": int(config["seed"]),
        "parameter_count_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameter_count_trainable": trainable,
        "shared_encoder_parameter_count_counted_once": shared,
        "fusion_parameter_count": fusion,
        "cost_profile": _cost(model, tuple(config["data"]["target_size"]), use_auxiliary, device),
        "test_assets_opened": 0,
    }
    output = Path(args.output).resolve()
    write_strict_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
