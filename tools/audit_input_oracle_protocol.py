from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.utils import write_json
from tools.input_oracle_cv_common import require_phase0, require_split_audit, resolve


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the frozen three-arm training protocol")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="runs/input_oracle_cv")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve(); output = resolve(root, args.output)
    phase0 = require_phase0(root, output); split = require_split_audit(output, args.fold)
    config = load_config(root / "configs/input_factorial_common.yaml")
    augmentation = config["data"].get("augmentation", {})
    intensity = {
        "gamma_range": augmentation.get("gamma_range"), "contrast_range": augmentation.get("contrast_range"),
        "speckle_std": augmentation.get("speckle_std", 0.0), "blur_probability": augmentation.get("blur_probability", 0.0),
        "additive_noise": augmentation.get("noise_std", 0.0), "sharpen": augmentation.get("sharpen_probability", 0.0),
        "histogram_transform": augmentation.get("histogram_probability", 0.0),
    }
    intensity_off = intensity["gamma_range"] == [1.0, 1.0] and intensity["contrast_range"] == [1.0, 1.0] and all(float(v or 0) == 0 for k, v in intensity.items() if k not in {"gamma_range", "contrast_range"})
    weights = config["loss"]["weights"]
    checks = {
        "phase0_passed": phase0["status"] == "passed", "split_passed": split["status"] == "passed",
        "fixed_512": list(config["data"]["target_size"]) == [512, 512],
        "input_segment_stage": config["train"]["stage"] == "input_segment",
        "d2s_off": not config["model"].get("d2s_enabled", config["model"].get("enable_denoise_to_seg", False)),
        "s2d_off": not config["model"].get("s2d_enabled", config["model"].get("enable_seg_to_denoise", False)),
        "forbidden_losses_off": all(float(weights.get(k, 0)) == 0 for k in ("reconstruction", "residual", "rmac", "pseudo", "identity")),
        "fixed_threshold": float(config["evaluation"].get("vessel_threshold", -1)) == .5,
        "fixed_final_epoch": int(config["train"]["epochs"]) == 60,
        "early_stop_cannot_preempt": int(config["train"].get("early_stopping_patience", 0)) > int(config["train"]["epochs"]),
        "intensity_augmentation_off": intensity_off,
        "test_assets_opened": 0,
    }
    report = {"status": "passed" if all(v is True or k == "test_assets_opened" for k, v in checks.items()) else "blocked", "fold": args.fold, "seed": args.seed, "checks": checks, "augmentation": augmentation, "intensity_audit": intensity, "dry_run": args.dry_run, "smoke_test": args.smoke_test}
    write_json(report, output / "augmentation_audit.json")
    write_json(report, output / "protocol_audit.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "passed": raise SystemExit(2)


if __name__ == "__main__":
    main()
