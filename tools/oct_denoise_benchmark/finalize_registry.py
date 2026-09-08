from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import yaml

from .io import sha256_file
from .registry import lock_run, save_yaml


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path(".")); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--main-seed", type=int, default=42); args = parser.parse_args(argv)
    root, run = args.project_root.resolve(), args.run_dir.resolve()
    classical = yaml.safe_load((run / "configs" / "locked_classical_configs.yaml").read_text(encoding="utf-8"))
    if classical.get("status") != "locked_on_pku37_validation": raise RuntimeError("classical configs are not locked")
    methods = {key: {"config": value, "seed": 0} for key, value in classical["methods"].items()}
    methods["noisy_identity"] = {"config": {"method_id": "noisy_identity"}, "seed": 0}
    deep_locked = {"status": "locked_on_pku37_validation", "main_seed": args.main_seed, "methods": {}}
    deep_selection = []
    for method in ("dncnn_paired", "nafnet_paired"):
        config = yaml.safe_load((root / "configs" / f"{method}.yaml").read_text(encoding="utf-8"))
        inventory = []
        for seed in config["seeds"]:
            checkpoint = run / "checkpoints" / method / f"seed_{seed}" / "best_psnr.pth"
            if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
            inventory_csv = checkpoint.parent / "checkpoint_inventory.csv"
            details = {}
            if inventory_csv.is_file():
                frame = pd.read_csv(inventory_csv)
                match = frame[frame.checkpoint.astype(str).map(lambda value: Path(value).name) == checkpoint.name]
                if not match.empty:
                    details = {key: match.iloc[-1][key] for key in ("selected_best_update", "selected_val_position_macro_psnr", "selected_val_position_macro_ssim") if key in match.columns}
            inventory.append({"seed": seed, "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint), **details})
        chosen = next(item for item in inventory if item["seed"] == args.main_seed)
        deep_locked["methods"][method] = {"config": config, "checkpoints": inventory, "main_checkpoint": chosen}
        methods[method] = {"config": config, "seed": args.main_seed, "checkpoint": chosen["checkpoint"], "checkpoint_sha256": chosen["sha256"], "evaluation_checkpoints": inventory}
        deep_selection.append({"method_id": method, "status": "selected_on_complete_PKU37_validation", "selection_rule": "position-macro PSNR; SSIM tie-break within tolerance", "main_seed": args.main_seed,
                               "checkpoint": chosen["checkpoint"], "checkpoint_sha256": chosen["sha256"],
                               "best_optimizer_update": chosen.get("selected_best_update"), "val_position_macro_psnr": chosen.get("selected_val_position_macro_psnr"), "val_position_macro_ssim": chosen.get("selected_val_position_macro_ssim")})
    save_yaml(run / "configs" / "locked_deep_configs.yaml", deep_locked)
    save_yaml(run / "configs" / "inference_registry.yaml", {"status": "locked", "methods": methods})
    selected_path = run / "metrics" / "selected_parameters.csv"
    existing = pd.read_csv(selected_path) if selected_path.is_file() else pd.DataFrame()
    combined = pd.concat([existing[~existing.method_id.isin({"dncnn_paired", "nafnet_paired"})] if not existing.empty else existing, pd.DataFrame(deep_selection)], ignore_index=True)
    combined.to_csv(selected_path, index=False)
    lock_run(root, run, test_started=False)
    print(run / "audit" / "config_lock.json")


if __name__ == "__main__": main()
