#!/usr/bin/env python
"""Build immutable 16-position grouped CV manifests/configs without opening test assets."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.engine.trainer import build_model
from sabids.experiments.dose_response import stable_sha, tensor_sha, write_strict_json_exclusive
from sabids.experiments.protocol_lock import sha256_file
from sabids.experiments.seg_guided import CV_ARMS, grouped_four_fold_assignment
from sabids.utils import seed_everything


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _write_once(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8-sig") != payload:
            raise FileExistsError(f"Existing immutable artifact differs: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def _arm_table(source: pd.DataFrame, arm: str) -> pd.DataFrame:
    table = source.copy()
    table["primary_path"] = table["noisy_path"]
    table["auxiliary_path"] = ""
    if arm == "B1":
        table["primary_path"] = table["mild_path"]
    elif arm in {"B3", "B3R"}:
        table["auxiliary_path"] = table["mild_path"]
    elif arm in {"B6", "B6R"}:
        table["auxiliary_path"] = table["noisy_path"]
    table["dual_view_arm"] = arm
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-manifest", required=True,
                        help="Evidence-bound dual-view development manifest (train+val only).")
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--assignment-seed", type=int, default=20260929)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    resolve = lambda value: Path(value).resolve() if Path(value).is_absolute() else (root / value).resolve()
    source_path, lock_path, split_path = map(resolve, (
        args.source_manifest, args.protocol_lock, args.split_contract,
    ))
    output = resolve(args.output)
    if (output / "fold_assignment.json").exists():
        raise FileExistsError(f"Refusing existing CV protocol: {output}")
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    split = yaml.safe_load(split_path.read_text(encoding="utf-8"))
    sealed_declared = list(lock.get("sealed_test_positions", lock.get("test_positions", [])))
    sealed_declared += list(split.get("test_positions", []))
    sealed_canonical = {_canonical(value) for value in sealed_declared}
    source = pd.read_csv(source_path, dtype=str).fillna("")
    required = {
        "sample_id", "group_id", "split", "noisy_path", "mild_path", "strong_path",
        "vessel_mask_path", "layer_mask_path", "spatial_valid_mask_path",
    }
    if missing := required - set(source.columns):
        raise ValueError(f"Source manifest missing columns: {sorted(missing)}")
    if source["group_id"].map(_canonical).isin(sealed_canonical).any():
        raise RuntimeError("BLOCKED: sealed test group appears in development source")
    if not source["split"].isin(["train", "val"]).all():
        raise RuntimeError("BLOCKED: source contains non-development split")
    assignment = grouped_four_fold_assignment(
        source, sealed_groups=sealed_declared, seed=args.assignment_seed,
        expected_groups=16,
    )
    assignment.update({
        "run_id": args.run_id, "protocol_id": "seg_guided_adaptive_v1",
        "source_manifest": str(source_path), "source_manifest_sha256": sha256_file(source_path),
        "protocol_lock": str(lock_path), "protocol_lock_sha256": sha256_file(lock_path),
        "split_contract": str(split_path), "split_contract_sha256": sha256_file(split_path),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                      capture_output=True, text=True, check=True).stdout.strip(),
    })
    assignment["protocol_sha256"] = stable_sha({
        key: value for key, value in assignment.items() if key != "assignment_sha256"
    })
    output.mkdir(parents=True)
    write_strict_json_exclusive(output / "fold_assignment.json", assignment)
    assignment_rows = [
        {"group_id": group, "validation_fold": fold,
         "assignment_seed": args.assignment_seed}
        for group, fold in sorted(assignment["assignment"].items())
    ]
    _write_once(output / "fold_assignment.csv", pd.DataFrame(assignment_rows).to_csv(index=False, lineterminator="\n"))

    config_paths = []
    initialization_audit = {}
    templates = {
        "B0": root / "configs/adaptive_denoising/dual_view/b0_noisy.yaml",
        "B1": root / "configs/adaptive_denoising/dual_view/b1_mild.yaml",
        "B3": root / "configs/adaptive_denoising/dual_view/b3_noisy_mild.yaml",
        "B6": root / "configs/adaptive_denoising/dual_view/b6_noisy_noisy.yaml",
        "B3R": root / "configs/adaptive_denoising/seg_guided_v1/b3r_noisy_mild_residual.yaml",
        "B6R": root / "configs/adaptive_denoising/seg_guided_v1/b6r_noisy_noisy_zero_residual.yaml",
    }
    for fold in range(4):
        val_groups = set(assignment["folds"][str(fold)]["val_groups"])
        folded = source.copy()
        folded["split"] = folded["group_id"].map(lambda group: "val" if group in val_groups else "train")
        if set(folded.loc[folded.split.eq("val"), "group_id"]) != val_groups:
            raise AssertionError("Fold validation membership mismatch")
        manifests = {}
        for arm in CV_ARMS:
            manifest = output / "manifests" / f"fold{fold}_{arm.lower()}.csv"
            table = _arm_table(folded, arm).sort_values(["split", "group_id", "sample_id"])
            _write_once(manifest, table.to_csv(index=False, lineterminator="\n"))
            manifests[arm] = manifest
        for seed in args.seeds:
            init_by_family = {}
            for family, template_arm in (("dual", "B3"), ("residual", "B3R")):
                config = load_config(templates[template_arm])
                config.update(seed=seed, device="cpu")
                seed_everything(seed, True, use_cuda=False)
                model = build_model(config)
                path = output / "initializations" / f"fold{fold}_seed{seed}_{family}.pth"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "epoch": -1, "config": config}, path)
                init_by_family[family] = path
                initialization_audit[f"fold{fold}_seed{seed}_{family}"] = {
                    "path": str(path), "sha256": sha256_file(path),
                    "common_state_sha256": stable_sha({
                        name: tensor_sha(value) for name, value in model.state_dict().items()
                        if not name.startswith("dual_fusions.")
                    }),
                }
            for arm in CV_ARMS:
                config = load_config(templates[arm])
                config.pop("runtime", None)
                family = "residual" if arm.endswith("R") else "dual"
                config.update(seed=seed, device="cuda", protocol_id="seg_guided_adaptive_v1")
                config["data"].update(
                    manifest=str(manifests[arm]), root=str(root), target_size=[512, 512],
                    train_split="train", val_split="val",
                )
                config["train"].update(
                    epochs=20, fixed_checkpoint_epochs=[12], early_stopping_patience=21,
                    pretrained=str(init_by_family[family]), strict_pretrained=True,
                    output_dir=str(root / "runs/adaptive_denoising/seg_guided_adaptive_v1"
                                   / args.run_id / f"fold{fold}" / f"{arm.lower()}_seed{seed}"),
                )
                config.setdefault("dual_view", {}).update(
                    arm=arm, protocol_id="seg_guided_adaptive_v1", record_git_commit=True,
                    fold=fold, fold_assignment_sha256=sha256_file(output / "fold_assignment.json"),
                )
                config["seg_guided"] = {
                    "enabled": True, "project_root": str(root), "run_id": args.run_id,
                    "fold": fold, "primary_epoch": 12, "test_assets_opened": 0,
                }
                path = output / "configs" / f"fold{fold}_{arm.lower()}_seed{seed}.yaml"
                _write_once(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
                config_paths.append(str(path))
    registry = {
        "status": "passed", "version": "seg-guided-adaptive-v1",
        "run_id": args.run_id, "fold_assignment": str(output / "fold_assignment.json"),
        "fold_assignment_sha256": sha256_file(output / "fold_assignment.json"),
        "config_paths": config_paths,
        "config_sha256": {path: sha256_file(Path(path)) for path in config_paths},
        "initializations": initialization_audit, "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output / "cv_registry.json", registry)
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
