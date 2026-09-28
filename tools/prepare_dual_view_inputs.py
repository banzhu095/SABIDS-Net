#!/usr/bin/env python
"""Prepare evidence-bound train/val manifests for noisy+mild dual-view v1."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.engine.trainer import build_model
from sabids.experiments.dual_view import (
    BLOCKED, PRIMARY_ARMS, audit_formal_input_evidence, deterministic_shuffle,
)
from sabids.experiments.dose_response import (
    stable_sha, tensor_sha, write_strict_json_exclusive,
)
from sabids.experiments.protocol_lock import sha256_file
from sabids.utils import seed_everything


def _new_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Refusing mismatched existing artifact: {path}")
        return
    with path.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)


def _array_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def _load_dose_manifests(registry: dict) -> dict[float, pd.DataFrame]:
    found: dict[float, pd.DataFrame] = {}
    for config_path in registry.get("configs", []):
        config = load_config(config_path)
        dose = config.get("dose_response", {})
        if dose.get("curve_type") != "d1":
            continue
        alpha = float(dose["alpha"])
        if alpha not in {0.0, 0.25, 1.0} or alpha in found:
            continue
        table = pd.read_csv(config["data"]["manifest"], dtype=str).fillna("")
        if not table["split"].isin(("train", "val")).all():
            raise ValueError("Dual-view preparation refuses non-train/val dose rows")
        found[alpha] = table
    if set(found) != {0.0, 0.25, 1.0}:
        raise ValueError(f"Dose registry lacks exact D1 alpha 0/.25/1 manifests: {sorted(found)}")
    return found


def _validate_cached_path(path: str, expected_alpha: float) -> dict:
    asset = Path(path).resolve()
    sidecar = asset.with_suffix(".json")
    if not asset.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Incomplete dose cache: {asset}")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    array = np.load(asset, allow_pickle=False)
    if (
        array.dtype != np.float32
        or array.ndim != 2
        or not np.isfinite(array).all()
        or float(metadata.get("alpha")) != expected_alpha
        or metadata.get("curve_type") != "d1"
        or metadata.get("array_sha256") != _array_sha(array)
        or metadata.get("split") not in {"train", "val"}
    ):
        raise ValueError(f"Invalid dose cache identity/content: {asset}")
    return metadata


def prepare(root: Path, evidence: dict, registry_path: Path, seeds: list[int], budget: str, tag: str) -> dict:
    if evidence.get("status") != "passed":
        raise RuntimeError(BLOCKED)
    registry = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    dose_tables = _load_dose_manifests(registry)
    indexed = {
        alpha: table.set_index("sample_id", drop=False)
        for alpha, table in dose_tables.items()
    }
    ids = list(indexed[0.0].index)
    if any(set(table.index) != set(ids) for table in indexed.values()):
        raise ValueError("D1 dose manifests do not describe the same cohort")
    base_rows = []
    output_root = root / "cache/adaptive_denoising" / evidence["protocol_id"] / "dual_view_v1" / tag
    generated_at = datetime.now(timezone.utc).isoformat()
    for sample_id in ids:
        rows = {alpha: indexed[alpha].loc[sample_id].to_dict() for alpha in indexed}
        identity = {(row["split"], row["group_id"]) for row in rows.values()}
        if len(identity) != 1:
            raise ValueError(f"Dose sample identity mismatch: {sample_id}")
        noisy_path, mild_path, strong_path = (rows[a]["dose_path"] for a in (0.0, 0.25, 1.0))
        metadata = {
            alpha: _validate_cached_path(rows[alpha]["dose_path"], alpha)
            for alpha in (0.0, 0.25, 1.0)
        }
        noisy = np.load(noisy_path, allow_pickle=False)
        mild = np.load(mild_path, allow_pickle=False)
        strong = np.load(strong_path, allow_pickle=False)
        expected_mild = np.clip(
            noisy - np.float32(0.25) * (noisy - strong), 0.0, 1.0
        ).astype(np.float32)
        if not np.array_equal(mild, expected_mild):
            raise AssertionError("D1 alpha=.25 is not the registered noisy-to-alpha1 interpolation")
        residual = np.ascontiguousarray(noisy - mild, dtype=np.float32)
        # alpha=0 is the dose generator's explicit noisy identity endpoint;
        # its cache identity and bytes were checked by _validate_cached_path.
        if not np.array_equal(noisy - mild, residual):
            raise AssertionError("Residual cache identity failed")
        # Keep development-train and validation derivatives in disjoint
        # namespaces even when sample ids are globally unique.
        sample_dir = output_root / "residuals" / str(rows[0.0]["split"]) / str(sample_id)
        residual_path = sample_dir / "noisy_minus_mild.npy"
        residual_sidecar = residual_path.with_suffix(".json")
        residual_meta = {
            "version": "noisy-mild-dual-view-v1",
            "protocol_id": evidence["protocol_id"],
            "split_contract_sha256": evidence["sha256"]["split_contract"],
            "denoiser_checkpoint_sha256": evidence["sha256"]["d1_checkpoint"],
            "checkpoint_binding_sha256": evidence["sha256"]["checkpoint_binding"],
            "sample_id": str(sample_id),
            "split": rows[0.0]["split"],
            "alpha": 0.25,
            "normalization": metadata[0.25].get("normalization"),
            "geometry": metadata[0.25].get("geometry"),
            "geometry_sha256": metadata[0.25].get("geometry_sha256"),
            "noisy_sha256": _array_sha(noisy),
            "denoised_sha256": _array_sha(mild),
            "residual_sha256": _array_sha(residual),
            "valid_mask_sha256": sha256_file(Path(rows[0.0]["spatial_valid_mask_path"])),
            "created_at": generated_at,
            "test_assets_opened": 0,
        }
        if residual_path.exists() or residual_sidecar.exists():
            if not residual_path.is_file() or not residual_sidecar.is_file():
                raise FileExistsError(f"Incomplete residual cache: {residual_path}")
            if (
                not np.array_equal(np.load(residual_path, allow_pickle=False), residual)
                or json.loads(residual_sidecar.read_text(encoding="utf-8")) != residual_meta
            ):
                raise FileExistsError(f"Residual identity conflict: {residual_path}")
        else:
            sample_dir.mkdir(parents=True, exist_ok=True)
            with residual_path.open("xb") as handle:
                np.save(handle, residual, allow_pickle=False)
            write_strict_json_exclusive(residual_sidecar, residual_meta)
        base_rows.append({
            **rows[0.0],
            "primary_path": noisy_path,
            "noisy_path": noisy_path,
            "mild_path": mild_path,
            "strong_path": strong_path,
            "residual_path": str(residual_path),
        })

    shuffle = deterministic_shuffle(base_rows, seeds[0])
    by_id = {str(row["sample_id"]): row for row in base_rows}
    manifests = {}
    for arm in PRIMARY_ARMS:
        arm_rows = []
        for row in base_rows:
            value = dict(row)
            if arm == "B1":
                value["primary_path"] = row["mild_path"]
                value["auxiliary_path"] = ""
            elif arm == "B3":
                value["auxiliary_path"] = row["mild_path"]
            elif arm == "B6":
                value["auxiliary_path"] = row["noisy_path"]
            elif arm == "C1":
                auxiliary_sample_id = shuffle[str(row["sample_id"])]
                value["auxiliary_path"] = by_id[auxiliary_sample_id]["mild_path"]
                value["auxiliary_sample_id"] = auxiliary_sample_id
            else:
                value["auxiliary_path"] = ""
            value["dual_view_arm"] = arm
            arm_rows.append(value)
        manifest = output_root / f"manifest_{arm.lower()}.csv"
        _new_text(manifest, pd.DataFrame(arm_rows).to_csv(index=False, lineterminator="\n"))
        manifests[arm] = str(manifest)

    config_paths = []
    init_by_seed = {}
    template_by_arm = {
        "B0": "b0_noisy.yaml", "B1": "b1_mild.yaml", "B3": "b3_noisy_mild.yaml",
        "B6": "b6_noisy_noisy.yaml", "C1": "c1_shuffled_mild.yaml",
    }
    epochs = {"overfit": 3, "pilot": 20, "full": 60}[budget]
    for seed in seeds:
        init_config = load_config(root / "configs/adaptive_denoising/dual_view/b3_noisy_mild.yaml")
        init_config.update(seed=seed, device="cpu", protocol_id=evidence["protocol_id"])
        seed_everything(seed, True, use_cuda=False)
        initialization = build_model(init_config)
        init_path = output_root / f"common_initialization_seed{seed}.pth"
        init_payload = {"model": initialization.state_dict(), "config": init_config, "epoch": -1}
        if init_path.exists():
            existing = torch.load(init_path, map_location="cpu", weights_only=False)
            existing_sha = {name: tensor_sha(value) for name, value in existing["model"].items()}
            expected_sha = {name: tensor_sha(value) for name, value in init_payload["model"].items()}
            if existing_sha != expected_sha or existing.get("config") != init_payload["config"]:
                raise FileExistsError(f"Initialization conflict: {init_path}")
        else:
            init_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(init_payload, init_path)
        init_by_seed[str(seed)] = {"path": str(init_path), "sha256": sha256_file(init_path)}
        for arm in PRIMARY_ARMS:
            config = load_config(root / "configs/adaptive_denoising/dual_view" / template_by_arm[arm])
            config.pop("runtime", None)
            config.update(seed=seed, device="cuda", protocol_id=evidence["protocol_id"])
            if budget == "overfit":
                # The overfit diagnostic is intentionally train-only.  The
                # Trainer still needs a validation loader for its ordinary
                # epoch plumbing, so use a disjoint, deterministic subset of
                # the train split rather than opening validation or test.
                config["data"].update(
                    max_train_samples=8,
                    max_val_samples=4,
                    val_split="train",
                )
            config["data"].update(manifest=manifests[arm], root=str(root), target_size=[512, 512])
            run = root / "runs/adaptive_denoising" / evidence["protocol_id"] / "dual_view_v1" / tag / f"{arm.lower()}_seed{seed}"
            config["train"].update(
                epochs=epochs,
                early_stopping_patience=epochs + 1,
                output_dir=str(run),
                pretrained=str(init_path),
            )
            config["dual_view"].update(
                protocol_id=evidence["protocol_id"], budget=budget, tag=tag,
                input_evidence_sha256=stable_sha(evidence),
                manifest_sha256=sha256_file(Path(manifests[arm])),
                common_initialization_sha256=sha256_file(init_path),
            )
            path = output_root / f"config_{arm.lower()}_seed{seed}.yaml"
            _new_text(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            config_paths.append(str(path))
    result = {
        "version": "noisy-mild-dual-view-v1",
        "status": "passed",
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=True,
        ).stdout.strip(),
        "evidence": evidence,
        "manifests": manifests,
        "manifest_sha256": {
            arm: sha256_file(Path(path)) for arm, path in manifests.items()
        },
        "shuffle_mapping": shuffle,
        "shuffle_mapping_sha256": stable_sha(shuffle),
        "initializations": init_by_seed,
        "configs": config_paths,
        "config_sha256": {path: sha256_file(Path(path)) for path in config_paths},
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(output_root / "preparation_registry.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=("audit", "prepare"), required=True)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--d1-checkpoint", required=True)
    parser.add_argument("--d1-checkpoint-binding", required=True)
    parser.add_argument("--d1-training-asset-inventory", required=True)
    parser.add_argument("--d1-dose-registry", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--budget", choices=("overfit", "pilot", "full"), default="pilot")
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    resolve = lambda value: (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    evidence = audit_formal_input_evidence(
        resolve(args.protocol_lock), resolve(args.split_contract), resolve(args.d1_checkpoint),
        resolve(args.d1_checkpoint_binding), resolve(args.d1_training_asset_inventory),
        resolve(args.d1_dose_registry),
    )
    if evidence["status"] != "passed":
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    if args.mode == "audit":
        registry = json.loads(resolve(args.d1_dose_registry).read_text(encoding="utf-8-sig"))
        tables = _load_dose_manifests(registry)
        path_checks = []
        for alpha, table in tables.items():
            for row in table.to_dict("records"):
                for column in ("dose_path", "spatial_valid_mask_path"):
                    path = Path(str(row[column])).expanduser().resolve()
                    path_checks.append(path.is_file())
        result = {
            **evidence,
            "d1_alpha_025_only_for_phase1": True,
            "dose_manifest_rows_by_alpha": {
                str(alpha): int(len(table)) for alpha, table in tables.items()
            },
            "all_train_val_cache_paths_present": bool(path_checks) and all(path_checks),
        }
        if not result["all_train_val_cache_paths_present"]:
            result.update(status="blocked", message=BLOCKED)
            result.setdefault("issues", []).append("missing train/val D1 dose cache path")
    else:
        result = prepare(
            root, evidence, resolve(args.d1_dose_registry), args.seeds, args.budget, args.tag
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
