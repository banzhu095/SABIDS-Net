"""Generate opt-in dose caches/configs; only --cpu-smoke runs tiny training."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import yaml

from sabids.config import load_config, save_config
from sabids.data.io import read_gray
from sabids.experiments.dose_response import (
    ALPHAS, DOSE_CURVES, SMOKE_NOTICE, VERSION, alpha_code, asset_inventory, cache_key,
    code_fingerprint, dose_deterministic_algorithms, dose_input, formal_preflight, git_commit, read_binary_mask, resolve, save_cache,
    sha256_file, stable_sha, to_model_grid, write_strict_json,
)


def synthetic_assets(root: Path, tag: str) -> dict:
    """Four samples: 2 train/2 val, separate positions, no test rows/assets."""
    from sabids.engine.trainer import build_model
    folder = root / "cache/adaptive_denoising" / f"synthetic_smoke_{tag}" / "dose_v1/source"
    if folder.exists():
        raise FileExistsError(f"Smoke source already exists: {folder}; choose new --tag")
    folder.mkdir(parents=True)
    rows = []
    yy, xx = np.mgrid[:24, :40]
    layer = ((yy >= 8) & (yy < 19)).astype(np.float32)
    vessel = ((yy >= 11) & (yy < 16) & (xx >= 12) & (xx < 21)).astype(np.float32)
    clean = np.clip(.3 + .3 * layer - .15 * vessel + xx / 400, 0, 1).astype(np.float32)
    rng = np.random.default_rng(123)
    for i in range(4):
        paths = {}
        for name, arr in {"image": np.clip(clean + rng.normal(0, .035, clean.shape), 0, 1).astype(np.float32),
                          "clean": clean, "layer_mask": layer, "vessel_mask": vessel,
                          "label_valid_mask": np.ones_like(clean), "vessel_valid_mask": np.ones_like(clean)}.items():
            path = folder / f"s{i}_{name}.npy"
            np.save(path, arr, allow_pickle=False)
            paths[f"{name}_path"] = str(path)
        group = "synthetic_train" if i < 2 else "synthetic_val"
        rows.append({"sample_id": f"synthetic_{i}", "group_id": group, "patient_id": group,
                     "dataset": "SYNTHETIC", "split": "train" if i < 2 else "val", **paths})
    manifest = folder / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    cfg = load_config(root / "configs/adaptive_denoising/dose_d1.yaml")
    cfg["model"].update(channels=[4, 8], encoder_depths=[1, 1], decoder_depth=1, interaction_levels=[1])
    cfg["data"]["target_size"] = [32, 48]
    cfg["dose_response"].update(mode="smoke", scientific_evaluation=False, notice=SMOKE_NOTICE)
    cfg["seed"] = 123
    torch.manual_seed(123)
    model = build_model(cfg).eval()
    cp = folder / "synthetic_d1.pth"
    torch.save({"model": model.state_dict(), "config": cfg, "epoch": 0}, cp)
    save_config(cfg, folder / "resolved_config.yaml")
    return {"mode": "smoke", "status": "passed", "notice": SMOKE_NOTICE,
            "checkpoint_path": str(cp), "checkpoint_sha256": sha256_file(cp),
            "d1_resolved_config": str(folder / "resolved_config.yaml"),
            "resolved_config_sha256": sha256_file(folder / "resolved_config.yaml"),
            "restoration_mode": "synthetic_untrained", "input_resolution": [32, 48],
            "segmentation_manifest": str(manifest), "segmentation_manifest_sha256": sha256_file(manifest),
            "protocol": {"protocol_id": f"synthetic_smoke_{tag}"}, "test_assets_opened": 0}


def _new_json(path: Path, value: dict) -> None:
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != value:
            raise FileExistsError(f"Refusing mismatched artifact: {path}")
    else:
        write_strict_json(path, value)


@dose_deterministic_algorithms()
def prepare(root: Path, preflight: dict, *, curves: list[str], alphas: list[float],
            seeds: list[int], budget: str, tag: str, device_name: str,
            split_contract: str | None = None, training_asset_inventory: str | None = None,
            checkpoint_binding: str | None = None) -> dict:
    from sabids.engine.trainer import build_model
    from sabids.utils import get_device
    if preflight["status"] != "passed":
        raise RuntimeError(f"Formal preflight blocked: {preflight['issues']}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("tag must be a portable, non-path token")
    for a in alphas:
        alpha_code(a)
    if not curves or len(set(curves)) != len(curves) or not set(curves) <= set(DOSE_CURVES):
        raise ValueError(f"Nonempty unique curves from {DOSE_CURVES} required")
    model_curves = set(curves) - {"oracle"}
    if len(model_curves) > 1:
        raise ValueError("One preparation binds one denoiser checkpoint; compare model curves in separate registries")
    if len(set(alphas)) != len(alphas) or len(set(seeds)) != len(seeds) or not alphas or not seeds:
        raise ValueError("Nonempty, unique alpha/seed lists required")
    mode = preflight["mode"]
    if mode == "smoke" and budget != "smoke":
        raise ValueError("Synthetic artifacts cannot generate formal/pilot budgets")
    if mode == "formal" and budget == "smoke":
        raise ValueError("Use the synthetic path for smoke")
    protocol = preflight["protocol"]["protocol_id"]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", protocol):
        raise ValueError("Invalid protocol directory identity")
    cache_root = root / "cache/adaptive_denoising" / protocol / "dose_v1"
    run_root = root / "runs/adaptive_denoising" / protocol / "dose_v1"
    name = f"{budget}_s{'-'.join(map(str, seeds))}_{tag}"
    prep_dir = cache_root / "preparations" / name
    registry_path = prep_dir / "preparation_registry.json"
    source = pd.read_csv(preflight["segmentation_manifest"], dtype=str).fillna("")
    if (sha256_file(preflight["checkpoint_path"]) != preflight["checkpoint_sha256"]
            or sha256_file(preflight["d1_resolved_config"]) != preflight["resolved_config_sha256"]
            or sha256_file(preflight["segmentation_manifest"]) != preflight["segmentation_manifest_sha256"]):
        raise ValueError("Source fingerprint changed after preflight; do not generate cache")
    if not source.split.isin(["train", "val"]).all():
        raise ValueError("Preparation accepts train/val only, not test")
    source = source.sort_values(["split", "group_id", "sample_id"], kind="stable").reset_index(drop=True)
    source_inventory = asset_inventory(root, source, include_labels=True)
    if mode == "formal" and source_inventory != preflight["segmentation_asset_inventory"]:
        raise ValueError("Train/val pixels changed after preflight")
    code = code_fingerprint(root)
    identity = {"mode": mode, "protocol_id": protocol, "code_version": code,
                "checkpoint_sha256": preflight["checkpoint_sha256"],
                "resolved_config_sha256": preflight["resolved_config_sha256"],
                "source_manifest_sha256": preflight["segmentation_manifest_sha256"],
                "source_assets_sha256": stable_sha(source_inventory),
                "curves": curves, "alphas": alphas, "seeds": seeds, "budget": budget, "tag": tag,
                "device": device_name, "geometry_version": VERSION}
    if registry_path.exists():
        existing = json.loads(registry_path.read_text(encoding="utf-8"))
        if existing["identity"] != identity:
            raise FileExistsError("Preparation identity mismatch; choose new tag, never overwrite")
    device = get_device(device_name)
    raw = torch.load(preflight["checkpoint_path"], map_location="cpu", weights_only=False)
    d1_cfg = raw["config"]
    model = build_model(d1_cfg).to(device).eval()
    model.load_state_dict(raw["model"], strict=True)
    target = tuple(preflight["input_resolution"])
    target_stride = 2 ** (len(d1_cfg["model"]["channels"]) - 1)
    if any(n % target_stride for n in target):
        raise ValueError("D1 model grid must be divisible by encoder stride")
    by_sample = {}
    arm_rows = {(c, a): [] for c in curves for a in alphas}
    generated_at = datetime.now(timezone.utc).isoformat()
    commit = git_commit(root)
    for row in source.to_dict("records"):
        arrays = {"noisy": read_gray(resolve(root, row["image_path"])),
                  "clean": read_gray(resolve(root, row["clean_path"]))}
        for role in ("layer", "vessel", "label_valid", "vessel_valid"):
            column = f"{role}_mask_path"
            if role in {"layer", "vessel"} and not row.get(column):
                raise ValueError(f"Missing required GT: {row['sample_id']} {column}")
            arrays[role] = read_binary_mask(resolve(root, row[column])) if row.get(column) else np.ones_like(arrays["noisy"])
        grid, geometry = to_model_grid(arrays, target)
        # Unknown layer pixels must not contribute vessel/containment supervision.
        grid["vessel_valid"] *= grid["label_valid"]
        noisy_sha = sha256_file(resolve(root, row["image_path"]))
        clean_sha = sha256_file(resolve(root, row["clean_path"]))
        source_label_sha = stable_sha([r for r in source_inventory if r["sample_id"] == row["sample_id"]])
        base_meta = {"protocol_id": protocol, "sample_id": row["sample_id"], "group_id": row["group_id"],
                     "case_id": row.get("patient_id") or row["group_id"], "split": row["split"],
                     "source_noisy_path": str(resolve(root, row["image_path"])), "source_noisy_sha256": noisy_sha,
                     "source_clean_path": str(resolve(root, row["clean_path"])), "source_clean_sha256": clean_sha,
                     "checkpoint_path": preflight["checkpoint_path"], "checkpoint_sha256": preflight["checkpoint_sha256"],
                     "resolved_config_sha256": preflight["resolved_config_sha256"],
                     "restoration_mode": preflight["restoration_mode"], "geometry_sha256": geometry["geometry_sha256"],
                     "geometry": geometry, "code_version": code, "code_commit": commit,
                     "source_label_assets_sha256": source_label_sha, "generation_time_utc": generated_at,
                     "input_shape": list(target), "output_shape": list(target), "dtype": "float32",
                     "normalization": "fixed", "range": [0., 1.], "mode": mode,
                     "d1_output_definition": "forward_denoise_only clipped output, crop valid content and repad with zero; exact alpha=1 on valid pixels",
                     "scientific_evaluation": mode == "formal", "notice": SMOKE_NOTICE if mode == "smoke" else ""}
        sample_folder = cache_root / "samples" / stable_sha({k: base_meta[k] for k in
            ("sample_id", "source_noisy_sha256", "source_clean_sha256", "source_label_assets_sha256",
             "checkpoint_sha256", "resolved_config_sha256", "geometry_sha256", "code_version")})
        common = {}
        for role in ("clean", "layer", "vessel", "label_valid", "vessel_valid", "spatial_valid"):
            path = sample_folder / f"common_{role}.npy"
            save_cache(path, grid[role].astype(np.float32), {**base_meta, "curve_type": "oracle", "alpha": 0., "cache_content_role": role})
            common[role] = str(path)
        with torch.inference_mode():
            denoiser_output = model.forward_denoise_only(torch.from_numpy(grid["noisy"][None, None]).to(device))["denoised"][0, 0].cpu().numpy().astype(np.float32)
        if denoiser_output.shape != target or not np.isfinite(denoiser_output).all():
            raise ValueError("Denoiser prediction invalid")
        denoiser_output = denoiser_output * grid["spatial_valid"]
        by_sample[row["sample_id"]] = {"geometry": geometry, "common_assets": common,
                                       "source_assets_sha256": source_label_sha}
        for c in curves:
            for a in alphas:
                value, stats = dose_input(grid["noisy"], grid["clean"] if c == "oracle" else denoiser_output,
                                          a, c, valid=grid["spatial_valid"])
                meta = {**base_meta, **stats, "curve_type": c, "alpha": a,
                        "output_min": float(value.min()), "output_max": float(value.max())}
                key = cache_key(meta)
                path = sample_folder / f"{c}_{alpha_code(a)}_{key[:16]}.npy"
                save_cache(path, value, meta)
                arm_rows[c, a].append({**row, "dose_path": str(path), "clean_path": common["clean"],
                    "layer_mask_path": common["layer"], "vessel_mask_path": common["vessel"],
                    "label_valid_mask_path": common["label_valid"], "vessel_valid_mask_path": common["vessel_valid"],
                    "spatial_valid_mask_path": common["spatial_valid"], "input_role": f"{c}_{alpha_code(a)}"})
    configs = []
    prep_dir.mkdir(parents=True, exist_ok=True)
    for (curve, alpha), rows in arm_rows.items():
        manifest = prep_dir / f"manifest_{curve}_{alpha_code(alpha)}.csv"
        payload = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")
        if manifest.exists() and manifest.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"Manifest mismatch: {manifest}")
        if not manifest.exists():
            with manifest.open("x", encoding="utf-8", newline="") as f:
                f.write(payload)
        for seed in seeds:
            cfg = load_config(root / f"configs/adaptive_denoising/dose_{'oracle' if curve == 'oracle' else 'd1'}.yaml")
            cfg.pop("runtime", None)
            if mode == "smoke":
                cfg["model"] = copy.deepcopy(d1_cfg["model"])
                cfg["train"].update(gradient_accumulation_steps=1)
            epochs = {"smoke": 1, "overfit": 2, "pilot": 20, "full": 60}[budget]
            cfg.update(seed=seed, device=device_name)
            cfg["data"].update(manifest=str(manifest), root=str(root), target_size=list(target), train_datasets=None, val_datasets=None)
            cfg["train"].update(epochs=epochs, early_stopping_patience=epochs + 1,
                                output_dir=str(run_root / name / f"{curve}_{alpha_code(alpha)}_seed{seed}"))
            if budget in {"overfit", "smoke"}:
                cfg["data"].update(max_train_samples=2, max_val_samples=2)
            cfg["dose_response"].update(project_root=str(root), protocol_id=protocol, mode=mode,
                curve_type=curve, alpha=alpha,
                budget=budget, preparation_registry=str(registry_path), manifest_sha256=sha256_file(manifest),
                scientific_evaluation=mode == "formal" and budget not in {"overfit", "smoke"},
                notice=SMOKE_NOTICE if budget in {"smoke", "overfit"} else "")
            config_path = prep_dir / f"config_{curve}_{alpha_code(alpha)}_seed{seed}.yaml"
            cfg["dose_response"]["generated_config_path"] = str(config_path)
            text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
            if config_path.exists() and config_path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"Config mismatch: {config_path}")
            if not config_path.exists():
                with config_path.open("x", encoding="utf-8", newline="") as f:
                    f.write(text)
            configs.append(str(config_path))
    result = {"identity": identity, "preflight": preflight, "code_version": code,
              "split_contract": str(resolve(root, split_contract)) if split_contract else None,
              "training_asset_inventory": str(resolve(root, training_asset_inventory)) if training_asset_inventory else None,
              "checkpoint_binding": str(resolve(root, checkpoint_binding)) if checkpoint_binding else None,
              "samples": by_sample, "configs": configs,
              "config_sha256": {p: sha256_file(p) for p in configs}, "test_assets_opened": 0,
              "notice": SMOKE_NOTICE if mode == "smoke" else "",
              "evaluation": "P0 0.5; model grid only; original boundary metrics NOT IMPLEMENTED"}
    _new_json(registry_path, result)
    return result


def cpu_smoke(configs: list[str], output: Path) -> dict:
    from sabids.engine.trainer import Trainer
    audits = []
    for p in configs:
        cfg = load_config(p)
        if cfg["dose_response"]["mode"] != "smoke" or cfg["device"] != "cpu":
            raise ValueError("--cpu-smoke requires synthetic CPU configs only")
        trainer = Trainer(cfg)
        save_config(cfg, trainer.output_dir / "resolved_config.yaml")
        trainer.fit()
        audits.append(json.loads((trainer.output_dir / "initialization_audit.json").read_text(encoding="utf-8")))
        for name in ("last.pth", "best.pth", "dose_training_metadata.json"):
            if not (trainer.output_dir / name).is_file():
                raise AssertionError(f"Smoke missing {name}")
    fields = ("model_state_sha256", "sampler_plan_sha256", "actual_augmentation_plan_sha256",
              "trainable_parameter_names_sha256", "paired_cohort_sha256")
    paired = {k: len({a[k] for a in audits}) == 1 for k in fields}
    if not all(paired.values()):
        raise AssertionError(f"CPU pairing failure: {paired}")
    result = {"status": "passed", "notice": SMOKE_NOTICE, "config_count": len(configs),
              "paired_checks": paired, "test_assets_opened": 0, "scientific_evaluation": False}
    write_strict_json(output, result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=".")
    p.add_argument("--mode", choices=("formal", "smoke"), required=True)
    p.add_argument("--denoiser-checkpoint")
    p.add_argument("--protocol-lock")
    p.add_argument("--split-contract")
    p.add_argument("--selection-rule", choices=("best_validation_psnr", "fixed_final"))
    p.add_argument("--training-asset-inventory")
    p.add_argument("--checkpoint-binding")
    p.add_argument("--d2-checkpoint-kind", choices=("d2_pixel", "d2_task", "d2_last"))
    p.add_argument("--d2-checkpoint-binding")
    p.add_argument("--curves", nargs="+", choices=DOSE_CURVES, default=["oracle", "d1"])
    p.add_argument("--alphas", nargs="+", type=float, default=list(ALPHAS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--budget", choices=("smoke", "overfit", "pilot", "full"), required=True)
    p.add_argument("--tag", default="v1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--synthetic-smoke", action="store_true")
    p.add_argument("--cpu-smoke", action="store_true")
    a = p.parse_args()
    root = Path(a.project_root).resolve()
    if a.cpu_smoke and (a.mode != "smoke" or a.device != "cpu"):
        p.error("CPU smoke must explicitly use --mode smoke --device cpu")
    if a.mode == "smoke":
        if not a.synthetic_smoke or a.budget != "smoke":
            p.error("Minimal smoke supports only --synthetic-smoke --budget smoke")
        torch.set_num_threads(1)
        pf = synthetic_assets(root, a.tag)
    else:
        if a.synthetic_smoke:
            p.error("Synthetic assets cannot be formal")
        non_oracle = set(a.curves) - {"oracle"}
        if a.d2_checkpoint_kind:
            if non_oracle != {a.d2_checkpoint_kind}:
                p.error("D2 checkpoint kind must exactly match the single non-oracle curve")
            if not a.denoiser_checkpoint or not a.d2_checkpoint_binding:
                p.error("D2 formal preparation requires checkpoint and D2 checkpoint binding")
            from sabids.experiments.d2 import formal_d2_preflight
            pf = formal_d2_preflight(
                root, resolve(root, a.denoiser_checkpoint),
                resolve(root, a.d2_checkpoint_binding), a.d2_checkpoint_kind,
                resolve(root, a.protocol_lock), resolve(root, a.split_contract),
            )
        else:
            pf = formal_preflight(root, a.denoiser_checkpoint, a.protocol_lock,
                                  a.split_contract, a.selection_rule, a.training_asset_inventory,
                                  a.checkpoint_binding)
        if pf["status"] != "passed":
            print(json.dumps(pf, ensure_ascii=False, indent=2, allow_nan=False))
            raise SystemExit(2)
    effective_binding = a.d2_checkpoint_binding if a.d2_checkpoint_kind else a.checkpoint_binding
    result = prepare(root, pf, curves=a.curves, alphas=a.alphas, seeds=a.seeds,
                     budget=a.budget, tag=a.tag, device_name=a.device,
                     split_contract=a.split_contract, training_asset_inventory=a.training_asset_inventory,
                     checkpoint_binding=effective_binding)
    if a.cpu_smoke:
        result["cpu_smoke"] = cpu_smoke(result["configs"],
            root / "runs/adaptive_denoising" / pf["protocol"]["protocol_id"] / "dose_v1/cpu_smoke_report.json")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
