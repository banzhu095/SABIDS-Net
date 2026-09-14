from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import torch

from .data import audit_protocol, load_protocol_manifest
from .io import sha256_file
from .registry import git_commit, load_yaml, save_yaml, stable_sha256
from .statistics import aggregate, bootstrap_confidence_intervals, paired_differences
from .table_store import atomic_write_csv


BASE_METHODS = {"noisy_identity", "bm3d_standard", "tv_chambolle", "nlm", "ksvd_self", "dncnn_paired", "nafnet_paired"}
NEW_METHODS = {"sabids_current", "tcfl_dncnn"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_status(root: Path) -> str:
    result = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    return result.stdout


def discover_base(runs: Path, preferred: Path | None = None) -> Path:
    candidates = ([preferred] if preferred else []) + sorted((path for path in runs.glob("denoise_benchmark_pku_protocol_*") if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None: continue
        candidate = candidate.resolve()
        if candidate in seen: continue
        seen.add(candidate)
        required = [candidate / "metrics" / "per_image_metrics.csv", candidate / "metrics" / "per_dataset_metrics.csv", candidate / "audit" / "config_lock.json"]
        if not all(path.is_file() for path in required): continue
        table = pd.read_csv(required[0])
        successful = table if "status" not in table else table[table.status == "success"]
        if BASE_METHODS.issubset(set(successful.method_id)) and {"PKU37", "Duke17", "Duke28"}.issubset(set(successful.dataset)):
            return candidate
    raise FileNotFoundError("no completed immutable BASE_RUN satisfying the seven-method protocol was found")


def _inventory_tree(root: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        rows.append({"relative_path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return pd.DataFrame(rows)


def verify_base(ext_run: Path) -> None:
    metadata = json.loads((ext_run / "audit" / "extension_init.json").read_text(encoding="utf-8"))
    base = Path(metadata["base_run"])
    expected = pd.read_csv(ext_run / "audit" / "base_run_inventory.csv")
    current = _inventory_tree(base)
    merged = expected.merge(current, on="relative_path", how="outer", suffixes=("_before", "_after"), indicator=True)
    changed = merged[(merged._merge != "both") | (merged.sha256_before != merged.sha256_after) | (merged.bytes_before != merged.bytes_after)]
    if not changed.empty:
        changed.to_csv(ext_run / "audit" / "base_run_mutation_detected.csv", index=False)
        raise RuntimeError("BASE_RUN changed after extension initialization")


def init(root: Path, ext_run: Path, base_run: Path | None) -> Path:
    root, ext_run = root.resolve(), ext_run.resolve()
    base = discover_base(root / "runs", base_run)
    for child in ("audit", "configs", "logs", "status", "reports", "metrics", "manifests", "smoke", "tracks/sabids_current", "tracks/tcfl_dncnn"):
        (ext_run / child).mkdir(parents=True, exist_ok=True)
    base_metrics = pd.read_csv(base / "metrics" / "per_image_metrics.csv")
    manifest = root / "Manifests" / "manifest_denoise.csv"
    protocol_audit = audit_protocol(load_protocol_manifest(root, manifest))
    if not protocol_audit["passed"]: raise RuntimeError(f"manifest protocol audit failed: {protocol_audit}")
    inventory = _inventory_tree(base); inventory.to_csv(ext_run / "audit" / "base_run_inventory.csv", index=False)
    shutil.copy2(base / "audit" / "config_lock.json", ext_run / "audit" / "config_lock.json")
    environment = {"created_at_utc": _now(), "project_root": str(root), "base_run": str(base), "ext_run": str(ext_run), "git_commit": git_commit(root), "git_status_short": _git_status(root), "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None, "manifest": str(manifest), "manifest_sha256": sha256_file(manifest), "base_success_rows": int((base_metrics.status == "success").sum()) if "status" in base_metrics else len(base_metrics), "base_methods": sorted(base_metrics.method_id.unique()), "base_inventory_sha256": stable_sha256(inventory.to_dict("records")), "data_audit": protocol_audit}
    (ext_run / "audit" / "extension_init.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_tcfl_audit(ext_run)
    save_yaml(ext_run / "configs" / "tcfl_dncnn.yaml", {"method_id": "tcfl_dncnn", "source_commit": "7320fc84fec37280d9643b1010d8afc4381f5a48", "num_layers": 10, "features": 64, "learning_rate": 2e-5, "betas": [0.5, 0.999], "lambda_pixel": 6.0, "epochs": 100, "batch_size": 2, "patch_size": 640, "seeds": [42, 123, 2026], "primary_seed": 42, "training_data": "PKU37 train noisy and clean pools sampled independently by position", "selection": "full PKU37 validation position-macro PSNR; SSIM tie-break within 0.0001 dB"})
    return ext_run


def _write_tcfl_audit(ext_run: Path) -> None:
    text = """# TCFL-OCT source audit

- Source: https://github.com/gengmufeng/TCFL-OCT
- Audited commit: `7320fc84fec37280d9643b1010d8afc4381f5a48`
- License: the audited repository contains no LICENSE file; the complete source is therefore kept in ignored `third_party_external/` and is not redistributed.
- Generator: official grayscale 10-layer, 64-feature DnCNN predicting noise; inference is `noisy - generator(noisy)`.
- Discriminator: official four-downsampling-block PatchGAN with InstanceNorm and a final patch prediction.
- Objective: six least-squares adversarial clean-domain terms plus four L1 cross-fusion/identity terms weighted by 6; discriminator real loss plus 0.1 times six fake losses.
- Optimizer: Adam for generator and discriminator, learning rate 2e-5, betas (0.5, 0.999), 100 epochs, batch size 2; no official scheduler.
- Official preprocessing: grayscale tensors divided by 255; this benchmark instead uses its common single-channel float32 [0,1] decoder, without histogram/gamma/min-max transforms.
- Official checkpoint selection is not specified. This extension preregisters validation position-macro PSNR with SSIM tie-break within 0.0001 dB.
"""
    (ext_run / "audit" / "tcfl_source_audit.md").write_text(text, encoding="utf-8")
    pd.DataFrame([
        {"item": "data loader", "official": "indexed A/B/C lists", "extension": "independent position-balanced PKU37-train A/B/C sampling", "reason": "enforce true unpaired benchmark protocol"},
        {"item": "crop", "official": "640x640", "extension": "640x640 native-scale input", "reason": "no numerical deviation"},
        {"item": "validation", "official": "not specified", "extension": "full PKU37 validation position-macro", "reason": "common checkpoint rule"},
        {"item": "engineering", "official": "epoch generator dumps", "extension": "atomic resumable G/D/optimizer/RNG checkpoints and audit logs", "reason": "cloud interruption recovery"},
    ]).to_csv(ext_run / "audit" / "tcfl_deviation_log.csv", index=False)


def resolve_sabids(root: Path, ext_run: Path) -> list[dict[str, Any]]:
    root, ext_run = root.resolve(), ext_run.resolve(); candidates = []
    patterns = ["runs/current/stage1_denoise_standalone/best.pth", "runs/current/stage1_denoise_fold0/best.pth", "runs/current/**/stage1_denoise*/best.pth"]
    paths: list[Path] = []
    for pattern in patterns: paths.extend(root.glob(pattern))
    for checkpoint in dict.fromkeys(path.resolve() for path in paths if "smoke" not in path.parts):
        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False); cfg = payload.get("config", {})
            stage = cfg.get("stage") or cfg.get("train", {}).get("stage") or cfg.get("training", {}).get("stage")
            data_cfg = cfg.get("data", {})
            train_datasets = data_cfg.get("train_datasets")
            val_datasets = data_cfg.get("val_datasets")
            pku_only = train_datasets == ["PKU37"] and val_datasets == ["PKU37"]
            config_path = checkpoint.parent / "resolved_config.yaml"
            metadata_path = checkpoint.parent / "run_metadata.json"
            compatible = stage == "denoise" and pku_only
            candidates.append({"method_id": "sabids_current" if compatible else "sabids_current_legacy_diagnostic", "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "config_path": str(config_path) if config_path.is_file() else cfg.get("runtime", {}).get("config_path", "embedded"), "metadata_path": str(metadata_path) if metadata_path.is_file() else "", "seed": cfg.get("seed"), "stage": stage, "train_datasets": train_datasets, "val_datasets": val_datasets, "uses_segmentation_labels": stage != "denoise", "pku37_only": pku_only, "legacy_diagnostic_reason": "" if pku_only else "training/validation datasets are not explicitly restricted to PKU37; Duke split-labelled rows may have entered development", "priority": 0 if checkpoint.as_posix().endswith("stage1_denoise_standalone/best.pth") else 1, "compatible_candidate": compatible})
        except Exception as exc:
            candidates.append({"checkpoint": str(checkpoint), "error": f"{type(exc).__name__}: {exc}", "compatible_candidate": False, "priority": 99})
    frame = pd.DataFrame(candidates); frame.to_csv(ext_run / "audit" / "sabids_checkpoint_inventory.csv", index=False)
    compatible = sorted((row for row in candidates if row.get("compatible_candidate")), key=lambda row: row["priority"])
    selected = compatible[0] if compatible and (len(compatible) == 1 or compatible[0]["priority"] < compatible[1]["priority"]) else None
    lines = ["# SABIDS current Stage-1 method resolution", "", "- Formal method ID: `sabids_current`", "- Model: `sabids.models.sabids_net.SABIDSNet`; adapter calls `forward_denoise_only`.", "- Intended identity: current formal independent Stage-1/D0 denoiser, not D1/joint/segmentation variants.", "- Segmentation labels: not used when checkpoint stage is exactly `denoise`.", "- Test/Duke use: forbidden for training and selection by this extension.", ""]
    if selected:
        lines += [f"- Automatically selected checkpoint: `{selected['checkpoint']}`", f"- SHA256: `{selected['checkpoint_sha256']}`", f"- Config: `{selected['config_path']}`", f"- Seed: `{selected['seed']}`", "- Objective basis: unique highest-priority formal standalone Stage-1 path with a denoise-stage checkpoint."]
        (ext_run / "audit" / "sabids_selected.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        lines += ["- Status: RETRAIN_REQUIRED_OR_BLOCKED_FOR_FORMAL_SELECTION.", "- Reason: no unique compatible formal checkpoint explicitly restricted to PKU37 train/validation was found. Smoke/joint checkpoints and checkpoints that may include Duke development rows are legacy diagnostics only; retrain the same current Stage-1 architecture/loss on PKU37 train/validation or pass a separately reviewed compatible checkpoint to the lock command."]
    (ext_run / "audit" / "sabids_method_resolution.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return candidates


def lock(root: Path, ext_run: Path, sabids_checkpoints: list[Path], sabids_config: Path | None) -> None:
    root, ext_run = root.resolve(), ext_run.resolve(); verify_base(ext_run)
    if _git_status(root).strip(): raise RuntimeError("formal extension lock requires a clean committed checkout")
    if not sabids_checkpoints:
        selected_path = ext_run / "audit" / "sabids_selected.json"
        if not selected_path.is_file(): raise RuntimeError("no uniquely resolved SABIDS checkpoint")
        selected = json.loads(selected_path.read_text(encoding="utf-8")); sabids_checkpoints = [Path(selected["checkpoint"])]
        if sabids_config is None and selected.get("config_path") not in {"", "embedded"}: sabids_config = Path(selected["config_path"])
    sabids_inventory = _checkpoint_inventory(sabids_checkpoints)
    tcfl_paths = [ext_run / "tracks" / "tcfl_dncnn" / f"seed_{seed}" / "best_psnr.pth" for seed in (42, 123, 2026)]
    if not all(path.is_file() for path in tcfl_paths): raise FileNotFoundError(f"TCFL best checkpoints incomplete: {[str(p) for p in tcfl_paths if not p.is_file()]}")
    tcfl_inventory = _checkpoint_inventory(tcfl_paths)
    sabids_cfg = {"method_id": "sabids_current", "config_path": str(sabids_config.resolve()) if sabids_config else None, "architecture": "current formal independent Stage-1/D0", "selection": "pre-existing formal registry checkpoint or validation-selected retraining"}
    save_yaml(ext_run / "configs" / "sabids_current.yaml", sabids_cfg)
    registry = {"status": "locked_extension", "methods": {"sabids_current": {"config": sabids_cfg, "seed": sabids_inventory[0]["seed"], "checkpoint": sabids_inventory[0]["path"], "evaluation_checkpoints": [{"seed": row["seed"], "checkpoint": row["path"]} for row in sabids_inventory]}, "tcfl_dncnn": {"config": load_yaml(ext_run / "configs" / "tcfl_dncnn.yaml"), "seed": 42, "checkpoint": tcfl_inventory[0]["path"], "evaluation_checkpoints": [{"seed": row["seed"], "checkpoint": row["path"]} for row in tcfl_inventory]}}}
    save_yaml(ext_run / "configs" / "inference_registry.yaml", registry)
    config_hashes = {path.name: sha256_file(path) for path in sorted((ext_run / "configs").glob("*.yaml"))}
    init_data = json.loads((ext_run / "audit" / "extension_init.json").read_text(encoding="utf-8"))
    value = {"status": "locked", "git_commit": git_commit(root), "locked_at_utc": _now(), "test_started_at_utc": None, "duke_evaluation_started_at_utc": None, "manifest_sha256": init_data["manifest_sha256"], "base_run": init_data["base_run"], "base_inventory_sha256": init_data["base_inventory_sha256"], "method_identity": {"sabids_current": sabids_cfg, "tcfl_dncnn": registry["methods"]["tcfl_dncnn"]["config"]}, "config_sha256": config_hashes, "checkpoint_sha256": {"sabids_current": sabids_inventory, "tcfl_dncnn": tcfl_inventory}}
    (ext_run / "audit" / "extension_config_lock.json").write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _checkpoint_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        path = path.resolve()
        if not path.is_file(): raise FileNotFoundError(path)
        payload = torch.load(path, map_location="cpu", weights_only=False); config = payload.get("config", {}) if isinstance(payload, dict) else {}
        rows.append({"seed": int(config.get("seed", payload.get("seed", 42))), "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def merge(root: Path, ext_run: Path) -> None:
    ext_run = ext_run.resolve(); verify_base(ext_run)
    base = Path(json.loads((ext_run / "audit" / "extension_init.json").read_text(encoding="utf-8"))["base_run"])
    base_table = pd.read_csv(base / "metrics" / "per_image_metrics.csv")
    extension_path = ext_run / "metrics" / "per_image_metrics.csv"
    if not extension_path.is_file(): raise FileNotFoundError(extension_path)
    extension = pd.read_csv(extension_path); extension = extension[extension.method_id.isin(NEW_METHODS)]
    combined = pd.concat([base_table, extension], ignore_index=True)
    keys = ["dataset", "split", "sample_id", "method_id", "seed"]
    if combined.duplicated(keys).any(): raise RuntimeError("duplicate per-image logical keys while merging BASE_RUN and extension")
    if combined.select_dtypes(include="number").replace([float("inf"), -float("inf")], pd.NA).isna().all(axis=None): raise RuntimeError("invalid numeric result table")
    atomic_write_csv(combined, extension_path, ext_run)
    summaries = aggregate(combined)
    for name, frame in summaries.items(): atomic_write_csv(frame, ext_run / "metrics" / f"{name}.csv", ext_run)
    atomic_write_csv(paired_differences(summaries["per_position_metrics"]), ext_run / "metrics" / "paired_method_differences.csv", ext_run)
    atomic_write_csv(bootstrap_confidence_intervals(summaries["per_position_metrics"], 10_000, 42), ext_run / "metrics" / "bootstrap_confidence_intervals.csv", ext_run)
    registry = load_yaml(ext_run / "configs" / "inference_registry.yaml")
    primary = {"dncnn_paired": 42, "nafnet_paired": 42, "tcfl_dncnn": 42}
    primary.update({method: int(entry.get("seed", 0)) for method, entry in registry.get("methods", {}).items()})
    combined["is_primary_seed"] = [int(seed) == primary.get(method, 0) for method, seed in zip(combined.method_id, combined.seed)]
    combined["status"] = combined.get("status", "success")
    manifest_columns = [column for column in ("dataset", "split", "position_id", "frame_id", "sample_id", "noisy_path", "reference_path", "denoised_path", "method_id", "seed", "is_primary_seed", "config_sha256", "checkpoint_sha256", "width", "height", "bit_depth", "output_sha256", "status") if column in combined]
    (ext_run / "manifests").mkdir(exist_ok=True)
    atomic_write_csv(combined[manifest_columns], ext_run / "manifests" / "denoised_dataset_manifest.csv", ext_run)
    atomic_write_csv(combined.loc[combined.is_primary_seed, manifest_columns], ext_run / "manifests" / "denoised_dataset_manifest_primary.csv", ext_run)
    for name in ("runtime_records.csv", "runtime_summary.csv"):
        base_path, ext_path = base / "metrics" / name, ext_run / "metrics" / name
        frames = [pd.read_csv(path) for path in (base_path, ext_path) if path.is_file()]
        if frames: atomic_write_csv(pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(), ext_path, ext_run)
    verify_base(ext_run)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); p.add_argument("action", choices=["init", "resolve-sabids", "lock", "merge", "verify-base"]); p.add_argument("--project-root", type=Path, default=Path(".")); p.add_argument("--ext-run", type=Path, required=True); p.add_argument("--base-run", type=Path); p.add_argument("--sabids-checkpoint", type=Path, action="append", default=[]); p.add_argument("--sabids-config", type=Path); return p


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.action == "init": print(init(args.project_root, args.ext_run, args.base_run))
    elif args.action == "resolve-sabids": resolve_sabids(args.project_root, args.ext_run)
    elif args.action == "lock": lock(args.project_root, args.ext_run, args.sabids_checkpoint, args.sabids_config)
    elif args.action == "merge": merge(args.project_root, args.ext_run)
    else: verify_base(args.ext_run)


if __name__ == "__main__": main()
