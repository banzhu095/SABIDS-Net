from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from . import METHOD_ORDER
from .image_io import sha256_file
from .manifest_builder import primary_seeds


def _resolve_path(value: object, root: Path, run: Path) -> Path | None:
    if not value: return None
    path = Path(str(value)).expanduser()
    for candidate in (path, root / path, run / path):
        if candidate.is_file(): return candidate.resolve()
    return path.resolve() if path.is_absolute() else None


def _runtime_config(entry: dict[str, Any], root: Path, run: Path) -> tuple[dict[str, Any], str]:
    registry_config = entry.get("config", {}) if isinstance(entry.get("config"), dict) else {}
    path = _resolve_path(registry_config.get("config_path") or entry.get("config_path"), root, run)
    if path and path.is_file():
        try:
            from sabids.config import load_config
            return load_config(path), str(path)
        except (ImportError, KeyError, TypeError, ValueError):
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}, str(path)
    return registry_config, ""


def load_literature(config: str | Path | None = None) -> pd.DataFrame:
    path = Path(config) if config else Path(__file__).parent / "configs" / "method_literature.yaml"
    methods = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("methods", [])
    frame = pd.DataFrame(methods)
    if frame.method_id.duplicated().any(): raise ValueError("duplicate method_id in literature configuration")
    missing = set(METHOD_ORDER) - set(frame.method_id)
    if missing: raise ValueError(f"literature configuration missing methods: {sorted(missing)}")
    for column in ("main_components", "strengths", "limitations"):
        frame[column] = frame[column].map(lambda value: "; ".join(map(str, value)) if isinstance(value, list) else value)
    return frame


def implementation_summary(project_root: str | Path, run_dir: str | Path) -> pd.DataFrame:
    root, run = Path(project_root).resolve(), Path(run_dir).resolve()
    registry = yaml.safe_load((run / "configs" / "inference_registry.yaml").read_text(encoding="utf-8")) or {}
    complexity_path = run / "metrics" / "model_complexity.csv"
    complexity = pd.read_csv(complexity_path) if complexity_path.is_file() else pd.DataFrame()
    checkpoints_path = run / "metrics" / "checkpoint_inventory.csv"
    checkpoints = pd.read_csv(checkpoints_path) if checkpoints_path.is_file() else pd.DataFrame()
    per_image_path = run / "metrics" / "per_image_metrics.csv"
    per_image = pd.read_csv(per_image_path, low_memory=False) if per_image_path.is_file() else pd.DataFrame()
    if not per_image.empty and "status" in per_image:
        per_image = per_image[per_image.status.astype(str).str.lower().isin({"success", "completed", "ok"})]
    successful_methods = set(per_image.method_id.dropna().astype(str)) if not per_image.empty and "method_id" in per_image else set()
    seeds = primary_seeds(run)
    rows = []
    for method in METHOD_ORDER:
        entry = registry.get("methods", {}).get(method)
        row: dict[str, Any] = {
            "method_id": method, "status": "completed" if entry is not None or method in successful_methods else "missing/not_completed",
            "completion_evidence": ";".join(filter(None, ("locked_registry" if entry is not None else "", "successful_per_image" if method in successful_methods else ""))),
            "primary_seed": seeds.get(method, 0), "registry_entry": json.dumps(entry, ensure_ascii=False, sort_keys=True) if entry is not None else "",
            "model_class": "", "stage_identity": "", "uses_segmentation_labels": "", "training_supervision": "",
            "network_structure": "", "loss_configuration": "", "runtime_source": "",
            "checkpoint": "", "checkpoint_sha256": "", "checkpoint_sha256_matches_registry": "", "selection_metric": "", "parameters": None,
            "input_range": "float32 [0,1]", "output_range": "float32 [0,1] before lossless export",
        }
        if isinstance(entry, dict):
            config = entry.get("config", {}) if isinstance(entry.get("config"), dict) else {}
            row.update({
                "checkpoint": str(entry.get("checkpoint", "")),
                "checkpoint_sha256": str(entry.get("checkpoint_sha256", "")),
                "selection_metric": str(config.get("selection", entry.get("selection", ""))),
                "training_supervision": str(config.get("training_data", config.get("supervision", ""))),
            })
            checkpoint = _resolve_path(row["checkpoint"], root, run)
            if checkpoint and checkpoint.is_file():
                actual_sha = sha256_file(checkpoint)
                row["checkpoint_sha256_matches_registry"] = not row["checkpoint_sha256"] or row["checkpoint_sha256"] == actual_sha
                row["checkpoint_sha256"] = actual_sha
        if method == "sabids_current":
            if isinstance(entry, dict):
                resolved, config_path = _runtime_config(entry, root, run)
                model_config = resolved.get("model", {}) if isinstance(resolved.get("model"), dict) else {}
                train_config = resolved.get("train", {}) if isinstance(resolved.get("train"), dict) else {}
                loss_config = resolved.get("loss", {}) if isinstance(resolved.get("loss"), dict) else {}
                stage = str(entry.get("stage", train_config.get("stage", config.get("architecture", ""))))
                row.update({
                    "stage_identity": stage,
                    "uses_segmentation_labels": bool(entry.get("uses_segmentation_labels", stage not in {"denoise", "D0", "d0"})),
                    "training_supervision": str(train_config.get("supervision", row["training_supervision"])),
                    "network_structure": json.dumps(model_config, ensure_ascii=False, sort_keys=True),
                    "loss_configuration": json.dumps(loss_config, ensure_ascii=False, sort_keys=True),
                    "runtime_source": f"{config_path}; tools/oct_denoise_benchmark/methods/sabids_adapter.py",
                })
                if model_config:
                    try:
                        from sabids.engine.trainer import build_model
                        network = build_model(resolved)
                        row["model_class"] = f"{type(network).__module__}.{type(network).__name__}"
                        if row["parameters"] is None: row["parameters"] = sum(parameter.numel() for parameter in network.parameters())
                    except (ImportError, KeyError, TypeError, ValueError, RuntimeError) as exc:
                        row["model_class"] = f"unresolved_from_runtime:{type(exc).__name__}"
        elif method == "tcfl_dncnn":
            if isinstance(entry, dict):
                config = entry.get("config", {}) if isinstance(entry.get("config"), dict) else {}
                row.update({"model_class": "tools.oct_denoise_benchmark.methods.tcfl_adapter.TCFLGenerator", "stage_identity": str(config.get("architecture", "generator inference")), "uses_segmentation_labels": False, "network_structure": json.dumps({key: config.get(key) for key in ("num_layers", "features") if key in config}, ensure_ascii=False, sort_keys=True), "loss_configuration": json.dumps({key: config.get(key) for key in ("lambda_pixel", "adversarial_loss") if key in config}, ensure_ascii=False, sort_keys=True), "runtime_source": "tools/oct_denoise_benchmark/methods/tcfl_adapter.py"})
        if not complexity.empty and "method_id" in complexity:
            hit = complexity[complexity.method_id.astype(str) == method]
            if not hit.empty: row["parameters"] = hit.iloc[-1].get("parameters")
        if not checkpoints.empty and "method_id" in checkpoints and not row["checkpoint_sha256"]:
            hit = checkpoints[checkpoints.method_id.astype(str) == method]
            if not hit.empty:
                row["checkpoint_sha256"] = hit.iloc[-1].get("sha256", hit.iloc[-1].get("checkpoint_sha256", ""))
        rows.append(row)
    return pd.DataFrame(rows)
