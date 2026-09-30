from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import yaml
import pandas as pd
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config, save_config
from sabids.data.io import read_gray
from sabids.experiments.dual_task_adaptive import audit_adaptive_inputs
from sabids.experiments.dual_task_adaptive import sha256_file


def _matching_supplement(root: Path, checkpoint_sha: str) -> dict:
    candidates: list[dict] = []
    for path in sorted(root.glob("exports/**/*best_checkpoint_supplement*.tar.gz")):
        with tarfile.open(path, "r:gz") as archive:
            members = [m for m in archive.getmembers() if m.name.endswith("b3_checkpoint_binding_best.json")]
            for member in members:
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                value = json.load(io.TextIOWrapper(handle, encoding="utf-8"))
                if value.get("checkpoint_sha256") == checkpoint_sha:
                    candidates.append(value)
    unique = {json.dumps(value, sort_keys=True): value for value in candidates}
    if len(unique) != 1:
        raise RuntimeError(
            "Expected exactly one matching best-checkpoint supplement in exports; "
            f"found {len(unique)}. Copy the verified supplement tar.gz into exports/."
        )
    return next(iter(unique.values()))


def _materialize_coarse_binding(root: Path, config: dict, registry: Path) -> None:
    evidence = config["dual_task_adaptive"]["evidence"]
    configured = (root / evidence["coarse_binding"]).resolve()
    if configured.is_file():
        return
    binding = _matching_supplement(
        root, config["dual_task_adaptive"]["anchors"]["coarse_checkpoint_sha256"]
    )
    destination = registry / "evidence" / "b3_checkpoint_binding_best.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(binding, indent=2, ensure_ascii=False) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"Existing materialized B3 binding differs: {destination}"
            )
    else:
        destination.write_text(serialized, encoding="utf-8")
    evidence["coarse_binding"] = str(destination)


def _record_input_inventory(root: Path, config: dict, registry: Path) -> Path:
    table = pd.read_csv(config["data"]["manifest"], dtype=str).fillna("")
    table = table[table["split"].isin(["train", "val"])].copy()
    records = []
    columns = [
        "primary_path", "image_path", "clean_path", "layer_mask_path",
        "vessel_mask_path", "label_valid_mask_path", "vessel_valid_mask_path",
        "spatial_valid_mask_path",
    ]
    for row in table.sort_values(["split", "sample_id"]).to_dict("records"):
        for column in columns:
            value = str(row.get(column, "")).strip()
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = (root / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"Missing adaptive train/val asset: {path}")
            records.append({"sample_id": str(row["sample_id"]), "split": str(row["split"]),
                            "column": column, "sha256": sha256_file(path)})
    records.sort(key=lambda item: (item["sample_id"], item["split"], item["column"], item["sha256"]))
    records_sha = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = {"schema_version": "dual-task-adaptive-input-assets-v1",
               "recorded_before_optimizer_step": True, "records": records,
               "records_sha256": records_sha, "test_assets_opened": 0}
    destination = registry / "training_input_inventory.json"
    serialized = json.dumps(payload, indent=2) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Existing adaptive input inventory differs: {destination}")
    if not destination.exists():
        destination.write_text(serialized, encoding="utf-8")
    return destination


def _asset(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _component_rows(root: Path, row: dict) -> list[dict]:
    assets = {
        "vessel_mask_path": _asset(root, row["vessel_mask_path"]),
        "layer_mask_path": _asset(root, row["layer_mask_path"]),
        "image_path": _asset(root, row["image_path"]),
    }
    try:
        vessel = read_gray(assets["vessel_mask_path"])
        layer = read_gray(assets["layer_mask_path"])
        noisy = read_gray(assets["image_path"])
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        details = ", ".join(f"{name}={path}" for name, path in assets.items())
        raise RuntimeError(
            f"Cannot read train/val component assets for {row['sample_id']}: "
            f"{details}"
        ) from error
    if vessel.shape != layer.shape:
        raise ValueError(
            f"Layer/vessel geometry mismatch for {row['sample_id']}: "
            f"layer={layer.shape}, vessel={vessel.shape}"
        )
    if noisy.shape != vessel.shape:
        noisy = cv2.resize(
            noisy,
            (vessel.shape[1], vessel.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    vessel_mask, layer_mask = vessel > 0.5, layer > 0.5
    count, labels = cv2.connectedComponents(vessel_mask.astype(np.uint8), connectivity=8)
    output = []
    for component_id in range(1, count):
        component = labels == component_id
        stroma = layer_mask & ~vessel_mask
        contrast_value = (
            abs(float(noisy[component].mean()) - float(noisy[stroma].mean()))
            if component.any() and stroma.any() else float("nan")
        )
        output.append({"sample_id": str(row["sample_id"]), "group_id": str(row["group_id"]),
                       "split": str(row["split"]), "component_id": int(component_id),
                       "area_model_grid_px": int(component.sum()),
                       "contrast_model_grid": (float(contrast_value) if np.isfinite(contrast_value) else None)})
    return output


def _record_component_inventory(root: Path, config: dict, registry: Path) -> Path:
    table = pd.read_csv(config["data"]["manifest"], dtype=str).fillna("")
    rows = []
    for row in table[table["split"].isin(["train", "val"])].sort_values(["split", "sample_id"]).to_dict("records"):
        rows.extend(_component_rows(root, row))
    train = [row for row in rows if row["split"] == "train"]
    if not train:
        raise RuntimeError("Cannot define component strata without train components")
    areas = np.asarray([row["area_model_grid_px"] for row in train], dtype=np.float64)
    contrasts = np.asarray([row["contrast_model_grid"] for row in train
                            if row["contrast_model_grid"] is not None], dtype=np.float64)
    if not contrasts.size:
        raise RuntimeError("Cannot define low-contrast stratum from train data")
    small_max = float(np.quantile(areas, 1.0 / 3.0))
    low_contrast_max = float(np.quantile(contrasts, 1.0 / 3.0))
    validation = []
    for row in rows:
        if row["split"] != "val":
            continue
        low_contrast = row["contrast_model_grid"] is not None and row["contrast_model_grid"] <= low_contrast_max
        validation.append({**row, "small": row["area_model_grid_px"] <= small_max,
                           "low_contrast": low_contrast,
                           "small_low_contrast": (
                               row["area_model_grid_px"] <= small_max
                               and low_contrast
                           )})
    payload = {"schema_version": "dual-task-adaptive-fixed-components-v1",
               "threshold_source": "train_only_model_grid",
               "coordinate_system": "model_grid_px",
               "small_area_max_model_grid_px": small_max,
               "low_contrast_max": low_contrast_max,
               "validation_components": validation, "test_assets_opened": 0}
    destination = registry / "fixed_component_inventory.json"
    serialized = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != serialized:
        raise FileExistsError(f"Existing fixed component inventory differs: {destination}")
    if not destination.exists():
        destination.write_text(serialized, encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=("preflight", "overfit", "formal"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = Path(args.project_root).expanduser().resolve()
    template = root / "configs/adaptive_denoising/dual_task_adaptive_v1/seed42.yaml"
    config = load_config(template)
    registry = root / "cache/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v1" / args.run_id
    registry.mkdir(parents=True, exist_ok=True)
    _materialize_coarse_binding(root, config, registry)
    config["device"] = args.device
    config["data"]["root"] = str(root)
    config["data"]["manifest"] = str((root / config["data"]["manifest"]).resolve())
    for section, keys in (
        ("anchors", ("d2_checkpoint", "coarse_checkpoint")),
        ("evidence", ("d2_binding", "d2_inventory", "protocol_lock", "split_contract")),
    ):
        values = config["dual_task_adaptive"][section]
        for key in keys:
            path = Path(values[key])
            if not path.is_absolute():
                values[key] = str((root / path).resolve())

    report = audit_adaptive_inputs(config, root)
    report.update({"mode": args.mode, "run_id": args.run_id})
    report_path = registry / f"preflight_{args.mode}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "passed":
        print(json.dumps(report, indent=2))
        raise SystemExit(2)
    if args.mode == "preflight":
        print(json.dumps(report, indent=2))
        return
    inventory_path = _record_input_inventory(root, config, registry)
    component_path = _record_component_inventory(root, config, registry)
    config["dual_task_adaptive"]["evidence"]["training_input_inventory"] = str(inventory_path)
    config["dual_task_adaptive"]["evidence"]["training_input_inventory_sha256"] = sha256_file(inventory_path)
    config["dual_task_adaptive"]["evidence"]["fixed_component_inventory"] = str(component_path)
    config["dual_task_adaptive"]["evidence"]["fixed_component_inventory_sha256"] = sha256_file(component_path)
    report["training_input_inventory"] = str(inventory_path)
    report["training_input_inventory_sha256"] = sha256_file(inventory_path)
    report["fixed_component_inventory"] = str(component_path)
    report["fixed_component_inventory_sha256"] = sha256_file(component_path)

    if args.mode != "preflight":
        prepared = copy.deepcopy(config)
        output = root / "runs/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v1" / args.run_id / "seed42"
        if args.mode == "overfit":
            output = output.with_name("overfit_seed42")
            prepared["train"].update({"epochs": 3, "early_stopping_patience": 4, "evaluate_epoch0": False})
            prepared["data"].update({
                "val_split": "train", "max_train_samples": 8, "max_val_samples": 8,
                "samples_per_epoch": 8,
            })
            prepared["dual_task_adaptive"]["validation_only"] = False
            prepared["dual_task_adaptive"]["run_mode"] = "overfit_train_only"
        else:
            prepared["dual_task_adaptive"]["run_mode"] = "formal_seed42_validation_only"
        prepared["train"]["output_dir"] = str(output)
        if output.exists() and not args.resume:
            raise FileExistsError(f"Refusing to overwrite adaptive run: {output}")
        if args.resume:
            last = output / "last.pth"
            if args.mode != "formal" or not last.is_file():
                raise FileNotFoundError("Resume requires an existing formal last.pth")
            prepared["train"]["resume"] = str(last)
        config_path = registry / f"config_{args.mode}{'_resume' if args.resume else ''}_seed42.yaml"
        if config_path.exists():
            existing = load_config(config_path)
            if existing != prepared:
                raise FileExistsError(
                    f"Existing prepared config differs; refusing reuse: {config_path}"
                )
        else:
            save_config(prepared, config_path)
        report["config"] = str(config_path)
        report["output_dir"] = str(output)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
