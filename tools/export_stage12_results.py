"""Stage 1/2 registry, non-test inference and validation archive.

The default scope never opens or evaluates reserved-test images.  Validation
tables use fixed P0 thresholds (0.5/0.5); this tool does not tune thresholds or
select checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import torch
import cv2
import numpy as np
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.data import OCTManifestDataset
from sabids.data.io import write_gray
from sabids.engine.evaluator import evaluate_model
from sabids.engine.trainer import _make_transform, build_model
from sabids.utils import get_device, write_json
from sabids.postprocessing import clean_layer_mask, hard_contain_vessel, regularize_lower_boundary


TOOL_VERSION = "stage12-export-v1"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
RUNS = [
    (0, "S1-Denoise", "stage1_denoise_fold0", "denoise"),
    (10, "E1-current", "stage2_segment_safe_current_fold0", "segment"),
    (20, "E3-current", "stage2_segment_roi_current_fold0", "segment"),
    (30, "E3b", "stage2_segment_roi_outside_fold0", "segment"),
    (40, "E3b-noD2S", "stage2_segment_roi_outside_no_d2s_fold0", "segment"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-selection", default=None,
                        help="Optional JSON list overriding the built-in current Stage 1/2 registry.")
    parser.add_argument("--all-manifest", default="Manifests/joint_folds/manifest_joint_fold0.csv")
    parser.add_argument("--segmentation-manifest", default="Manifests/segmentation_folds/manifest_seg_fold0.csv")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--full-non-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-all-float", action="store_true")
    parser.add_argument("--include-sealed-test-export", action="store_true",
                        help="Pure forward export to a sealed directory; never included in metrics/gallery.")
    parser.add_argument("--skip-input-hashes", action="store_true")
    return parser.parse_args()


def resolve_path(value: object, root: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_runs(path: str | None) -> list[Dict[str, Any]]:
    if path is None:
        return [
            {"display_order": order, "alias": alias, "run_dir": run_dir, "stage": stage}
            for order, alias, run_dir, stage in RUNS
        ]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("--run-selection must contain a JSON list")
    return data


def history_best(run_dir: Path, monitor: str) -> Dict[str, Any]:
    history_path = run_dir / "history.csv"
    if not history_path.is_file():
        return {"history_status": "missing"}
    history = pd.read_csv(history_path)
    column = f"val_{monitor}" if f"val_{monitor}" in history else monitor
    if history.empty or column not in history:
        return {"history_status": "monitor_missing", "history_rows": len(history)}
    row = history.loc[history[column].idxmax()]
    return {
        "history_status": "available", "history_rows": int(len(history)),
        "history_best_epoch": int(row.get("epoch", history[column].idxmax() + 1)),
        "history_best_metric": float(row[column]), "history_monitor_column": column,
    }


def build_registry(root: Path, selections: list[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in selections:
        run_dir = root / "runs" / "current" / str(item["run_dir"])
        config_path = run_dir / "resolved_config.yaml"
        checkpoint_path = run_dir / "best.pth"
        metadata_path = run_dir / "run_metadata.json"
        config = load_config(config_path) if config_path.is_file() else {}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
        monitor = str(config.get("train", {}).get("monitor", "psnr" if item["stage"] == "denoise" else "vessel_soft_dice"))
        model_cfg = config.get("model", {})
        loss_cfg = config.get("loss", {})
        row = {
            **item, "run_path": str(run_dir.relative_to(root)),
            "status": "ready" if config_path.is_file() and checkpoint_path.is_file() else "blocked_missing_config_or_checkpoint",
            "resolved_config": str(config_path.relative_to(root)),
            "checkpoint": str(checkpoint_path.relative_to(root)),
            "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path.is_file() else "missing",
            "checkpoint_epoch": metadata.get("best_epoch", "unknown"),
            "monitor": monitor, "monitor_direction": "max", "seed": config.get("seed", "unknown"),
            "fold": metadata.get("fold", 0), "git_commit": metadata.get("git_commit", "unknown"),
            "target_size": "x".join(map(str, config.get("data", {}).get("target_size", []))),
            "normalization": config.get("data", {}).get("normalization", "unknown"),
            "d2s_enabled": model_cfg.get("enable_denoise_to_seg", False),
            "s2d_enabled": model_cfg.get("enable_seg_to_denoise", False),
            "freeze_shared_encoder": model_cfg.get("stage2_freeze_shared_encoder", "n/a"),
            "vessel_supervision_mode": loss_cfg.get("vessel_supervision_mode", "composite"),
            "outside_weight": loss_cfg.get("weights", {}).get("vessel_outside", 0.0),
            "checkpoint_selection": "original best.pth only; no fallback to last.pth",
        }
        row.update(history_best(run_dir, monitor))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("display_order")


def build_training_best(root: Path, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in registry.to_dict("records"):
        history_path = root / item["run_path"] / "history.csv"
        if not history_path.is_file():
            continue
        history = pd.read_csv(history_path)
        if history.empty:
            continue
        epoch = item.get("checkpoint_epoch")
        selected = history[history["epoch"] == int(epoch)] if str(epoch).isdigit() and "epoch" in history else pd.DataFrame()
        if selected.empty:
            column = item.get("history_monitor_column")
            selected = history.loc[[history[column].idxmax()]] if column in history else history.tail(1)
        row = selected.iloc[0].to_dict()
        rows.append({"display_order": item["display_order"], "experiment": item["alias"],
                     "selection_source": "checkpoint metadata epoch" if str(epoch).isdigit() else "history monitor fallback",
                     **row})
    return pd.DataFrame(rows).sort_values("display_order") if rows else pd.DataFrame()


def build_inventory(root: Path, manifest_path: Path, skip_hashes: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    required = {"sample_id", "group_id", "dataset", "split", "image_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    records: Dict[str, Dict[str, Any]] = {}
    associations = []
    for row in manifest.to_dict("records"):
        split = str(row["split"]).lower()
        for column, role in (("image_path", "noisy_input"), ("clean_path", "paired_clean_reference")):
            path = resolve_path(row.get(column, ""), root)
            if path is None:
                continue
            key = os.path.normcase(str(path))
            associations.append({"physical_path": key, "split": split, "role": role,
                                 "dataset": row["dataset"], "group_id": row["group_id"],
                                 "sample_id": row["sample_id"]})
            record = records.setdefault(key, {
                "sample_uid": hashlib.sha256(key.encode("utf-8")).hexdigest()[:20],
                "dataset": row["dataset"], "relative_path": (
                    path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
                ), "physical_path": key, "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else None,
                "file_sha256": "skipped", "roles": set(), "splits": set(), "groups": set(),
                "manifest_status": "indexed", "inference_status": "pending",
            })
            record["roles"].add(role); record["splits"].add(split); record["groups"].add(str(row["group_id"]))
    for path in (root / "Data").rglob("*") if (root / "Data").is_dir() else []:
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        key = os.path.normcase(str(path.resolve()))
        records.setdefault(key, {
            "sample_uid": hashlib.sha256(key.encode("utf-8")).hexdigest()[:20], "dataset": "unknown",
            "relative_path": path.relative_to(root).as_posix(), "physical_path": key, "exists": True,
            "bytes": path.stat().st_size, "file_sha256": "skipped", "roles": {"unresolved_bscan_candidate"},
            "splits": {"split_unresolved"}, "groups": {"unknown"}, "manifest_status": "not_in_manifest",
            "inference_status": "excluded_unresolved",
        })
    for record in records.values():
        if "test" in record["splits"]:
            record["inference_status"] = "excluded_reserved_test_and_linked_asset"
        elif not record["exists"]:
            record["inference_status"] = "excluded_missing_file"
        elif record["manifest_status"] == "indexed":
            record["inference_status"] = "eligible_non_test"
        if not skip_hashes and record["exists"] and record["inference_status"] != "excluded_reserved_test_and_linked_asset":
            record["file_sha256"] = sha256_file(Path(record["physical_path"]))
        for key in ("roles", "splits", "groups"):
            record[key] = ";".join(sorted(record[key]))
    inventory = pd.DataFrame(records.values()).sort_values(["dataset", "relative_path"])
    links = pd.DataFrame(associations)
    return inventory, links


def write_dry_run_files(output: Path, registry: pd.DataFrame, inventory: pd.DataFrame,
                        links: pd.DataFrame, manifest: Path,
                        segmentation_manifest: Path, root: Path) -> None:
    registry.to_csv(output / "experiment_registry.csv", index=False, encoding="utf-8-sig")
    write_json(registry.to_dict("records"), output / "experiment_registry.json")
    build_training_best(root, registry).to_csv(
        output / "training_best.csv", index=False, encoding="utf-8-sig"
    )
    inventory.drop(columns=["physical_path"]).to_csv(output / "data_inventory.csv", index=False, encoding="utf-8-sig")
    counts = inventory.groupby(["dataset", "splits", "roles", "inference_status"], dropna=False).size().rename("n_files").reset_index()
    counts.to_csv(output / "input_role_counts.csv", index=False, encoding="utf-8-sig")
    coverage = inventory.groupby(["dataset", "inference_status"], dropna=False).agg(
        discovered=("sample_uid", "size"), bytes=("bytes", "sum")
    ).reset_index()
    coverage.to_csv(output / "coverage.csv", index=False, encoding="utf-8-sig")
    test_groups = sorted(set(links.loc[links["split"] == "test", "group_id"].astype(str)))
    val = links[(links["split"] == "val") & (links["role"] == "noisy_input")]
    segmentation = pd.read_csv(segmentation_manifest, dtype=str).fillna("")
    seg_counts = segmentation.groupby("split").agg(
        rows=("sample_id", "size"), groups=("group_id", "nunique")
    ).to_dict("index")
    seg_val = segmentation[segmentation["split"].str.lower() == "val"]
    expected_current = len(seg_val) == 141 and seg_val["group_id"].nunique() == 3
    text = f"""# Split audit

- Source manifest: `{manifest}`
- Validation noisy rows: {len(val)}; groups: {val['group_id'].nunique()}
- Stage 2 labelled split counts: `{json.dumps(seg_counts, ensure_ascii=False)}`
- Stage 2 validation manifest matches the recorded current 141-frame/3-group protocol: `{expected_current}`
- Stage 2 validation groups in this manifest: `{';'.join(sorted(seg_val['group_id'].unique()))}`
- Reserved-test groups excluded by manifest metadata: {len(test_groups)}
- Test group identifiers are retained only in this local audit count and are not opened for filtering.
- Files linked to any test row, including shared clean references, are excluded from default inference.
- `split_unresolved` files are inventoried but excluded pending identity resolution.
- Patient identity remains unknown unless independently evidenced; group_id is not promoted to patient_id.
"""
    (output / "split_audit.md").write_text(text, encoding="utf-8")
    ready = int((registry["status"] == "ready").sum())
    readme = f"""# Stage 1/2 inference and validation archive

- Tool: `{TOOL_VERSION}`
- Current status: dry-run inventory complete; {ready}/{len(registry)} checkpoints ready.
- Default scope: all manifest-indexed non-test noisy B-scans; unresolved files excluded.
- Validation: complete labelled validation split from each resolved run config.
- Primary thresholds: layer=0.5, vessel=0.5, raw P0. No threshold search is performed.
- Coordinate protocol: deterministic training-grid forward, remove padding, restore probabilities/images to original grid, then threshold.
- Reserved test: excluded by default together with linked clean assets. Sealed export is pure forward only and is never included in metrics or gallery.
- Missing checkpoints/configs remain `blocked`; `last.pth` is never substituted for `best.pth`.

## Commands

```bash
python tools/export_stage12_results.py --project-root /mnt/SABIDS-Net \\
  --output /mnt/SABIDS-Net/runs/reports/stage12_validation_<timestamp> \\
  --dry-run --skip-input-hashes

python tools/export_stage12_results.py --project-root /mnt/SABIDS-Net \\
  --output /mnt/SABIDS-Net/runs/reports/stage12_validation_<timestamp> \\
  --device cuda --batch-size 1 --num-workers 4 --full-non-test
```

Do not use `--include-sealed-test-export` during development unless test unsealing is explicitly authorized.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "architecture_stage12.md").write_text("""# Stage 1/2 architecture truth

- Shared NAF-style encoder and denoising/layer/vessel decoders: `sabids/models/sabids_net.py`.
- Denoising is residual: `denoised_raw=noisy-residual`, clipped only for display/evaluation.
- Stage 1 uses only the denoising result; its untrained segmentation heads are not exported.
- E1-current freezes the Stage 1 upstream/denoising function and disables D→S/S→D.
- E3-current enables detached-source D→S and ROI vessel supervision; outside BCE=0.
- E3b changes E3-current by adding outside-GT-layer logit BCE weight 0.5.
- E3b-noD2S retains E3b supervision and disables D→S.
- All four Stage 2 runs use one original best checkpoint for both layer and vessel outputs.
- S→D is disabled in the current Stage 2 controls, so Stage 2 is expected to preserve—not improve—the Stage 1 denoiser.
""", encoding="utf-8")
    (output / "metric_definitions.md").write_text("""# Metric definitions

- Image range is fixed `[0,1]`; metrics use float arrays, never re-read display PNGs.
- PSNR is computed per image as `10 log10(1/MSE)`; exact identity is `+inf`.
- SSIM uses the repository 7-pixel uniform-window implementation with C1=0.01² and C2=0.03².
- EPI is the correlation of gradient magnitudes implemented in `sabids/metrics.py`.
- Hard layer/vessel metrics include Dice, IoU, Precision, Recall and explicit TP/FP/FN/TN.
- Vessel full-image metrics use the vessel-valid mask; GT-layer ROI and oracle/gated diagnostics are separately named.
- Primary aggregation is frame metric → mean within anatomical group → equal-weight group macro.
- HD95/ASSD and boundary/thickness errors use pixel units unless verified physical spacing is supplied.
""", encoding="utf-8")
    write_json({
        "tool_version": TOOL_VERSION, "forward_grid": "resolved_config.data.target_size",
        "evaluation_grid": "restored_original_grid", "inverse_order": "remove padding; resize continuous probability/image; threshold",
        "interpolation": {"image_probability": "linear", "mask": "nearest"},
        "thresholds": {"layer": 0.5, "vessel": 0.5}, "augmentation": False,
        "checkpoint_selection": "existing best.pth only", "threshold_search": False,
        "primary_segmentation_split": "val", "test_metrics": False,
    }, output / "evaluation_protocol.json")
    write_json({
        "P0": "raw layer/vessel probability threshold at 0.5",
        "P1": "main layer component plus enclosed-hole fill",
        "P2b": "P1 followed by fixed lower-boundary regularization (smoothness=2,max_displacement=8 px)",
        "P3": "P0 vessel intersected with P2b final predicted layer; no vessel component pruning",
    }, output / "postprocess_config.json")
    missing = registry.loc[registry["status"] != "ready", ["alias", "status"]]
    (output / "missing_and_issues.md").write_text(
        "# Missing and issues\n\n" +
        (missing.to_markdown(index=False) if not missing.empty else "All configured runs are ready.\n") +
        "\n\n- Manifest protocol mismatch, unresolved identities and missing labels must remain explicit.\n"
        "- Full inference, Excel and gallery are generated only after checkpoints load successfully on the cloud.\n",
        encoding="utf-8",
    )
    (output / "run_log.txt").write_text(
        f"{datetime.now().isoformat()} dry-run inventory and checkpoint registry completed\n",
        encoding="utf-8",
    )


def make_scope_manifest(source: Path, output: Path, include_test: bool,
                        root: Path | None = None,
                        test_linked_paths: set[str] | None = None) -> pd.DataFrame:
    table = pd.read_csv(source, dtype=str).fillna("")
    permitted = (
        table[table["split"].str.lower() == "test"].copy()
        if include_test else table[table["split"].str.lower() != "test"].copy()
    )
    if not include_test and test_linked_paths and root is not None:
        resolved = permitted["image_path"].map(
            lambda value: os.path.normcase(str(resolve_path(value, root)))
        )
        permitted = permitted.loc[~resolved.isin(test_linked_paths)].copy()
    permitted.insert(permitted.columns.get_loc("split"), "original_split", permitted["split"])
    permitted["split"] = "sealed_test" if include_test else "non_test_inference"
    permitted.to_csv(output, index=False, encoding="utf-8-sig")
    return permitted


def load_model(config: Dict[str, Any], checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, Dict[str, Any]]:
    model = build_model(config).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, {"epoch": state.get("epoch"), "best_metric": state.get("best_metric"), "state": "model", "strict": True}


def loader(config: Dict[str, Any], manifest: Path, split: str, root: Path,
           batch_size: int, workers: int) -> DataLoader:
    dataset = OCTManifestDataset(manifest, split=split, transform=_make_transform(config, False),
                                 sample_repeat=False, root=root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                      pin_memory=True)


def write_gray16(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.round(np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)
    success, encoded = cv2.imencode(".png", array)
    if not success:
        raise RuntimeError(f"Could not encode 16-bit PNG: {path}")
    encoded.tofile(str(path))


def restore(array: np.ndarray, valid: np.ndarray, height: int, width: int,
            is_mask: bool = False) -> np.ndarray:
    coordinates = np.argwhere(valid)
    if coordinates.size:
        y0, x0 = coordinates.min(axis=0); y1, x1 = coordinates.max(axis=0) + 1
        array = array[y0:y1, x0:x1]
    if array.shape != (height, width):
        array = cv2.resize(array.astype(np.float32), (width, height),
                           interpolation=cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR)
    return array > 0.5 if is_mask else array


@torch.inference_mode()
def predict_only(model: torch.nn.Module, data_loader: DataLoader, device: torch.device,
                 output: Path, report_root: Path, project_root: Path, alias: str,
                 stage: str, save_all_float: bool, resume: bool) -> pd.DataFrame:
    rows = []
    for batch in data_loader:
        result = model(batch["image"].to(device, non_blocking=True),
                       return_features=False, return_auxiliary=False)
        denoised = result["denoised"].cpu().numpy()
        denoised_raw = result["denoised_raw"].cpu().numpy()
        layer_prob = result["layer_prob"].cpu().numpy()
        vessel_prob = result["vessel_prob"].cpu().numpy()
        for index in range(len(batch["sample_id"])):
            dataset, split = str(batch["dataset"][index]), str(batch["source_split"][index])
            group, sample = str(batch["group_id"][index]), str(batch["sample_id"][index])
            sample_root = output / alias / dataset / split / group / sample
            final_path = sample_root / ("denoised_clipped_u16.png" if stage == "denoise" else "vessel_mask_p3.png")
            valid_canvas = batch["valid_mask"][index, 0].numpy() > 0.5
            height, width = int(batch["original_height"][index]), int(batch["original_width"][index])
            if not (resume and final_path.is_file()):
                noisy = restore(batch["image"][index, 0].numpy(), valid_canvas, height, width)
                den = restore(denoised[index, 0], valid_canvas, height, width)
                den_raw = restore(denoised_raw[index, 0], valid_canvas, height, width)
                p_layer = restore(layer_prob[index, 0], valid_canvas, height, width)
                p_vessel = restore(vessel_prob[index, 0], valid_canvas, height, width)
                valid = np.ones((height, width), dtype=bool)
                write_gray(sample_root / "noisy.png", noisy)
                write_gray16(sample_root / "denoised_clipped_u16.png", den)
                if stage != "denoise":
                    layer_p0, vessel_p0 = p_layer >= 0.5, p_vessel >= 0.5
                    layer_p1, _ = clean_layer_mask(layer_p0, valid)
                    layer_p2a, _ = regularize_lower_boundary(layer_p1, valid, smoothness=0.0)
                    layer_p2b, _ = regularize_lower_boundary(layer_p2a, valid, smoothness=2.0, max_displacement=8)
                    vessel_p3, _ = hard_contain_vessel(vessel_p0, layer_p2b, valid)
                    write_gray(sample_root / "layer_mask_p0.png", layer_p0.astype(np.float32))
                    write_gray(sample_root / "vessel_mask_p0.png", vessel_p0.astype(np.float32))
                    write_gray(sample_root / "layer_mask_p2a.png", layer_p2a.astype(np.float32))
                    write_gray(sample_root / "layer_mask_final_p2b.png", layer_p2b.astype(np.float32))
                    write_gray(sample_root / "vessel_mask_p3.png", vessel_p3.astype(np.float32))
                if split == "val" or save_all_float:
                    arrays = {
                        "denoised_raw": den_raw.astype(np.float32),
                        "denoised_clipped": den.astype(np.float32),
                        "valid_mask": valid,
                    }
                    if stage != "denoise":
                        arrays.update(layer_probability=p_layer.astype(np.float32),
                                      vessel_probability=p_vessel.astype(np.float32))
                    np.savez_compressed(sample_root / "raw_outputs_float32.npz", **arrays)
            rows.append({
                "experiment": alias, "dataset": dataset, "split": split, "group_id": group,
                "sample_id": sample,
                "source_image": (
                    Path(str(batch["original_path"][index])).relative_to(project_root).as_posix()
                    if Path(str(batch["original_path"][index])).is_relative_to(project_root)
                    else "external_path_redacted"
                ),
                "denoised": (sample_root / "denoised_clipped_u16.png").relative_to(report_root).as_posix(),
                "layer_p0": "" if stage == "denoise" else (sample_root / "layer_mask_p0.png").relative_to(report_root).as_posix(),
                "vessel_p0": "" if stage == "denoise" else (sample_root / "vessel_mask_p0.png").relative_to(report_root).as_posix(),
                "layer_p2a": "" if stage == "denoise" else (sample_root / "layer_mask_p2a.png").relative_to(report_root).as_posix(),
                "layer_final": "" if stage == "denoise" else (sample_root / "layer_mask_final_p2b.png").relative_to(report_root).as_posix(),
                "vessel_p3": "" if stage == "denoise" else (sample_root / "vessel_mask_p3.png").relative_to(report_root).as_posix(),
                "threshold_layer": 0.5 if stage != "denoise" else None,
                "threshold_vessel": 0.5 if stage != "denoise" else None,
                "forward_grid": "training_grid", "export_grid": "restored_original_grid",
            })
    return pd.DataFrame(rows)


def _relative_paths(table: pd.DataFrame, root: Path) -> pd.DataFrame:
    for column in [name for name in table.columns if name.endswith("_path") or name == "original_path"]:
        table[column] = table[column].map(
            lambda value: (
                Path(str(value)).relative_to(root).as_posix()
                if str(value).strip() and Path(str(value)).is_absolute() and Path(str(value)).is_relative_to(root)
                else ("external_path_redacted" if str(value).strip() and Path(str(value)).is_absolute() else value)
            )
        )
    return table


def aggregate_outputs(output: Path, registry: pd.DataFrame, root: Path) -> None:
    frame_parts, group_parts, summaries = [], [], []
    for item in registry.to_dict("records"):
        alias = item["alias"]
        directory = output / "validation" / alias
        frame_path, group_path, summary_path = directory / "frame_metrics.csv", directory / "group_metrics.csv", directory / "summary.json"
        if frame_path.is_file():
            part = _relative_paths(pd.read_csv(frame_path), root); part.insert(0, "experiment", alias); part.insert(0, "display_order", item["display_order"]); part["stage"] = item["stage"]; frame_parts.append(part)
        if group_path.is_file():
            part = pd.read_csv(group_path); part.insert(0, "experiment", alias); part.insert(0, "display_order", item["display_order"]); part["stage"] = item["stage"]; group_parts.append(part)
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8")); summaries.append({"display_order": item["display_order"], "experiment": alias, **{k:v for k,v in summary.items() if not isinstance(v,(dict,list))}})
    frames = pd.concat(frame_parts, ignore_index=True) if frame_parts else pd.DataFrame()
    groups = pd.concat(group_parts, ignore_index=True) if group_parts else pd.DataFrame()
    summary = pd.DataFrame(summaries).sort_values("display_order") if summaries else pd.DataFrame()
    segmentation_frames = frames[frames["stage"] == "segment"].copy() if not frames.empty else frames
    segmentation_groups = groups[groups["stage"] == "segment"].copy() if not groups.empty else groups
    segmentation_frames.to_csv(output / "segmentation_per_frame.csv", index=False, encoding="utf-8-sig")
    segmentation_groups.to_csv(output / "segmentation_per_group.csv", index=False, encoding="utf-8-sig")
    denoise_columns = [c for c in frames.columns if c.startswith(("psnr", "ssim", "mse", "rmse", "mae", "epi", "snr", "cnr", "layer_roi"))]
    id_columns = [c for c in ("display_order", "experiment", "stage", "sample_id", "group_id", "dataset") if c in frames]
    denoise_frames = frames.loc[frames.get("psnr", pd.Series(index=frames.index, dtype=float)).notna(), id_columns + denoise_columns] if not frames.empty else frames
    denoise_frames.to_csv(output / "denoise_per_frame.csv", index=False, encoding="utf-8-sig")
    denoise_group_columns = [c for c in groups.columns if c.startswith(("psnr", "ssim", "mse", "rmse", "mae", "epi", "snr", "cnr", "layer_roi"))]
    group_ids = [c for c in ("display_order", "experiment", "stage", "group_id", "dataset", "n_evaluated_frames") if c in groups]
    denoise_groups = groups.loc[groups.get("psnr", pd.Series(index=groups.index, dtype=float)).notna(), group_ids + denoise_group_columns] if not groups.empty else groups
    denoise_groups.to_csv(output / "denoise_per_group.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(output / "validation_summary_by_experiment.csv", index=False, encoding="utf-8-sig")
    if not summary.empty:
        denoise_cols = [c for c in summary if c.startswith(("psnr", "ssim", "rmse", "epi", "snr", "cnr", "layer_roi"))]
        summary[["display_order", "experiment", "n_groups", "n_frames", *denoise_cols]].to_csv(output / "denoise_summary.csv", index=False, encoding="utf-8-sig")
        raw_cols = [c for c in summary if c.startswith(("layer_", "vessel_", "pred_layer"))]
        summary[["display_order", "experiment", "n_groups", "n_frames", *raw_cols]].to_csv(output / "segmentation_raw_summary.csv", index=False, encoding="utf-8-sig")
        processed_cols = [c for c in summary if c.startswith(("p1_", "p2_", "p3_"))]
        summary[["display_order", "experiment", "n_groups", "n_frames", *processed_cols]].to_csv(output / "segmentation_processed_summary.csv", index=False, encoding="utf-8-sig")
        postprocess_rows = []
        for row in summary.to_dict("records"):
            if row.get("experiment") == "S1-Denoise":
                continue
            postprocess_rows.append({
                "display_order": row["display_order"], "experiment": row["experiment"],
                "layer_p0_dice": row.get("p0_layer_dice", row.get("layer_dice")),
                "layer_p2b_dice": row.get("p2_layer_dice"),
                "layer_dice_delta": (
                    row.get("p2_layer_dice") - row.get("p0_layer_dice")
                    if pd.notna(row.get("p2_layer_dice")) and pd.notna(row.get("p0_layer_dice")) else None
                ),
                "vessel_p0_dice": row.get("p0_vessel_dice", row.get("vessel_dice")),
                "vessel_p3_dice": row.get("p3_vessel_dice"),
                "vessel_dice_delta": (
                    row.get("p3_vessel_dice") - row.get("p0_vessel_dice")
                    if pd.notna(row.get("p3_vessel_dice")) and pd.notna(row.get("p0_vessel_dice")) else None
                ),
                "p3_removed_tp": row.get("p3_removed_tp"), "p3_removed_fp": row.get("p3_removed_fp"),
            })
        pd.DataFrame(postprocess_rows).sort_values("display_order").to_csv(
            output / "postprocess_comparison.csv", index=False, encoding="utf-8-sig"
        )
    if not segmentation_groups.empty:
        comparisons = []
        pairs = [("E3b", "E3-current", "outside_bce"), ("E3b", "E3b-noD2S", "d2s_retrained")]
        numeric = segmentation_groups.select_dtypes(include=[np.number]).columns.difference(["display_order"])
        for candidate, reference, contrast in pairs:
            left = segmentation_groups[segmentation_groups["experiment"] == candidate]
            right = segmentation_groups[segmentation_groups["experiment"] == reference]
            merged = left.merge(right, on=["dataset", "group_id"], suffixes=("_candidate", "_reference"))
            for row in merged.to_dict("records"):
                for metric in numeric:
                    a, b = row.get(f"{metric}_candidate"), row.get(f"{metric}_reference")
                    if pd.notna(a) and pd.notna(b):
                        comparisons.append({"contrast": contrast, "candidate": candidate, "reference": reference,
                                            "dataset": row["dataset"], "group_id": row["group_id"],
                                            "metric": metric, "candidate_value": a, "reference_value": b,
                                            "paired_difference": a - b})
        pd.DataFrame(comparisons).to_csv(output / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
        metric_columns = [c for c in segmentation_frames.select_dtypes(include=[np.number]).columns if c != "display_order"]
        long = segmentation_frames.melt(
            id_vars=[c for c in ("display_order", "experiment", "sample_id", "group_id", "dataset") if c in segmentation_frames],
            value_vars=metric_columns, var_name="metric", value_name="value",
        )
        long.to_csv(output / "metrics_long.csv", index=False, encoding="utf-8-sig")


def write_denoise_drift(output: Path) -> None:
    prediction_root = output / "predictions"
    reference_root = prediction_root / "S1-Denoise"
    rows = []
    if not reference_root.is_dir():
        pd.DataFrame(rows).to_csv(output / "denoise_drift.csv", index=False, encoding="utf-8-sig")
        return
    references = {
        path.relative_to(reference_root): path
        for path in reference_root.glob("*/val/*/*/raw_outputs_float32.npz")
    }
    for alias in ("E1-current", "E3-current", "E3b", "E3b-noD2S"):
        candidate_root = prediction_root / alias
        for relative, reference in references.items():
            candidate = candidate_root / relative
            if not candidate.is_file():
                rows.append({"experiment": alias, "sample": relative.as_posix(), "status": "missing_candidate"})
                continue
            with np.load(reference) as ref_data, np.load(candidate) as cand_data:
                difference = np.abs(
                    ref_data["denoised_clipped"].astype(np.float64)
                    - cand_data["denoised_clipped"].astype(np.float64)
                )
            rows.append({"experiment": alias, "sample": relative.as_posix(), "status": "compared",
                         "max_abs": float(difference.max()), "mean_abs": float(difference.mean()),
                         "exact_equal": bool(np.array_equal(difference, np.zeros_like(difference)))})
    pd.DataFrame(rows).to_csv(output / "denoise_drift.csv", index=False, encoding="utf-8-sig")


def write_validation_gallery(output: Path) -> None:
    frame_path = output / "segmentation_per_frame.csv"
    if not frame_path.is_file():
        return
    frames = pd.read_csv(frame_path, dtype=str).fillna("")
    samples = frames[["dataset", "group_id", "sample_id"]].drop_duplicates().sort_values(
        ["group_id", "sample_id"]
    )
    aliases = [alias for _, alias, _, stage in RUNS if stage == "segment"]
    sections = []
    for row in samples.itertuples(index=False):
        cells = []
        for alias in aliases:
            base = output / "validation" / alias / "predictions" / row.dataset
            paths = [
                ("denoised", base / f"{row.sample_id}_denoised.png"),
                ("layer P0", base / f"{row.sample_id}_layer_mask.png"),
                ("vessel P0", base / f"{row.sample_id}_vessel_mask.png"),
                ("layer P2b", base / f"{row.sample_id}_p2_layer_mask.png"),
                ("vessel P3", base / f"{row.sample_id}_p3_vessel_mask.png"),
                ("error", base / f"{row.sample_id}_error_overlay.png"),
            ]
            images = "".join(
                f'<figure><img loading="lazy" src="{path.relative_to(output).as_posix()}"><figcaption>{label}</figcaption></figure>'
                for label, path in paths if path.is_file()
            )
            cells.append(f"<td><h4>{alias}</h4><div class='images'>{images}</div></td>")
        sections.append(f"<tr><th>{row.group_id}<br>{row.sample_id}</th>{''.join(cells)}</tr>")
    html = """<!doctype html><meta charset='utf-8'><title>Stage 1/2 validation gallery</title>
<style>body{font-family:Arial;margin:1rem}table{border-collapse:collapse}td,th{border:1px solid #bbb;vertical-align:top;padding:.4rem}.images{display:grid;grid-template-columns:repeat(2,180px);gap:.3rem}figure{margin:0}img{width:180px;max-height:180px;object-fit:contain;background:#111}figcaption{font-size:11px}</style>
<h1>Complete validation gallery</h1><p>Validation only; fixed frame order; no test images.</p><table>""" + "".join(sections) + "</table>"
    (output / "gallery.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output).expanduser().resolve() if args.output else root / "runs" / "reports" / f"stage12_validation_{timestamp}"
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selections = load_runs(args.run_selection)
    registry = build_registry(root, selections)
    manifest = resolve_path(args.all_manifest, root)
    if manifest is None or not manifest.is_file():
        raise FileNotFoundError(f"Missing all-data manifest: {manifest}")
    segmentation_manifest = resolve_path(args.segmentation_manifest, root)
    if segmentation_manifest is None or not segmentation_manifest.is_file():
        raise FileNotFoundError(f"Missing segmentation manifest: {segmentation_manifest}")
    inventory, links = build_inventory(root, manifest, args.skip_input_hashes)
    write_dry_run_files(output, registry, inventory, links, manifest, segmentation_manifest, root)
    run_selection = [{k: row[k] for k in ("display_order", "alias", "run_dir", "stage")} for row in selections]
    write_json(run_selection, output / "run_selection.json")
    if args.dry_run:
        write_json({"tool_version": TOOL_VERSION, "dry_run": True,
                    "created_at": datetime.now().isoformat(), "test_images_opened": False,
                    "checkpoint_ready": int((registry["status"] == "ready").sum())},
                   output / "provenance.json")
        print(output); return
    ready = registry[registry["status"] == "ready"]
    if ready.empty:
        raise RuntimeError("No run has both resolved_config.yaml and best.pth; dry-run files were written")
    device = get_device(args.device)
    load_audit, prediction_indices = [], []
    non_test_manifest = output / "non_test_inference_manifest.csv"
    test_linked_paths = set(links.loc[links["split"] == "test", "physical_path"].astype(str))
    non_test = make_scope_manifest(
        manifest, non_test_manifest, include_test=False, root=root,
        test_linked_paths=test_linked_paths,
    )
    for item in ready.to_dict("records"):
        alias = item["alias"]
        run_dir = root / item["run_path"]
        config = load_config(run_dir / "resolved_config.yaml")
        model, audit = load_model(config, run_dir / "best.pth", device)
        load_audit.append({"alias": alias, "checkpoint_sha256": item["checkpoint_sha256"], **audit})
        validation_loader = loader(config, resolve_path(config["data"]["manifest"], root), "val", root, args.batch_size, args.num_workers)
        tasks = ("denoise",) if item["stage"] == "denoise" else ("denoise", "layer", "vessel")
        evaluate_model(model, validation_loader, device, output_dir=output / "validation" / alias,
                       layer_threshold=0.5, vessel_threshold=0.5, save_predictions=True,
                       stage=item["stage"], input_normalization=config["data"].get("normalization", "fixed"),
                       tasks=tasks, postprocess_modes=("p0", "p1", "p2", "p3"), restore_original_geometry=True)
        if args.full_non_test and not args.validation_only:
            inference_loader = loader(config, non_test_manifest, "non_test_inference", root, args.batch_size, args.num_workers)
            prediction_indices.append(predict_only(
                model, inference_loader, device, output / "predictions", output,
                root, alias, item["stage"], args.save_all_float, args.resume,
            ))
        if args.include_sealed_test_export:
            sealed_manifest = output / "sealed_test_manifest.csv"
            sealed = make_scope_manifest(manifest, sealed_manifest, include_test=True)
            if not sealed.empty:
                sealed_loader = loader(config, sealed_manifest, "sealed_test", root, args.batch_size, args.num_workers)
                # Pure forward only: no metrics, ranking, calibration or gallery.
                predict_only(model, sealed_loader, device, output / "sealed_test_predictions",
                             output, root, alias, item["stage"], args.save_all_float, args.resume)
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
    write_json(load_audit, output / "checkpoint_load_audit.json")
    predictions_index = pd.concat(prediction_indices, ignore_index=True) if prediction_indices else pd.DataFrame()
    predictions_index.to_csv(output / "predictions_index.csv", index=False, encoding="utf-8-sig")
    aggregate_outputs(output, registry, root)
    write_denoise_drift(output)
    write_validation_gallery(output)
    write_json({"tool_version": TOOL_VERSION, "created_at": datetime.now().isoformat(),
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "device": str(device), "fixed_thresholds": {"layer": 0.5, "vessel": 0.5},
                "test_used_for_metrics_or_selection": False, "n_manifest_rows": len(pd.read_csv(manifest)),
                "n_non_test_rows": len(non_test)}, output / "provenance.json")
    print(output)


if __name__ == "__main__":
    main()
