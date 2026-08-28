from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw, ImageFont


CURRENT_RUNS = {
    "E3b": "stage2_segment_roi_outside_fold0",
    "E3-current": "stage2_segment_roi_current_fold0",
    "E3b-no-D2S": "stage2_segment_roi_outside_no_d2s_fold0",
    "E1-current": "stage2_segment_safe_current_fold0",
}
ARCHIVE_PREFIX = "runs/current/"
METRIC_VERSION = "sabids-evaluator-v0-20260828"
POSTPROCESS_VERSION = "p0-p3-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a strict records-only, validation-only SABIDS experiment archive."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--source-archive", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--legacy-history", action="append", default=[], metavar="ALIAS=CSV",
        help="Optional historical training CSV; kept separate from current protocol.",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--strict-val-only", action="store_true")
    scope.add_argument(
        "--include-test-results", action="store_true",
        help="Archive existing test result files without using them for selection, calibration, or comparisons.",
    )
    parser.add_argument("--records-only", action="store_true", default=True)
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SafeArchive:
    def __init__(self, path: Path, allow_test_results: bool = False):
        self.path = path
        self.allow_test_results = allow_test_results
        self.tar = tarfile.open(path, "r:gz")
        self.names = set(self.tar.getnames())
        forbidden = [name for name in self.names if self._is_test_result(name)]
        if forbidden and not allow_test_results:
            self.tar.close()
            raise RuntimeError(f"Archive contains forbidden test result members: {forbidden[:5]}")

    @staticmethod
    def _is_test_result(name: str) -> bool:
        parts = [part.lower() for part in PurePosixPath(name).parts]
        return any(part in {"test", "tests", "test_results", "predictions_test"} for part in parts)

    def read(self, name: str) -> bytes:
        if self._is_test_result(name) and not self.allow_test_results:
            raise RuntimeError(f"Refusing test member: {name}")
        member = self.tar.getmember(name)
        if not member.isfile():
            raise ValueError(f"Not a regular file: {name}")
        handle = self.tar.extractfile(member)
        if handle is None:
            raise OSError(f"Cannot read archive member: {name}")
        return handle.read()

    def json(self, name: str) -> Dict[str, Any]:
        return json.loads(self.read(name))

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(self.read(name)))

    def yaml(self, name: str) -> Dict[str, Any]:
        return yaml.safe_load(self.read(name))

    def close(self) -> None:
        self.tar.close()


def export_test_results(archive: SafeArchive, output: Path) -> pd.DataFrame:
    """Copy existing test artifacts verbatim and index them; never aggregate or rank test metrics."""
    rows: list[Dict[str, Any]] = []
    destination_root = output / "test_results" / "source_members"
    excluded_suffixes = {".pth", ".pt", ".ckpt"}
    for name in sorted(archive.names):
        if not archive._is_test_result(name):
            continue
        member = archive.tar.getmember(name)
        if not member.isfile() or PurePosixPath(name).suffix.lower() in excluded_suffixes:
            continue
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe test result member: {name}")
        payload = archive.read(name)
        destination = destination_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        rows.append({
            "scope": "test_archival_only",
            "source_member": name,
            "exported_path": destination.relative_to(output).as_posix(),
            "bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "selection_or_calibration_use": "forbidden",
        })
    table = pd.DataFrame(rows, columns=[
        "scope", "source_member", "exported_path", "bytes", "sha256",
        "selection_or_calibration_use",
    ])
    table.to_csv(output / "test_results_index.csv", index=False, encoding="utf-8-sig")
    return table


def archive_member(run_dir: str, filename: str) -> str:
    return f"{ARCHIVE_PREFIX}{run_dir}/{filename}"


def clean_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value


def write_json(data: Any, path: Path) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=clean_scalar), encoding="utf-8")


def safe_relative_path(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.replace("\\", "/")
    for prefix in ("/mnt/SABIDS-Net/", "E:/1-脉络膜/OCT降噪/SABIDS-Net/SABIDS-Net/"):
        if normalized.startswith(prefix):
            return normalized[len(prefix):]
    return Path(normalized).name if re.match(r"^[A-Za-z]:/", normalized) else normalized


def git_info(root: Path) -> tuple[str, str]:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], cwd=root, text=True).strip()
        return commit, status
    except Exception:
        return "unknown", "unknown"


def parse_legacy(specifications: Iterable[str]) -> list[tuple[str, Path]]:
    result = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Expected ALIAS=CSV, got {specification!r}")
        alias, raw_path = specification.split("=", 1)
        result.append((alias, Path(raw_path).expanduser().resolve()))
    return result


def best_row(history: pd.DataFrame, metadata: Dict[str, Any]) -> tuple[pd.Series, str]:
    epoch = metadata.get("best_epoch")
    if epoch is not None and "epoch" in history and bool((history["epoch"] == epoch).any()):
        return history.loc[history["epoch"] == epoch].iloc[0], "metadata_verified"
    monitor = "val_vessel_soft_dice"
    if monitor in history and history[monitor].notna().any():
        return history.loc[history[monitor].idxmax()], "history_inferred"
    return history.iloc[-1], "last_row_fallback"


def selected_metrics(row: pd.Series) -> Dict[str, Any]:
    names = [
        "val_layer_dice", "val_layer_precision", "val_layer_recall",
        "val_vessel_dice", "val_vessel_soft_dice", "val_vessel_precision",
        "val_vessel_recall", "val_vessel_roi_dice",
        "val_vessel_outside_gt_layer_fraction", "val_vessel_area_fraction_pred",
        "val_vessel_area_fraction_true", "val_vessel_area_fraction_mae",
        "val_pred_layer_vessel_dice", "val_psnr", "val_denoise_probe_max_abs_diff",
        "train_total", "train_layer", "train_vessel", "train_vessel_outside",
        "train_vessel_outside_weighted", "train_optimizer_steps",
        "train_unique_groups_seen", "train_d2s_gradient_norm",
        "train_d2s_scale_update_abs_mean",
        "val_d2s_disabled_vessel_soft_dice",
        "val_d2s_vessel_probability_mean_abs_change",
    ]
    return {name: clean_scalar(row[name]) for name in names if name in row and pd.notna(row[name])}


def metric_record(
    *, run_id: str, eval_id: str, alias: str, epoch: Any, split: str,
    aggregation: str, metric: str, value: Any, source_file: str,
    source_key: str, checkpoint: str = "", group_id: str = "",
    frame_id: str = "", input_hw: str = "512x512", evaluation_hw: str = "512x512",
    mode: str = "raw", vessel_threshold: Any = 0.5, layer_threshold: Any = 0.5,
    n_frames: Any = None, n_groups: Any = None, status: str = "observed",
) -> Dict[str, Any]:
    unit = "px" if any(token in metric for token in ("pixels", "mae", "hd95", "assd", "roughness", "displacement")) else "ratio"
    if metric in {"psnr", "psnr_noisy", "psnr_gain_db", "snr_gain_db", "snr_noisy_db", "snr_denoised_db", "layer_roi_psnr", "layer_roi_psnr_noisy"}:
        unit = "dB"
    return {
        "run_id": run_id, "eval_id": eval_id, "experiment_alias": alias,
        "stage": "segment", "seed": 42, "fold": 0,
        "checkpoint_sha256": checkpoint, "epoch": clean_scalar(epoch), "split": split,
        "dataset": "PKU37", "group_id": group_id, "frame_id": frame_id,
        "aggregation_level": aggregation, "metric": metric,
        "value": clean_scalar(value), "unit": unit, "input_hw": input_hw,
        "evaluation_hw": evaluation_hw, "spacing": "1 px (physical spacing unverified)",
        "mode": mode, "layer_threshold": clean_scalar(layer_threshold),
        "vessel_threshold": clean_scalar(vessel_threshold),
        "valid_mask_definition": "spatial valid AND annotation-valid where available",
        "postprocess_version": POSTPROCESS_VERSION if mode in {"p0", "p1", "p2", "p3"} else "none",
        "metric_version": METRIC_VERSION, "n_frames_total": clean_scalar(n_frames),
        "n_frames_evaluated": clean_scalar(n_frames), "n_groups_valid": clean_scalar(n_groups),
        "source_file": source_file, "source_row_or_key": source_key,
        "status": status, "missing_reason": "",
    }


def extract_current(archive: SafeArchive) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, pd.DataFrame]]:
    registry, training, histories = [], [], {}
    for alias, run_dir in CURRENT_RUNS.items():
        metadata_name = archive_member(run_dir, "run_metadata.json")
        history_name = archive_member(run_dir, "history.csv")
        config_name = archive_member(run_dir, "resolved_config.yaml")
        metadata, history, config = archive.json(metadata_name), archive.csv(history_name), archive.yaml(config_name)
        histories[alias] = history
        row, selection = best_row(history, metadata)
        run_id = f"current-f0-{run_dir}"
        train = config.get("train", {})
        data = config.get("data", {})
        registry.append({
            "experiment_alias": alias, "run_id": run_id, "eval_id": f"{run_id}-history-val",
            "role": "current_matched_protocol" if alias != "E1-current" else "current_multifactor_baseline",
            "stage": train.get("stage", "segment"), "fold": 0, "seed": config.get("seed", 42),
            "git_commit": metadata.get("git_commit", "unknown"),
            "status": "training_complete", "history_rows": len(history),
            "best_epoch": metadata.get("best_epoch"), "last_epoch": int(history["epoch"].iloc[-1]),
            "monitor": metadata.get("monitor", "unknown"), "selection_status": selection,
            "checkpoint_sha256": metadata.get("best_checkpoint_sha256", "unknown"),
            "manifest_sha256": metadata.get("manifest_sha256", "unknown"),
            "effective_split_sha256": metadata.get("effective_split_sha256", "unknown"),
            "label_raw_sha256": metadata.get("label_assets_raw_sha256", "unknown"),
            "label_decoded_sha256": metadata.get("label_assets_decoded_sha256", "unknown"),
            "initialization_sha256": metadata.get("initialization_checkpoint_sha256", "unknown"),
            "train_groups": len(metadata.get("effective_groups", {}).get("train", [])),
            "val_groups": len(metadata.get("effective_groups", {}).get("val", [])),
            "train_frames": metadata.get("rows_by_split", {}).get("train"),
            "val_frames": metadata.get("rows_by_split", {}).get("val"),
            "input_hw": "x".join(map(str, data.get("target_size", [None, None]))),
            "patient_identity_status": "unverified: patient_id duplicates group_id in V0 rows",
            "source_file": history_name,
        })
        training.append({
            "experiment_alias": alias, "run_id": run_id, "epoch": int(row["epoch"]),
            "selection_status": selection, "monitor": metadata.get("monitor"),
            "monitor_value": metadata.get("best_metric"), "last_epoch": int(history["epoch"].iloc[-1]),
            "early_stop_status": "evidence_present" if len(history) < int(train.get("epochs", len(history))) else "not_triggered_or_unknown",
            "batch_size": train.get("batch_size"), "gradient_accumulation_steps": train.get("gradient_accumulation_steps"),
            "samples_per_epoch": data.get("samples_per_epoch"), "learning_rate": train.get("learning_rate"),
            "checkpoint_sha256": metadata.get("best_checkpoint_sha256"),
            **selected_metrics(row),
        })
    return registry, training, histories


def extract_legacy(specifications: list[tuple[str, Path]]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, pd.DataFrame]]:
    registry, training, histories = [], [], {}
    for alias, path in specifications:
        if not path.is_file():
            registry.append({"experiment_alias": alias, "run_id": f"legacy-{alias}", "status": "missing", "source_file": path.name})
            continue
        history = pd.read_csv(path)
        histories[alias] = history
        row, selection = best_row(history, {})
        run_id = f"legacy-{alias}-{sha256_file(path)[:10]}"
        registry.append({
            "experiment_alias": alias, "run_id": run_id, "eval_id": f"{run_id}-history-val",
            "role": "historical_different_or_unknown_protocol", "stage": "segment", "fold": 0,
            "seed": None, "git_commit": "unknown", "status": "history_only",
            "history_rows": len(history), "best_epoch": int(row.get("epoch", 0)),
            "last_epoch": int(history["epoch"].iloc[-1]), "monitor": "vessel_soft_dice inferred",
            "selection_status": selection, "checkpoint_sha256": "unknown",
            "manifest_sha256": "unknown", "effective_split_sha256": "unknown",
            "label_raw_sha256": "unknown", "label_decoded_sha256": "unknown",
            "initialization_sha256": "unknown", "source_file": path.name,
        })
        training.append({
            "experiment_alias": alias, "run_id": run_id, "epoch": int(row.get("epoch", 0)),
            "selection_status": selection, "monitor": "val_vessel_soft_dice inferred",
            "monitor_value": clean_scalar(row.get("val_vessel_soft_dice")),
            "last_epoch": int(history["epoch"].iloc[-1]), "checkpoint_sha256": "unknown",
            **selected_metrics(row),
        })
    return registry, training, histories


def build_protocol_table(archive: SafeArchive) -> tuple[pd.DataFrame, Dict[str, Any]]:
    name = f"{ARCHIVE_PREFIX}stage2_protocol_comparison_fold0/comparison.json"
    source = archive.json(name)
    rows = []
    role = {
        "E3b": "self_reference", "E3_current": "single_factor: outside BCE",
        "E3b_no_D2S": "single_factor: D→S with necessary switches",
        "E1_current": "multifactor_baseline",
    }
    for raw_alias, comparison in source.get("comparisons", {}).items():
        alias = raw_alias.replace("_current", "-current").replace("E3b_no_D2S", "E3b-no-D2S")
        identity = comparison.get("identity", {})
        protocol = comparison.get("training_protocol", {})
        if alias == "E1-current":
            contrast = "multifactor_baseline"
        elif protocol.get("status") == "matched":
            contrast = "single_factor"
        elif protocol.get("status") == "unknown":
            contrast = "unknown"
        else:
            contrast = "different_protocol"
        rows.append({
            "reference": "E3b", "candidate": alias,
            "data_identity_match": identity.get("status", "unknown"),
            "training_contrast_status": contrast,
            "declared_role": role.get(raw_alias, "unknown"),
            "actual_differences": json.dumps(protocol.get("all_differences", {}), ensure_ascii=False),
            "unexpected_differences": json.dumps(protocol.get("unexpected_differences", {}), ensure_ascii=False),
            "evaluation_comparable": "unknown: only E3b has V0 exported; history metrics share 0.5/model coordinates",
            "patient_identity": "unverified",
            "source_file": name,
        })
    return pd.DataFrame(rows), source


def add_history_metrics(metrics: list[Dict[str, Any]], registry: list[Dict[str, Any]], training: list[Dict[str, Any]], histories: Dict[str, pd.DataFrame]) -> None:
    registry_by_alias = {row["experiment_alias"]: row for row in registry}
    training_by_alias = {row["experiment_alias"]: row for row in training}
    for alias, history in histories.items():
        if alias not in registry_by_alias or alias not in training_by_alias:
            continue
        info, selected = registry_by_alias[alias], training_by_alias[alias]
        epoch = selected.get("epoch")
        chosen = history.loc[history["epoch"] == epoch].iloc[0]
        for key, value in chosen.items():
            if not key.startswith("val_") or key.startswith("val_n_groups_") or pd.isna(value):
                continue
            metrics.append(metric_record(
                run_id=info["run_id"], eval_id=info.get("eval_id", ""), alias=alias,
                epoch=epoch, split="val", aggregation="group_macro_training_scale",
                metric=key[4:], value=value, source_file=str(info.get("source_file", "")),
                source_key=f"epoch={epoch};column={key}", checkpoint=str(info.get("checkpoint_sha256", "")),
                n_frames=info.get("val_frames"), n_groups=info.get("val_groups"), status=selected.get("selection_status", "observed"),
            ))


def extract_v0(archive: SafeArchive, output: Path, metrics: list[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = archive_member(CURRENT_RUNS["E3b"], "validity_v0")
    frame_name, group_name = f"{base}/frame_metrics.csv", f"{base}/group_metrics.csv"
    frame, group = archive.csv(frame_name), archive.csv(group_name)
    for table in (frame, group):
        table.insert(0, "split", "val")
        table.insert(0, "eval_id", "current-f0-e3b-v0-one-frame-original-geometry")
        table.insert(0, "run_id", f"current-f0-{CURRENT_RUNS['E3b']}")
        for column in table.columns:
            if column.endswith("_path"):
                table[column] = table[column].map(safe_relative_path)
    frame.to_csv(output / "frame_metrics.csv", index=False, encoding="utf-8-sig")
    group.to_csv(output / "group_metrics.csv", index=False, encoding="utf-8-sig")
    checkpoint = archive.json(archive_member(CURRENT_RUNS["E3b"], "run_metadata.json")).get("best_checkpoint_sha256", "")
    for aggregation, table, source in (("frame", frame, frame_name), ("group", group, group_name)):
        id_columns = {"run_id", "eval_id", "split", "sample_id", "group_id", "patient_id", "dataset", "scan_protocol"}
        for _, row in table.iterrows():
            for key, value in row.items():
                if key in id_columns or not isinstance(value, (int, float, np.number)) or pd.isna(value):
                    continue
                mode = key.split("_", 1)[0] if re.match(r"p[0-3]_", key) else ("soft_gate" if "soft_gate" in key else "raw")
                metrics.append(metric_record(
                    run_id=str(row["run_id"]), eval_id=str(row["eval_id"]), alias="E3b", epoch=52,
                    split="val", aggregation=aggregation, metric=key, value=value, source_file=source,
                    source_key=f"{row.get('sample_id', row.get('group_id'))};column={key}", checkpoint=checkpoint,
                    group_id=str(row.get("group_id", "")), frame_id=str(row.get("sample_id", "")),
                    input_hw="512x512", evaluation_hw="640x640", mode=mode,
                    vessel_threshold=0.35, layer_threshold=0.5,
                    n_frames=row.get("n_evaluated_frames", 1), n_groups=1,
                ))
    summary = archive.json(f"{base}/summary.json")
    post_metrics = [key for key in summary if key.startswith(("p0_", "p1_", "p2_", "p3_"))]
    post = pd.DataFrame([{"metric": key, "value": clean_scalar(summary[key]), "source": f"{base}/summary.json"} for key in sorted(post_metrics)])
    thresholds = []
    for mode, folder in (("raw", "threshold_calibration"), ("soft_gate", "threshold_calibration_soft_gate")):
        name = archive_member(CURRENT_RUNS["E3b"], f"{folder}/best_vessel_threshold_{mode}.json")
        item = archive.json(name)
        thresholds.append({"mode": mode, **item["best"], "selection_split": item["selection_split"],
                           "one_frame_per_group": item["one_frame_per_group"], "checkpoint_sha256": item["checkpoint_sha256"],
                           "coordinate_status": "512x512 model coordinates", "source_file": name})
    threshold_table = pd.DataFrame(thresholds)
    threshold_table.to_csv(output / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
    post.to_csv(output / "postprocess_comparison.csv", index=False, encoding="utf-8-sig")
    return frame, group, post, threshold_table


def _line_panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                title: str, series: list[tuple[str, list[float]]], colors: list[str]) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, fill="white", outline="#B8C2CC")
    draw.text((left + 10, top + 8), title, fill="#17212B")
    plot = (left + 44, top + 32, right - 14, bottom - 32)
    draw.line((plot[0], plot[3], plot[2], plot[3]), fill="#607080")
    draw.line((plot[0], plot[1], plot[0], plot[3]), fill="#607080")
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = plot[3] - int(tick * (plot[3] - plot[1]))
        draw.line((plot[0], y, plot[2], y), fill="#E5E9ED")
        draw.text((left + 4, y - 5), f"{tick:.2f}", fill="#607080")
    maximum = max((len(values) for _, values in series), default=1) - 1
    for index, (label, values) in enumerate(series):
        points = []
        for offset, value in enumerate(values):
            if not np.isfinite(value):
                continue
            x = plot[0] + int(offset / max(maximum, 1) * (plot[2] - plot[0]))
            y = plot[3] - int(np.clip(value, 0, 1) * (plot[3] - plot[1]))
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=colors[index], width=2)
        draw.text((left + 10 + (index % 2) * 165, bottom - 20 - (index // 2) * 12), label, fill=colors[index])


def plot_training(histories: Dict[str, pd.DataFrame], registry: list[Dict[str, Any]], figures: Path) -> None:
    aliases = [alias for alias in CURRENT_RUNS if alias in histories]
    specs = [
        ("val_vessel_soft_dice", "Validation vessel soft Dice"),
        ("val_vessel_dice", "Validation vessel Dice @0.5"),
        ("val_vessel_precision", "Validation vessel precision @0.5"),
        ("val_vessel_recall", "Validation vessel recall @0.5"),
    ]
    image = Image.new("RGB", (1400, 920), "#F3F5F7")
    draw = ImageDraw.Draw(image)
    draw.text((25, 15), "Current matched-data Stage 2 histories (validation, 512x512)", fill="#17212B")
    colors = ["#2678B2", "#E07A1F", "#2A9D67", "#C44E52"]
    boxes = [(25, 50, 690, 465), (710, 50, 1375, 465), (25, 485, 690, 900), (710, 485, 1375, 900)]
    for box, (column, title) in zip(boxes, specs):
        series = [(alias, histories[alias][column].astype(float).tolist()) for alias in aliases if column in histories[alias]]
        _line_panel(draw, box, title, series, colors)
    image.save(figures / "training_curves_current.png")


def _bar_chart(values: Dict[str, float], title: str, destination: Path,
               minimum: float = 0.0, maximum: float = 1.0) -> None:
    image = Image.new("RGB", (1000, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.text((25, 18), title, fill="#17212B")
    left, top, right, bottom = 70, 60, 970, 450
    draw.line((left, bottom, right, bottom), fill="#607080")
    draw.line((left, top, left, bottom), fill="#607080")
    width = (right - left) / max(len(values), 1)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"]
    for index, (label, value) in enumerate(values.items()):
        x0 = int(left + index * width + width * 0.18)
        x1 = int(left + (index + 1) * width - width * 0.18)
        height = (float(value) - minimum) / max(maximum - minimum, 1e-9) * (bottom - top)
        y0 = int(bottom - np.clip(height, 0, bottom - top))
        draw.rectangle((x0, y0, x1, bottom), fill=colors[index % len(colors)])
        draw.text((x0, y0 - 16), f"{value:.4f}", fill="#17212B")
        draw.text((x0, bottom + 8), label, fill="#17212B")
    image.save(destination)


def plot_v0(group: pd.DataFrame, post: pd.DataFrame, threshold: pd.DataFrame, figures: Path) -> None:
    paired = {}
    for _, row in group.iterrows():
        paired[f"{row['group_id']} noisy"] = float(row["psnr_noisy"]) / 50.0
        paired[f"{row['group_id']} den"] = float(row["psnr"]) / 50.0
    _bar_chart(paired, "PSNR paired by validation group (bar height normalized by 50 dB)", figures / "v0_group_and_thresholds.png")
    summary_values = {
        "Layer P0": float(group["p0_layer_dice"].mean()),
        "Layer P1": float(group["p1_layer_dice"].mean()),
        "Layer P2": float(group["p2_layer_dice"].mean()),
        "Vessel P0": float(group["p0_vessel_dice"].mean()),
        "Vessel P3": float(group["p3_vessel_dice"].mean()),
    }
    _bar_chart(summary_values, "P0-P3 validation change (3 groups, one frame/group, 640x640)", figures / "postprocess_dice.png", 0.65, 1.0)


def extract_images_and_gallery(archive: SafeArchive, frame: pd.DataFrame, output: Path) -> pd.DataFrame:
    figures = output / "qualitative"
    figures.mkdir(parents=True, exist_ok=True)
    base = archive_member(CURRENT_RUNS["E3b"], "validity_v0/predictions/PKU37")
    suffixes = ["noisy", "denoised", "layer_gt", "layer_mask", "p1_layer_mask", "p2_layer_mask",
                "vessel_gt", "vessel_mask", "p3_vessel_mask", "vessel_fp", "vessel_fn", "error_overlay"]
    rows, html_sections = [], []
    font = ImageFont.load_default()
    for _, metric_row in frame.iterrows():
        sample = str(metric_row["sample_id"])
        tiles = []
        sample_rows = []
        for suffix in suffixes:
            member = f"{base}/{sample}_{suffix}.png"
            if member not in archive.names:
                sample_rows.append((suffix, "", "missing"))
                continue
            data = archive.read(member)
            destination = figures / f"{sample}_{suffix}.png"
            destination.write_bytes(data)
            image = Image.open(io.BytesIO(data)).convert("RGB")
            image.thumbnail((260, 260))
            tile = Image.new("RGB", (280, 300), "white")
            tile.paste(image, ((280 - image.width) // 2, 28))
            ImageDraw.Draw(tile).text((8, 8), suffix, fill="black", font=font)
            tiles.append(tile)
            sample_rows.append((suffix, destination.relative_to(output).as_posix(), "observed"))
        columns = 4
        contact = Image.new("RGB", (columns * 280, int(np.ceil(len(tiles) / columns)) * 300), "#EEEEEE")
        for index, tile in enumerate(tiles):
            contact.paste(tile, ((index % columns) * 280, (index // columns) * 300))
        contact_path = figures / f"{sample}_contact.png"
        contact.save(contact_path)
        html_sections.append(f"<section><h2>{html.escape(sample)}</h2><p>group={html.escape(str(metric_row['group_id']))}; frame=f01; checkpoint=E3b epoch52; raw threshold=0.35; input=512×512; evaluation=640×640.</p><img class='contact' src='{contact_path.relative_to(output).as_posix()}'></section>")
        for suffix, relative, status in sample_rows:
            rows.append({
                "run_id": metric_row["run_id"], "eval_id": metric_row["eval_id"],
                "group_id": metric_row["group_id"], "frame_id": sample,
                "image_role": suffix, "source_image": f"{base}/{sample}_{suffix}.png",
                "display_image": relative, "crop": "full original geometry",
                "source_hw": "640x640", "mode": "P0/P1/P2/P3 as suffix",
                "selection_reason": "all validation groups; fixed first frame",
                "validity_status": status,
            })
    gallery = """<!doctype html><meta charset='utf-8'><title>SABIDS validation gallery</title>
<style>body{font-family:Arial,sans-serif;margin:24px;color:#17212b}section{margin:0 0 40px}.contact{max-width:100%;border:1px solid #bbb}p{color:#445}</style>
<h1>SABIDS validation qualitative gallery</h1><p>Records-only. Three independent validation groups, one fixed frame per group. No test data. Clean PNG was not exported and is shown as missing in the index.</p>""" + "\n".join(html_sections)
    (output / "gallery.html").write_text(gallery, encoding="utf-8")
    result = pd.DataFrame(rows)
    result.to_csv(output / "qualitative_index.csv", index=False, encoding="utf-8-sig")
    return result


def methods_text() -> str:
    return """# 当前真实方法与证据状态

## 网络图

```text
noisy B-scan
  └─ shared stem + four-scale encoder
       ├─ task adapters → denoise decoder → residual head → noisy - 0.5*tanh(residual)
       ├─ task adapters → layer decoder   → layer + two-channel boundary heads
       └─ task adapters → vessel decoder  → vessel head
             ↕ UGBI at levels 3/2/1 (only enabled directions act)
```

分割不是对 `denoised` 图像执行第二次显式 forward；三条路径共享 noisy 编码特征，D→S 通过 UGBI 将 restoration context 注入层/血管特征。源码：`sabids/models/sabids_net.py`、`sabids/models/ugbi.py`。

| 条目 | proposed | implemented | run_evidence | validation_supported | 说明/源码 |
|---|---|---|---|---|---|
| 共享编码器、三任务适配器/解码器 | yes | yes | yes | Stage 2 only | `sabids/models/sabids_net.py` |
| 残差降噪 | yes | yes | Stage 1初始化及V0输出 | 固定3组支持 | `denoised_raw=image-residual`，展示输出clamp |
| D→S UGBI | yes | yes | E3b及完整no-D2S训练 | 未显示稳定优势 | no-D2S hard Dice接近E3b；需seed/完整配对评价 |
| S→D UGBI | yes | yes | 当前四Stage 2关闭 | no | 不确定性置信主要作用于S→D anatomy融合，当前结果不能证明其有效 |
| E3b ROI BCE+Dice | yes | yes | yes | 当前validation支持 | GT layer ROI内逐图BCE+Dice |
| outside BCE | yes | yes | E3b/E3-current单因素 | 支持抑制层外FP | FP32 softplus(logit)，空outside跳过 |
| containment | yes | yes | yes | 与outside共同存在 | 有GT层用GT，否则detach预测层；不能单独防整层血管 |
| layer boundary loss | yes | yes | yes | 有layer指标 | boundary_weight=0.2为层loss内部边界BCE权重，不是独立总loss权重 |
| RMAC | yes | yes | 当前Stage 2权重为0 | no completed Joint evidence | `sabids/losses/rmac.py` |
| memory-safe repeat/clean stop-gradient | yes | yes | 仅smoke/未来Joint | no | `sabids/engine/trainer.py` |
| EMA+暗腔双源伪标签 | yes | yes | Stage 5尚无正式结果 | no | `sabids/losses/pseudo.py` |
| P1/P2/P3后处理 | later plan | yes | V0 3固定帧 | limited | P3严格相交；越界=0是构造性质 |

## Stage 2 冻结与梯度

E3b/E3-current冻结去噪分支和完整共享上游，S→D关闭；D→S来源被detach，仅训练接收侧交互和分割路径。no-D2S关闭该交互。E1-current是复合监督整体基线，不是单因素loss对照。Stage 2 reconstruction、RMAC、pseudo均不在active集合中。

## 损失与评价边界

Stage 1 reconstruction = Charbonnier + 0.2 MS-SSIM loss + 0.1 wavelet + 0.1边缘项，另有residual L1。E3b分割主目标为layer、ROI vessel、outside与containment；不同loss定义的total不能横向排名。阈值校准在512模型坐标，而V0恢复到640原图坐标；二者不可视为已完全复现的一致评价。
"""


def write_narrative(output: Path, registry: pd.DataFrame, training: pd.DataFrame, group: pd.DataFrame, threshold: pd.DataFrame) -> None:
    e3b = training.loc[training["experiment_alias"] == "E3b"].iloc[0]
    removed_tp, removed_fp = group["p3_removed_tp"].sum(), group["p3_removed_fp"].sum()
    readme = f"""# SABIDS 实验记录归档

- 范围：records-only、validation-only；未运行训练、推理或test。
- 当前正式运行：E3b、E3-current、E3b-no-D2S、E1-current；历史E0–E3仅按找到的CSV单列。
- 当前四组使用相同manifest/effective split/标签raw+decoded/Stage 1初始化指纹；E3b vs E3-current、E3b vs no-D2S为声明的单因素对照，E1-current为多因素基线。
- E3b由metadata确认best epoch={int(e3b['epoch'])}，history固定0.5 vessel Dice={e3b.get('val_vessel_dice', float('nan')):.6f}。
- V0仅3个validation group、每组f01一帧，输入512×512、评价640×640；不是完整141帧验证。
- raw阈值校准最佳={float(threshold.loc[threshold['mode']=='raw','threshold'].iloc[0]):.3f}（512坐标）；V0使用0.35（640坐标），两者评价条件不一致。
- P3三帧共删除TP={removed_tp:.0f}、FP={removed_fp:.0f}；收益限于固定帧，预测层外为0是算法构造。

## 复现

```bash
python tools/export_experiment_record.py --project-root . \
  --source-archive stage2_validity_fold0_20260828.tar.gz \
  --output reports/experiment_records/<timestamp> \
  --legacy-history E0=stage2_overfit_safe_fold0.csv \
  --legacy-history E1-old=stage2_segment_safe_fold0.csv \
  --legacy-history E2-old=stage2_segment_d2s_fold0.csv \
  --legacy-history E3-old=stage2_segment_roi_fold0.csv \
  --legacy-history E3b-old=stage2_segment_roi_outside_fold0.csv \
  --strict-val-only --records-only
```

`experiment_record.xlsx`由生成后的CSV构建。所有NA/unknown保留，不以0填充。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "methods.md").write_text(methods_text(), encoding="utf-8")
    evidence = f"""# Evidence ledger

| 结论 | 证据 | 限制/反例 | 状态 |
|---|---|---|---|
| 固定PKU validation样本降噪有效 | V0三组 PSNR noisy→denoised、EPI gain均为正 | 仅f01三帧，Stage 2冻结去噪功能 | validation_supported_limited |
| outside BCE主要抑制层外FP | E3b vs E3-current数据身份匹配，E3b full Dice/Precision更高而ROI Dice接近 | 单seed、history为512模型坐标 | validation_supported_limited |
| D→S稳定有效 | E3b与no-D2S hard Dice接近，soft Dice差异很小 | 无重复seed，E1是多因素基线 | not_supported |
| P1/P2改善层mask | 同三帧P0→P1→P2 group-macro Dice上升 | P2同时重建层带并平滑，作用未拆分 | validation_supported_limited |
| P3改善血管 | raw→P3 Dice上升；删除FP={removed_fp:.0f}、TP={removed_tp:.0f} | 阈值不是完整P3流程独立校准；只3帧 | validation_supported_limited |
| RMAC/Joint优于Stage 2 | 当前包无完成Joint验证 | 只有历史探索且协议不严谨 | no_evidence |
"""
    (output / "evidence_ledger.md").write_text(evidence, encoding="utf-8")
    issues = """# Missing and issues

## P0

1. V0只有每组f01，共3帧；manifest原始组帧为50/50/41，尚无完整141帧评价。
2. calibration在512模型坐标，V0在640恢复坐标；raw最佳0.425与V0 raw阈值0.35不可直接核对。
3. component bins `[126,301]`在512坐标推导，却用于640 V0；面积阈值不可直接跨坐标沿用，小/中/大Recall标为定义不匹配。
4. P3沿用raw阈值，未对最终P3流程单独校准。
5. patient_id在V0等于group_id，只能确认group隔离，真实患者身份未核验。

## P1

6. 归档包只含E3b的V0预测，无法对四模型做同帧、同坐标定性比较。
7. clean PNG未导出，图册只能展示noisy/denoised；数值来自已有浮点评价日志。
8. 历史CSV缺少metadata/config/checkpoint/指纹，协议和权重保存状态为unknown。
9. `p1_component_count`是清理前计数；hole为二值填洞，开放凹口不计为hole。
10. P2从列上下边界重建层带并平滑下边界，现有结果不能把填带与平滑贡献拆开。

## 待评价补丁（未执行）

- 全141 validation帧 records-compatible推理，独立eval_id；保持test未触碰。
- 在同一恢复几何下重新校准raw/soft-gate/P3，并在640坐标重新推导component bins。
- 导出浮点概率或无损数组、clean图、完整TP/FP/FN及P3删除TP/FP图。
- 四模型使用同一固定帧集合导出定性结果。
"""
    (output / "missing_and_issues.md").write_text(issues, encoding="utf-8")
    next_text = """# Next experiments

1. 评价闭环：完成141帧validation、统一640坐标阈值和component bins、单独校准P3。
2. 同帧四模型导出：E3b/E3-current/no-D2S/E1-current，先配对组/帧再汇总。
3. 重复seed：重点检验outside BCE和D→S；报告seed与group波动。
4. 仅在Stage 2门禁通过后启动Joint；随后做no-RMAC/no-UGBI单因素消融。
5. test保持封存，直到模型、阈值、后处理版本全部冻结。
"""
    (output / "next_experiments.md").write_text(next_text, encoding="utf-8")


def write_workbook_sources(output: Path, registry: pd.DataFrame, protocol: pd.DataFrame,
                           training: pd.DataFrame, group: pd.DataFrame,
                           post: pd.DataFrame, thresholds: pd.DataFrame,
                           qualitative: pd.DataFrame) -> None:
    def subset(name: str, columns: list[str]) -> None:
        group[[column for column in columns if column in group]].to_csv(
            output / name, index=False, encoding="utf-8-sig"
        )

    subset("denoising_val.csv", [
        "run_id", "eval_id", "split", "group_id", "n_evaluated_frames", "manifest_group_frames",
        "psnr_noisy", "psnr", "psnr_gain_db", "ssim_noisy", "ssim", "ssim_gain",
        "rmse_noisy", "rmse", "rmse_reduction", "epi_noisy", "epi", "epi_gain",
        "snr_gain_db", "layer_roi_mse_noisy", "layer_roi_mse", "layer_roi_psnr_noisy",
        "layer_roi_psnr", "vessel_stroma_cnr_noisy", "vessel_stroma_cnr_denoised",
        "vessel_stroma_cnr_clean",
    ])
    subset("layer_val.csv", [
        "run_id", "eval_id", "split", "group_id", "n_evaluated_frames", "layer_dice", "layer_iou",
        "layer_precision", "layer_recall", "layer_hd95", "layer_assd", "layer_surface_dice",
        "upper_boundary_mae", "lower_boundary_mae", "thickness_mae", "upper_boundary_signed_bias",
        "lower_boundary_signed_bias", "thickness_signed_bias", "layer_component_count",
        "layer_extra_component_area_ratio", "layer_hole_area_ratio", "lower_boundary_roughness_pred",
        "p0_layer_dice", "p1_layer_dice", "p2_layer_dice",
    ])
    subset("vessel_val.csv", [
        "run_id", "eval_id", "split", "group_id", "n_evaluated_frames", "vessel_dice", "vessel_soft_dice",
        "vessel_precision", "vessel_recall", "vessel_roi_dice", "vessel_area_fraction_pred",
        "vessel_area_fraction_true", "vessel_area_fraction_mae", "pred_layer_vessel_dice",
        "vessel_outside_gt_layer_fraction", "pred_vessel_outside_layer_fraction",
        "gt_vessel_outside_pred_layer_fraction", "vessel_boundary_band_fp_pixels",
        "vessel_boundary_band_fn_pixels", "vessel_gt_component_small_pixel_recall",
        "vessel_gt_component_medium_pixel_recall", "vessel_gt_component_large_pixel_recall",
        "p0_vessel_dice", "p0_vessel_precision", "p0_vessel_recall", "p3_vessel_dice",
        "p3_vessel_precision", "p3_vessel_recall", "p3_removed_tp", "p3_removed_fp",
    ])
    subset("groups_table.csv", [
        "run_id", "eval_id", "split", "group_id", "dataset", "n_evaluated_frames",
        "manifest_group_frames", "model_input_height", "model_input_width", "evaluation_height",
        "evaluation_width", "psnr_noisy", "psnr", "psnr_gain_db", "ssim_noisy", "ssim",
        "ssim_gain", "epi_noisy", "epi", "epi_gain", "layer_dice", "layer_hd95",
        "upper_boundary_mae", "lower_boundary_mae", "thickness_mae", "vessel_dice",
        "vessel_precision", "vessel_recall", "vessel_area_fraction_pred",
        "vessel_area_fraction_true", "vessel_outside_gt_layer_fraction",
        "pred_vessel_outside_layer_fraction", "p0_layer_dice", "p1_layer_dice",
        "p2_layer_dice", "p0_vessel_dice", "p3_vessel_dice", "p3_removed_tp",
        "p3_removed_fp",
    ])
    pd.DataFrame([
        {"scope": "Archive", "item": "Evaluation scope", "value": "records-only; validation-only; no test"},
        {"scope": "Archive", "item": "Current runs", "value": "E3b; E3-current; E3b-no-D2S; E1-current"},
        {"scope": "Archive", "item": "V0 coverage", "value": "3 groups; one f01 frame/group; 640x640 evaluation"},
        {"scope": "Archive", "item": "Primary limitation", "value": "not the complete 141-frame validation"},
        {"scope": "Archive", "item": "Threshold mismatch", "value": "calibration 512x512; V0 raw threshold 0.35 at 640x640"},
    ]).to_csv(output / "workbook_readme.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"method": "Shared encoder + three decoders", "proposed": "yes", "implemented": "yes", "run_evidence": "yes", "validation_supported": "Stage 2 only", "source": "sabids/models/sabids_net.py"},
        {"method": "Denoising residual head", "proposed": "yes", "implemented": "yes", "run_evidence": "yes", "validation_supported": "limited 3 groups", "source": "sabids/models/sabids_net.py"},
        {"method": "D->S UGBI", "proposed": "yes", "implemented": "yes", "run_evidence": "E3b/no-D2S", "validation_supported": "not stable/significant", "source": "sabids/models/ugbi.py"},
        {"method": "S->D uncertainty gate", "proposed": "yes", "implemented": "yes", "run_evidence": "disabled in current four", "validation_supported": "no", "source": "sabids/models/ugbi.py"},
        {"method": "E3b ROI+outside", "proposed": "yes", "implemented": "yes", "run_evidence": "yes", "validation_supported": "limited", "source": "sabids/losses/total.py"},
        {"method": "RMAC", "proposed": "yes", "implemented": "yes", "run_evidence": "no completed Joint", "validation_supported": "no", "source": "sabids/losses/rmac.py"},
        {"method": "EMA dark-lumen pseudo", "proposed": "yes", "implemented": "yes", "run_evidence": "no formal Stage 5", "validation_supported": "no", "source": "sabids/losses/pseudo.py"},
        {"method": "P0-P3 postprocess", "proposed": "later", "implemented": "yes", "run_evidence": "V0", "validation_supported": "limited 3 frames", "source": "sabids/postprocessing.py"},
    ]).to_csv(output / "methods_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"claim": "fixed validation samples denoising improved", "evidence": "PSNR/EPI paired gain on 3 groups", "status": "supported_limited", "limitation": "one frame/group; frozen denoising"},
        {"claim": "outside BCE suppresses exterior FP", "evidence": "matched E3b vs E3-current histories", "status": "supported_limited", "limitation": "single seed; model coordinates"},
        {"claim": "D->S is stably beneficial", "evidence": "E3b and no-D2S are close", "status": "not_supported", "limitation": "no repeated seeds"},
        {"claim": "P1/P2/P3 improve fixed-frame Dice", "evidence": "V0 group metrics", "status": "supported_limited", "limitation": "3 frames; P3 not separately calibrated"},
        {"claim": "Joint/RMAC outperforms Stage 2", "evidence": "none in current package", "status": "no_evidence", "limitation": "Joint deferred"},
    ]).to_csv(output / "evidence_table.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {"priority": "P0", "issue": "V0 only f01 for 3 groups, not 141 frames", "impact": "coverage insufficient"},
        {"priority": "P0", "issue": "threshold calibration 512 vs V0 640 coordinates", "impact": "evaluation_comparable unknown"},
        {"priority": "P0", "issue": "component bins 512 applied to 640", "impact": "small/medium/large recall invalidly defined"},
        {"priority": "P0", "issue": "P3 final pipeline not separately calibrated", "impact": "postprocess estimate incomplete"},
        {"priority": "P1", "issue": "patient_id duplicates group_id", "impact": "patient isolation unverified"},
        {"priority": "P1", "issue": "clean PNG and four-model matched images absent", "impact": "qualitative comparison incomplete"},
    ]).to_csv(output / "missing_table.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    source = Path(args.source_archive).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output).expanduser().resolve() if args.output else root / "reports" / "experiment_records" / timestamp
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    archive = SafeArchive(source, allow_test_results=args.include_test_results)
    registry, training, histories = extract_current(archive)
    legacy_registry, legacy_training, legacy_histories = extract_legacy(parse_legacy(args.legacy_history))
    registry.extend(legacy_registry)
    training.extend(legacy_training)
    histories.update(legacy_histories)
    protocol, protocol_source = build_protocol_table(archive)
    metrics: list[Dict[str, Any]] = []
    add_history_metrics(metrics, registry, training, histories)
    frame, group, post, thresholds = extract_v0(archive, output, metrics)
    registry_table, training_table, metrics_table = pd.DataFrame(registry), pd.DataFrame(training), pd.DataFrame(metrics)
    registry_table.to_csv(output / "experiment_registry.csv", index=False, encoding="utf-8-sig")
    write_json(registry, output / "experiment_registry.json")
    protocol.to_csv(output / "protocol_comparison.csv", index=False, encoding="utf-8-sig")
    write_json(protocol.to_dict("records"), output / "protocol_comparison.json")
    training_table.to_csv(output / "training_best.csv", index=False, encoding="utf-8-sig")
    metrics_table.to_csv(output / "metrics_long.csv", index=False, encoding="utf-8-sig")
    plot_training(histories, registry, figures)
    plot_v0(group, post, thresholds, figures)
    qualitative = extract_images_and_gallery(archive, frame, output)
    test_index = export_test_results(archive, output) if args.include_test_results else pd.DataFrame()
    write_narrative(output, registry_table, training_table, group, thresholds)
    if args.include_test_results:
        with (output / "README.md").open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n## Test结果归档\n\n已原样归档 {len(test_index)} 个test结果文件。"
                "这些文件未进入checkpoint选择、阈值校准、模型排名或当前证据结论。\n"
            )
    write_workbook_sources(output, registry_table, protocol, training_table, group, post, thresholds, qualitative)
    commit, dirty = git_info(root)
    inputs = [{"name": source.name, "sha256": sha256_file(source), "size": source.stat().st_size}]
    inputs.extend({"name": path.name, "sha256": sha256_file(path), "size": path.stat().st_size}
                  for _, path in parse_legacy(args.legacy_history) if path.is_file())
    provenance = {
        "scope": "records-only validation plus test archival" if args.include_test_results else "records-only validation-only",
        "test_metrics_exported": bool(args.include_test_results),
        "test_metrics_used_for_selection_or_calibration": False,
        "project_root_at_export": ".", "repository_name": root.name,
        "git_commit": commit, "git_dirty_status": dirty.splitlines(),
        "source_inputs": inputs, "source_archive_members_read": sorted(
            name for name in archive.names if not name.endswith(".pth") and not name.endswith("/")
        ),
        "excluded_members": [name for name in archive.names if name.endswith(".pth")],
        "metric_version": METRIC_VERSION, "postprocess_version": POSTPROCESS_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "counts": {"runs": len(registry_table), "metrics_long": len(metrics_table),
                   "v0_frames": len(frame), "v0_groups": len(group), "qualitative_rows": len(qualitative),
                   "test_artifacts_archived": len(test_index)},
    }
    write_json(provenance, output / "provenance.json")
    generated = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "provenance.json":
            continue
        generated.append({"path": path.relative_to(output).as_posix(), "sha256": sha256_file(path), "size": path.stat().st_size})
    provenance["generated_files"] = generated
    write_json(provenance, output / "provenance.json")
    archive.close()
    print(output)


if __name__ == "__main__":
    main()
