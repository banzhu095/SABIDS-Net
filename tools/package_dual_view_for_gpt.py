#!/usr/bin/env python
"""Create a small, validated GPT archive for one dual-view run id."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pth", ".pt", ".ckpt", ".npy", ".npz"}
FIXED_SAMPLE_IDS = ("pku_0006_f01", "pku_0012_f01", "pku_0040_f01")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve())
    except ValueError:
        return Path("external") / resolved.name


def _copy(source: Path, destination_root: Path, project_root: Path, selected: set[str]) -> None:
    if not source.is_file() or source.suffix.lower() in FORBIDDEN_SUFFIXES:
        return
    relative = _safe_relative(source, project_root)
    lowered = "/".join(relative.parts).lower()
    if any(token in lowered for token in ("/test/", "\\test\\", "test_results")):
        raise ValueError(f"Refusing possible test asset: {source}")
    key = relative.as_posix()
    if key in selected:
        return
    target = destination_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    selected.add(key)


def collect(project_root: Path, run_id: str, stage: Path) -> dict:
    selected: set[str] = set()
    report_base = project_root / "reports/adaptive_denoising/dual_view_v1" / run_id
    run_base = project_root / "runs/adaptive_denoising/pku37_binary_v3/dual_view_v1"
    cache_base = project_root / "cache/adaptive_denoising/pku37_binary_v3/dual_view_v1"

    for evidence in (
        project_root / "Manifests/pku37_binary_v3/active_protocol_lock.json",
        project_root / "configs/data/pku37_binary_v3_split.yaml",
        project_root / "runs/adaptive_denoising/pku37_binary_v3/d1_repro_fold0_seed42/checkpoint_binding_best_d2_v1.json",
        project_root / "runs/adaptive_denoising/pku37_binary_v3/d1_repro_fold0_seed42/training_asset_inventory_initial.json",
    ):
        _copy(evidence, stage, project_root, selected)

    for path in report_base.rglob("*") if report_base.is_dir() else ():
        if path.is_file():
            _copy(path, stage, project_root, selected)

    for tag in (f"{run_id}_pilot", f"{run_id}_formal", f"{run_id}_overfit"):
        prep = cache_base / tag
        for pattern in ("preparation_registry.json", "config_*.yaml", "manifest_*.csv"):
            for path in prep.glob(pattern):
                _copy(path, stage, project_root, selected)
        runs = run_base / tag
        if not runs.is_dir():
            continue
        for run in sorted(path for path in runs.iterdir() if path.is_dir()):
            for name in (
                "resolved_config.yaml", "history.csv", "initialization_audit.json",
                "data_plan.json", "dual_view_training_metadata.json",
                "interaction_strength.csv", "run_metadata.json",
            ):
                _copy(run / name, stage, project_root, selected)
            for selection in ("validation_last", "validation_best", "validation_c5_last"):
                folder = run / selection
                for name in (
                    "summary.json", "frame_metrics.csv", "group_metrics.csv",
                    "component_metrics.csv",
                ):
                    _copy(folder / name, stage, project_root, selected)
                prediction_root = folder / "predictions"
                if prediction_root.is_dir():
                    for image in prediction_root.rglob("*.png"):
                        if any(sample in image.name for sample in FIXED_SAMPLE_IDS):
                            _copy(image, stage, project_root, selected)
            log = run / "train.log"
            if log.is_file():
                tail = stage / _safe_relative(log, project_root)
                tail.parent.mkdir(parents=True, exist_ok=True)
                lines = log.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]
                tail.write_text("\n".join(lines) + "\n", encoding="utf-8")
                selected.add(_safe_relative(log, project_root).as_posix())

    gate_path = report_base / "pilot/gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    formal_complete = (report_base / "formal/summary/summary.json").is_file()
    pilot_complete = gate.get("formal_allowed") is True
    return {
        "status": (
            "formal_complete" if formal_complete else
            "pilot_complete" if pilot_complete else "incomplete"
        ),
        "formal_allowed": pilot_complete,
        "formal_results_present": formal_complete,
        "file_count_before_manifest": len(selected),
        "test_assets_opened": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    project_root = Path(args.project_root).resolve()
    requested = Path(args.output).expanduser().resolve()
    if requested.suffix.lower() != ".zip":
        raise ValueError("--output must be an absolute .zip path")
    if not Path(args.output).expanduser().is_absolute():
        raise ValueError("--output must be absolute")

    with tempfile.TemporaryDirectory(prefix="sabids_dual_view_package_") as temporary:
        stage = Path(temporary) / "GPT_light_dual_view"
        stage.mkdir()
        result = collect(project_root, args.run_id, stage)
        output = requested
        if result["status"] == "incomplete" and "incomplete" not in output.stem.lower():
            output = output.with_name(f"{output.stem}_incomplete.zip")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite archive: {output}")
        (stage / "PACKAGE_STATUS.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        rows = []
        for path in sorted(item for item in stage.rglob("*") if item.is_file()):
            rows.append({
                "path": path.relative_to(stage).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
        with (stage / "MANIFEST.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("path", "size_bytes", "sha256"))
            writer.writeheader()
            writer.writerows(rows)
        output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(item for item in stage.rglob("*") if item.is_file()):
                archive.write(path, Path("GPT_light_dual_view") / path.relative_to(stage))
        with zipfile.ZipFile(output) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise RuntimeError(f"ZIP integrity failure: {bad}")
            archived = set(archive.namelist())
            expected = {
                (Path("GPT_light_dual_view") / path.relative_to(stage)).as_posix()
                for path in stage.rglob("*") if path.is_file()
            }
            if archived != expected:
                raise RuntimeError("ZIP member manifest is incomplete")
        result.update({
            "output": str(output),
            "size_bytes": output.stat().st_size,
            "sha256": sha256_file(output),
            "zip_readable": True,
            "manifest_complete": True,
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
