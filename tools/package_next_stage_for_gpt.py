"""Create a validation-only lightweight GPT evidence archive."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file


FORBIDDEN_SUFFIX = {".pth", ".pt", ".ckpt", ".npy", ".npz"}
FORBIDDEN_PARTS = {"data", "label", "test", "tests", "test_results", "cache", "input_probe_cache"}
RUN_FILES = {"resolved_config.yaml", "config_resolved.yaml", "run_metadata.json", "initialization_audit.json", "data_plan_audit.json", "parameter_audit.json", "history.csv", "gradient_audit.csv", "interaction_strength.csv"}


def forbidden(relative: Path) -> bool:
    return relative.suffix.lower() in FORBIDDEN_SUFFIX or any(part.lower() in FORBIDDEN_PARTS or part.lower().startswith("test_") for part in relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--report-roots", nargs="+", required=True)
    parser.add_argument("--output-dir", default="exports")
    parser.add_argument("--exclude-test", action="store_true")
    args = parser.parse_args()
    if not args.exclude_test: raise SystemExit("BLOCKED: --exclude-test is mandatory")
    root = Path(args.project_root).resolve()
    lock_path = Path(args.protocol_lock)
    if not lock_path.is_absolute(): lock_path = root / lock_path
    lock = load_protocol_lock(lock_path)
    sources: set[Path] = {lock_path}
    manifest_root = Path(str(lock["manifest_root"])); manifest_root = manifest_root if manifest_root.is_absolute() else (root / manifest_root).resolve()
    for name in ("protocol_audit.json", "split_by_position.csv", "dataset_inventory.csv", "label_inventory.csv"):
        if (manifest_root / name).is_file(): sources.add(manifest_root / name)
    config_root = root / "configs" / "next_stage_v3"
    if config_root.is_dir(): sources.update(path for path in config_root.rglob("*") if path.is_file())
    run_ids = set()
    for report_arg in args.report_roots:
        report = Path(report_arg); report = report if report.is_absolute() else root / report
        if not report.is_dir(): raise FileNotFoundError(f"Report root does not exist: {report}")
        sources.update(path for path in report.rglob("*") if path.is_file())
        manifest = report / "report_manifest.json"
        if manifest.is_file(): run_ids.update(json.loads(manifest.read_text(encoding="utf-8-sig")).get("run_ids", []))
    for run_id in run_ids:
        run = root / "runs" / "current" / run_id
        cfg_path = next((run / name for name in ("resolved_config.yaml", "config_resolved.yaml") if (run / name).is_file()), None)
        if cfg_path:
            cfg = load_config(cfg_path)
            if cfg.get("protocol_id") != lock["protocol_id"]: raise RuntimeError(f"Run {run_id} protocol mismatch")
        for name in RUN_FILES:
            if (run / name).is_file(): sources.add(run / name)
    selected = []
    for source in sources:
        try: relative = source.resolve().relative_to(root)
        except ValueError: raise RuntimeError(f"Refusing external archive member: {source}")
        if not forbidden(relative): selected.append((source, relative))
    output_dir = Path(args.output_dir); output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = str(lock["protocol_id"]).replace("_binary", "")
    output = output_dir / f"SABIDS_three_experiments_{tag}_GPT_{datetime.now():%Y%m%d_%H%M%S}.tar.gz"
    checksums = []
    with tarfile.open(output, "w:gz") as archive:
        for source, relative in sorted(selected, key=lambda item: str(item[1])):
            digest = sha256_file(source); checksums.append(f"{digest}  {relative.as_posix()}")
            archive.add(source, arcname=relative.as_posix(), recursive=False)
        payload = ("\n".join(checksums) + "\n").encode("utf-8")
        info = tarfile.TarInfo("SHA256SUMS.txt"); info.size = len(payload); info.mtime = int(datetime.now().timestamp())
        archive.addfile(info, io.BytesIO(payload))
    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
    checkpoint_count = sum(Path(name).suffix.lower() in {".pth", ".pt", ".ckpt"} for name in names)
    numpy_count = sum(Path(name).suffix.lower() in {".npy", ".npz"} for name in names)
    test_count = sum(any(part.lower() in FORBIDDEN_PARTS or part.lower().startswith("test_") for part in Path(name).parts) for name in names)
    if checkpoint_count or numpy_count or test_count: raise RuntimeError("Archive post-check failed")
    missing = 0; statuses = {}
    for report_arg in args.report_roots:
        report = Path(report_arg); report = report if report.is_absolute() else root / report
        if (report / "missing_assets.csv").is_file(): missing += len(pd.read_csv(report / "missing_assets.csv"))
        if (report / "completion_matrix.csv").is_file():
            table = pd.read_csv(report / "completion_matrix.csv")
            statuses[report.name] = table["status"].value_counts().to_dict() if "status" in table else {}
    result = {"absolute_path": str(output.resolve()), "size_bytes": output.stat().st_size, "file_count": len(names), "sha256": sha256_file(output), "protocol_id": lock["protocol_id"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "included_run_ids": sorted(run_ids), "completion_status": statuses, "test_file_count": test_count, "checkpoint_count": checkpoint_count, "npy_npz_count": numpy_count, "missing_asset_count": missing}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
