from __future__ import annotations

import json
import io
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from . import METHOD_ORDER
from .image_io import decode_lossless, sha256_file
from .manifest_builder import build_asset_manifest


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _copy_verified(source: Path, destination: Path, resume: bool, overwrite: bool) -> tuple[str, int]:
    digest = sha256_file(source)
    if destination.exists():
        if sha256_file(destination) == digest: return digest, destination.stat().st_size
        if not overwrite: raise FileExistsError(f"hash conflict at existing package asset: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    if sha256_file(temporary) != digest:
        temporary.unlink(missing_ok=True); raise IOError(f"copy hash mismatch: {source}")
    os.replace(temporary, destination)
    return digest, destination.stat().st_size


def _asset_record(source: Path, destination: Path, role: str, method: str, row: Any, package_root: Path,
                  resume: bool, overwrite: bool) -> dict[str, Any]:
    digest, size = _copy_verified(source, destination, resume, overwrite)
    raw = decode_lossless(source)
    return {
        "dataset": str(row.dataset), "split": str(row.split), "position_id": str(row.position_id),
        "sample_id": str(row.sample_id), "asset_role": role, "method_id": method,
        "seed": int(getattr(row, "seed", 0)), "is_primary_seed": bool(getattr(row, "is_primary_seed", True)),
        "checkpoint_sha256": str(getattr(row, "checkpoint_sha256", "") or ""),
        "source_path": str(source), "packaged_path": destination.relative_to(package_root).as_posix(),
        "sha256": digest, "bytes": size, "height": int(raw.shape[0]), "width": int(raw.shape[1]),
        "dtype": str(raw.dtype), "bit_depth": int(raw.dtype.itemsize * 8),
    }


def _zip_position(package_root: Path, position: str, records: pd.DataFrame, archive: Path, resume: bool) -> dict[str, Any]:
    if archive.is_file() and resume:
        with zipfile.ZipFile(archive) as bundle:
            bad = bundle.testzip()
            try:
                archived = pd.read_csv(io.BytesIO(bundle.read("IMAGE_MANIFEST.csv")), keep_default_na=False)
                columns = ["packaged_path", "sha256"]
                expected = records[columns].astype(str).sort_values(columns).reset_index(drop=True)
                observed = archived[columns].astype(str).sort_values(columns).reset_index(drop=True)
                same_payload = expected.equals(observed)
            except (KeyError, ValueError, pd.errors.ParserError):
                same_payload = False
        if bad is None and same_payload:
            return {"position_id": position, "archive_path": str(archive.resolve()), "sample_count": records.sample_id.nunique(), "method_count": records[records.asset_role == "method"].method_id.nunique(), "file_count": len(records), "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "crc_status": "passed", "status": "resumed_existing"}
    archive.parent.mkdir(parents=True, exist_ok=True)
    temporary = archive.with_name(f".{archive.name}.{os.getpid()}.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as bundle:
        for row in records.itertuples(index=False):
            path = package_root / row.packaged_path
            bundle.write(path, row.packaged_path)
        bundle.writestr("IMAGE_MANIFEST.csv", records.to_csv(index=False))
        bundle.writestr("README.md", f"# PKU37 test position {position}\n\nLossless source assets copied without pixel re-encoding. Quantitative analysis must use the image files, not previews.\n")
    os.replace(temporary, archive)
    with zipfile.ZipFile(archive) as bundle: bad = bundle.testzip()
    if bad: raise RuntimeError(f"ZIP CRC failure {archive}: {bad}")
    return {"position_id": position, "archive_path": str(archive.resolve()), "sample_count": records.sample_id.nunique(), "method_count": records[records.asset_role == "method"].method_id.nunique(), "file_count": len(records), "bytes": archive.stat().st_size, "sha256": sha256_file(archive), "crc_status": "passed", "status": "created"}


def package_test_images(project_root: str | Path, run_dir: str | Path, output_dir: str | Path,
                        dataset: str = "PKU37", split: str = "test", primary_only: bool = True,
                        include_noisy: bool = True, include_reference: bool = True,
                        archive_by_position: bool = True, archive_all: bool = False,
                        positions: list[str] | None = None, methods: list[str] | None = None,
                        samples: list[str] | None = None,
                        dry_run: bool = False, resume: bool = False, overwrite: bool = False) -> dict[str, Any]:
    root, run, output = Path(project_root).resolve(), Path(run_dir).resolve(), Path(output_dir).resolve()
    selected_methods = methods or METHOD_ORDER[1:]
    assets, failures = build_asset_manifest(run, dataset, split, primary_only, selected_methods)
    if positions: assets = assets[assets.position_id.astype(str).isin(positions)]
    if samples: assets = assets[assets.sample_id.astype(str).isin(samples)]
    sources = []
    for row in assets.itertuples(index=False):
        sources.append(Path(str(row.denoised_path)))
        if include_noisy: sources.append(Path(str(row.noisy_path)))
        if include_reference: sources.append(Path(str(row.reference_path)))
    unique_sources = list(dict.fromkeys(path.resolve() for path in sources if path.is_file()))
    estimate = {"source_files": len(unique_sources), "source_bytes": sum(path.stat().st_size for path in unique_sources), "estimated_zip_bytes": sum(path.stat().st_size for path in unique_sources), "free_bytes": shutil.disk_usage(output.parent if output.parent.exists() else root).free, "selected_metric_rows": len(assets), "failures": len(failures)}
    if dry_run: return {"status": "dry_run", "run_dir": str(run), "output_root": str(output), **estimate}
    package_root = output / f"{dataset}_{split}_{'primary' if primary_only else 'all_seeds'}"
    for child in ("positions", "manifests", "previews", "archives"): (package_root / child).mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    seen_common: set[tuple[str, str, str]] = set()
    package_failures = failures.to_dict("records")
    for row in assets.itertuples(index=False):
        sample_dir = package_root / "positions" / str(row.position_id) / str(row.sample_id)
        try:
            common_key = (str(row.position_id), str(row.sample_id), "common")
            if common_key not in seen_common:
                if include_noisy:
                    source = Path(str(row.noisy_path)); records.append(_asset_record(source, sample_dir / f"noisy{source.suffix.lower()}", "noisy", "noisy_identity", row, package_root, resume, overwrite))
                if include_reference:
                    source = Path(str(row.reference_path)); records.append(_asset_record(source, sample_dir / f"reference{source.suffix.lower()}", "reference", "clean_oracle", row, package_root, resume, overwrite))
                seen_common.add(common_key)
            source = Path(str(row.denoised_path))
            records.append(_asset_record(source, sample_dir / f"{row.method_id}{source.suffix.lower()}", "method", str(row.method_id), row, package_root, resume, overwrite))
        except Exception as exc:
            package_failures.append({"dataset": row.dataset, "split": row.split, "position_id": row.position_id, "sample_id": row.sample_id, "method_id": row.method_id, "seed": row.seed, "failure": f"{type(exc).__name__}: {exc}"})
    manifest = pd.DataFrame(records)
    _atomic_csv(manifest, package_root / "manifests" / "image_manifest.csv")
    failure_frame = pd.DataFrame(package_failures)
    _atomic_csv(failure_frame, package_root / "manifests" / "package_failures.csv")
    expected = set(selected_methods)
    completeness = []
    if not manifest.empty:
        for (position, sample), part in manifest.groupby(["position_id", "sample_id"]):
            actual = set(part.loc[part.asset_role == "method", "method_id"])
            completeness.append({"position_id": position, "sample_id": sample, "method_count": len(actual), "expected_method_count": len(expected), "missing_methods": ";".join(sorted(expected - actual)), "complete": actual == expected})
            _atomic_csv(part, package_root / "positions" / str(position) / str(sample) / "sample_manifest.csv")
    completeness_frame = pd.DataFrame(completeness)
    _atomic_csv(completeness_frame, package_root / "manifests" / "sample_completeness.csv")
    method_inventory = manifest.groupby(["method_id", "asset_role"], dropna=False).agg(files=("sha256", "size"), samples=("sample_id", "nunique"), bytes=("bytes", "sum")).reset_index() if not manifest.empty else pd.DataFrame()
    _atomic_csv(method_inventory, package_root / "manifests" / "method_inventory.csv")
    archives = []
    if archive_by_position and not manifest.empty:
        for position, part in manifest.groupby("position_id", sort=True):
            archives.append(_zip_position(package_root, str(position), part, package_root / "archives" / f"{dataset}_{split}_{position}_all_methods_{'primary' if primary_only else 'all_seeds'}.zip", resume))
    if archive_all and not manifest.empty:
        archives.append(_zip_position(package_root, "ALL", manifest, package_root / f"{dataset}_{split}_all_positions_all_methods_{'primary' if primary_only else 'all_seeds'}.zip", resume))
    archive_frame = pd.DataFrame(archives)
    _atomic_csv(archive_frame, package_root / "manifests" / "archive_inventory.csv")
    lines = ["# Download paths", "", f"Source run: `{run}`", ""]
    for item in archives:
        lines += [f"## {item['position_id']}", "", f"- Absolute path: `{item['archive_path']}`", f"- Samples: {item['sample_count']}; methods: {item['method_count']}; files: {item['file_count']}", f"- Bytes: {item['bytes']}; SHA256: `{item['sha256']}`; CRC: {item['crc_status']}", ""]
    (package_root / "download_paths.md").write_text("\n".join(lines), encoding="utf-8")
    return {"status": "completed_with_failures" if package_failures else "completed", "run_dir": str(run), "package_root": str(package_root), "image_files": len(manifest), "samples": manifest.sample_id.nunique() if not manifest.empty else 0, "positions": manifest.position_id.nunique() if not manifest.empty else 0, "archives": archives, "failures": len(package_failures), **estimate}
