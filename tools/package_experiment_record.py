"""Create a safe, checkpoint-free ZIP from an exported experiment record."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


EXCLUDED_DIRS = {"workbook_previews", "__pycache__"}
EXCLUDED_SUFFIXES = {".pth", ".pt", ".ckpt"}


def validate_val_only_csv(path: Path) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        scoped = [column for column in ("split", "selection_split") if column in (reader.fieldnames or [])]
        for row_number, row in enumerate(reader, start=2):
            for column in scoped:
                value = (row.get(column) or "").strip().lower()
                if value and value != "val":
                    raise ValueError(f"Non-validation row in {path.name}:{row_number}: {column}={value}")


def collect_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(part in EXCLUDED_DIRS for part in path.relative_to(source).parts):
            continue
        if path.name.endswith(".inspect.ndjson") or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        relative = path.relative_to(source)
        if any(part.lower() in {"test", "tests"} for part in relative.parts):
            raise ValueError(f"Refusing test path: {relative}")
        if path.suffix.lower() == ".csv":
            validate_val_only_csv(path)
        files.append(path)
    return files


def write_archive_manifest(source: Path, files: list[Path]) -> Path:
    manifest_path = source / "archive_manifest.json"
    entries = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"path": path.relative_to(source).as_posix(), "bytes": path.stat().st_size, "sha256": digest})
    manifest_path.write_text(
        json.dumps(
            {
                "scope": "records-only; validation-only; no test results",
                "excluded": ["model checkpoints", "workbook render previews", "artifact inspection support files"],
                "files": entries,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    files = collect_files(source)
    manifest_path = write_archive_manifest(source, [path for path in files if path.name != "archive_manifest.json"])
    files = [path for path in files if path.name != "archive_manifest.json"] + [manifest_path]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path(source.name) / path.relative_to(source))
    print(f"{output}\nfiles={len(files)}")


if __name__ == "__main__":
    main()
