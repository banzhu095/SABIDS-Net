#!/usr/bin/env python
"""Create a lightweight, test-free GPT archive for seg-guided experiments."""
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ALLOWED = {".json", ".csv", ".md", ".yaml", ".yml", ".log", ".txt", ".png"}
FORBIDDEN_PARTS = {"test", "test_results", "sealed_test", "checkpoints", "cache"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if not output.is_absolute() or output.suffix.lower() != ".zip":
        raise ValueError("--output must be an absolute .zip path")
    if output.exists():
        raise FileExistsError(f"Refusing existing archive: {output}")
    selected = []
    for raw in args.roots:
        root = Path(raw).resolve()
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in ALLOWED:
                continue
            relative = path.relative_to(root)
            lowered_dirs = {part.lower() for part in relative.parts[:-1]}
            if (lowered_dirs & FORBIDDEN_PARTS
                    or any(part.startswith("test_") or part.startswith("sealed_test")
                           for part in lowered_dirs)
                    or path.stat().st_size > 10 * 1024 * 1024):
                continue
            selected.append((root.name + "/" + relative.as_posix(), path))
    status = "complete" if any(path.name == "summary.json" and
                                json.loads(path.read_text(encoding="utf-8-sig")).get("status") == "passed"
                                for _, path in selected) else "incomplete"
    manifest = {"status": status, "test_assets_included": 0,
                "files": [{"path": name, "size": path.stat().st_size, "sha256": digest(path)}
                          for name, path in selected]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for name, path in selected:
            archive.write(path, name)
        archive.writestr("PACKAGE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Archive verification failed: {bad}")
    result = {"status": status, "path": str(output), "size": output.stat().st_size,
              "sha256": digest(output), "file_count": len(selected), "test_assets_included": 0}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
