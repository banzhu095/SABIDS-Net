from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite package: {output}")
    run = root / "runs/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v1" / args.run_id / "seed42"
    registry = root / "cache/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v1" / args.run_id
    report = root / "reports/adaptive_denoising/dual_task_adaptive_v1" / args.run_id
    allowed_suffixes = {".json", ".yaml", ".yml", ".csv", ".md", ".txt", ".log", ".png"}
    forbidden_tokens = ("/test/", "\\test\\", ".pth", ".npy", ".npz")
    files: list[Path] = []
    for base in (registry, run, report):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            relative_lower = str(path.relative_to(root)).lower()
            if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
                continue
            if any(token in relative_lower for token in forbidden_tokens):
                continue
            if path.stat().st_size > 8 * 1024 * 1024:
                continue
            files.append(path)
    if not files:
        raise RuntimeError("No lightweight adaptive artifacts were found")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = [{"path": str(path.relative_to(root)).replace("\\", "/"),
                 "size": path.stat().st_size, "sha256": sha256(path)} for path in sorted(set(files))]
    manifest_bytes = (json.dumps({"files": manifest, "test_assets_opened": 0}, indent=2) + "\n").encode()
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, str(path.relative_to(root)).replace("\\", "/"))
        archive.writestr("PACKAGE_MANIFEST.json", manifest_bytes)
    with zipfile.ZipFile(output, "r") as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure: {bad}")
    result = {"status": "passed", "path": str(output), "size_bytes": output.stat().st_size,
              "sha256": sha256(output), "file_count": len(manifest), "test_assets_opened": 0}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
