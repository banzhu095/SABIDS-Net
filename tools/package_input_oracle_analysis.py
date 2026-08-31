from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime
from pathlib import Path


EXCLUDED_SUFFIXES = {".pth", ".pt", ".npy", ".tif", ".tiff"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a lightweight GPT analysis package")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--runs", default="runs/input_oracle_cv")
    parser.add_argument("--report", default="runs/input_oracle_cv/report")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    runs, report = (root / args.runs).resolve(), (root / args.report).resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.output).expanduser().resolve() if args.output else root / f"SABIDS_PKU37_input_oracle_analysis_{stamp}.zip"
    if output.exists(): raise FileExistsError(f"Refusing to overwrite package: {output}")
    checklist_path = report / "missing_and_failure_checklist.json"
    checklist = json.loads(checklist_path.read_text(encoding="utf-8")) if checklist_path.is_file() else {}
    if not args.dry_run and (not checklist.get("complete") or checklist.get("missing_inputs")):
        raise RuntimeError(f"Refusing to package an incomplete analysis: {checklist_path}")
    guide = report / "ANALYSIS_GUIDE.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("""# GPT analysis guide\n\n1. Verify split/checkpoint/data-plan audits, fixed epoch, positions, and sealed test status first.\n2. Treat anatomical position as the independent unit; seeds and repeat frames are not patients.\n3. Analyze DENOISED-NOISY, CLEAN-NOISY, and CLEAN-DENOISED.\n4. Check agreement of layer and vessel effects, then Precision/Recall, ROI FP/FN and outside-layer fraction.\n5. Check small/medium/large vessels, boundaries, thickness, CNR and edge preservation.\n6. Determine whether pku_0006/0012/0040 are typical and whether one position/fold/seed drives the result.\n7. Incorporate blinded label review only if review_status is no longer PENDING.\n8. Inspect fixed atlas for smoothing, lumen breaks and hallucination.\n9. Do not use test, best epochs, threshold calibration, post-processing, or selective samples to create a positive result.\n""", encoding="utf-8")
    candidates: list[tuple[Path, str]] = []
    for base, prefix in ((report, "report"), (runs / "audit", "audit"), (runs / "label_audit", "label_audit"), (runs / "splits", "splits"), (root / "configs/current/input_oracle_cv", "configs/input_oracle_cv")):
        if not base.exists(): continue
        for path in base.rglob("*"):
            if path.is_file() and path.suffix.lower() not in EXCLUDED_SUFFIXES:
                candidates.append((path, f"{prefix}/{path.relative_to(base).as_posix()}"))
    # Training histories and small audit files; never prediction trees or caches.
    for path in runs.rglob("*"):
        if not path.is_file() or any(part in {"cache", "predictions"} for part in path.parts): continue
        if path.name in {"history.csv", "resolved_config.yaml", "initialization_audit.json", "parameter_audit.json", "run_metadata.json", "d0_leakage_audit.json", "evaluation_registry.json"}:
            candidates.append((path, f"runs/{path.relative_to(runs).as_posix()}"))
    unique = {arc: path for path, arc in candidates}
    manifest = [{"path": arc, "bytes": path.stat().st_size, "sha256": sha(path)} for arc, path in sorted(unique.items())]
    if any(Path(item["path"]).suffix.lower() in EXCLUDED_SUFFIXES for item in manifest):
        raise RuntimeError("Forbidden large/binary training asset entered package")
    if args.dry_run:
        print(json.dumps({"output": str(output), "files": len(manifest), "bytes": sum(x["bytes"] for x in manifest)}, indent=2)); return
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for arc, path in sorted(unique.items()): archive.write(path, arc)
        archive.writestr("MANIFEST.json", json.dumps({"created": stamp, "files": manifest, "excluded": ["checkpoints", "float cache", "full predictions", "sealed test assets"], "test_assets_opened": 0}, ensure_ascii=False, indent=2))
    print(json.dumps({"output": str(output), "sha256": sha(output), "files": len(manifest), "bytes": output.stat().st_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
