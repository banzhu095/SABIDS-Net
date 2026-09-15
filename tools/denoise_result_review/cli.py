from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from tools.oct_denoise_benchmark.data import audit_protocol, load_protocol_manifest

from .package_test_images import package_test_images
from .roi_evaluation import evaluate_rois
from .roi_registry import ROIRegistry
from .roi_report import build_roi_report
from .run_discovery import discover_runs, discovery_frame, resolve_run
from .stage_summary import build_stage_summary
from .validation import audit_local


def _list(value: str | None) -> list[str] | None:
    return [item.strip() for item in value.split(",") if item.strip()] if value else None


def _print(value) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, default=str))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traceable OCT denoising result review")
    sub = p.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover"); discover.add_argument("--project-root", type=Path, default=Path(".")); discover.add_argument("--output", type=Path)
    summarize = sub.add_parser("summarize"); summarize.add_argument("--project-root", type=Path, default=Path(".")); summarize.add_argument("--run-dir", default="auto"); summarize.add_argument("--output-dir", type=Path)
    package = sub.add_parser("package-test"); package.add_argument("--project-root", type=Path, default=Path(".")); package.add_argument("--run-dir", default="auto"); package.add_argument("--dataset", default="PKU37"); package.add_argument("--split", default="test"); package.add_argument("--output-dir", type=Path, required=True); package.add_argument("--primary-seeds-only", action="store_true"); package.add_argument("--include-all-seeds", action="store_true"); package.add_argument("--include-noisy", action="store_true"); package.add_argument("--include-reference", action="store_true"); package.add_argument("--archive-by-position", action="store_true"); package.add_argument("--archive-all", action="store_true"); package.add_argument("--positions"); package.add_argument("--samples"); package.add_argument("--methods"); package.add_argument("--dry-run", action="store_true"); package.add_argument("--resume", action="store_true"); package.add_argument("--overwrite", action="store_true"); package.add_argument("--allow-incomplete-run", action="store_true", help="Explicit development/smoke escape hatch; never changes the reported dataset split.")
    local = sub.add_parser("audit-local"); local.add_argument("--input-root", type=Path, required=True); local.add_argument("--output-root", type=Path, required=True)
    select = sub.add_parser("select-roi"); select.add_argument("--input-root", type=Path, required=True); select.add_argument("--selection-list", type=Path, required=True); select.add_argument("--output-root", type=Path, required=True); select.add_argument("--roi-size", type=int, choices=(32,48,64), default=48); select.add_argument("--unlock-rois", action="store_true"); select.add_argument("--unlock-reason", default="")
    evaluate = sub.add_parser("evaluate-roi"); evaluate.add_argument("--input-root", type=Path, required=True); evaluate.add_argument("--roi-registry", type=Path, required=True); evaluate.add_argument("--output-root", type=Path, required=True); evaluate.add_argument("--require-locked-rois", action="store_true"); evaluate.add_argument("--primary-seeds-only", action="store_true"); evaluate.add_argument("--include-all-seeds", action="store_true"); evaluate.add_argument("--resume", action="store_true")
    report = sub.add_parser("build-report"); report.add_argument("--project-root", type=Path, default=Path(".")); report.add_argument("--run-dir", default="auto"); report.add_argument("--input-root", type=Path, required=True); report.add_argument("--output-root", type=Path, required=True)
    all_local = sub.add_parser("run-all-local"); all_local.add_argument("--project-root", type=Path, default=Path(".")); all_local.add_argument("--run-dir", default="auto"); all_local.add_argument("--input-root", type=Path, required=True); all_local.add_argument("--output-root", type=Path, required=True); all_local.add_argument("--roi-size", type=int, choices=(32,48,64), default=48); all_local.add_argument("--primary-seeds-only", action="store_true"); all_local.add_argument("--include-all-seeds", action="store_true"); all_local.add_argument("--resume", action="store_true")
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "discover":
        candidates = discover_runs(args.project_root); frame = discovery_frame(candidates)
        if args.output: args.output.parent.mkdir(parents=True, exist_ok=True); frame.to_csv(args.output, index=False)
        manifest_audit = audit_protocol(load_protocol_manifest(args.project_root))
        _print({"candidates": [asdict(item) for item in candidates], "manifest_audit": manifest_audit})
    elif args.command == "summarize":
        run = resolve_run(args.project_root, args.run_dir, require_complete=False); _print(build_stage_summary(args.project_root, run, args.output_dir))
    elif args.command == "package-test":
        run = resolve_run(args.project_root, args.run_dir, require_complete=not (args.dry_run or args.allow_incomplete_run))
        primary = not args.include_all_seeds
        _print(package_test_images(args.project_root, run, args.output_dir, args.dataset, args.split, primary, args.include_noisy, args.include_reference, args.archive_by_position, args.archive_all, _list(args.positions), _list(args.methods), _list(args.samples), args.dry_run, args.resume, args.overwrite))
    elif args.command == "audit-local": _print(audit_local(args.input_root, args.output_root))
    elif args.command == "select-roi":
        from .roi_gui import select_rois
        select_rois(args.input_root, args.selection_list, args.output_root, args.roi_size, args.unlock_rois, args.unlock_reason)
    elif args.command == "evaluate-roi": _print(evaluate_rois(args.input_root, args.roi_registry, args.output_root, args.require_locked_rois, not args.include_all_seeds, args.resume))
    elif args.command == "build-report":
        from .roi_visualization import build_panels
        run = resolve_run(args.project_root, args.run_dir, require_complete=False); panels = build_panels(args.input_root, args.output_root / "roi_registry.csv", args.output_root); result = build_roi_report(args.project_root, run, args.output_root); _print({**result, **panels})
    elif args.command == "run-all-local":
        audit = audit_local(args.input_root, args.output_root); registry = ROIRegistry(args.output_root)
        if not registry.locked:
            from .roi_gui import select_rois
            select_rois(args.input_root, Path(audit["candidate_manifest"]), args.output_root, args.roi_size)
            registry = ROIRegistry(args.output_root)
        if not registry.locked:
            _print({"status": "waiting_for_locked_rois", "roi_registry": str(registry.csv_path), "next": "Reopen select-roi and press L only after reviewing all coordinates."}); return
        evaluation = evaluate_rois(args.input_root, registry.csv_path, args.output_root, True, not args.include_all_seeds, args.resume)
        from .roi_visualization import build_panels
        run = resolve_run(args.project_root, args.run_dir, require_complete=False); panels = build_panels(args.input_root, registry.csv_path, args.output_root); report = build_roi_report(args.project_root, run, args.output_root); _print({"audit": audit, "evaluation": evaluation, "panels": panels, "report": report})


if __name__ == "__main__": main()
