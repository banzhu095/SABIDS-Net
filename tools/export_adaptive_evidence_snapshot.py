#!/usr/bin/env python
"""Create a compact, metadata-only snapshot for adaptive follow-up analysis."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.adaptive_evidence import (  # noqa: E402
    discover_candidates,
    export_snapshot,
    resolve_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=("discover", "export"), required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--supplemental-dose-registry",
        action="append",
        default=[],
        help="Additional registry belonging to a recovery shard; repeat as needed.",
    )
    for name in (
        "dose-report", "dual-summary", "b3-run", "d2-binding",
        "dose-registry", "dual-registry", "protocol-lock", "split-contract",
    ):
        parser.add_argument(f"--{name}")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    candidates = discover_candidates(root)
    if args.mode == "discover":
        print(json.dumps({
            "status": "discovered",
            "candidates": candidates,
            "note": "No image, cache array, checkpoint tensor, or test asset was opened.",
            "test_assets_opened": 0,
        }, ensure_ascii=False, indent=2))
        return
    if not args.output:
        parser.error("--output is required for --mode export")
    overrides = {
        "dose_report": args.dose_report,
        "dual_summary": args.dual_summary,
        "b3_run": args.b3_run,
        "d2_binding": args.d2_binding,
        "dose_registry": args.dose_registry,
        "dual_registry": args.dual_registry,
        "protocol_lock": args.protocol_lock,
        "split_contract": args.split_contract,
    }
    selected, issues = resolve_selection(root, candidates, overrides)
    if issues:
        print(json.dumps({
            "status": "blocked",
            "blocked_message": "BLOCKED: BEST CHECKPOINT EVIDENCE",
            "issues": issues,
            "candidates": candidates,
            "test_assets_opened": 0,
        }, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = root / output
    supplemental = []
    for value in args.supplemental_dose_registry:
        path = Path(value).expanduser()
        supplemental.append(path.resolve() if path.is_absolute() else (root / path).resolve())
    result = export_snapshot(root, output, selected, tuple(supplemental))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
