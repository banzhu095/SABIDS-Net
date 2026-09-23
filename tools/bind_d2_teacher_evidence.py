#!/usr/bin/env python
"""Create an immutable native or fully evidenced legacy D2 teacher binding."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.d2_teacher import (
    bind_derived_legacy_teacher,
    bind_native_teacher,
)
from sabids.experiments.dose_response import resolve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--mode", choices=("native", "derived-legacy"), required=True)
    for name in (
        "checkpoint", "history", "resolved-config", "run-metadata",
        "protocol-lock", "split-contract", "parameter-audit", "output",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--initial-inventory")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--initialization-audit")
    parser.add_argument("--historical-protocol-evidence")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    path = lambda value: resolve(root, value)
    common = dict(
        root=root,
        checkpoint=path(args.checkpoint),
        history=path(args.history),
        resolved_config=path(args.resolved_config),
        run_metadata=path(args.run_metadata),
        protocol_lock=path(args.protocol_lock),
        split_contract=path(args.split_contract),
        parameter_audit=path(args.parameter_audit),
        output=path(args.output),
    )
    if args.mode == "native":
        missing = [name for name in (
            "initial_inventory", "initial_checkpoint", "initialization_audit"
        ) if not getattr(args, name)]
        if missing:
            raise ValueError(f"Native binding missing arguments: {missing}")
        result = bind_native_teacher(
            **common,
            initial_inventory=path(args.initial_inventory),
            initial_checkpoint=path(args.initial_checkpoint),
            initialization_audit=path(args.initialization_audit),
        )
    else:
        if not args.historical_protocol_evidence:
            raise ValueError("Derived legacy binding requires --historical-protocol-evidence")
        result = bind_derived_legacy_teacher(
            **common,
            historical_protocol_evidence=path(args.historical_protocol_evidence),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
