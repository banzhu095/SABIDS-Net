from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.d2 import bind_best_checkpoint_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Bind a D1 best checkpoint to immutable training evidence")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--initial-inventory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--history", required=True)
    parser.add_argument("--resolved-config", required=True)
    parser.add_argument("--run-metadata", required=True)
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--split-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    def path(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    result = bind_best_checkpoint_evidence(
        root, path(args.initial_inventory), path(args.checkpoint), path(args.history),
        path(args.resolved_config), path(args.run_metadata), path(args.protocol_lock),
        path(args.split_contract), path(args.output),
    )
    print(result)


if __name__ == "__main__":
    main()
