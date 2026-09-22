#!/usr/bin/env python
"""Fail-closed seed-42 gate; it never launches or prepares seed 43/44 runs."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.dose_response import write_strict_json_exclusive
from sabids.experiments.protocol_lock import sha256_file


BOOLEAN_CHECKS = (
    "all_metrics_finite",
    "teacher_changed_parameter_count_zero",
    "frozen_changed_parameter_count_zero",
    "checkpoint_bindings_passed",
    "alpha_zero_identity_passed",
    "d2_task_improves_structure_endpoint_over_d1",
    "no_structure_hallucination",
    "psnr_above_noisy",
    "at_least_two_of_three_positions_same_direction",
    "not_driven_by_single_frame",
    "fixed_final_and_predefined_best_direction_agree",
    "dose_and_checkpoint_rules_frozen",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Audited seed-42 gate evidence JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    evidence = json.loads(source.read_text(encoding="utf-8-sig"))
    failures = [name for name in BOOLEAN_CHECKS if evidence.get(name) is not True]
    if int(evidence.get("validation_position_count", -1)) != 3:
        failures.append("validation_position_count_is_three")
    if int(evidence.get("seed", -1)) != 42:
        failures.append("seed_is_42")
    numeric = evidence.get("numeric_endpoints", {})
    if not numeric or not all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in numeric.values()
    ):
        failures.append("numeric_endpoints_finite_and_present")
    result = {
        "schema_version": "d2-seed42-gate-v1",
        "status": "passed" if not failures else "blocked",
        "seed": 42,
        "checks": {name: evidence.get(name) is True for name in BOOLEAN_CHECKS},
        "failures": failures,
        "source": str(source),
        "source_sha256": sha256_file(source),
        "multi_seed_configs_authorized": not failures,
        "multi_seed_training_authorized": False,
        "note": "A passed gate permits later config generation only; training still needs user confirmation.",
        "test_assets_opened": 0,
    }
    write_strict_json_exclusive(Path(args.output).expanduser().resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
