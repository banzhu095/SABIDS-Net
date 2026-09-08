"""Extract the active protocol lock from D1 evidence and audit completion."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.experiments.protocol_lock import d1_completion_rows, extract_active_protocol_lock, find_d1_runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--write-lock", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    runs = find_d1_runs(root)
    lock = extract_active_protocol_lock(runs)
    lock_path = root / "Manifests" / str(lock["protocol_id"]) / "active_protocol_lock.json"
    if args.write_lock:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if lock_path.exists():
            previous = json.loads(lock_path.read_text(encoding="utf-8-sig"))
            immutable = ("protocol_id", "data_plan_sha256", "label_inventory_sha256", "split_contract_sha256")
            if any(previous.get(key) != lock.get(key) for key in immutable):
                raise FileExistsError(f"Refusing to replace a different protocol lock: {lock_path}")
        else:
            lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    report = root / "runs" / "reports" / f"d1_structure_{lock['protocol_id']}"
    report.mkdir(parents=True, exist_ok=True)
    rows = d1_completion_rows(root, lock, args.seeds)
    with (report / "d1_completion_matrix.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"status": "passed", "protocol_lock": str(lock_path), "source_runs": lock["source_run_ids"], "completion_matrix": str(report / "d1_completion_matrix.csv"), "rows": rows, "test_assets_opened": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
