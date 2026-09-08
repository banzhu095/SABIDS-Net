"""Create paired neutral initializations without opening any dataset asset."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.engine.trainer import build_model
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file, validate_checkpoint_config
from sabids.utils import save_checkpoint, seed_everything, write_json


TEMPLATES = {
    "order": "order_ds_pku37_v3.yaml",
    "input": "input_noisy_pku37_v3.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--expected-protocol-lock", required=True)
    parser.add_argument("--kind", choices=TEMPLATES, required=True)
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    lock_path = Path(args.expected_protocol_lock)
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    lock = load_protocol_lock(lock_path)
    records = []
    for fold in args.folds:
        for seed in args.seeds:
            cfg = load_config(root / "configs" / "next_stage_v3" / TEMPLATES[args.kind])
            cfg.update({"protocol_id": lock["protocol_id"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "fold": fold, "seed": seed})
            cfg.setdefault("runtime", {})["active_protocol_lock"] = lock
            seed_everything(seed, deterministic=True, use_cuda=False)
            model = build_model(cfg).cpu()
            output = root / "runs" / "anchors" / f"{args.kind}_common_{lock['protocol_id']}_fold{fold}_seed{seed}.pth"
            if output.exists():
                if not args.reuse_existing:
                    raise FileExistsError(f"Refusing to overwrite anchor: {output}; use --reuse-existing after audit")
                validate_checkpoint_config(torch.load(output, map_location="cpu", weights_only=False), lock, str(output))
            else:
                save_checkpoint(output, model, None, None, -1, float("-inf"), cfg)
            records.append({"kind": args.kind, "fold": fold, "seed": seed, "checkpoint": str(output), "model_state_sha256": sha256_file(output), "protocol_id": lock["protocol_id"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "test_assets_opened": 0})
    audit = root / "runs" / "anchors" / f"{args.kind}_common_initialization_audit.json"
    write_json({"status": "passed", "anchors": records, "test_assets_opened": 0}, audit)
    print(json.dumps({"status": "passed", "audit": str(audit), "anchors": records, "test_assets_opened": 0}, indent=2))


if __name__ == "__main__":
    main()
