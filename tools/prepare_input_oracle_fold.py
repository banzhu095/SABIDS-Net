from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.config import load_config
from sabids.engine.trainer import build_model
from sabids.utils import save_checkpoint, write_json
from tools.input_oracle_cv_common import (
    audit_d0_checkpoint, read_fold_groups, require_phase0, require_split_audit,
    resolve, sha256_file,
)
from tools.prepare_input_factorial import build_cache


def _state_sha(model: torch.nn.Module) -> str:
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit/cache/initialize one input-oracle CV fold")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="runs/input_oracle_cv")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--mode", choices=("audit", "cache", "initialize"), required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = resolve(root, args.output)
    phase0 = require_phase0(root, output)
    split = require_split_audit(output, args.fold)
    _, val = read_fold_groups(output, args.fold)
    fold_info = split["folds"][str(args.fold)]
    d0_manifest = root / fold_info["d0_manifest"]
    seg_manifest = root / fold_info["seg_manifest"]
    checkpoint = resolve(root, args.checkpoint) if args.checkpoint else output / f"d0_fold{args.fold}{'_smoke' if args.smoke_test else ''}" / "best.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing fold-specific D0 checkpoint: {checkpoint}")
    sealed = set(map(str, phase0["sealed_test_groups"]))
    audit = audit_d0_checkpoint(checkpoint, d0_manifest, val, sealed)
    fold_root = output / f"fold{args.fold}{'_smoke' if args.smoke_test else ''}"
    write_json(audit, fold_root / "d0_leakage_audit.json")
    if audit["status"] != "passed":
        raise RuntimeError(f"Fold D0 audit blocked: {fold_root / 'd0_leakage_audit.json'}")
    if args.mode == "audit" or args.dry_run:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    cache = fold_root / "cache"
    derived = root / f"Manifests/input_oracle_cv/fold_{args.fold}_input{'_smoke' if args.smoke_test else ''}.csv"
    if args.mode == "cache":
        build_cache(root, checkpoint, cache, args.device, seg_manifest, d0_manifest, derived, sealed, 2 if args.smoke_test else None)
        return
    snapshot = fold_root / "preseg_initialization.pth"
    if snapshot.exists():
        raise FileExistsError(f"Refusing to overwrite common initialization: {snapshot}")
    payload = torch.load(checkpoint, map_location="cpu")
    model = build_model(payload["config"])
    model.load_state_dict(payload["model"], strict=True)
    config = copy.deepcopy(load_config(root / "configs/input_factorial_common.yaml"))
    config["data"]["manifest"] = str(derived)
    config["train"]["pretrained"] = str(snapshot)
    config.setdefault("runtime", {}).update({
        "d0_checkpoint": str(checkpoint), "d0_checkpoint_sha256": sha256_file(checkpoint),
        "fold": args.fold, "fold_seg_manifest_sha256": sha256_file(seg_manifest),
    })
    save_checkpoint(snapshot, model, None, None, -1, float("-inf"), config)
    write_json({
        "snapshot": str(snapshot), "snapshot_sha256": sha256_file(snapshot),
        "model_state_sha256": _state_sha(model), "d0_checkpoint_sha256": sha256_file(checkpoint),
        "fold": args.fold, "test_assets_opened": 0,
    }, fold_root / "initialization_audit.json")


if __name__ == "__main__":
    main()
