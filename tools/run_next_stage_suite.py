"""Fail-closed launcher for the three PKU37 next-stage experiments."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sabids.config import load_config, save_config
from sabids.experiments.protocol_lock import find_training_pid, load_protocol_lock, process_alive, run_matches_protocol_lock, sha256_file


SUITES = {
    "d1_structure": ["d1_d0_pku37_v3.yaml", "d1_structure_pku37_v3.yaml"],
    "input_image": [f"input_{arm}_pku37_v3.yaml" for arm in ("noisy", "d0", "d1", "clean")],
    "training_order": [f"order_{arm}_pku37_v3.yaml" for arm in ("ds", "sd", "alt")],
    "decoder_interaction_strength": [f"interaction_{arm}_pku37_v3.yaml" for arm in ("a00", "ad05", "ad10", "ad20", "as05", "as10", "as20")],
    "decoder_interaction_confirm": [f"interaction_j{arm}_strong_pku37_v3.yaml" for arm in ("00", "10", "01", "11")],
    "decoder_interaction_controls": ["interaction_j10_shuffle_pku37_v3.yaml", "interaction_j01_shuffle_pku37_v3.yaml", "interaction_j_self_adapter_pku37_v3.yaml"],
}


def _current_epoch(history: Path) -> int:
    if not history.is_file():
        return 0
    try:
        return max((int(float(row["epoch"])) for row in csv.DictReader(history.open("r", encoding="utf-8-sig")) if row.get("epoch")), default=0)
    except Exception:
        return 0


def _run_state(run_dir: Path, expected_epochs: int) -> tuple[str, dict[str, Any]]:
    lock_path = run_dir / ".run.lock"
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig")) if lock_path.is_file() else {}
    detected_pid = find_training_pid(run_dir.name)
    if (lock.get("pid") and process_alive(lock["pid"], lock.get("hostname"))) or detected_pid:
        if detected_pid and not lock.get("pid"):
            lock = {**lock, "pid": detected_pid, "source": "process_table"}
        return "running", lock
    epoch = _current_epoch(run_dir / "history.csv")
    if expected_epochs and epoch >= expected_epochs and (run_dir / "last.pth").is_file():
        return "completed", lock
    if run_dir.exists() and any(run_dir.iterdir()):
        return "interrupted", lock
    return "missing", lock


def _project_training_running(root: Path) -> bool:
    if os.name == "nt":
        return False
    process = subprocess.run(["pgrep", "-af", "train.py"], text=True, capture_output=True, check=False)
    return any(str(root) in line or " train.py " in f" {line} " for line in process.stdout.splitlines())


def _load_anchor(root: Path, lock: dict[str, Any], suite: str, fold: int, seed: int) -> tuple[Path | None, str | None, str | None]:
    tag = str(lock["protocol_id"])
    if suite in {"training_order", "input_image"}:
        kind = "order" if suite == "training_order" else "input"
        path = root / "runs" / "anchors" / f"{kind}_common_{tag}_fold{fold}_seed{seed}.pth"
        return path, sha256_file(path) if path.is_file() else None, f"missing {kind} common anchor" if not path.is_file() else None
    if suite in {"decoder_interaction_strength", "decoder_interaction_confirm", "decoder_interaction_controls"}:
        selection_path = root / "runs" / "anchors" / "interaction_anchor_selection.json"
        if not selection_path.is_file():
            return None, None, "missing interaction_anchor_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
        for key in ("protocol_id", "data_plan_sha256", "label_inventory_sha256"):
            if selection.get(key) != lock.get(key):
                return None, None, f"interaction anchor {key} mismatch"
        candidates = selection.get("anchors", {})
        value = candidates.get(f"fold{fold}_seed{seed}") or selection.get("anchor_checkpoint")
        path = Path(value) if value else None
        if path is not None and not path.is_absolute():
            path = (root / path).resolve()
        if path is None or not path.is_file():
            return path, None, "selected interaction anchor checkpoint is missing"
        digest = sha256_file(path)
        expected = selection.get("anchor_checkpoint_sha256_by_seed", {}).get(f"fold{fold}_seed{seed}") or selection.get("anchor_checkpoint_sha256")
        if expected and expected != digest:
            return path, digest, "selected interaction anchor SHA mismatch"
        if suite in {"decoder_interaction_confirm", "decoder_interaction_controls"}:
            strength = root / "runs" / "reports" / f"decoder_interaction_strength_{tag}" / "interaction_strength_lock.yaml"
            if not strength.is_file():
                return path, digest, "missing interaction_strength_lock.yaml"
        return path, digest, None
    return None, None, None


def _bind_config(root: Path, template: Path, lock: dict[str, Any], suite: str, fold: int, seed: int, mode: str, device: str) -> tuple[dict[str, Any], str, Path | None, str | None]:
    cfg = load_config(template)
    cfg.update({"protocol_id": lock["protocol_id"], "manifest_root": lock["manifest_root"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "seed": seed, "fold": fold, "device": device})
    cfg.setdefault("runtime", {})["active_protocol_lock"] = lock
    manifest_root = Path(str(lock["manifest_root"]))
    if not manifest_root.is_absolute():
        manifest_root = (root / manifest_root).resolve()
    manifest_name = Path(str(cfg["data"]["manifest"])).name
    cfg["data"]["manifest"] = str(manifest_root / manifest_name)
    cfg["data"]["root"] = str(root)
    if lock.get("input_resolution"):
        cfg["data"]["target_size"] = list(lock["input_resolution"])
    if lock.get("normalization"):
        cfg["data"]["normalization"] = lock["normalization"]
    anchor, anchor_sha, blocker = _load_anchor(root, lock, suite, fold, seed)
    input_arm = cfg.get("input_arm")
    if suite == "input_image":
        probe_manifest = manifest_root / f"input_probe_manifest_seed{seed}.csv"
        cfg["data"]["manifest"] = str(probe_manifest)
        cfg["data"]["input_column"] = {"noisy": "image_path", "clean": "clean_path", "d0": "d0_path", "d1": "d1_path"}[input_arm]
    if suite in {"decoder_interaction_confirm", "decoder_interaction_controls"}:
        strength_path = root / "runs" / "reports" / f"decoder_interaction_strength_{lock['protocol_id']}" / "interaction_strength_lock.yaml"
        if strength_path.is_file():
            strength = yaml.safe_load(strength_path.read_text(encoding="utf-8-sig")) or {}
            for key in ("protocol_id", "data_plan_sha256", "label_inventory_sha256"):
                if strength.get(key) != lock.get(key):
                    raise RuntimeError(f"Interaction strength lock {key} mismatch")
            if cfg.get("model", {}).get("d2s_enabled"):
                cfg["model"]["strong_d2s_rho"] = float(strength["selected_d2s_rho"])
            if cfg.get("model", {}).get("s2d_enabled"):
                cfg["model"]["strong_s2d_rho"] = float(strength["selected_s2d_rho"])
    if cfg.get("model", {}).get("d2s_source_mode") == "shuffled_cross" or cfg.get("model", {}).get("s2d_source_mode") == "shuffled_cross":
        mapping = manifest_root / f"interaction_shuffle_seed{seed}.csv"
        cfg["data"]["guidance_mapping"] = str(mapping)
        if not mapping.is_file():
            blocker = blocker or "missing deterministic interaction shuffle mapping"
    original_stem = template.stem
    compact_tag = str(lock["protocol_id"]).replace("_binary", "")
    run_stem = re.sub(r"pku37_v\d+", compact_tag, original_stem)
    mode_tag = "_pilot" if mode == "pilot" else ""
    run_id = f"{run_stem}{mode_tag}_fold{fold}_seed{seed}"
    cfg["train"]["output_dir"] = str(root / "runs" / "current" / run_id)
    cfg["train"]["early_stopping_patience"] = int(cfg["train"].get("epochs", 60)) + 1
    if mode == "pilot":
        cfg["data"]["max_train_samples"] = 2
        cfg["data"]["max_val_samples"] = 2
        cfg["train"]["num_workers"] = 0
        cfg.setdefault("evaluation", {})["num_workers"] = 0
        if suite == "training_order":
            cfg["train"]["epochs"] = 3
            cfg["train"]["schedule_epochs"] = (
                [2, 0, 1]
                if cfg["train"].get("schedule") == "order_alt"
                else [1, 1, 1]
            )
        else:
            cfg["train"]["epochs"] = 2
        cfg["train"]["early_stopping_patience"] = cfg["train"]["epochs"] + 1
    if anchor is not None and blocker is None:
        cfg["train"]["pretrained"] = str(anchor)
        cfg["train"]["strict_pretrained"] = True
        cfg["runtime"]["anchor_checkpoint_sha256"] = anchor_sha
    if suite == "input_image" and not Path(cfg["data"]["manifest"]).is_file():
        blocker = blocker or "missing input_probe_manifest.csv"
    return cfg, run_id, anchor, blocker


def _launch(command: list[str], root: Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, cwd=root)
    lock_path = run_dir / ".run.lock"
    record = {"pid": process.pid, "hostname": socket.gethostname(), "start_time": datetime.now(timezone.utc).isoformat(), "command": command, "status": "running"}
    lock_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    code = process.wait()
    record.update({"status": "completed" if code == 0 else "failed", "returncode": code, "finished_at": datetime.now(timezone.utc).isoformat()})
    lock_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if code:
        raise subprocess.CalledProcessError(code, command)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol-id")
    parser.add_argument("--expected-protocol-lock", required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--folds", nargs="+", type=int, default=[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-running", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--resume-interrupted", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-concurrent-training", action="store_true")
    args = parser.parse_args()
    if args.plan_only and args.execute:
        parser.error("--plan-only and --execute are mutually exclusive")
    root = Path(args.project_root).resolve()
    lock_path = Path(args.expected_protocol_lock)
    if not lock_path.is_absolute():
        lock_path = root / lock_path
    lock = load_protocol_lock(lock_path)
    if args.protocol_id and args.protocol_id != lock["protocol_id"]:
        raise SystemExit("BLOCKED: --protocol-id differs from active protocol lock")
    other_training = _project_training_running(root)
    global_block = args.suite != "d1_structure" and other_training and not args.allow_concurrent_training
    plans: list[dict[str, Any]] = []
    commands: list[tuple[list[str], Path]] = []
    for fold in args.folds:
        for seed in args.seeds:
            for name in SUITES[args.suite]:
                cfg, run_id, anchor, blocker = _bind_config(root, root / "configs" / "next_stage_v3" / name, lock, args.suite, fold, seed, args.mode, args.device)
                run_dir = Path(cfg["train"]["output_dir"])
                if args.suite == "d1_structure":
                    prefix = "d1_denoise_struct" if "d1_structure" in name else "d1_denoise_d0"
                    semantic_matches = []
                    for path in (root / "runs" / "current").glob(f"{prefix}*seed{seed}"):
                        if args.mode == "full" and "_pilot_" in path.name: continue
                        if args.mode == "pilot" and "_pilot_" not in path.name: continue
                        candidate_path = next((path / item for item in ("resolved_config.yaml", "config_resolved.yaml") if (path / item).is_file()), None)
                        candidate = load_config(candidate_path) if candidate_path else {}
                        if args.mode == "full" and int(candidate.get("train", {}).get("epochs", 0)) < 60: continue
                        if run_matches_protocol_lock(path, lock):
                            semantic_matches.append(path)
                    if len(semantic_matches) == 1:
                        run_dir = semantic_matches[0]
                        run_id = run_dir.name
                        cfg["train"]["output_dir"] = str(run_dir)
                    elif len(semantic_matches) > 1:
                        blocker = "multiple existing D1 semantic matches; audit manually"
                state, stale_lock = _run_state(run_dir, int(cfg["train"]["epochs"]))
                allowed = blocker is None and not global_block
                action = "run" if allowed else "blocked"
                if state == "running":
                    allowed = False; action = "skip" if args.skip_running else "blocked"
                elif state == "completed":
                    allowed = False; action = "skip" if args.skip_completed else "blocked"
                elif state == "interrupted":
                    if args.resume_interrupted and (run_dir / "last.pth").is_file():
                        cfg["train"]["resume"] = str(run_dir / "last.pth"); action = "resume"
                    else:
                        allowed = False; action = "blocked"
                        blocker = blocker or "existing interrupted run; pass --resume-interrupted after checking SHA"
                if global_block:
                    blocker = "another project training process is active; full suite remains plan-only"
                resolved = root / "runs" / "next_stage_launch_configs" / f"{run_id}.yaml"
                save_config(cfg, resolved)
                command = [sys.executable, "train.py", "--config", str(resolved)]
                plan = {"run_id": run_id, "config": str(resolved), "source_anchor": str(anchor or ""), "anchor_sha256": cfg.get("runtime", {}).get("anchor_checkpoint_sha256", ""), "protocol_id": lock["protocol_id"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "requested_d2s_rho": cfg.get("model", {}).get("strong_d2s_rho", 0.0), "requested_s2d_rho": cfg.get("model", {}).get("strong_s2d_rho", 0.0), "epochs": cfg["train"]["epochs"], "output_dir": str(run_dir), "existing_state": state, "action": action, "allowed": allowed, "blocker": blocker or "", "stale_lock_preserved": bool(stale_lock and state != "running")}
                plans.append(plan)
                if allowed and action in {"run", "resume"}:
                    commands.append((command, run_dir))
    if args.execute:
        blocked = [item for item in plans if item["action"] == "blocked"]
        if blocked:
            raise SystemExit("BLOCKED: " + "; ".join(f"{item['run_id']}: {item['blocker'] or item['existing_state']}" for item in blocked))
        for command, run_dir in commands:
            _launch(command, root, run_dir)
    print(json.dumps({"status": "executed" if args.execute else "planned", "protocol_lock": str(lock_path), "suite": args.suite, "run_count": len(plans), "plans": plans, "test_assets_opened": 0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
