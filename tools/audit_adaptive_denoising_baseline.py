"""Read-only provenance audit; reports are written only to a fresh report dir."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sabids.experiments.dose_response import SMOKE_NOTICE, formal_preflight, write_strict_json
from sabids.experiments.dose_response import sha256_file
from sabids.config import load_config
import pandas as pd


def readonly_inventory(root: Path) -> dict:
    """Inventory is informational, NEVER used to select a checkpoint/protocol."""
    candidates = []
    for cp in sorted((root / "runs").rglob("*.pth")):
        item = {"path": str(cp), "sha256": sha256_file(cp), "checkpoint_kind": cp.name,
                "smoke_or_pilot_name": "smoke" in str(cp).lower() or "pilot" in str(cp).lower()}
        config = cp.parent / "resolved_config.yaml"
        item["resolved_config"] = str(config) if config.is_file() else None
        if config.is_file():
            try:
                cfg = load_config(config)
                item.update(stage=cfg.get("train", {}).get("stage"),
                            restoration_mode=cfg.get("loss", {}).get("restoration_mode", "legacy/default"),
                            epochs=cfg.get("train", {}).get("epochs"), protocol_id=cfg.get("protocol_id"),
                            manifest=cfg.get("data", {}).get("manifest"),
                            target_size=cfg.get("data", {}).get("target_size"))
            except Exception as exc:
                item["config_error"] = str(exc)
        candidates.append(item)
    cohorts = []
    for manifest in sorted((root / "Manifests").rglob("manifest_seg_fold0.csv")):
        table = pd.read_csv(manifest, dtype=str).fillna("")
        cohorts.append({"manifest": str(manifest), "sha256": sha256_file(manifest),
                        "positions": table.groupby("split").group_id.nunique().to_dict(),
                        "frames": table.split.value_counts().to_dict()})
    return {"checkpoint_candidates_not_selected": candidates, "legacy_cohorts_not_selected": cohorts,
            "active_lock_paths": [str(p) for p in sorted((root / "Manifests").rglob("active_protocol_lock.json"))],
            "historical_cohort_metadata_only": {"train_positions": 13, "val_positions": 3, "train_frames": 588, "val_frames": 141},
            "test_assets_opened": 0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", default=".")
    p.add_argument("--denoiser-checkpoint")
    p.add_argument("--protocol-lock")
    p.add_argument("--split-contract")
    p.add_argument("--selection-rule", choices=("best_validation_psnr", "fixed_final"))
    p.add_argument("--training-asset-inventory")
    p.add_argument("--checkpoint-binding")
    p.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    p.add_argument("--output", default="reports/adaptive_denoising/preflight_v1")
    a = p.parse_args()
    root = Path(a.project_root).resolve()
    if a.mode == "smoke":
        result = {"status": "passed", "mode": "smoke", "notice": SMOKE_NOTICE,
                  "scope": "Synthetic chain only; formal provenance checks NOT passed",
                  "test_assets_opened": 0}
    else:
        result = formal_preflight(root, a.denoiser_checkpoint, a.protocol_lock,
                                  a.split_contract, a.selection_rule, a.training_asset_inventory,
                                  a.checkpoint_binding)
    result["readonly_inventory"] = readonly_inventory(root)
    out = (root / a.output).resolve()
    if (root / "reports/adaptive_denoising").resolve() not in out.parents:
        raise ValueError("Audit output must be a child of reports/adaptive_denoising")
    if out.exists():
        raise FileExistsError(f"Refusing existing audit output: {out}; choose a new --output")
    write_strict_json(out / "preflight_report.json", result)
    (out / "preflight_report.md").write_text(
        "# Dose preflight\n\nStatus: " + result["status"] + "\n\n" +
        (SMOKE_NOTICE + "\n\n" if a.mode == "smoke" else "") +
        "```json\n" + json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n```\n",
        encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
