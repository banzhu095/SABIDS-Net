from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from sabids.experiments.adaptive_evidence import (
    discover_candidates,
    export_snapshot,
    summarize_dose,
)


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_discovery_is_shallow_and_does_not_enumerate_oracle_grid(tmp_path: Path) -> None:
    report = tmp_path / "reports/adaptive_denoising/dose_seed42_best_sensitivity_x"
    report.mkdir(parents=True)
    (report / "metrics_by_position.csv").write_text("arm,vessel_dice\n", encoding="utf-8")
    noisy = tmp_path / "reports/adaptive_denoising/seg_guided_adaptive_v1/x/oracle/evaluations/a"
    noisy.mkdir(parents=True)
    (noisy / "summary.json").write_text("{}", encoding="utf-8")
    found = discover_candidates(tmp_path)
    assert found["dose_report"] == ["reports/adaptive_denoising/dose_seed42_best_sensitivity_x"]
    assert "seg_guided_adaptive_v1" not in json.dumps(found)


def test_position_equal_dose_summary_parses_arm_alpha(tmp_path: Path) -> None:
    source = tmp_path / "metrics_by_position.csv"
    pd.DataFrame([
        {"arm": "d2_task:alpha=0.25", "group_id": "a", "split": "val", "layer_dice": .8, "vessel_dice": .6},
        {"arm": "d2_task:alpha=0.25", "group_id": "b", "split": "val", "layer_dice": .9, "vessel_dice": .7},
        {"arm": "d2_task:alpha=0.5", "group_id": "a", "split": "val", "layer_dice": .7, "vessel_dice": .8},
    ]).to_csv(source, index=False)
    summary, audit = summarize_dose(source)
    assert summary["alpha"].tolist() == [.25, .5]
    assert audit["position_count"] == 2
    assert audit["best_layer_alpha"] == .25
    assert audit["best_vessel_alpha"] == .5


def test_export_hashes_opaque_checkpoint_and_never_copies_it(tmp_path: Path) -> None:
    root = tmp_path
    checkpoint = root / "runs/d2/best_task_preserving.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    binding = checkpoint.parent / "checkpoint_binding_best_task_preserving.json"
    _json(binding, {"checkpoint_path": "runs/d2/best_task_preserving.pth", "checkpoint_sha256": digest})
    b3 = root / "runs/b3"
    b3.mkdir(parents=True)
    (b3 / "resolved_config.yaml").write_text("seed: 42\n", encoding="utf-8")
    dose = root / "reports/dose"
    dose.mkdir(parents=True)
    pd.DataFrame([
        {"arm": "d2_task:alpha=0.25", "group_id": "p1", "split": "val", "layer_dice": .8, "vessel_dice": .7}
    ]).to_csv(dose / "metrics_by_position.csv", index=False)
    dual = root / "reports/dual/summary.json"
    _json(dual, {"status": "passed"})
    protocol = root / "protocol.json"
    _json(protocol, {"protocol_id": "pku37_binary_v3"})
    split = root / "split.yaml"
    split.write_text(yaml.safe_dump({"validation": ["p1"]}), encoding="utf-8")
    dose_registry = root / "dose_registry.json"
    dual_registry = root / "dual_registry.json"
    _json(dose_registry, {"test_assets_opened": 0})
    _json(dual_registry, {"test_assets_opened": 0})
    recovery_registry = root / "recovery/preparation_registry.json"
    _json(recovery_registry, {"test_assets_opened": 0, "alphas": [1.25]})
    output = root / "snapshot"
    result = export_snapshot(root, output, {
        "dose_report": dose,
        "dual_summary": dual,
        "b3_run": b3,
        "d2_binding": binding,
        "dose_registry": dose_registry,
        "dual_registry": dual_registry,
        "protocol_lock": protocol,
        "split_contract": split,
    }, (recovery_registry,))
    assert result["status"] == "passed"
    assert result["d2_checkpoint"]["sha256_matches"] is True
    assert result["checkpoint_bytes_copied"] is False
    assert not list(output.rglob("*.pth"))
    assert (output / "dose_registry_supplemental_01/preparation_registry.json").is_file()
    assert result["supplemental_dose_registries"] == ["recovery/preparation_registry.json"]
    assert result["image_assets_opened"] == result["test_assets_opened"] == 0


def test_export_blocks_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runs/d2/best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"changed")
    binding = checkpoint.parent / "checkpoint_binding_best_task_preserving.json"
    _json(binding, {"checkpoint_path": "runs/d2/best.pth", "checkpoint_sha256": "0" * 64})
    b3 = tmp_path / "b3"; b3.mkdir()
    dose = tmp_path / "dose"; dose.mkdir()
    pd.DataFrame([{"arm": "d2_task:alpha=0.25", "group_id": "p1", "split": "val", "layer_dice": .8, "vessel_dice": .7}]).to_csv(dose / "metrics_by_position.csv", index=False)
    dual = tmp_path / "dual.json"; _json(dual, {})
    protocol = tmp_path / "protocol.json"; _json(protocol, {})
    split = tmp_path / "split.yaml"; split.write_text("{}", encoding="utf-8")
    registry = tmp_path / "registry.json"; _json(registry, {"test_assets_opened": 0})
    result = export_snapshot(tmp_path, tmp_path / "out", {
        "dose_report": dose, "dual_summary": dual, "b3_run": b3,
        "d2_binding": binding, "dose_registry": registry, "dual_registry": registry,
        "protocol_lock": protocol, "split_contract": split,
    })
    assert result["status"] == "blocked"
    assert result["blocked_message"] == "BLOCKED: BEST CHECKPOINT EVIDENCE"
