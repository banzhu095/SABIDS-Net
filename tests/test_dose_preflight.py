"""Fabricated provenance fixtures test gates, never scientific checkpoints."""
import csv
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from sabids.config import load_config, save_config
from sabids.engine.trainer import build_model
from sabids.experiments import dose_response as dose

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def evidence(tmp_path):
    protocol = tmp_path / "Manifests/fixture_protocol"
    protocol.mkdir(parents=True)
    source = tmp_path / "assets"
    source.mkdir()
    rows, labels = [], []
    for index, split in enumerate(("train", "val", "test")):
        group = f"pku_{index + 1:04d}"
        row = {"sample_id": f"{group}_f01", "group_id": group, "patient_id": group,
               "dataset": "PKU37", "split": split}
        labelrow = {"group_id": group}
        for role in ("image", "clean", "layer_mask", "vessel_mask"):
            p = source / f"{index}_{role}.npy" if split != "test" else tmp_path / "sealed" / f"{index}_{role}.npy"
            if split != "test":
                np.save(p, np.ones((24, 32), np.float32) * (.5 if role in {"image", "clean"} else 1))
            row[f"{role}_path"] = str(p)
            if role in {"layer_mask", "vessel_mask"}:
                task = role.split("_")[0]
                labelrow[f"{task}_path"] = str(p)
                labelrow[f"{task}_sha256"] = dose.sha256_file(p) if split != "test" else "sealed-metadata-only"
        rows.append(row)
        labels.append(labelrow)
    allrows = pd.DataFrame(rows)
    joint = allrows[allrows.split.ne("test")].copy()
    sealed = allrows[allrows.split.eq("test")].copy()
    inventory = allrows[["group_id", "split", "clean_path", "layer_mask_path", "vessel_mask_path"]]
    labeltable = pd.DataFrame(labels)
    contract = tmp_path / "contract.yaml"
    contract.write_text("protocol_id: fixture_protocol\nvalidation_positions: [pku_0002]\ntest_positions: [pku_0003]\n")
    for name, df in {"train_denoise": joint, "train_segment": joint, "train_joint": joint,
                     "test_sealed": sealed, "label_inventory": labeltable, "dataset_inventory": inventory}.items():
        df.to_csv(protocol / f"{name}.csv", index=False)
    sha_table = lambda df: hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()
    lock = {"protocol_id": "fixture_protocol", "manifest_root": str(protocol),
            "data_plan_sha256": sha_table(allrows), "label_inventory_sha256": sha_table(labeltable),
            "dataset_inventory_sha256": sha_table(inventory), "split_contract_sha256": dose.sha256_file(contract),
            "train_positions": ["pku_0001"], "validation_positions": ["pku_0002"],
            "sealed_test_positions": ["pku_0003"], "input_resolution": [32, 32],
            "git_commit_at_d1_start": "fabricated-unit-fixture", "test_assets_opened": 0}
    lp = protocol / "active_protocol_lock.json"
    dose.write_strict_json(lp, lock)
    dose.write_strict_json(protocol / "protocol_audit.json", {**lock, "status": "passed"})
    cfg = load_config(ROOT / "configs/base.yaml")
    cfg["model"].update(channels=[4, 8], encoder_depths=[1, 1], decoder_depth=1, interaction_levels=[1])
    cfg["train"].update(stage="denoise", epochs=60, monitor="psnr", output_dir=str(tmp_path / "formal_run"))
    cfg["loss"]["restoration_mode"] = "structure_d1"
    cfg["data"].update(manifest=str(protocol / "train_denoise.csv"), root=str(tmp_path), target_size=[32, 32], normalization="fixed")
    cfg["runtime"].update(active_protocol_lock=lock,
        manifest_sha256=dose.sha256_file(protocol / "train_denoise.csv"),
        effective_split_sha256=hashlib.sha256(b"train:pku_0001\nval:pku_0002").hexdigest(),
        train_val_noisy_clean_asset_sha256=dose.stable_sha(dose.asset_inventory(tmp_path, joint, False)))
    run = tmp_path / "formal_run"
    run.mkdir()
    save_config(cfg, run / "resolved_config.yaml")
    cp = run / "last.pth"
    torch.save({"model": build_model(cfg).state_dict(), "config": cfg, "epoch": 59}, cp)
    pd.DataFrame([{"epoch": i, "val_psnr": 30.} for i in range(1, 61)]).to_csv(run / "history.csv", index=False)
    return {"root": tmp_path, "checkpoint": cp, "lock_path": lp, "contract": contract,
            "cfg": cfg, "lock": lock, "joint": joint, "protocol": protocol}


def audit(e):
    return dose.formal_preflight(e["root"], str(e["checkpoint"]), str(e["lock_path"]),
                                 str(e["contract"]), "fixed_final")


def rewrite_checkpoint(e, cfg):
    raw = torch.load(e["checkpoint"], map_location="cpu", weights_only=False)
    raw["config"] = cfg
    torch.save(raw, e["checkpoint"])
    save_config(cfg, e["checkpoint"].parent / "resolved_config.yaml")


def write_dynamic_history(path):
    """Legacy 287 -> 295 rows: the old val_psnr slot becomes 0.13005."""
    header = ["epoch", "legacy_metric", "val_psnr"] + [f"legacy_{i}" for i in range(284)]
    assert len(header) == 287
    rows = []
    for epoch in range(1, 61):
        if epoch <= 34:
            row = [str(epoch), "0.2", str(30.0 + epoch / 100)] + [""] * 284
        else:
            # Eight new, unlabelled values precede the actual PSNR. The stale
            # header therefore maps index 2 to 0.13005, not to the true PSNR.
            row = [str(epoch), "0.2", "0.13005"] + ["new"] * 7
            row += [str(35.0 + epoch / 100)] + [""] * (295 - 11)
        rows.append(row)
    assert [len(row) for row in rows[:34]] == [287] * 34
    assert [len(row) for row in rows[34:]] == [295] * 26
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows([header, *rows])


def write_epoch_history(path, epochs, val_psnr=None):
    header = ["epoch"] + (["val_psnr"] if val_psnr is not None else [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for index, epoch in enumerate(epochs):
            writer.writerow([epoch] + ([val_psnr[index]] if val_psnr is not None else []))


def test_formal_pass_never_opens_sealed_assets(evidence, monkeypatch):
    original = dose.sha256_file
    def guarded(path):
        assert "sealed" not in Path(path).parts, f"Forbidden test asset opened: {path}"
        return original(path)
    monkeypatch.setattr(dose, "sha256_file", guarded)
    result = audit(evidence)
    assert result["status"] == "passed", result
    assert result["test_assets_opened"] == 0
    assert result["d1_epoch"] == 60


def test_formal_passes_new_bound_training_inventory(evidence):
    e = evidence
    cfg = copy.deepcopy(e["cfg"])
    cfg["runtime"].pop("train_val_noisy_clean_asset_sha256")
    rewrite_checkpoint(e, cfg)
    filtered = e["joint"][e["joint"].split.isin(["train", "val"])]
    initial = e["checkpoint"].parent / "training_asset_inventory_initial.json"
    cfg["training_asset_evidence"] = {"enabled": True, "project_root": str(e["root"])}
    dose.create_training_asset_evidence(e["root"], cfg, filtered, initial)
    records = dose.asset_inventory(e["root"], filtered, False)
    bound = e["checkpoint"].parent / "training_asset_inventory_last.json"
    dose.bind_training_asset_evidence(
        initial, bound, e["checkpoint"], Path(cfg["data"]["manifest"]),
        records, completed_epochs=60, configured_epochs=60,
    )
    result = dose.formal_preflight(
        e["root"], str(e["checkpoint"]), str(e["lock_path"]),
        str(e["contract"]), "fixed_final", str(bound),
    )
    assert result["status"] == "passed", result


def test_formal_rejects_forged_marker_without_initial_chain(evidence):
    e = evidence
    cfg = copy.deepcopy(e["cfg"])
    cfg["runtime"].pop("train_val_noisy_clean_asset_sha256")
    rewrite_checkpoint(e, cfg)
    records = dose.asset_inventory(e["root"], e["joint"], False)
    forged = e["checkpoint"].parent / "forged_inventory.json"
    dose.write_strict_json(forged, {
        "recorded_at_training": True,
        "checkpoint_sha256": dose.sha256_file(e["checkpoint"]),
        "manifest_sha256": dose.sha256_file(cfg["data"]["manifest"]),
        "records": records,
        "records_sha256": dose.stable_sha(records),
        "train_val_noisy_clean_asset_sha256": dose.stable_sha(records),
        "completed_epochs": 60,
        "selection_rule": "fixed_final",
    })
    result = dose.formal_preflight(
        e["root"], str(e["checkpoint"]), str(e["lock_path"]),
        str(e["contract"]), "fixed_final", str(forged),
    )
    assert result["status"] == "blocked"
    assert "initial evidence chain" in result["issues"][0]


def test_dynamic_history_fixed_final_passes_but_best_psnr_fails_closed(evidence):
    history = evidence["checkpoint"].parent / "history.csv"
    write_dynamic_history(history)
    with history.open("r", newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.reader(handle))
    assert len(raw_rows) == 61
    assert len(raw_rows[1]) == 287 and len(raw_rows[34]) == 287
    assert len(raw_rows[35]) == 295 and len(raw_rows[-1]) == 295
    assert float(raw_rows[35][2]) == pytest.approx(0.13005)

    fixed = audit(evidence)
    assert fixed["status"] == "passed", fixed
    assert fixed["history_schema_drift_detected"] is True
    assert fixed["history_row_width_distribution"] == {"287": 34, "295": 26}
    assert fixed["val_psnr_trusted"] is False
    assert fixed["selection_history_audit"]["history_row_count"] == 60
    assert fixed["selection_history_audit"]["last_epoch"] == 60

    best = evidence["checkpoint"].with_name("best.pth")
    payload = torch.load(evidence["checkpoint"], map_location="cpu", weights_only=False)
    torch.save(payload, best)
    dose.write_strict_json(
        best.parent / "run_metadata.json", {"best_checkpoint_sha256": dose.sha256_file(best)}
    )
    blocked = dose.formal_preflight(
        evidence["root"], str(best), str(evidence["lock_path"]),
        str(evidence["contract"]), "best_validation_psnr"
    )
    assert blocked["status"] == "blocked"
    assert blocked["history_schema_drift_detected"] is True
    assert blocked["val_psnr_trusted"] is False
    assert "schema drift prevents trustworthy val_psnr" in blocked["issues"][0]


@pytest.mark.parametrize("epochs,match", [
    ([*range(1, 30), *range(31, 62)], "strictly equal"),
    ([*range(1, 60), 59], "duplicate epoch"),
    (list(range(1, 60)), "row count"),
    (["NaN", *range(2, 61)], "not an integer"),
])
def test_selection_history_rejects_missing_duplicate_short_or_nonfinite_epoch(tmp_path, epochs, match):
    path = tmp_path / "history.csv"
    write_epoch_history(path, epochs)
    with pytest.raises(dose.SelectionHistoryError, match=match):
        dose.audit_selection_history(path, 60, "fixed_final")


def test_rectangular_best_history_is_trusted_and_nonfinite_psnr_is_rejected(tmp_path):
    path = tmp_path / "history.csv"
    values = [32.0 if epoch == 3 else 30.0 for epoch in range(1, 61)]
    write_epoch_history(path, list(range(1, 61)), values)
    result = dose.audit_selection_history(path, 60, "best_validation_psnr")
    assert result["history_schema_drift_detected"] is False
    assert result["history_row_width_distribution"] == {"2": 60}
    assert result["val_psnr_trusted"] is True
    assert result["best_epoch"] == 3
    values[9] = float("nan")
    write_epoch_history(path, list(range(1, 61)), values)
    with pytest.raises(dose.SelectionHistoryError, match="nonfinite"):
        dose.audit_selection_history(path, 60, "best_validation_psnr")


def test_fixed_final_requires_only_complete_integer_epoch_column(tmp_path):
    path = tmp_path / "history.csv"
    write_epoch_history(path, list(range(1, 61)))
    result = dose.audit_selection_history(path, 60, "fixed_final")
    assert result["history_schema_drift_detected"] is False
    assert result["val_psnr_trusted"] is False
    assert "val_psnr_column_index" not in result and "best_epoch" not in result


@pytest.mark.parametrize("kind,match", [
    ("missing_checkpoint", "Missing checkpoint"), ("auto", "auto forbidden"),
    ("smoke", "Smoke/pilot"), ("missing_lock", "missing/nonunique"),
    ("duplicate_lock", "missing/nonunique"), ("protocol", "protocol_id mismatch"),
    ("data_sha", "data_plan_sha256 mismatch"), ("missing_manifest", "Missing checkpoint manifest"),
    ("manifest_sha", "manifest SHA"), ("untraceable", "untraceable"),
    ("split_leak", "leakage"), ("label_sha", "label_inventory.csv SHA"),
    ("pixel_sha_missing", "pixel fingerprint"), ("pixel_drift", "pixel fingerprint"),
    ("short_budget", "Smoke/short"), ("selection_missing", "Freeze selection"),
    ("resolved_missing", "resolved_config"), ("contract_sha", "contract SHA"),
    ("wrong_source_asset", "D1 manifest row differs"),
    ("incomplete_history", "complete fixed budget"),
    ("fixed_wrong_checkpoint_name", "Incomplete fixed-final"),
    ("fixed_wrong_checkpoint_epoch", "Incomplete fixed-final"),
])
def test_formal_refuses_invalid_evidence(evidence, kind, match):
    e = evidence
    cfg = copy.deepcopy(e["cfg"])
    cp, lp, rule = str(e["checkpoint"]), str(e["lock_path"]), "fixed_final"
    if kind == "missing_checkpoint":
        cp = str(e["root"] / "absent.pth")
    elif kind == "auto":
        cp = "auto"
    elif kind == "smoke":
        new = e["root"] / "smoke.pth"
        new.write_bytes(e["checkpoint"].read_bytes())
        cp = str(new)
    elif kind == "missing_lock":
        e["lock_path"].unlink()
    elif kind == "duplicate_lock":
        dose.write_strict_json(e["root"] / "Manifests/other/active_protocol_lock.json", e["lock"])
    elif kind in {"protocol", "data_sha"}:
        cfg["runtime"]["active_protocol_lock"]["protocol_id" if kind == "protocol" else "data_plan_sha256"] = "different"
        rewrite_checkpoint(e, cfg)
    elif kind == "missing_manifest":
        Path(cfg["data"]["manifest"]).unlink()
    elif kind == "manifest_sha":
        cfg["runtime"]["manifest_sha256"] = "wrong"
        rewrite_checkpoint(e, cfg)
    elif kind == "untraceable":
        torch.save({"model": {"random": torch.ones(1)}}, e["checkpoint"])
    elif kind == "split_leak":
        table = e["joint"].copy()
        table.loc[table.split.eq("val"), "group_id"] = "pku_0001"
        table.to_csv(cfg["data"]["manifest"], index=False)
        cfg["runtime"]["manifest_sha256"] = dose.sha256_file(cfg["data"]["manifest"])
        rewrite_checkpoint(e, cfg)
    elif kind == "label_sha":
        p = e["protocol"] / "label_inventory.csv"
        p.write_text(p.read_text() + "\nanything\n")
    elif kind == "pixel_sha_missing":
        cfg["runtime"].pop("train_val_noisy_clean_asset_sha256")
        rewrite_checkpoint(e, cfg)
    elif kind == "pixel_drift":
        np.save(e["joint"].iloc[0].image_path, np.zeros((24, 32), np.float32))
    elif kind == "short_budget":
        cfg["train"]["epochs"] = 2
        rewrite_checkpoint(e, cfg)
    elif kind == "selection_missing":
        rule = None
    elif kind == "resolved_missing":
        (e["checkpoint"].parent / "resolved_config.yaml").unlink()
    elif kind == "contract_sha":
        e["contract"].write_text("different")
    elif kind == "wrong_source_asset":
        table = e["joint"].copy()
        table.loc[0, "image_path"] = table.loc[1, "image_path"]
        table.to_csv(cfg["data"]["manifest"], index=False)
        cfg["runtime"]["manifest_sha256"] = dose.sha256_file(cfg["data"]["manifest"])
        rewrite_checkpoint(e, cfg)
    elif kind == "incomplete_history":
        pd.DataFrame([{"epoch": 60, "val_psnr": 30.}]).to_csv(e["checkpoint"].parent / "history.csv", index=False)
    elif kind == "fixed_wrong_checkpoint_name":
        wrong = e["checkpoint"].with_name("best.pth")
        wrong.write_bytes(e["checkpoint"].read_bytes())
        cp = str(wrong)
    elif kind == "fixed_wrong_checkpoint_epoch":
        raw = torch.load(e["checkpoint"], map_location="cpu", weights_only=False)
        raw["epoch"] = 58
        torch.save(raw, e["checkpoint"])
    result = dose.formal_preflight(e["root"], cp, lp, str(e["contract"]), rule)
    assert result["status"] == "blocked"
    assert any(match in issue for issue in result["issues"]), result
    assert result["test_assets_opened"] == 0


def test_segmentation_manifest_cannot_override_locked_cohort(evidence):
    table = evidence["joint"].copy()
    table.loc[0, "image_path"] = "unregistered_asset.npy"
    table.to_csv(evidence["protocol"] / "train_segment.csv", index=False)
    result = audit(evidence)
    assert result["status"] == "blocked"
    assert "differs from locked data plan" in result["issues"][0]


def _best_binding_sources(e):
    cfg = copy.deepcopy(e["cfg"])
    cfg["protocol_id"] = e["lock"]["protocol_id"]
    cfg["runtime"]["git_commit"] = dose.git_commit(e["root"])
    cfg["training_asset_evidence"] = {
        "enabled": True, "project_root": str(e["root"])
    }
    save_config(cfg, e["checkpoint"].parent / "resolved_config.yaml")
    cp = e["checkpoint"].with_name("best.pth")
    raw = torch.load(e["checkpoint"], map_location="cpu", weights_only=False)
    raw["epoch"] = 2
    raw["config"] = cfg
    raw["best_metric"] = 32.0
    raw["optimizer"] = {"state": {0: {"step": torch.tensor(3)}}, "param_groups": []}
    raw["scheduler"] = {"last_epoch": 3}
    torch.save(raw, cp)
    pd.DataFrame([{"epoch": i, "val_psnr": 32. if i == 3 else 30.} for i in range(1, 61)]).to_csv(cp.parent / "history.csv", index=False)
    dose.write_strict_json(cp.parent / "run_metadata.json", {
        "best_checkpoint_sha256": dose.sha256_file(cp), "best_epoch": 3,
        "monitor": "psnr", "best_metric": 32.0,
        "best_checkpoint": str(cp.resolve()), "run_id": cp.parent.name,
        "selection_rule": "best_validation_psnr",
    })
    filtered = e["joint"][e["joint"].split.isin(["train", "val"])]
    initial = cp.parent / "training_asset_inventory_initial.json"
    dose.create_training_asset_evidence(e["root"], cfg, filtered, initial)
    return {
        "config": cfg, "checkpoint": cp, "history": cp.parent / "history.csv",
        "resolved": cp.parent / "resolved_config.yaml",
        "metadata": cp.parent / "run_metadata.json", "initial": initial,
        "binding": cp.parent / "checkpoint_binding_best.json",
    }


def test_best_selection_is_frozen_and_sha_bound(evidence):
    e = evidence
    sources = _best_binding_sources(e)
    from sabids.experiments.d2 import bind_best_checkpoint_evidence
    bind_best_checkpoint_evidence(
        e["root"], sources["initial"], sources["checkpoint"], sources["history"],
        sources["resolved"], sources["metadata"], e["lock_path"], e["contract"],
        sources["binding"],
    )
    result = dose.formal_preflight(
        e["root"], str(sources["checkpoint"]), str(e["lock_path"]), str(e["contract"]),
        "best_validation_psnr", str(sources["initial"]), str(sources["binding"]),
    )
    assert result["status"] == "passed", result
    assert result["history_schema_drift_detected"] is False
    assert result["history_row_width_distribution"] == {"2": 60}
    assert result["val_psnr_trusted"] is True
    assert result["selection_history_audit"]["best_epoch"] == 3
    history_path = sources["history"]
    original_history = history_path.read_bytes()
    changed_bytes = original_history.replace(b"60,30.0", b"60,30.1")
    assert changed_bytes != original_history
    history_path.write_bytes(changed_bytes)
    changed_history = dose.formal_preflight(
        e["root"], str(sources["checkpoint"]), str(e["lock_path"]), str(e["contract"]),
        "best_validation_psnr", str(sources["initial"]), str(sources["binding"]),
    )
    assert changed_history["status"] == "blocked"
    assert "source changed: history" in changed_history["issues"][0]
    history_path.write_bytes(original_history)
    dose.write_strict_json(sources["metadata"], {
        "best_checkpoint_sha256": "incorrect", "best_epoch": 3,
        "monitor": "psnr", "best_metric": 32.0,
        "best_checkpoint": str(sources["checkpoint"].resolve()),
        "run_id": sources["checkpoint"].parent.name,
        "selection_rule": "best_validation_psnr",
    })
    result = dose.formal_preflight(
        e["root"], str(sources["checkpoint"]), str(e["lock_path"]), str(e["contract"]),
        "best_validation_psnr", str(sources["initial"]), str(sources["binding"]),
    )
    assert result["status"] == "blocked" and "provenance SHA" in result["issues"][0]


@pytest.mark.parametrize("failure,match", [
    ("checkpoint_sha", "checkpoint SHA"),
    ("history_best_epoch", "epoch differs"),
    ("checkpoint_epoch", "epoch differs"),
    ("run_id", "run ID"),
    ("data_plan", "data_plan_sha256"),
    ("selection_rule", "selection rule"),
    ("missing_initial", "Missing binding source initial_inventory"),
    ("overwritten_best", "checkpoint SHA"),
])
def test_best_binding_fail_closed_for_each_provenance_break(evidence, failure, match):
    from sabids.experiments.d2 import bind_best_checkpoint_evidence
    e = evidence
    sources = _best_binding_sources(e)
    if failure in {"checkpoint_sha", "overwritten_best"}:
        raw = torch.load(sources["checkpoint"], map_location="cpu", weights_only=False)
        raw["tampered_after_training"] = True
        torch.save(raw, sources["checkpoint"])
    elif failure == "history_best_epoch":
        pd.DataFrame([
            {"epoch": i, "val_psnr": 32.0 if i == 4 else 30.0}
            for i in range(1, 61)
        ]).to_csv(sources["history"], index=False)
    elif failure == "checkpoint_epoch":
        raw = torch.load(sources["checkpoint"], map_location="cpu", weights_only=False)
        raw["epoch"] = 3
        torch.save(raw, sources["checkpoint"])
        metadata = json.loads(sources["metadata"].read_text())
        metadata["best_checkpoint_sha256"] = dose.sha256_file(sources["checkpoint"])
        dose.write_strict_json(sources["metadata"], metadata)
    elif failure == "run_id":
        metadata = json.loads(sources["metadata"].read_text())
        metadata["run_id"] = "different_run"
        dose.write_strict_json(sources["metadata"], metadata)
    elif failure == "data_plan":
        lock = json.loads(e["lock_path"].read_text())
        lock["data_plan_sha256"] = "different"
        dose.write_strict_json(e["lock_path"], lock)
    elif failure == "selection_rule":
        metadata = json.loads(sources["metadata"].read_text())
        metadata["selection_rule"] = "fixed_final"
        dose.write_strict_json(sources["metadata"], metadata)
    elif failure == "missing_initial":
        sources["initial"].unlink()
    with pytest.raises(ValueError, match=match):
        bind_best_checkpoint_evidence(
            e["root"], sources["initial"], sources["checkpoint"], sources["history"],
            sources["resolved"], sources["metadata"], e["lock_path"], e["contract"],
            sources["binding"],
        )


def test_teacher_audit_derives_selection_and_training_cohort_identity(evidence):
    e = evidence
    run = e["root"] / "teacher_run"
    run.mkdir()
    config = copy.deepcopy(e["cfg"])
    config["protocol_id"] = e["lock"]["protocol_id"]
    config["train"].update(
        stage="segment", monitor="vessel_soft_dice", output_dir=str(run)
    )
    resolved = run / "resolved_config.yaml"
    save_config(config, resolved)
    checkpoint = run / "best.pth"
    torch.save({
        "model": build_model(config).state_dict(), "config": config,
        "epoch": 2, "best_metric": .75,
    }, checkpoint)
    dose.write_strict_json(run / "run_metadata.json", {
        "best_checkpoint_sha256": dose.sha256_file(checkpoint),
        "best_checkpoint": str(checkpoint.resolve()),
        "best_epoch": 3, "monitor": "vessel_soft_dice", "best_metric": .75,
    })
    output = run / "teacher_evidence.json"
    process = subprocess.run([
        sys.executable, str(ROOT / "tools/audit_d2_teacher.py"),
        "--project-root", str(e["root"]), "--checkpoint", str(checkpoint),
        "--resolved-config", str(resolved), "--protocol-lock", str(e["lock_path"]),
        "--output", str(output),
    ], cwd=ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["selection_rule"] == "best_validation_vessel_soft_dice"
    assert result["training_data"].startswith("teacher-development-cohort:")
    assert result["train_groups"] == ["pku_0001"]
    assert result["validation_groups"] == ["pku_0002"]
    assert result["test_assets_opened"] == 0
