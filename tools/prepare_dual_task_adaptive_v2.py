from __future__ import annotations

import argparse, copy, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from sabids.config import load_config, save_config
from sabids.experiments.dual_task_adaptive import audit_adaptive_inputs, sha256_file
from sabids.experiments.dual_task_adaptive_v2 import audit_v2_inputs
from tools.prepare_dual_task_adaptive import _record_component_inventory, _record_input_inventory


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", default=".")
    p.add_argument("--mode", choices=("preflight", "overfit", "cuda-check", "pilot"), required=True)
    p.add_argument("--run-id", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    a = p.parse_args(); root = Path(a.project_root).resolve()
    cfg = load_config(root / "configs/adaptive_denoising/dual_task_adaptive_v2/seed42.yaml")
    cfg["device"] = a.device; cfg["data"]["root"] = str(root)
    for section, keys in (("anchors", ("d2_checkpoint", "coarse_checkpoint", "v1_checkpoint")),):
        for key in keys:
            value = Path(cfg["dual_task_adaptive_v2"][section][key])
            cfg["dual_task_adaptive_v2"][section][key] = str(value if value.is_absolute() else (root / value).resolve())
    cfg["dual_task_adaptive_v2"]["v1_checkpoint"] = cfg["dual_task_adaptive_v2"]["anchors"]["v1_checkpoint"]
    binding = Path(cfg["dual_task_adaptive_v2"]["evidence"]["v1_binding"])
    cfg["dual_task_adaptive_v2"]["evidence"]["v1_binding"] = str(binding if binding.is_absolute() else (root / binding).resolve())
    manifest = Path(cfg["data"]["manifest"])
    cfg["data"]["manifest"] = str(manifest if manifest.is_absolute() else (root / manifest).resolve())
    for section, keys in (("anchors", ("d2_checkpoint", "coarse_checkpoint")),
                          ("evidence", ("d2_binding", "d2_inventory", "coarse_binding", "protocol_lock", "split_contract"))):
        values = cfg["dual_task_adaptive"][section]
        for key in keys:
            value = Path(values[key]); values[key] = str(value if value.is_absolute() else (root / value).resolve())
    registry = root / "cache/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v2" / a.run_id
    registry.mkdir(parents=True, exist_ok=True)
    report = audit_v2_inputs(cfg, root)
    anchor_report = audit_adaptive_inputs(cfg, root)
    report["bound_anchor_audit"] = anchor_report
    if anchor_report["status"] != "passed":
        report["status"] = "blocked"; report["issues"].extend(anchor_report["issues"])
    report.update({"mode": a.mode, "run_id": a.run_id})
    (registry / f"preflight_{a.mode}.json").write_text(json.dumps(report, indent=2)+"\n")
    if report["status"] != "passed": print(json.dumps(report, indent=2)); raise SystemExit(2)
    if a.mode == "preflight": print(json.dumps(report, indent=2)); return
    inventory = _record_input_inventory(root, cfg, registry)
    components = _record_component_inventory(root, cfg, registry)
    component_data = json.loads(components.read_text())
    v2 = cfg["dual_task_adaptive_v2"]
    v2.setdefault("evidence", {}).update({"training_input_inventory": str(inventory),
                      "training_input_inventory_sha256": sha256_file(inventory),
                      "fixed_component_inventory": str(components),
                      "fixed_component_inventory_sha256": sha256_file(components)})
    v2["strata"] = {key: component_data[key] for key in
                    ("small_area_max_model_grid_px", "low_contrast_max")}
    cfg["loss"].setdefault("vessel_protect", {})["strata"] = dict(v2["strata"])
    prepared = copy.deepcopy(cfg)
    run_base = root / "runs/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v2" / a.run_id
    name = "seed42"
    if a.mode == "overfit":
        name = "overfit_seed42"; prepared["train"].update(epochs=3, early_stopping_patience=4, evaluate_epoch0=False, monitor="vessel_dice")
        prepared["data"].update(val_split="train", max_train_samples=1, max_val_samples=1, samples_per_epoch=1)
        prepared["dual_task_adaptive_v2"]["validation_only"] = False
    elif a.mode == "cuda-check":
        name = "cuda_check_seed42"; prepared["train"].update(epochs=2, early_stopping_patience=3)
        prepared["data"].update(max_train_samples=16, samples_per_epoch=16)
    output = run_base / name; prepared["train"]["output_dir"] = str(output)
    prepared["dual_task_adaptive_v2"]["run_mode"] = a.mode
    if output.exists() and not a.resume: raise FileExistsError(f"Refusing overwrite: {output}")
    if a.resume:
        last = output / "last.pth"
        if not last.is_file(): raise FileNotFoundError("Resume requires last.pth")
        prepared["train"]["resume"] = str(last)
    path = registry / f"config_{a.mode}{'_resume' if a.resume else ''}_seed42.yaml"
    if path.exists() and load_config(path) != prepared: raise FileExistsError(f"Prepared config differs: {path}")
    if not path.exists(): save_config(prepared, path)
    report.update(config=str(path), output_dir=str(output), fixed_component_inventory=str(components))
    (registry / f"preflight_{a.mode}.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))

if __name__ == "__main__": main()
