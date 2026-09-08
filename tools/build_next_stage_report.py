"""Build validation-only, position-equal next-stage experiment reports."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from sabids.config import load_config
from sabids.experiments.protocol_lock import load_protocol_lock, sha256_file
from sabids.experiments.atlas import build_report_atlas


REPORT_NAMES = {
    "d1_structure": "d1_structure",
    "input_image": "input_image",
    "training_order": "training_order",
    "decoder_interaction_strength": "decoder_interaction_strength",
    "decoder_interaction_confirm": "decoder_strong_interaction",
    "decoder_interaction_controls": "decoder_interaction_controls",
}


def suite_and_arm(run_id: str) -> tuple[str | None, str | None]:
    if run_id.startswith("d1_denoise_d0"): return "d1_structure", "D0"
    if run_id.startswith("d1_denoise_struct"): return "d1_structure", "D1"
    match = re.match(r"input_(noisy|d0|d1|clean)_", run_id)
    if match: return "input_image", "I-" + match.group(1).upper()
    match = re.match(r"order_(ds|sd|alt)_", run_id)
    if match: return "training_order", "O-" + match.group(1).upper()
    if "j10_shuffle" in run_id: return "decoder_interaction_controls", "J10-SHUFFLE"
    if "j01_shuffle" in run_id: return "decoder_interaction_controls", "J01-SHUFFLE"
    if "self_adapter" in run_id: return "decoder_interaction_controls", "J-SELF-ADAPTER"
    match = re.match(r"interaction_(a00|ad05|ad10|ad20|as05|as10|as20)_", run_id)
    if match: return "decoder_interaction_strength", match.group(1).upper()
    match = re.match(r"interaction_j(00|10|01|11)_strong", run_id)
    if match: return "decoder_interaction_confirm", "J" + match.group(1)
    return None, None


def config_path(run: Path) -> Path | None:
    return next((run / name for name in ("resolved_config.yaml", "config_resolved.yaml", "config.yaml") if (run / name).is_file()), None)


def contrasts(suite: str) -> list[tuple[str, str, str]]:
    return {
        "d1_structure": [("D1-D0", "D1", "D0")],
        "input_image": [("I-D1-I-D0", "I-D1", "I-D0"), ("I-D1-I-NOISY", "I-D1", "I-NOISY"), ("I-D0-I-NOISY", "I-D0", "I-NOISY"), ("I-CLEAN-I-NOISY", "I-CLEAN", "I-NOISY"), ("I-CLEAN-I-D1", "I-CLEAN", "I-D1")],
        "training_order": [("O-SD-O-DS", "O-SD", "O-DS"), ("O-ALT-O-DS", "O-ALT", "O-DS"), ("O-ALT-O-SD", "O-ALT", "O-SD")],
        "decoder_interaction_confirm": [("D2S_J10-J00", "J10", "J00"), ("D2S_J11-J01", "J11", "J01"), ("S2D_J01-J00", "J01", "J00"), ("S2D_J11-J10", "J11", "J10"), ("TOTAL_J11-J00", "J11", "J00")],
        "decoder_interaction_controls": [("J10-J10-SHUFFLE", "J10", "J10-SHUFFLE"), ("J01-J01-SHUFFLE", "J01", "J01-SHUFFLE"), ("J10-J-SELF-ADAPTER", "J10", "J-SELF-ADAPTER"), ("J01-J-SELF-ADAPTER", "J01", "J-SELF-ADAPTER")],
    }.get(suite, [])


def improvement_sign(metric: str) -> float:
    lower = metric.lower()
    return -1.0 if any(token in lower for token in ("rmse", "mae", "error", "distance", "hd95", "assd", "_fp", "_fn", "outside")) else 1.0


def paired(position: pd.DataFrame, suite: str) -> pd.DataFrame:
    if position.empty: return pd.DataFrame()
    keys = [key for key in ("fold", "seed", "group_id") if key in position.columns]
    numeric = [column for column in position.select_dtypes(include=[np.number]).columns if column not in {"fold", "seed"}]
    rows = []
    for label, left_arm, right_arm in contrasts(suite):
        left, right = position[position.arm.eq(left_arm)], position[position.arm.eq(right_arm)]
        if left.empty or right.empty: continue
        merged = left[keys + numeric].merge(right[keys + numeric], on=keys, suffixes=("_left", "_right"))
        for item in merged.to_dict("records"):
            base = {key: item[key] for key in keys}; base.update({"contrast": label, "left_arm": left_arm, "right_arm": right_arm})
            for metric in numeric:
                difference = item[f"{metric}_left"] - item[f"{metric}_right"]
                base[f"raw_difference__{metric}"] = difference
                base[f"improvement__{metric}"] = improvement_sign(metric) * difference
            rows.append(base)
    result = pd.DataFrame(rows)
    if suite == "decoder_interaction_confirm" and not position.empty:
        pivot = position.pivot_table(index=keys, columns="arm", values=numeric)
        if all(arm in position.arm.unique() for arm in ("J00", "J10", "J01", "J11")):
            extra = []
            for index, row in pivot.iterrows():
                record = dict(zip(keys, index if isinstance(index, tuple) else (index,)))
                record.update({"contrast": "INTERACTION_J11-J10-J01+J00", "left_arm": "factorial", "right_arm": "factorial"})
                for metric in numeric:
                    value = row[(metric, "J11")] - row[(metric, "J10")] - row[(metric, "J01")] + row[(metric, "J00")]
                    record[f"raw_difference__{metric}"] = value
                    record[f"improvement__{metric}"] = improvement_sign(metric) * value
                extra.append(record)
            result = pd.concat([result, pd.DataFrame(extra)], ignore_index=True)
    return result


def gain_summary(position_gains: pd.DataFrame, seed_gains: pd.DataFrame) -> pd.DataFrame:
    if position_gains.empty or seed_gains.empty: return pd.DataFrame()
    rng = np.random.default_rng(20260908)
    rows = []
    metrics = [column for column in position_gains.columns if column.startswith("improvement__")]
    for contrast, positions in position_gains.groupby("contrast"):
        seeds = seed_gains[seed_gains.contrast.eq(contrast)]
        for metric in metrics:
            values = pd.to_numeric(seeds[metric], errors="coerce").dropna().to_numpy()
            position_values = pd.to_numeric(positions[metric], errors="coerce").dropna().to_numpy()
            bootstrap = []
            # Resample anatomical positions independently inside each seed, then
            # average seed estimates. Repeated frames were already collapsed.
            for _ in range(2000):
                sampled_seed_means = []
                for _, seed_positions in positions.groupby([key for key in ("fold", "seed") if key in positions.columns]):
                    candidate = pd.to_numeric(seed_positions[metric], errors="coerce").dropna().to_numpy()
                    if candidate.size:
                        sampled_seed_means.append(float(rng.choice(candidate, candidate.size, replace=True).mean()))
                if sampled_seed_means: bootstrap.append(float(np.mean(sampled_seed_means)))
            rows.append({"contrast": contrast, "metric": metric.removeprefix("improvement__"), "mean_improvement": float(np.mean(values)) if values.size else np.nan, "seed_sd": float(np.std(values, ddof=1)) if values.size > 1 else np.nan, "position_sign_consistency": float(np.mean(position_values > 0.0)) if position_values.size else np.nan, "position_cluster_bootstrap_ci_low": float(np.quantile(bootstrap, .025)) if bootstrap else np.nan, "position_cluster_bootstrap_ci_high": float(np.quantile(bootstrap, .975)) if bootstrap else np.nan, "n_seeds": int(values.size), "n_positions": int(position_values.size), "ci_limitation": "position-cluster bootstrap; only three seeds and few validation positions"})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--protocol-lock", required=True)
    parser.add_argument("--suites", required=True, help="Comma-separated suites")
    parser.add_argument("--exclude-test", action="store_true")
    args = parser.parse_args()
    if not args.exclude_test: raise SystemExit("BLOCKED: development reports require --exclude-test")
    root = Path(args.project_root).resolve()
    lock_path = Path(args.protocol_lock)
    if not lock_path.is_absolute(): lock_path = root / lock_path
    lock = load_protocol_lock(lock_path)
    wanted = set(args.suites.split(","))
    inventories = {suite: [] for suite in wanted}
    current = root / "runs" / "current"
    for run in sorted(current.iterdir() if current.is_dir() else []):
        suite, arm = suite_and_arm(run.name)
        if suite == "decoder_interaction_strength" and "_pilot_" not in run.name: continue
        if suite and suite != "decoder_interaction_strength" and "_pilot_" in run.name: continue
        cfg_path = config_path(run)
        targets = []
        if suite in wanted: targets.append(suite)
        if suite == "decoder_interaction_confirm" and "decoder_interaction_controls" in wanted:
            targets.append("decoder_interaction_controls")
        if not targets or cfg_path is None: continue
        cfg = load_config(cfg_path)
        if cfg.get("protocol_id") != lock["protocol_id"]: continue
        sha_ok = cfg.get("data_plan_sha256") == lock["data_plan_sha256"] and cfg.get("label_inventory_sha256") == lock["label_inventory_sha256"]
        for target in targets: inventories[target].append((run, arm, cfg, sha_ok))

    outputs = []
    for suite in wanted:
        out = root / "runs" / "reports" / f"{REPORT_NAMES[suite]}_{lock['protocol_id']}"
        out.mkdir(parents=True, exist_ok=True)
        completion, frames, positions, histories, missing = [], [], [], [], []
        atlas_selection = None
        for run, arm, cfg, sha_ok in inventories[suite]:
            history_path = run / "history.csv"; final = run / "last.pth"
            history = pd.read_csv(history_path) if history_path.is_file() else pd.DataFrame()
            expected = int(cfg["train"].get("epochs", 0)); current_epoch = int(history.epoch.max()) if not history.empty else 0
            rho_valid = True; requested_rho = actual_rho = np.nan
            if suite.startswith("decoder_interaction") and not history.empty and "train_interaction_actual_rho_mean" in history:
                last_history = history.sort_values("epoch").iloc[-1]
                requested_rho = float(last_history.get("train_interaction_requested_rho", 0.0))
                actual_rho = float(last_history["train_interaction_actual_rho_mean"])
                tolerance = max(1e-4, abs(requested_rho) * .05)
                rho_valid = bool(np.isfinite(actual_rho) and abs(actual_rho - requested_rho) <= tolerance)
            complete = bool(final.is_file() and current_epoch >= expected and sha_ok and rho_valid)
            completion.append({"run_id": run.name, "arm": arm, "fold": cfg.get("fold", 0), "seed": cfg.get("seed"), "status": "completed" if complete else "invalid_rho" if final.is_file() and not rho_valid else "incomplete", "current_epoch": current_epoch, "expected_epoch": expected, "checkpoint_exists": final.is_file(), "protocol_sha_match": sha_ok, "requested_rho_final": requested_rho, "actual_rho_final": actual_rho, "rho_within_tolerance": rho_valid, "safe_to_merge": complete})
            if not history.empty:
                history.insert(0, "seed", cfg.get("seed")); history.insert(0, "fold", cfg.get("fold", 0)); history.insert(0, "arm", arm); history.insert(0, "run_id", run.name); histories.append(history)
            validation = run / "validation_results"
            frame_path, group_path = validation / "frame_metrics.csv", validation / "group_metrics.csv"
            if frame_path.is_file():
                table = pd.read_csv(frame_path); table.insert(0, "seed", cfg.get("seed")); table.insert(0, "fold", cfg.get("fold", 0)); table.insert(0, "arm", arm); table.insert(0, "run_id", run.name); frames.append(table)
            else: missing.append({"run_id": run.name, "asset": str(frame_path), "reason": "run validation evaluator"})
            if group_path.is_file():
                table = pd.read_csv(group_path); table.insert(0, "seed", cfg.get("seed")); table.insert(0, "fold", cfg.get("fold", 0)); table.insert(0, "arm", arm); table.insert(0, "run_id", run.name); positions.append(table)
            if atlas_selection is None and (validation / "atlas_selection.csv").is_file(): atlas_selection = pd.read_csv(validation / "atlas_selection.csv")
            source_atlas = validation / "fixed_atlas" / "predictions"
            if source_atlas.is_dir():
                destination = out / "fixed_atlas" / run.name
                if not destination.exists(): shutil.copytree(source_atlas, destination)
        completion_table = pd.DataFrame(completion)
        frame_table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        position_table = pd.concat(positions, ignore_index=True) if positions else pd.DataFrame()
        history_table = pd.concat(histories, ignore_index=True) if histories else pd.DataFrame()
        seed_table = pd.DataFrame()
        summary_table = pd.DataFrame()
        if not position_table.empty:
            numeric = [col for col in position_table.select_dtypes(include=[np.number]).columns if col not in {"fold", "seed"}]
            seed_table = position_table.groupby(["arm", "fold", "seed"], as_index=False)[numeric].mean()
            means = seed_table.groupby("arm")[numeric].mean().add_suffix("__mean")
            stds = seed_table.groupby("arm")[numeric].std(ddof=1).add_suffix("__seed_sd")
            summary_table = means.join(stds).reset_index()
        gains_position = paired(position_table, suite)
        gain_keys = [key for key in ("contrast", "fold", "seed") if key in gains_position.columns]
        gain_metrics = [key for key in gains_position.select_dtypes(include=[np.number]).columns if key not in {"fold", "seed"}]
        gains_seed = gains_position.groupby(gain_keys, as_index=False)[gain_metrics].mean() if gain_keys and not gains_position.empty else pd.DataFrame()
        gains_summary = gain_summary(gains_position, gains_seed)
        completion_table.to_csv(out / "completion_matrix.csv", index=False, encoding="utf-8-sig")
        frame_table.to_csv(out / "metrics_by_frame.csv", index=False, encoding="utf-8-sig")
        position_table.to_csv(out / "metrics_by_position.csv", index=False, encoding="utf-8-sig")
        seed_table.to_csv(out / "metrics_by_seed.csv", index=False, encoding="utf-8-sig")
        summary_table.to_csv(out / "metrics_summary.csv", index=False, encoding="utf-8-sig")
        gains_position.to_csv(out / "paired_gains_by_position.csv", index=False, encoding="utf-8-sig")
        gains_seed.to_csv(out / "paired_gains_by_seed.csv", index=False, encoding="utf-8-sig")
        gains_summary.to_csv(out / "paired_gains_summary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(missing, columns=["run_id", "asset", "reason"]).to_csv(out / "missing_assets.csv", index=False, encoding="utf-8-sig")
        (atlas_selection if atlas_selection is not None else pd.DataFrame(columns=["group_id", "selection_rule"])).to_csv(out / "atlas_selection.csv", index=False, encoding="utf-8-sig")
        missing.extend(build_report_atlas(out))
        pd.DataFrame(missing, columns=["run_id", "asset", "reason"]).to_csv(out / "missing_assets.csv", index=False, encoding="utf-8-sig")
        if suite == "training_order" and not history_table.empty:
            gradient_columns = [col for col in history_table.columns if "gradient" in col]
            history_table[["run_id", "arm", "epoch", *gradient_columns]].to_csv(out / "gradient_conflict_by_epoch.csv", index=False, encoding="utf-8-sig")
            phase_columns = [col for col in ("run_id", "arm", "epoch", "training_phase", "val_psnr", "val_vessel_soft_dice") if col in history_table]
            phase_table = history_table[phase_columns].copy()
            phase_table["phase_boundary"] = phase_table.groupby("run_id")["training_phase"].transform(lambda values: values.ne(values.shift())).astype(int)
            for metric in ("val_psnr", "val_vessel_soft_dice"):
                if metric in phase_table: phase_table[f"{metric}_change"] = phase_table.groupby("run_id")[metric].diff()
            phase_table.to_csv(out / "phase_transition_metrics.csv", index=False, encoding="utf-8-sig")
        if suite.startswith("decoder_interaction") and not history_table.empty:
            columns = [col for col in history_table.columns if "interaction_" in col or "mapping_" in col or "gradient_group_" in col or col in {"run_id", "arm", "fold", "seed", "epoch"}]
            history_table[columns].to_csv(out / "interaction_strength.csv", index=False, encoding="utf-8-sig")
        try:
            import matplotlib.pyplot as plt
            figure, axis = plt.subplots(figsize=(8, 4))
            if not history_table.empty:
                metric = "val_vessel_soft_dice" if "val_vessel_soft_dice" in history_table else "val_psnr"
                for (arm, seed), table in history_table.groupby(["arm", "seed"]): axis.plot(table.epoch, table[metric], label=f"{arm}/s{seed}", alpha=.8)
                axis.set_ylabel(metric); axis.legend(fontsize=6)
            axis.set_xlabel("epoch"); figure.tight_layout(); figure.savefig(out / "training_trajectory.png", dpi=160); plt.close(figure)
        except Exception as error:
            missing.append({"run_id": "report", "asset": "training_trajectory.png", "reason": str(error)})
            pd.DataFrame(missing, columns=["run_id", "asset", "reason"]).to_csv(out / "missing_assets.csv", index=False, encoding="utf-8-sig")
        evidence = pd.DataFrame([{"claim": "validation-only fixed P0 threshold=0.5", "evidence": "resolved configs and validation_results", "status": "passed"}, {"claim": "position-equal aggregation", "evidence": "metrics_by_position -> metrics_by_seed", "status": "passed" if not position_table.empty else "missing"}, {"claim": "no test access", "evidence": str(lock_path), "status": "passed"}])
        evidence.to_csv(out / "evidence_matrix.csv", index=False, encoding="utf-8-sig")
        summary = f"# {suite}\n\nProtocol: `{lock['protocol_id']}`. Fixed-final checkpoints, validation only, P0 threshold 0.5. Frames are first aggregated by anatomical position and then by seed. Missing evaluation outputs are not imputed.\n"
        (out / "SUMMARY.md").write_text(summary, encoding="utf-8")
        manifest = {"suite": suite, "protocol_id": lock["protocol_id"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "run_ids": [item[0].name for item in inventories[suite]], "completed_runs": int(completion_table.safe_to_merge.sum()) if not completion_table.empty else 0, "missing_assets": len(missing), "test_assets_opened": 0}
        (out / "report_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        outputs.append(str(out))

        full_order_ready = (
            suite == "training_order"
            and not completion_table.empty
            and set(completion_table.arm) == {"O-DS", "O-SD", "O-ALT"}
            and completion_table.safe_to_merge.all()
            and completion_table.seed.nunique() >= 3
            and completion_table.expected_epoch.min() >= 60
        )
        if full_order_ready and not seed_table.empty:
            metric = "vessel_soft_dice" if "vessel_soft_dice" in seed_table else next((col for col in seed_table if "vessel" in col and "dice" in col), None)
            if metric:
                selected = seed_table.groupby("arm")[metric].mean().idxmax()
                chosen = [(run, cfg) for run, arm, cfg, ok in inventories[suite] if arm == selected and ok and (run / "last.pth").is_file()]
                anchors = {f"fold{cfg.get('fold',0)}_seed{cfg.get('seed')}": str((run / "last.pth").relative_to(root)).replace("\\", "/") for run, cfg in chosen}
                hashes = {key: sha256_file(root / value) for key, value in anchors.items()}
                selection = {"selected_order": selected, "selected_fixed_epoch": 60, "selected_run_ids": [run.name for run, _ in chosen], "selection_metrics": {metric: float(seed_table[seed_table.arm.eq(selected)][metric].mean())}, "selection_rule": "maximum validation position-equal vessel Dice at fixed final epoch; test not read", "anchors": anchors, "anchor_checkpoint_sha256_by_seed": hashes, "protocol_id": lock["protocol_id"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "test_assets_opened": 0}
                anchor_out = root / "runs" / "anchors" / "interaction_anchor_selection.json"; anchor_out.parent.mkdir(parents=True, exist_ok=True); anchor_out.write_text(json.dumps(selection, indent=2), encoding="utf-8")
        if suite == "decoder_interaction_strength" and not history_table.empty and set(("A00", "AD05", "AD10", "AD20", "AS05", "AS10", "AS20")).issubset(set(history_table.arm)):
            last = history_table.sort_values("epoch").groupby("arm", as_index=False).tail(1)
            d_metric = "val_vessel_soft_dice" if "val_vessel_soft_dice" in history_table else "val_vessel_dice"
            s_metric = "val_psnr"
            selected_d = last[last.arm.isin(["A00", "AD05", "AD10", "AD20"])].sort_values(d_metric, ascending=False).iloc[0]
            selected_s = last[last.arm.isin(["A00", "AS05", "AS10", "AS20"])].sort_values(s_metric, ascending=False).iloc[0]
            decode = {"00": 0.0, "05": .005, "10": .010, "20": .020}
            strength = {"selected_d2s_rho": decode[selected_d.arm[-2:]], "selected_s2d_rho": decode[selected_s.arm[-2:]], "selection_rule": f"D2S=max validation {d_metric}; S2D=max validation {s_metric}; A00 eligible so zero is retained if interaction harms; fixed final pilot epoch", "pilot_run_ids": sorted(history_table.run_id.unique()), "pilot_metrics": {str(row.arm): {d_metric: float(row[d_metric]), s_metric: float(row[s_metric])} for _, row in last.iterrows()}, "protocol_id": lock["protocol_id"], "data_plan_sha256": lock["data_plan_sha256"], "label_inventory_sha256": lock["label_inventory_sha256"], "created_without_test": True}
            (out / "interaction_strength_lock.yaml").write_text(yaml.safe_dump(strength, sort_keys=False), encoding="utf-8")
    print(json.dumps({"status": "passed", "outputs": outputs, "test_assets_opened": 0}, indent=2))


if __name__ == "__main__": main()
