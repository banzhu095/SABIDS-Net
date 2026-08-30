from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose why learned J interaction injections remain small")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--output", default="runs/interaction_suppression_diagnosis")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite diagnosis report: {output}")
    output.mkdir(parents=True, exist_ok=True)
    history_rows, scale_rows, feature_rows, clip_rows, final_scales, missing = [], [], [], [], [], []
    artifact_audits = []
    for seed in args.seeds:
        for variant in ("j00", "j10", "j01", "j11"):
            run = root / "runs/current" / f"interaction_{variant}_fold0_seed{seed}"
            history_path, checkpoint = run / "history.csv", run / "last.pth"
            initialization_path = run / "initialization_audit.json"
            parameter_path = run / "parameter_audit.json"
            diagnostic_metadata = run / "dependence_diagnostics_v2/diagnostic_metadata.json"
            if not diagnostic_metadata.is_file():
                diagnostic_metadata = run / "dependence_diagnostics/diagnostic_metadata.json"
            audit_row = {"seed": seed, "variant": variant.upper(), "run": str(run)}
            for label, path in (
                ("resolved_config", run / "resolved_config.yaml"),
                ("initialization_audit", initialization_path),
                ("parameter_audit", parameter_path),
                ("history", history_path), ("checkpoint", checkpoint),
                ("fixed_validation", run / "final_validation/group_metrics.csv"),
                ("dependence_diagnostics", diagnostic_metadata),
            ):
                audit_row[f"{label}_present"] = path.is_file()
                audit_row[f"{label}_sha256"] = sha256_file(path) if path.is_file() else None
            if initialization_path.is_file():
                initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
                for key in ("model_state_sha256", "common_state_sha256", "interaction_state_sha256", "data_plan_sha256", "initialization_checkpoint_sha256"):
                    audit_row[key] = initialization.get(key)
            if parameter_path.is_file():
                parameters = json.loads(parameter_path.read_text(encoding="utf-8"))
                trainable = sorted(parameters.get("trainable", []))
                audit_row["trainable_parameter_count"] = len(trainable)
                audit_row["trainable_parameter_names_sha256"] = hashlib.sha256("\n".join(trainable).encode()).hexdigest()
            artifact_audits.append(audit_row)
            if not history_path.is_file():
                missing.append(str(history_path))
                continue
            history = pd.read_csv(history_path)
            history["seed"], history["variant"] = seed, variant.upper()
            history_rows.append(history)
            base = [column for column in ("seed", "variant", "epoch", "seconds", "lr") if column in history]
            scale_columns = [column for column in history if any(token in column for token in (
                "scale_signed", "scale_abs", "scale_rms", "scale_gradient", "scale_update",
                "mapping_update", "mapping_parameter_rms", "mapping_gradient",
            ))]
            if scale_columns:
                scale_rows.append(history[base + scale_columns])
            feature_columns = [column for column in history if column.startswith("train_interaction_") and any(token in column for token in (
                "rms", "gate_", "guidance", "receiver", "source", "transformed", "injection",
            ))]
            if feature_columns:
                feature_rows.append(history[base + feature_columns])
            clip_columns = [column for column in history if any(token in column for token in (
                "gradient_norm", "gradient_clip", "gradient_group", "weight_decay",
            ))]
            if clip_columns:
                clip_rows.append(history[base + clip_columns])
            if checkpoint.is_file():
                payload = torch.load(checkpoint, map_location="cpu")
                for name, tensor in payload["model"].items():
                    if name.endswith(("seg_scale", "layer_scale", "vessel_scale")):
                        value = tensor.detach().float()
                        final_scales.append({
                            "seed": seed, "variant": variant.upper(), "parameter": name,
                            "signed_mean": float(value.mean()), "abs_mean": float(value.abs().mean()),
                            "rms": float(value.square().mean().sqrt()), "minimum": float(value.min()),
                            "maximum": float(value.max()),
                        })
            else:
                missing.append(str(checkpoint))
    if history_rows:
        pd.concat(history_rows, ignore_index=True).to_csv(output / "history_long.csv", index=False, encoding="utf-8-sig")
    pd.concat(scale_rows, ignore_index=True).to_csv(output / "interaction_scale_trajectory.csv", index=False, encoding="utf-8-sig") if scale_rows else pd.DataFrame().to_csv(output / "interaction_scale_trajectory.csv", index=False)
    pd.concat(feature_rows, ignore_index=True).to_csv(output / "interaction_feature_statistics.csv", index=False, encoding="utf-8-sig") if feature_rows else pd.DataFrame().to_csv(output / "interaction_feature_statistics.csv", index=False)
    pd.concat(clip_rows, ignore_index=True).to_csv(output / "gradient_clipping_audit.csv", index=False, encoding="utf-8-sig") if clip_rows else pd.DataFrame().to_csv(output / "gradient_clipping_audit.csv", index=False)
    pd.DataFrame(final_scales).to_csv(output / "final_checkpoint_scales.csv", index=False, encoding="utf-8-sig")
    artifact_table = pd.DataFrame(artifact_audits)
    artifact_table.to_csv(output / "run_artifact_audit.csv", index=False, encoding="utf-8-sig")

    position_findings, position_gains = [], []
    for seed in args.seeds:
        j10_path = root / "runs/current" / f"interaction_j10_fold0_seed{seed}" / "final_validation/group_metrics.csv"
        j00_path = root / "runs/current" / f"interaction_j00_fold0_seed{seed}" / "final_validation/group_metrics.csv"
        if not j10_path.is_file():
            continue
        table = pd.read_csv(j10_path)
        control = pd.read_csv(j00_path) if j00_path.is_file() else pd.DataFrame()
        for group in ("pku_0006", "pku_0012", "pku_0040"):
            part = table[table["group_id"].astype(str).str.lower() == group]
            if part.empty:
                continue
            for metric in ("p0_vessel_dice", "p0_vessel_precision", "p0_vessel_recall", "vessel_roi_dice"):
                if metric in part:
                    position_findings.append({"seed": seed, "group_id": group, "metric": metric, "value": float(part.iloc[0][metric])})
                    control_part = control[control["group_id"].astype(str).str.lower() == group] if not control.empty else pd.DataFrame()
                    if not control_part.empty and metric in control_part:
                        gain = float(part.iloc[0][metric]) - float(control_part.iloc[0][metric])
                        all_gains = table[["group_id", metric]].merge(control[["group_id", metric]], on="group_id", suffixes=("_j10", "_j00"))
                        all_gains["gain"] = all_gains[f"{metric}_j10"] - all_gains[f"{metric}_j00"]
                        without = all_gains[all_gains["group_id"].astype(str).str.lower() != group]["gain"]
                        position_gains.append({
                            "seed": seed, "group_id": group, "metric": metric,
                            "j10_minus_j00": gain,
                            "mean_all_positions": float(all_gains["gain"].mean()),
                            "mean_without_this_position": float(without.mean()) if len(without) else float("nan"),
                            "is_negative": gain < 0,
                        })
    pd.DataFrame(position_findings).to_csv(output / "j10_selected_position_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(position_gains).to_csv(output / "j10_selected_position_gains.csv", index=False, encoding="utf-8-sig")
    available_new = bool(scale_rows and feature_rows and clip_rows)
    markdown = [
        "# Interaction suppression diagnosis", "",
        "This report diagnoses optimization and dependence; it is not a retraining gain.", "",
        f"Runs requested: seeds {args.seeds}; missing artifacts: {len(missing)}.",
        "",
        "## Audit interpretation", "",
        "- A non-zero scale is insufficient evidence of meaningful injection; inspect injection/receiver RMS together.",
        "- AdamW applies the configured weight decay to scale parameters because the current optimizer uses one undifferentiated parameter group.",
        "- Zero scales give mapping parameters zero first-step gradient; persistent mapping gradient proportional to a tiny scale is gradient starvation.",
        "- A clipping coefficient below one indicates global clipping. Compare J10 seed 42 against seeds 43/44 and the D2S share of the pre-clip norm.",
        "- Constant/saturated gates, transformed/source RMS mismatch, or near-zero target-RMS sensitivity indicate representational redundancy or poor scaling.",
        "- S->D statistics must be stratified by vessel-labelled versus unlabelled/Duke samples; unconditional guidance without reliable segmentation is a failure risk.",
        "",
        "New per-scale fields were found." if available_new else "Older histories lack some requested fields. The new Trainer records them prospectively; existing runs are not overwritten or backfilled.",
    ]
    if final_scales:
        scales = pd.DataFrame(final_scales)
        active = scales[scales["variant"].isin(["J10", "J01", "J11"])]
        markdown.extend(["", "## Observed checkpoint scales", "", (
            f"Active-direction scale absolute mean range: {active['abs_mean'].min():.6g}--{active['abs_mean'].max():.6g}. "
            "This only proves that scale parameters left zero; injection/receiver RMS is required to judge effective interaction."
        )])
    if position_gains:
        gains_table = pd.DataFrame(position_gains)
        vessel = gains_table[gains_table["metric"] == "p0_vessel_dice"]
        markdown.extend(["", "## J10 selected-position attribution", "", (
            f"Negative J10-J00 vessel-Dice effects: {int(vessel['is_negative'].sum())}/{len(vessel)} selected seed-position rows. "
            "Use j10_selected_position_gains.csv to compare the all-position mean with the leave-one-position-out mean."
        )])
    (output / "interaction_suppression_diagnosis.md").write_text("\n".join(markdown), encoding="utf-8")
    (output / "missing_and_failure_checklist.json").write_text(json.dumps({
        "missing_artifacts": missing, "prospective_diagnostics_available": available_new,
        "test_used": False, "claims_from_missing_history_backfilled": False,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
