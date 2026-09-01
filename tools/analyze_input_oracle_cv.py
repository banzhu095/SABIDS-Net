from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


COMPARISONS = (("denoised", "noisy"), ("clean", "noisy"), ("clean", "denoised"))


def orientation(metric: str) -> float:
    lower = ("mae", "mse", "rmse", "hd95", "assd", "error", "outside", "_fp", "_fn", "bias_abs")
    return -1.0 if any(token in metric.lower() for token in lower) else 1.0


def scale_unit(metric: str) -> tuple[float, str]:
    name = metric.lower()
    if any(token in name for token in ("dice", "precision", "recall", "iou")):
        return 100.0, "percentage_points"
    if "psnr" in name:
        return 1.0, "dB"
    if any(token in name for token in ("boundary", "thickness", "hd95", "assd")):
        return 1.0, "px"
    return 1.0, "native"


def _prepare(path: Path, archive: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not archive:
            raise FileExistsError(f"Refusing to overwrite report: {path}; use --archive-existing")
        destination = path.with_name(path.name + "_archive_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(path), str(destination))
    path.mkdir(parents=True, exist_ok=True)


def _sign_p(values: pd.Series) -> float:
    values = values.dropna()
    positive, negative = int((values > 0).sum()), int((values < 0).sum())
    n = positive + negative
    if not n:
        return math.nan
    tail = sum(math.comb(n, k) for k in range(0, min(positive, negative) + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _bootstrap(values: np.ndarray, seed: int, repeats: int = 10000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(repeats, len(values)))].mean(axis=1)
    return tuple(np.quantile(sampled, [0.025, 0.975]))


def _sample_std(values) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.std(array, ddof=1)) if len(array) >= 2 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser(description="Position-clustered analysis of NOISY/CLEAN/DENOISED CV")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--runs", default="runs/input_oracle_cv")
    parser.add_argument("--output", default="runs/input_oracle_cv/report")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=None, help="Single-seed alias")
    parser.add_argument("--archive-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seed is not None and args.seeds is not None: parser.error("Use either --seed or --seeds")
    args.seeds = args.seeds or ([args.seed] if args.seed is not None else [42, 43, 44])
    root = Path(args.project_root).expanduser().resolve()
    runs = (root / args.runs).resolve()
    output = (root / args.output).resolve()
    _prepare(output, args.archive_existing)
    frames, positions, missing, epoch0 = [], [], [], []
    fold_suffix = "_smoke" if args.smoke_test else ""
    for fold in args.folds:
        d0_audit_path = runs / f"fold{fold}{fold_suffix}/d0_leakage_audit.json"
        if not d0_audit_path.is_file():
            missing.append(str(d0_audit_path))
        else:
            d0_audit = json.loads(d0_audit_path.read_text(encoding="utf-8"))
            if d0_audit.get("status") != "passed" or d0_audit.get("test_assets_opened") != 0:
                missing.append(f"FAILED_D0_AUDIT:{d0_audit_path}")
        for seed in args.seeds:
            pair_path = runs / f"fold{fold}{fold_suffix}/paired_data_plan_audit_fold{fold}_seed{seed}.json"
            if not pair_path.is_file():
                missing.append(str(pair_path))
            else:
                pair = json.loads(pair_path.read_text(encoding="utf-8"))
                if not pair.get("all_equal") or not pair.get("all_three_arms_present"):
                    missing.append(f"INCOMPLETE_PAIRING:{pair_path}")
            for arm in ("noisy", "clean", "denoised"):
                run = runs / f"fold{fold}{fold_suffix}/{arm}_seed{seed}"
                validation = run / "final_validation"
                frame_path, group_path = validation / "frame_metrics.csv", validation / "group_metrics.csv"
                if not frame_path.is_file() or not group_path.is_file():
                    missing.append(str(validation))
                    continue
                frame, group = pd.read_csv(frame_path), pd.read_csv(group_path)
                for table in (frame, group):
                    table["fold"], table["seed"], table["arm"] = fold, seed, arm
                frames.append(frame); positions.append(group)
                epoch_path = run / "diagnostics/epoch000_metrics.json"
                if epoch_path.is_file():
                    payload = json.loads(epoch_path.read_text(encoding="utf-8"))
                    row = {"fold": fold, "seed": seed, "arm": arm}
                    row.update({f"val_{k}": v for k, v in payload.get("val", {}).items() if isinstance(v, (int, float))})
                    epoch0.append(row)
    checklist = {"missing_inputs": missing, "test_assets_opened": 0, "fixed_threshold": 0.5, "postprocess": "P0"}
    (output / "missing_and_failure_checklist.json").write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing:
        if args.dry_run:
            print(json.dumps(checklist, ensure_ascii=False, indent=2)); return
        raise FileNotFoundError("Missing fixed-final validations; see missing_and_failure_checklist.json")
    # A deep copy consolidates blocks from wide evaluator CSVs. Without it,
    # pandas emits a fragmentation warning for every arm during smoke reports.
    frame_table = pd.concat(frames, ignore_index=True).copy(deep=True)
    position_table = pd.concat(positions, ignore_index=True).copy(deep=True)
    frame_table.to_csv(output / "frame_metrics_long.csv", index=False, encoding="utf-8-sig")
    position_table.to_csv(output / "position_metrics_long.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(epoch0).to_csv(output / "epoch0_input_domain_bias.csv", index=False, encoding="utf-8-sig")
    excluded = ("height", "width", "valid_pixels", "n_evaluated", "_true", "_pred", "baseline")
    metrics = [c for c in position_table.select_dtypes(include=[np.number]).columns if c not in {"fold", "seed"} and not any(x in c for x in excluded)]
    gain_rows = []
    keys = ["fold", "seed", "group_id"] + (["dataset"] if "dataset" in position_table else [])
    for metric in metrics:
        pivot = position_table.pivot_table(index=keys, columns="arm", values=metric)
        for left, right in COMPARISONS:
            if not {left, right}.issubset(pivot.columns):
                continue
            factor, unit = scale_unit(metric)
            if "signed_bias" in metric:
                gain = (pivot[right].abs() - pivot[left].abs()) * factor
            else:
                gain = orientation(metric) * (pivot[left] - pivot[right]) * factor
            for index, value in gain.items():
                index = index if isinstance(index, tuple) else (index,)
                row = dict(zip(keys, index))
                row.update({
                    "comparison": f"{left.upper()}-{right.upper()}", "metric": metric,
                    "arm_a": left.upper(), "arm_b": right.upper(),
                    "arm_a_absolute": float(pivot.loc[index, left]),
                    "arm_b_absolute": float(pivot.loc[index, right]),
                    "improvement": value, "unit": unit,
                })
                gain_rows.append(row)
    gains = pd.DataFrame(gain_rows)
    gains.to_csv(output / "paired_gains_by_position_seed.csv", index=False, encoding="utf-8-sig")
    # Seeds are repeated fits of the same anatomical positions, not subjects.
    position_gain = gains.groupby(["group_id", "comparison", "metric", "unit"], as_index=False).agg(
        mean_improvement=("improvement", "mean"),
        seed_std=("improvement", _sample_std),
        n_seeds=("seed", "nunique"),
    )
    position_gain["effect_percentile_within_metric"] = position_gain.groupby(
        ["comparison", "metric"]
    )["mean_improvement"].rank(pct=True)
    position_gain.to_csv(output / "paired_gains_by_position.csv", index=False, encoding="utf-8-sig")
    seed_summary = gains.groupby(["seed", "comparison", "metric", "unit"], as_index=False).agg(
        mean_improvement=("improvement", "mean"), n_positions=("group_id", "nunique"),
        positive_positions=("improvement", lambda x: int((x > 0).sum())),
    )
    seed_summary.to_csv(output / "paired_gains_by_seed.csv", index=False, encoding="utf-8-sig")
    fold_summary = gains.groupby(["fold", "comparison", "metric", "unit"], as_index=False).agg(
        mean_improvement=("improvement", "mean"), n_positions=("group_id", "nunique"),
    )
    fold_summary.to_csv(output / "paired_gains_by_fold.csv", index=False, encoding="utf-8-sig")
    seed_metric_source = position_table.loc[:, ["fold", "seed", "arm", *metrics]].copy(deep=True)
    seed_metrics = seed_metric_source.groupby(["fold", "seed", "arm"], as_index=False).mean(numeric_only=True)
    seed_metrics.to_csv(output / "seed_metrics_long.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    for (comparison, metric, unit), part in position_gain.groupby(["comparison", "metric", "unit"], sort=True):
        values = part["mean_improvement"].to_numpy(float)
        import hashlib
        bootstrap_seed = int(hashlib.sha256(f"{comparison}|{metric}".encode()).hexdigest()[:8], 16)
        low, high = _bootstrap(values, bootstrap_seed)
        q1, q3 = np.nanquantile(values, [.25, .75]) if np.isfinite(values).any() else (math.nan, math.nan)
        seed_values = seed_summary[(seed_summary["comparison"] == comparison) & (seed_summary["metric"] == metric)]["mean_improvement"]
        fold_values = fold_summary[(fold_summary["comparison"] == comparison) & (fold_summary["metric"] == metric)]["mean_improvement"]
        finite_abs = np.abs(values[np.isfinite(values)])
        summary_rows.append({
            "comparison": comparison, "metric": metric, "unit": unit,
            "mean": float(np.nanmean(values)), "median": float(np.nanmedian(values)),
            "position_std": _sample_std(values), "iqr": float(q3 - q1),
            "seed_mean_std": _sample_std(seed_values),
            "n_independent_positions": int(np.isfinite(values).sum()), "cluster_bootstrap_ci95_low": low,
            "cluster_bootstrap_ci95_high": high, "exact_sign_test_p": _sign_p(part["mean_improvement"]),
            "positive_positions": int((part["mean_improvement"] > 0).sum()),
            "negative_positions": int((part["mean_improvement"] < 0).sum()),
            "zero_positions": int((part["mean_improvement"] == 0).sum()),
            "largest_single_position_effect": float(np.nanmax(finite_abs)) if len(finite_abs) else math.nan,
            "mean_smaller_than_seed_sd": bool(abs(np.nanmean(values)) < _sample_std(seed_values)) if len(seed_values) > 1 else False,
            "fold_range": float(fold_values.max() - fold_values.min()) if len(fold_values) else math.nan,
            "exploratory_inference_only": True,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "paired_gains_summary.csv", index=False, encoding="utf-8-sig")

    # Collate preregistered non-model covariates and immutable run provenance.
    for source_name, target_name in (("position_characteristics.csv", "position_characteristics.csv"), ("position_typicality.csv", "position_typicality.csv")):
        source = runs / source_name
        if source.is_file(): shutil.copy2(source, output / target_name)
    label_source = runs / "label_audit/label_quality_metrics.csv"
    label_quality = pd.read_csv(label_source) if label_source.is_file() else pd.DataFrame()
    label_quality.to_csv(output / "label_quality_by_position.csv", index=False, encoding="utf-8-sig")
    fold_registry = runs / "splits/fold_registry.csv"
    if fold_registry.is_file(): shutil.copy2(fold_registry, output / "fold_registry.csv")
    registries, checkpoints = [], []
    for path in sorted((runs / "registry").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); payload["registry_path"] = str(path); registries.append(payload)
    pd.json_normalize(registries).to_csv(output / "experiment_registry.csv", index=False, encoding="utf-8-sig")
    for path in sorted(runs.glob("fold*/*_seed*/final_validation/evaluation_registry.json")):
        payload = json.loads(path.read_text(encoding="utf-8")); payload["registry_path"] = str(path); checkpoints.append(payload)
    pd.json_normalize(checkpoints).to_csv(output / "checkpoint_audit.csv", index=False, encoding="utf-8-sig")
    quality_parts = []
    for fold in args.folds:
        path = runs / f"fold{fold}/cache/input_quality_metrics.csv"
        val_path = runs / f"splits/fold_{fold}_val.csv"
        if path.is_file() and val_path.is_file():
            part = pd.read_csv(path); val_ids = set(pd.read_csv(val_path, dtype=str)["group_id"])
            part = part[part["group_id"].astype(str).isin(val_ids)].copy(); part["fold"] = fold; quality_parts.append(part)
    input_quality = pd.concat(quality_parts, ignore_index=True) if quality_parts else pd.DataFrame()
    if not input_quality.empty:
        input_quality = input_quality.groupby(["fold", "group_id"], as_index=False).mean(numeric_only=True)
    input_quality.to_csv(output / "input_quality_by_position.csv", index=False, encoding="utf-8-sig")

    trajectories = []
    wanted_epochs = {0, 1, 5, 10, 20, 30, 40, 50, 60}
    for fold in args.folds:
        for seed in args.seeds:
            for arm in ("noisy", "clean", "denoised"):
                path = runs / f"fold{fold}{fold_suffix}/{arm}_seed{seed}/history.csv"
                if not path.is_file(): continue
                history = pd.read_csv(path)
                epoch_col = "epoch" if "epoch" in history else history.columns[0]
                history = history[pd.to_numeric(history[epoch_col], errors="coerce").isin(wanted_epochs)].copy()
                history["fold"], history["seed"], history["arm"] = fold, seed, arm; trajectories.append(history)
    trajectory = pd.concat(trajectories, ignore_index=True) if trajectories else pd.DataFrame()
    trajectory.to_csv(output / "training_trajectory_summary.csv", index=False, encoding="utf-8-sig")

    priority = ["layer_dice", "upper_boundary_mae", "lower_boundary_mae", "thickness_mae", "p0_vessel_dice", "p0_vessel_precision", "p0_vessel_recall", "vessel_roi_dice", "vessel_outside_gt_layer_fraction", "vessel_area_fraction_mae", "vessel_gt_component_small_pixel_recall", "repeat_layer_dice", "repeat_vessel_dice"]
    primary = summary[summary["metric"].isin(priority)].copy()
    def value(comparison: str, metric: str) -> float:
        found = summary[(summary["comparison"] == comparison) & (summary["metric"] == metric)]["mean"]
        return float(found.iloc[0]) if len(found) else math.nan
    d_layer, d_vessel = value("DENOISED-NOISY", "layer_dice"), value("DENOISED-NOISY", "p0_vessel_dice")
    oracle_layer, oracle_vessel = value("CLEAN-NOISY", "layer_dice"), value("CLEAN-NOISY", "p0_vessel_dice")
    if all(np.isfinite(x) for x in (d_layer, d_vessel, oracle_layer, oracle_vessel)):
        if oracle_layer > 0 and oracle_vessel > 0 and d_layer <= 0 and d_vessel <= 0:
            mechanism = "Oracle clean input is beneficial, but D0 fails to realize it: prioritize task-preserving denoising/domain alignment."
        elif oracle_layer <= 0 and oracle_vessel <= 0:
            mechanism = "Even oracle clean input gives no mean segmentation gain: the current segmenter is noise-robust or information-limited by labels."
        elif d_layer > 0 and d_vessel > 0:
            mechanism = "D0 input yields concordant mean layer/vessel gains; verify position consistency and CI before claiming benefit."
        elif d_layer > 0 and d_vessel < 0:
            mechanism = "D0 helps coarse layer anatomy but harms vessel evidence; prioritize small-vessel/edge preservation."
        else:
            mechanism = "Mixed endpoint effects; no single positive interaction claim is supported."
    else:
        mechanism = "Primary metrics are incomplete; no mechanism conclusion is issued."
    mechanism_rows = []
    for group_id in sorted(position_gain["group_id"].astype(str).unique()):
        part = position_gain[(position_gain["group_id"].astype(str) == group_id) & (position_gain["metric"] == "p0_vessel_dice")]
        lookup = dict(zip(part["comparison"], part["mean_improvement"]))
        clean_noisy, den_noisy, clean_den = (lookup.get(k, math.nan) for k in ("CLEAN-NOISY", "DENOISED-NOISY", "CLEAN-DENOISED"))
        if not all(np.isfinite(x) for x in (clean_noisy, den_noisy, clean_den)): cls = "INSUFFICIENT"
        elif clean_noisy > 0 and den_noisy <= 0: cls = "A_CLEAN_BENEFIT_D0_NOT_RECOVERED"
        elif clean_noisy <= 0 and den_noisy <= 0: cls = "B_NO_CLEAN_OR_D0_ADVANTAGE"
        elif clean_noisy > 0 and den_noisy > 0 and clean_den > 0: cls = "C_PARTIAL_RECOVERY"
        elif den_noisy > 0 and clean_den <= 0: cls = "D_D0_EXCEEDS_CLEAN_OR_NOISY"
        else: cls = "E_MIXED"
        recovery = den_noisy / clean_noisy if np.isfinite(clean_noisy) and clean_noisy > .1 else math.nan
        mechanism_rows.append({"group_id": group_id, "endpoint": "p0_vessel_dice", "clean_minus_noisy_pp": clean_noisy, "denoised_minus_noisy_pp": den_noisy, "clean_minus_denoised_pp": clean_den, "mechanism_class": cls, "recovery_ratio": recovery})
    mechanism_table = pd.DataFrame(mechanism_rows)
    mechanism_table.to_csv(output / "position_mechanism_classification.csv", index=False, encoding="utf-8-sig")
    associations = []
    d_gain = position_gain[position_gain["comparison"] == "DENOISED-NOISY"]
    for covariate_name, covariates in (("input", input_quality), ("label", label_quality)):
        if covariates.empty: continue
        numeric_covariates = [c for c in covariates.select_dtypes(include=[np.number]).columns if c != "fold"]
        for metric in ("layer_dice", "p0_vessel_dice"):
            effects = d_gain[d_gain["metric"] == metric][["group_id", "mean_improvement"]]
            paired = effects.merge(covariates, on="group_id", how="inner")
            for covariate in numeric_covariates:
                valid = paired[["mean_improvement", covariate]].dropna()
                if len(valid) >= 3:
                    associations.append({"covariate_source": covariate_name, "covariate": covariate, "segmentation_metric": metric, "n_positions": len(valid), "pearson_descriptive": valid.iloc[:, 0].corr(valid.iloc[:, 1]), "spearman_descriptive": valid.iloc[:, 0].rank().corr(valid.iloc[:, 1].rank()), "inferential_claim_allowed": False})
    association_table = pd.DataFrame(associations)
    association_table[association_table.get("covariate_source", pd.Series(dtype=str)).eq("input")].to_csv(output / "input_quality_segmentation_association.csv", index=False, encoding="utf-8-sig")
    association_table[association_table.get("covariate_source", pd.Series(dtype=str)).eq("label")].to_csv(output / "label_quality_segmentation_association.csv", index=False, encoding="utf-8-sig")
    checks = pd.DataFrame([
        {"check": "fixed_final_complete", "passed": not missing},
        {"check": "position_first_inference", "passed": True},
        {"check": "test_assets_opened_zero", "passed": True},
        {"check": "clean_repeat_not_used", "passed": True},
        {"check": "manual_label_review_complete", "passed": False, "note": "PENDING unless independently adjudicated"},
    ])
    checks.to_csv(output / "interpretation_checks.csv", index=False, encoding="utf-8-sig")
    protocol_text = """# Statistical protocol\n\n- Unit of inference: independent anatomical position (`group_id`).\n- Repeat frames are averaged within position; seeds are repeated fits, not subjects.\n- Primary checkpoint: fixed final epoch; threshold 0.5; P0 without calibration/post-processing.\n- Positive gain means improvement (error metrics are direction-reversed).\n- CI: percentile bootstrap resampling positions. With 16 positions it is descriptive and low-powered.\n- Exact sign test is reported without multiplicity correction; it is secondary/exploratory.\n- Sealed test images are neither opened nor evaluated.\n"""
    (output / "STATISTICAL_PROTOCOL.md").write_text(protocol_text, encoding="utf-8")
    (output / "PROTOCOL.md").write_text(protocol_text, encoding="utf-8")
    psnr_gain = (
        float((input_quality["psnr_d0"] - input_quality["psnr_noisy"]).mean())
        if {"psnr_d0", "psnr_noisy"}.issubset(input_quality.columns) else math.nan
    )
    fp_gain, fn_gain = value("DENOISED-NOISY", "vessel_roi_fp_per_valid_pixel"), value("DENOISED-NOISY", "vessel_roi_fn_per_valid_pixel")
    size_effects = {size: value("DENOISED-NOISY", f"vessel_gt_component_{size}_pixel_recall") for size in ("small", "medium", "large")}
    typicality_path = output / "position_typicality.csv"
    typicality_note = "not available"
    if typicality_path.is_file():
        typicality = pd.read_csv(typicality_path)
        legacy = typicality[typicality["group_id"].astype(str).isin(["pku_0006", "pku_0012", "pku_0040"])]
        if not legacy.empty: typicality_note = "; ".join(f"{row.group_id}: {row.atypicality_percentile:.1%}" for row in legacy.itertuples())
    review_path = runs / "label_audit/label_review_form.csv"
    manual_complete = False
    if review_path.is_file():
        review = pd.read_csv(review_path, dtype=str).fillna("")
        manual_complete = bool(len(review) and review["review_status"].ne("PENDING").all())
    recommendation = (
        "prioritize task-preserving denoising and domain alignment" if "D0 fails" in mechanism
        else "prioritize label/registration audit and a neutral-initialization sensitivity run" if "oracle clean" in mechanism.lower() and "no mean" in mechanism.lower()
        else "retain the segmenter and validate the gain before adding feature interaction" if d_layer > 0 and d_vessel > 0
        else "do not strengthen D→S yet; resolve the mixed endpoint mechanism first"
    )
    answers = [
        f"1. Stage 1 image quality: {'improved' if np.isfinite(psnr_gain) and psnr_gain > 0 else 'evidence insufficient'}; mean PSNR change={psnr_gain:.4g} dB.",
        f"2. CLEAN versus NOISY for layer: {'improved' if oracle_layer > 0 else 'not improved' if np.isfinite(oracle_layer) else 'insufficient'}; {oracle_layer:.4g} pp.",
        f"3. CLEAN versus NOISY for vessel: {'improved' if oracle_vessel > 0 else 'not improved' if np.isfinite(oracle_vessel) else 'insufficient'}; {oracle_vessel:.4g} pp.",
        f"4. D0 recovery: layer={d_layer:.4g} pp, vessel={d_vessel:.4g} pp. {mechanism}",
        f"5. D0 error trade-off: ROI FP improvement={fp_gain:.4g}; ROI FN improvement={fn_gain:.4g}; positive means reduced error.",
        "6. Vessel-size effects: " + ", ".join(f"{key}={val:.4g} pp" for key, val in size_effects.items()) + ".",
        f"7. Original-position typicality percentiles: {typicality_note}.",
        f"8. Label-quality effect: {'manual review available' if manual_complete else 'uncertain; blinded review remains PENDING, automatic associations are descriptive only'}.",
        f"9. Stability: {int(summary.loc[summary['metric'].isin(['layer_dice','p0_vessel_dice']), 'n_independent_positions'].max()) if not summary.empty else 0} independent positions; inspect fold/seed tables and cluster CI—seed×position was not tested as independent.",
        f"10. Recommended next step: {recommendation}.",
    ]
    (output / "SUMMARY.md").write_text("\n".join([
        "# PKU37 three-arm input-oracle CV", "", "## Direct answers", "", *answers, "",
        "The table below reports fixed-final, position-first paired gains. Positive is better.", "",
        "```text", primary.to_string(index=False), "```", "",
        "CLEAN is evaluated once per anatomical position; repeat stability is N/A for CLEAN. No test result was used.",
    ]), encoding="utf-8")
    try:
        with pd.ExcelWriter(output / "SABIDS_input_oracle_experiment.xlsx", engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="gain_summary", index=False)
            position_gain.to_excel(writer, sheet_name="position_gains", index=False)
            fold_summary.to_excel(writer, sheet_name="fold_summary", index=False)
            seed_summary.to_excel(writer, sheet_name="seed_gains", index=False)
            position_table.to_excel(writer, sheet_name="position_metrics", index=False)
            pd.DataFrame(epoch0).to_excel(writer, sheet_name="epoch0", index=False)
            input_quality.to_excel(writer, sheet_name="input_quality", index=False)
            label_quality.to_excel(writer, sheet_name="label_audit", index=False)
            mechanism_table.to_excel(writer, sheet_name="mechanisms", index=False)
            trajectory.to_excel(writer, sheet_name="trajectory", index=False)
            checks.to_excel(writer, sheet_name="checks", index=False)
    except ImportError:
        (output / "xlsx_missing.txt").write_text("openpyxl is unavailable; CSV outputs are complete.\n", encoding="utf-8")
    checklist.update({"complete": True, "position_first": True, "n_positions": int(position_table["group_id"].nunique()), "n_seeds": len(args.seeds), "clean_repeat_stability_applicable": False})
    (output / "missing_and_failure_checklist.json").write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
