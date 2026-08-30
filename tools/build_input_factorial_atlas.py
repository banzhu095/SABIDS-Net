from __future__ import annotations

import argparse
import html
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.build_interaction_atlas import read_crop


def resolve(root: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else (root / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed I_NOISY/I_DENOISED validation atlas")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="runs/input_factorial_report/atlas_seed42")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    noisy_eval = root / f"runs/current/input_noisy_fold0_seed{args.seed}/final_validation"
    denoised_eval = root / f"runs/current/input_denoised_fold0_seed{args.seed}/final_validation"
    required = [noisy_eval / "frame_metrics.csv", noisy_eval / "qualitative_crops.csv", denoised_eval / "frame_metrics.csv"]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Both fixed-final evaluations with --save-predictions are required")
    frames = pd.read_csv(noisy_eval / "frame_metrics.csv")
    crops = pd.read_csv(noisy_eval / "qualitative_crops.csv")
    manifest = pd.read_csv(root / "Manifests/input_factorial/manifest_input_fold0.csv", dtype=str).fillna("")
    quality_path = root / "runs/current/input_factorial_common_fold0/cache/input_quality_metrics.csv"
    quality = pd.read_csv(quality_path) if quality_path.is_file() else pd.DataFrame()
    vessel_metric = "p0_vessel_dice" if "p0_vessel_dice" in frames else "vessel_dice"
    labelled = frames[frames[vessel_metric].notna()].copy()
    chosen = []
    for group_id, part in labelled.groupby("group_id"):
        part = part.sort_values("sample_id")
        ids = [str(part.iloc[0]["sample_id"]), str(part.iloc[len(part) // 2]["sample_id"]), str(part.iloc[-1]["sample_id"])]
        if not quality.empty:
            q = quality[quality["group_id"].astype(str) == str(group_id)].sort_values("psnr_noisy")
            if not q.empty:
                ids[1] = str(q.iloc[0]["sample_id"])  # difficult input, selected without model metrics
        for role, sample_id in zip(("ordinary_first", "difficult_low_noisy_psnr", "repeat_last"), ids):
            chosen.append({"group_id": group_id, "sample_id": sample_id, "selection_role": role})
    selection = pd.DataFrame(chosen).drop_duplicates(["group_id", "sample_id"])
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite atlas: {output}")
    output.mkdir(parents=True, exist_ok=True)
    columns = (
        "input", "layer_prob", "vessel_prob", "layer_mask", "vessel_mask",
        "vessel_tp", "vessel_fp", "vessel_fn", "boundary_overlay",
    )
    html_rows = []
    for _, selected in selection.iterrows():
        sample_id, group_id = str(selected["sample_id"]), str(selected["group_id"])
        frame = frames[frames["sample_id"].astype(str) == sample_id].iloc[0]
        dataset = str(frame["dataset"])
        crop_row = crops[crops["sample_id"].astype(str) == sample_id].iloc[0]
        x, y, width, height = [int(crop_row[key]) for key in ("x", "y", "width", "height")]
        manifest_row = manifest[(manifest["split"] == "val") & (manifest["sample_id"] == sample_id)].iloc[0]
        common_paths = {
            "noisy": resolve(root, manifest_row["noisy_cache_path"]),
            "clean": resolve(root, manifest_row["clean_path"]) if manifest_row.get("clean_path", "") else Path(""),
            "D0": resolve(root, manifest_row["denoised_cache_path"]),
            "layer_GT": noisy_eval / "predictions" / dataset / f"{sample_id}_layer_gt.png",
            "vessel_GT": noisy_eval / "predictions" / dataset / f"{sample_id}_vessel_gt.png",
        }
        run_rows = {
            "I_NOISY": noisy_eval / "predictions" / dataset,
            "I_DENOISED": denoised_eval / "predictions" / dataset,
        }
        tile_h = tile_w = 128
        headers = list(common_paths) + ["D0_abs_error"] + [f"{run}:{column}" for run in run_rows for column in columns]
        canvas = np.full((tile_h + 22, tile_w * len(headers), 3), 255, np.uint8)
        index = 0
        for label, path in common_paths.items():
            image, _ = read_crop(path, x, y, width, height)
            canvas[22:, index * tile_w:(index + 1) * tile_w] = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
            cv2.putText(canvas, label, (index * tile_w + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
            index += 1
        d0_crop, d0_status = read_crop(common_paths["D0"], x, y, width, height)
        clean_crop, clean_status = read_crop(common_paths["clean"], x, y, width, height)
        if d0_status == clean_status == "ok":
            common_height = min(d0_crop.shape[0], clean_crop.shape[0])
            common_width = min(d0_crop.shape[1], clean_crop.shape[1])
            difference = cv2.absdiff(
                d0_crop[:common_height, :common_width],
                clean_crop[:common_height, :common_width],
            )
        else:
            difference = np.zeros((max(height, 1), max(width, 1), 3), dtype=np.uint8)
            cv2.putText(difference, "MISSING", (4, max(height // 2, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        canvas[22:, index * tile_w:(index + 1) * tile_w] = cv2.resize(difference, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas, "D0_abs_error", (index * tile_w + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)
        index += 1
        for run, directory in run_rows.items():
            for suffix in columns:
                actual_suffix = "noisy" if suffix == "input" else suffix
                image, _ = read_crop(directory / f"{sample_id}_{actual_suffix}.png", x, y, width, height)
                canvas[22:, index * tile_w:(index + 1) * tile_w] = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                cv2.putText(canvas, f"{run}:{suffix}", (index * tile_w + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 0), 1)
                index += 1
        filename = f"{group_id}_{sample_id}.png"
        encoded, buffer = cv2.imencode(".png", canvas)
        if not encoded:
            raise RuntimeError("Atlas encoding failed")
        (output / filename).write_bytes(buffer.tobytes())
        html_rows.append(f"<h3>{html.escape(group_id)} / {html.escape(sample_id)} / {html.escape(selected['selection_role'])}</h3><img src='{html.escape(filename)}'>")
    selection.to_csv(output / "atlas_selection.csv", index=False, encoding="utf-8-sig")
    (output / "index.html").write_text("<html><body>" + "\n".join(html_rows) + "</body></html>", encoding="utf-8")


if __name__ == "__main__":
    main()
