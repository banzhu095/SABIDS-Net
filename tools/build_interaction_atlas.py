from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed, model-independent interaction atlas")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--labelled-frames-per-group", type=int, default=1)
    parser.add_argument("--output", default="runs/interaction_factorial_report/atlas")
    parser.add_argument(
        "--archive-existing", action="store_true",
        help="Move an existing non-empty atlas directory to a timestamped sibling before rebuilding.",
    )
    return parser.parse_args()


def _placeholder(shape: tuple[int, int], label: str) -> np.ndarray:
    height, width = max(int(shape[0]), 32), max(int(shape[1]), 64)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        image, label, (5, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
        0.35, (0, 0, 255), 1, cv2.LINE_AA,
    )
    return image


def read_crop(
    path: Path, x: int, y: int, width: int, height: int,
) -> tuple[np.ndarray, str]:
    """Read an original-grid image and safely apply the shared atlas crop."""
    if not path.is_file():
        return _placeholder((height, width), "MISSING"), "missing"
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False).astype(np.float32)
        image = cv2.cvtColor(np.round(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return _placeholder((height, width), "DECODE FAIL"), "decode_failed"
    image_height, image_width = image.shape[:2]
    x0, y0 = max(int(x), 0), max(int(y), 0)
    x1 = min(int(x + width), image_width)
    y1 = min(int(y + height), image_height)
    if x0 >= x1 or y0 >= y1:
        # Keep the atlas inspectable and make the geometry mismatch explicit.
        fallback = cv2.resize(
            image, (max(int(width), 1), max(int(height), 1)),
            interpolation=cv2.INTER_AREA,
        )
        cv2.putText(
            fallback, "CROP OOB", (5, max(fallback.shape[0] // 2, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA,
        )
        return fallback, "crop_out_of_bounds"
    return image[y0:y1, x0:x1], "ok"


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    b0 = root / "runs" / "current" / f"interaction_b0_fold0_seed{args.seed}" / "validation"
    frame_path, crop_path = b0 / "frame_metrics.csv", b0 / "qualitative_crops.csv"
    if not frame_path.is_file() or not crop_path.is_file():
        raise FileNotFoundError("B0 validation predictions/crop registry are required; run B0 with --save-predictions")
    frames, crops = pd.read_csv(frame_path, dtype=str), pd.read_csv(crop_path)
    availability_columns = [
        column for column in ("psnr", "layer_dice", "p0_vessel_dice")
        if column in frames.columns
    ]
    frames["_availability_score"] = sum(
        frames[column].notna() & (frames[column].astype(str).str.strip() != "")
        for column in availability_columns
    ) if availability_columns else 0
    dataset_selected = (
        frames.sort_values(
            ["dataset", "_availability_score", "group_id", "sample_id"],
            ascending=[True, False, True, True],
        )
        .groupby("dataset", as_index=False, group_keys=False)
        .head(args.samples_per_dataset)
    )
    vessel_metric = "p0_vessel_dice" if "p0_vessel_dice" in frames else "vessel_dice"
    labelled = frames[frames[vessel_metric].notna()].sort_values(["group_id", "sample_id"])
    labelled_selected = labelled.groupby("group_id", as_index=False, group_keys=False).head(
        args.labelled_frames_per_group
    )
    selected = pd.concat([dataset_selected, labelled_selected], ignore_index=True).drop_duplicates(
        ["dataset", "sample_id"]
    )
    runs = {
        "B0": b0,
        **{
            variant.upper(): root / "runs" / "current" / f"interaction_{variant}_fold0_seed{args.seed}" / "final_validation"
            for variant in ("j00", "j10", "j01", "j11")
        },
    }
    columns = (
        "noisy", "clean", "denoised", "reference_abs_error", "layer_gt", "vessel_gt",
        "layer_prob", "vessel_prob", "layer_mask", "vessel_mask", "vessel_tp", "vessel_fp",
        "vessel_fn", "boundary_overlay",
    )
    output = (root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)
    if output.exists() and any(output.iterdir()):
        if not args.archive_existing:
            raise FileExistsError(
                f"Atlas output is non-empty; rerun with --archive-existing to preserve it: {output}"
            )
        archive = output.with_name(
            output.name + "_archive_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        if archive.exists():
            raise FileExistsError(f"Timestamped atlas archive already exists: {archive}")
        shutil.move(str(output), str(archive))
        print(f"Archived existing atlas without deletion: {archive}")
    output.mkdir(parents=True, exist_ok=True)
    selection_rows = []
    asset_rows = []
    tile_h, tile_w, header = 128, 128, 20
    for _, row in selected.iterrows():
        dataset, sample_id = str(row["dataset"]), str(row["sample_id"])
        crop_match = crops[(crops["dataset"].astype(str) == dataset) & (crops["sample_id"].astype(str) == sample_id)]
        if crop_match.empty:
            continue
        crop = crop_match.iloc[0]
        x, y, width, height = (int(crop[key]) for key in ("x", "y", "width", "height"))
        canvas = np.full((len(runs) * (tile_h + header), len(columns) * tile_w, 3), 255, dtype=np.uint8)
        for run_index, (run_name, run_dir) in enumerate(runs.items()):
            prediction_dir = run_dir / "predictions" / dataset
            for column_index, suffix in enumerate(columns):
                asset_path = prediction_dir / f"{sample_id}_{suffix}.png"
                image, status = read_crop(asset_path, x, y, width, height)
                asset_rows.append({
                    "dataset": dataset, "group_id": row["group_id"],
                    "sample_id": sample_id, "run": run_name, "asset": suffix,
                    "status": status, "path": str(asset_path),
                })
                image = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                y0, x0 = run_index * (tile_h + header), column_index * tile_w
                canvas[y0 + header:y0 + header + tile_h, x0:x0 + tile_w] = image
                cv2.putText(canvas, f"{run_name}:{suffix}", (x0 + 2, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1)
        encoded, buffer = cv2.imencode(".png", canvas)
        if not encoded:
            raise RuntimeError("Failed to encode atlas")
        (output / f"{dataset}_{sample_id}.png").write_bytes(buffer.tobytes())
        selection_rows.append({
            "dataset": dataset, "group_id": row["group_id"], "sample_id": sample_id,
            "x": x, "y": y, "width": width, "height": height,
            "availability_score": int(row["_availability_score"]),
            "selection_rule": "highest B0 GT/clean availability, then lexical IDs; crop defined once from B0 GT",
        })
    if not selection_rows:
        raise RuntimeError("No atlas samples matched the B0 crop registry")
    pd.DataFrame(selection_rows).to_csv(output / "atlas_selection.csv", index=False, encoding="utf-8-sig")
    asset_table = pd.DataFrame(asset_rows)
    asset_table.to_csv(output / "atlas_asset_inventory.csv", index=False, encoding="utf-8-sig")
    missing_table = asset_table[asset_table["status"] != "ok"]
    missing_table.to_csv(output / "atlas_missing_assets.csv", index=False, encoding="utf-8-sig")
    (output / "atlas_build_summary.json").write_text(json.dumps({
        "seed": args.seed,
        "n_atlas_samples": len(selection_rows),
        "n_expected_assets": len(asset_table),
        "n_missing_or_invalid_assets": len(missing_table),
        "status_counts": asset_table["status"].value_counts().astype(int).to_dict(),
        "note": "Missing clean/GT products may be structurally expected for incompletely labelled datasets; inspect atlas_missing_assets.csv.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
