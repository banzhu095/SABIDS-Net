from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a fixed, model-independent interaction atlas")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-dataset", type=int, default=2)
    parser.add_argument("--output", default="runs/interaction_factorial_report/atlas")
    return parser.parse_args()


def read_image(path: Path, shape: tuple[int, int]) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR) if path.is_file() else None
    if image is None:
        image = np.zeros((*shape, 3), dtype=np.uint8)
        cv2.putText(image, "MISSING", (5, shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    return image


def main() -> None:
    args = parse_args()
    root = Path(args.project_root).expanduser().resolve()
    b0 = root / "runs" / "current" / f"interaction_b0_fold0_seed{args.seed}" / "validation"
    frame_path, crop_path = b0 / "frame_metrics.csv", b0 / "qualitative_crops.csv"
    if not frame_path.is_file() or not crop_path.is_file():
        raise FileNotFoundError("B0 validation predictions/crop registry are required; run B0 with --save-predictions")
    frames, crops = pd.read_csv(frame_path, dtype=str), pd.read_csv(crop_path)
    selected = (
        frames.sort_values(["dataset", "group_id", "sample_id"])
        .groupby("dataset", as_index=False, group_keys=False)
        .head(args.samples_per_dataset)
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
    output.mkdir(parents=True, exist_ok=True)
    selection_rows = []
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
                image = read_image(prediction_dir / f"{sample_id}_{suffix}.png", (height, width))
                image = image[y:y + height, x:x + width]
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
            "selection_rule": "lexicographically first fixed samples per dataset; crop defined once from B0 GT",
        })
    pd.DataFrame(selection_rows).to_csv(output / "atlas_selection.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
