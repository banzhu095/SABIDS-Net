from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PANELS = (
    "noisy", "clean", "denoised", "reference_abs_error",
    "layer_gt", "vessel_gt", "layer_prob", "vessel_prob",
    "layer_overlay", "vessel_overlay", "vessel_error_tp_fp_fn", "boundary_overlay",
)


def _placeholder(text: str, height: int = 240, width: int = 240) -> np.ndarray:
    tile = np.full((height, width, 3), 235, np.uint8)
    cv2.putText(tile, text[:24], (10, height // 2), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 180), 1, cv2.LINE_AA)
    return tile


def _read(path: Path) -> tuple[np.ndarray | None, str | None]:
    if not path.is_file(): return None, "MISSING"
    image = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
    return (image, None) if image is not None and image.size else (None, "DECODE FAIL")


def build_report_atlas(report: str | Path) -> list[dict[str, str]]:
    report = Path(report)
    panels = (
        ("noisy", "clean", "denoised", "reference_abs_error")
        if report.name.startswith("d1_structure_")
        else ("noisy", "layer_gt", "vessel_gt", "layer_prob", "vessel_prob", "layer_overlay", "vessel_overlay", "vessel_error_tp_fp_fn", "boundary_overlay")
    )
    frame_path, selection_path = report / "metrics_by_frame.csv", report / "atlas_selection.csv"
    if not frame_path.is_file() or frame_path.stat().st_size == 0: return [{"run_id": "report", "asset": str(frame_path), "reason": "MISSING"}]
    try: frames = pd.read_csv(frame_path)
    except pd.errors.EmptyDataError: return [{"run_id": "report", "asset": str(frame_path), "reason": "MISSING"}]
    if frames.empty or not {"run_id", "arm", "group_id", "sample_id"}.issubset(frames.columns): return [{"run_id": "report", "asset": str(frame_path), "reason": "MISSING columns"}]
    if selection_path.is_file():
        try: groups = pd.read_csv(selection_path)["group_id"].astype(str).tolist()
        except Exception: groups = []
    else: groups = []
    groups = groups or sorted(frames.group_id.astype(str).unique())[:3]
    destination = report / "fixed_atlas_sheets"; destination.mkdir(parents=True, exist_ok=True)
    missing = []
    for group in groups:
        part = frames[frames.group_id.astype(str).eq(str(group))]
        if part.empty: continue
        sample_id = sorted(part.sample_id.astype(str).unique())[0]
        rows = []
        reference_shape = None
        crop = None
        for run_id, arm in part[["run_id", "arm"]].drop_duplicates().sort_values("arm").itertuples(index=False):
            sample_dir = report / "fixed_atlas" / str(run_id) / sample_id
            vessel_gt, _ = _read(sample_dir / f"{sample_id}_vessel_gt.png")
            if crop is None and vessel_gt is not None:
                gray = cv2.cvtColor(vessel_gt, cv2.COLOR_BGR2GRAY); ys, xs = np.where(gray > 127)
                center_y = int(np.median(ys)) if ys.size else vessel_gt.shape[0] // 2
                center_x = int(np.median(xs)) if xs.size else vessel_gt.shape[1] // 2
                size = min(128, vessel_gt.shape[0], vessel_gt.shape[1]); crop = (max(0, center_y-size//2), max(0, center_x-size//2), size)
            tiles = []
            for suffix in panels:
                path = sample_dir / f"{sample_id}_{suffix}.png"
                image, error = _read(path)
                if image is None:
                    image = _placeholder(error or "MISSING")
                    missing.append({"run_id": str(run_id), "asset": str(path), "reason": error or "MISSING"})
                reference_shape = reference_shape or image.shape[:2]
                tile = cv2.resize(image, (240, 240), interpolation=cv2.INTER_NEAREST)
                cv2.rectangle(tile, (0, 0), (239, 22), (255, 255, 255), -1); cv2.putText(tile, suffix, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
                tiles.append(tile)
            row = np.concatenate(tiles, axis=1)
            cv2.putText(row, str(arm), (4, 238), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 255), 1)
            rows.append(row)
        if rows:
            sheet = np.concatenate(rows, axis=0); cv2.imencode(".png", sheet)[1].tofile(str(destination / f"{group}_{sample_id}_atlas.png"))
        if crop and rows:
            y, x, size = crop; zoom_rows = []
            for run_id, arm in part[["run_id", "arm"]].drop_duplicates().sort_values("arm").itertuples(index=False):
                tiles=[]; sample_dir=report/"fixed_atlas"/str(run_id)/sample_id
                for suffix in panels:
                    image,error=_read(sample_dir/f"{sample_id}_{suffix}.png")
                    if image is None or y+size>image.shape[0] or x+size>image.shape[1]: tile=_placeholder("CROP OOB" if image is not None else error or "MISSING")
                    else: tile=cv2.resize(image[y:y+size,x:x+size],(240,240),interpolation=cv2.INTER_NEAREST)
                    tiles.append(tile)
                zoom_rows.append(np.concatenate(tiles,axis=1))
            cv2.imencode(".png",np.concatenate(zoom_rows,axis=0))[1].tofile(str(destination/f"{group}_{sample_id}_zoom.png"))
    return missing
