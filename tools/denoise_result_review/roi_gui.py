from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

from .image_io import read_float01
from .roi_registry import ROIRegistry, TISSUES, square_from_center


TISSUE_NAMES = {"vitreous": "玻璃体", "retina": "视网膜", "choroid_vessel": "脉络膜血管", "choroid_stroma": "脉络膜基质", "custom": "自定义"}


def select_rois(input_root: str | Path, selection_list: str | Path, output_root: str | Path,
                roi_size: int = 48, unlock_rois: bool = False, unlock_reason: str = "") -> None:
    candidates = pd.read_csv(selection_list, dtype={"sample_id": str, "position_id": str})
    candidates = candidates[candidates.exists.astype(str).str.lower().isin({"true", "1"})].reset_index(drop=True)
    if candidates.empty: raise ValueError("selection list contains no existing exact-match candidates")
    registry = ROIRegistry(output_root)
    if unlock_rois: registry.unlock(unlock_reason)
    if registry.locked: raise RuntimeError("ROI registry is locked; methods remain hidden unless explicitly unlocked")
    state = {"index": 0, "tissue": "custom", "last_roi": None}
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    def save_overlays() -> None:
        overlay_dir = registry.output_root / "overlays"; overlay_dir.mkdir(parents=True, exist_ok=True)
        for item in candidates.itertuples(index=False):
            current = registry.frame[registry.frame.sample_id.astype(str) == str(item.sample_id)]
            if current.empty: continue
            noisy, _ = read_float01(item.noisy_path)
            overlay_fig, axis = plt.subplots(figsize=(7, 7)); axis.imshow(noisy, cmap="gray", vmin=0, vmax=1); axis.set_axis_off()
            for roi in current.itertuples(index=False):
                axis.add_patch(Rectangle((roi.x0, roi.y0), roi.x1-roi.x0, roi.y1-roi.y0, fill=False, linewidth=1.5))
                axis.text(roi.x0, roi.y0, roi.roi_id, color="yellow", fontsize=7)
            overlay_fig.tight_layout(); overlay_fig.savefig(overlay_dir / f"{item.sample_id}.png", dpi=160); plt.close(overlay_fig)

    def warn_missing_tissues() -> None:
        for sample, part in registry.frame.groupby("sample_id"):
            missing = set(TISSUES[:4]) - set(part.tissue.astype(str))
            if missing: print(f"warning: {sample} has no ROI for {sorted(missing)}")

    def draw() -> None:
        row = candidates.iloc[state["index"]]
        noisy, noisy_meta = read_float01(row.noisy_path); reference, _ = read_float01(row.reference_path)
        for axis, image, title in zip(axes, (noisy, reference), ("Noisy (blinded selection)", "Reference (blinded selection)")):
            axis.clear(); axis.imshow(image, cmap="gray", vmin=0, vmax=1); axis.set_title(title); axis.set_axis_off()
            current = registry.frame[registry.frame.sample_id.astype(str) == str(row.sample_id)]
            for item in current.itertuples(index=False):
                axis.add_patch(Rectangle((item.x0, item.y0), item.x1-item.x0, item.y1-item.y0, fill=False, linewidth=1.5))
                axis.text(item.x0, item.y0, item.roi_id, color="yellow", fontsize=7)
        fig.suptitle(f"{row.sample_id} | {TISSUE_NAMES[state['tissue']]} | ROI {roi_size}x{roi_size} | {state['index']+1}/{len(candidates)}")
        fig.canvas.draw_idle()

    def click(event) -> None:
        if event.inaxes not in axes or event.xdata is None or event.ydata is None: return
        row = candidates.iloc[state["index"]]
        noisy, meta = read_float01(row.noisy_path); _, ref_meta = read_float01(row.reference_path)
        try:
            square_from_center(event.xdata, event.ydata, roi_size, noisy.shape[1], noisy.shape[0])
            record = registry.add(dataset=row.dataset, split=row.split, position_id=str(row.position_id), sample_id=str(row.sample_id), tissue=state["tissue"], roi_size=roi_size, center_x=event.xdata, center_y=event.ydata, width=noisy.shape[1], height=noisy.shape[0], selection_image="noisy_and_reference_only", selection_reason="manual blinded anatomical selection", reference_sha256=ref_meta["sha256"], noisy_sha256=meta["sha256"])
            state["last_roi"] = record["roi_id"]; draw()
        except ValueError as exc: print(f"ROI rejected: {exc}")

    def key(event) -> None:
        mapping = {str(index+1): tissue for index, tissue in enumerate(TISSUES)}
        if event.key in mapping: state["tissue"] = mapping[event.key]; draw()
        elif event.key in {"n", "N"}: state["index"] = min(len(candidates)-1, state["index"]+1); draw()
        elif event.key in {"p", "P"}: state["index"] = max(0, state["index"]-1); draw()
        elif event.key in {"s", "S"}: registry.save(); save_overlays(); print(f"saved {registry.csv_path}")
        elif event.key in {"d", "D"} and state["last_roi"]: registry.delete(state["last_roi"]); state["last_roi"] = None; draw()
        elif event.key in {"l", "L"}: warn_missing_tissues(); registry.lock(); save_overlays(); print(f"locked {registry.csv_path}"); plt.close(fig)
        elif event.key in {"q", "Q"}: registry.save(); save_overlays(); plt.close(fig)
    fig.canvas.mpl_connect("button_press_event", click); fig.canvas.mpl_connect("key_press_event", key)
    draw(); plt.show()
