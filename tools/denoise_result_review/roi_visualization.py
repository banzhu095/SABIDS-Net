from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd

from . import METHOD_ORDER
from .image_io import decode_lossless, display_uint8, read_float01
from .validation import find_image_manifest


DISPLAY_NAMES = {"noisy_identity": "Noisy", "bm3d_standard": "BM3D", "tv_chambolle": "TV", "nlm": "NLM", "ksvd_self": "K-SVD", "dncnn_paired": "DnCNN", "nafnet_paired": "NAFNet", "sabids_current": "SABIDS-current", "tcfl_dncnn": "TCFL-DnCNN"}


def build_panels(input_root: str | Path, registry_path: str | Path, output_root: str | Path) -> dict[str, int]:
    root, output = Path(input_root).resolve(), Path(output_root).resolve()
    registry = pd.read_csv(registry_path, dtype={"sample_id": str, "position_id": str})
    if not registry.locked.astype(str).str.lower().isin({"true", "1"}).all(): raise RuntimeError("panels require locked ROIs")
    assets = pd.read_csv(find_image_manifest(root), dtype={"sample_id": str, "position_id": str}, low_memory=False)
    for child in ("comparison_panels", "crops/raw", "crops/display", "crops/arrays", "method_noise_panels", "absolute_error_panels"): (output / child).mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics" / "per_roi_metrics.csv"
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    count = 0
    for roi in registry.itertuples(index=False):
        part = assets[assets.sample_id.astype(str) == str(roi.sample_id)]
        noisy_row, ref_row = part[part.asset_role == "noisy"], part[part.asset_role == "reference"]
        if len(noisy_row) != 1 or len(ref_row) != 1: continue
        noisy_path, reference_path = root / noisy_row.iloc[0].packaged_path, root / ref_row.iloc[0].packaged_path
        noisy, _ = read_float01(noisy_path); reference, _ = read_float01(reference_path)
        bounds = np.s_[int(roi.y0):int(roi.y1), int(roi.x0):int(roi.x1)]
        noisy_crop, ref_crop = noisy[bounds], reference[bounds]
        method_images, raw_images = {}, {"noisy_identity": decode_lossless(noisy_path)[bounds], "reference": decode_lossless(reference_path)[bounds]}
        for method in METHOD_ORDER:
            if method == "noisy_identity": method_images[method] = noisy_crop; continue
            hit = part[(part.asset_role == "method") & (part.method_id.astype(str) == method)]
            if len(hit) == 1:
                method_path = root / hit.iloc[0].packaged_path
                image, _ = read_float01(method_path); method_images[method] = image[bounds]
                raw_images[method] = decode_lossless(method_path)[bounds]
        fig, axes = plt.subplots(4, len(METHOD_ORDER)+1, figsize=(2*(len(METHOD_ORDER)+1), 8), squeeze=False)
        columns = [("reference", ref_crop)] + [(method, method_images.get(method)) for method in METHOD_ORDER]
        for column, (method, crop) in enumerate(columns):
            title = "Reference" if method == "reference" else DISPLAY_NAMES[method]
            if crop is None:
                for row in range(4): axes[row, column].text(.5, .5, "MISSING", ha="center", va="center", color="red"); axes[row, column].set_axis_off()
                continue
            residual = noisy_crop - crop
            images = (crop, np.abs(crop-ref_crop), residual, np.clip(0.5 + 2*residual, 0, 1))
            cmaps = ("gray", "magma", "coolwarm", "gray")
            for row, (image, cmap) in enumerate(zip(images, cmaps)):
                if row == 2: axes[row, column].imshow(image, cmap=cmap, vmin=-1, vmax=1)
                else: axes[row, column].imshow(image, cmap=cmap, vmin=0, vmax=1)
                axes[row, column].set_axis_off()
            if method != "reference" and not metrics.empty:
                hit = metrics[(metrics.roi_id.astype(str) == str(roi.roi_id)) & (metrics.method_id.astype(str) == method)]
                if not hit.empty: title += f"\nPSNR {hit.iloc[0].psnr:.2f} | SSIM {hit.iloc[0].ssim:.3f}"
            axes[0, column].set_title(title, fontsize=7)
            np.save(output / "crops" / "arrays" / f"{roi.roi_id}__{method}.npy", crop.astype(np.float32))
            display_path = output / "crops" / "display" / f"{roi.roi_id}__{method}.png"
            ok, encoded = cv2.imencode(".png", display_uint8(crop, (0.0, 1.0)))
            if not ok: raise RuntimeError(f"failed to encode display crop: {display_path}")
            encoded.tofile(display_path)
            raw_crop = raw_images.get(method)
            if raw_crop is not None:
                raw_path = output / "crops" / "raw" / f"{roi.roi_id}__{method}.tif"
                ok, encoded = cv2.imencode(".tif", raw_crop)
                if not ok: raise RuntimeError(f"failed to encode raw crop: {raw_path}")
                encoded.tofile(raw_path)
        for row, label in enumerate(("crop", "absolute error", "noisy - output", "residual x2 display")): axes[row, 0].set_ylabel(label)
        fig.suptitle(f"{roi.roi_id} | {roi.tissue} | common display range [0,1]"); fig.tight_layout()
        fig.savefig(output / "comparison_panels" / f"{roi.roi_id}.png", dpi=160); plt.close(fig)
        for directory, transform, cmap, vmin, vmax in (
            ("absolute_error_panels", lambda crop: np.abs(crop-ref_crop), "magma", 0, 1),
            ("method_noise_panels", lambda crop: noisy_crop-crop, "coolwarm", -1, 1),
        ):
            small, small_axes = plt.subplots(1, len(METHOD_ORDER), figsize=(2*len(METHOD_ORDER), 2), squeeze=False)
            for column, method in enumerate(METHOD_ORDER):
                axis, crop = small_axes[0, column], method_images.get(method)
                if crop is None: axis.text(.5, .5, "MISSING", ha="center", va="center", color="red")
                else: axis.imshow(transform(crop), cmap=cmap, vmin=vmin, vmax=vmax)
                axis.set_title(DISPLAY_NAMES[method], fontsize=8); axis.set_axis_off()
            small.tight_layout(); small.savefig(output / directory / f"{roi.roi_id}.png", dpi=160); plt.close(small)
        count += 1
    return {"roi_panels": count}
