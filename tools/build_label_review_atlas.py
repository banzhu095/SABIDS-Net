from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sabids.data.io import read_gray, read_mask
from sabids.utils import write_json
from tools.input_oracle_cv_common import resolve


def _rgb(image: np.ndarray) -> np.ndarray:
    value = np.clip(image * 255 if image.max() <= 1.0 else image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR) if value.ndim == 2 else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Blinded, pre-model PKU37 label-review atlas")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", default="runs/input_oracle_cv/label_audit/atlas")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve(); output = resolve(root, args.output)
    audit = json.loads((root / "runs/input_oracle_cv/audit/audit_summary.json").read_text(encoding="utf-8"))
    # A blocked count/split audit may still produce a safe label atlas for the
    # metadata-defined non-sealed union; it never authorizes training.
    development, sealed = set(audit["development_groups"]), set(audit["sealed_test_groups"])
    manifest = pd.read_csv(audit["source_manifest"], dtype=str).fillna("")
    manifest = manifest[manifest["group_id"].isin(development) & ~manifest["group_id"].isin(sealed)]
    if output.exists() and any(output.iterdir()) and not args.resume: raise FileExistsError(output)
    png = output / "png"; png.mkdir(parents=True, exist_ok=True)
    pages, inventory, missing = [], [], []
    for group_id, group in manifest.groupby("group_id", sort=True):
        ordered = group.sort_values("sample_id"); indices = sorted(set([0, len(ordered)//2, len(ordered)-1])); selected = ordered.iloc[indices]
        first = ordered.iloc[0]
        clean_path = resolve(root, first["clean_path"]); layer_path = resolve(root, first["layer_mask_path"]); vessel_path = resolve(root, first["vessel_mask_path"])
        clean = read_gray(clean_path) if clean_path and clean_path.is_file() else None
        layer = read_mask(layer_path) > .5 if layer_path and layer_path.is_file() else None
        vessel = read_mask(vessel_path) > .5 if vessel_path and vessel_path.is_file() else None
        panels: list[tuple[str, np.ndarray | None]] = [("clean", clean)]
        for _, row in selected.iterrows():
            path = resolve(root, row["image_path"]); panels.append((f"noisy:{row['sample_id']}", read_gray(path) if path and path.is_file() else None))
        if clean is not None and layer is not None and clean.shape == layer.shape:
            overlay = _rgb(clean); boundary = layer & ~cv2.erode(layer.astype(np.uint8), np.ones((3,3), np.uint8)).astype(bool); overlay[boundary] = (0,255,0)
            panels.append(("layer_overlay", overlay))
        else: panels.append(("layer_overlay", None))
        if clean is not None and vessel is not None and clean.shape == vessel.shape:
            overlay = _rgb(clean); overlay[vessel] = (0,128,255); panels.append(("vessel_overlay", overlay))
        else: panels.append(("vessel_overlay", None))
        size = 180; canvas = np.full((size + 25, size * len(panels), 3), 255, np.uint8)
        for index, (name, image) in enumerate(panels):
            if image is None or image.size == 0:
                tile = np.full((size,size,3), 235, np.uint8); cv2.putText(tile,"MISSING",(35,90),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),1); status="missing"; missing.append({"group_id":group_id,"asset":name})
            else: tile=cv2.resize(_rgb(image),(size,size),interpolation=cv2.INTER_NEAREST); status="ok"
            canvas[25:, index*size:(index+1)*size]=tile; cv2.putText(canvas,name[:24],(index*size+3,17),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),1)
            inventory.append({"group_id":group_id,"asset":name,"status":status,"selection_uses_model_results":False})
        target=png/f"{group_id}.png"
        if not args.dry_run:
            ok, encoded=cv2.imencode('.png',canvas)
            if not ok: raise RuntimeError('Encoding failed')
            target.write_bytes(encoded.tobytes())
        pages.append(f"<h3>{html.escape(str(group_id))}</h3><img src='png/{target.name}'>")
    pd.DataFrame(inventory).to_csv(output/"asset_inventory.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(missing).to_csv(output/"missing_assets.csv",index=False,encoding="utf-8-sig")
    (output/"index.html").write_text("<html><body>"+"\n".join(pages)+"</body></html>",encoding="utf-8")
    write_json({"positions":len(pages),"missing_assets":len(missing),"selection_rule":"first_middle_last_by_sample_id","model_results_viewed":False,"test_assets_opened":0,"training_authorized":audit.get("training_authorized",False)},output/"atlas_summary.json")


if __name__ == "__main__": main()
