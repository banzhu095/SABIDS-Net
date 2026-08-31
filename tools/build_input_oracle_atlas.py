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

from tools.build_interaction_atlas import read_crop
from tools.input_oracle_cv_common import ARMS, resolve, sha256_file


def _tile(path: Path, x: int, y: int, width: int, height: int, size: int = 144) -> tuple[np.ndarray, str]:
    image, status = read_crop(path, x, y, width, height)
    if image is None or image.size == 0:
        image = np.full((size, size, 3), 235, np.uint8)
        cv2.putText(image, "MISSING", (15, size // 2), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 1)
        return image, "missing"
    return cv2.resize(image, (size, size), interpolation=cv2.INTER_NEAREST), status


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed, all-position three-arm input-oracle atlas")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--runs", default="runs/input_oracle_cv")
    parser.add_argument("--output", default="runs/input_oracle_cv/report/atlas")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).expanduser().resolve()
    runs, output = resolve(root, args.runs), resolve(root, args.output)
    fold_suffix = "_smoke" if args.smoke_test else ""
    split_audit = json.loads((runs / "splits/split_audit.json").read_text(encoding="utf-8"))
    folds = [args.fold] if args.fold is not None else sorted(map(int, split_audit["folds"]))
    selections, assets, missing, pages = [], [], [], []
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite atlas: {output}")
    png_dir = output / "png"; png_dir.mkdir(parents=True, exist_ok=True)
    suffixes = ("layer_prob", "vessel_prob", "layer_mask", "vessel_mask", "layer_error_tp_fp_fn", "vessel_error_tp_fp_fn", "boundary_overlay")
    for fold in folds:
        manifest = pd.read_csv(root / f"Manifests/input_oracle_cv/fold_{fold}_input{'_smoke' if args.smoke_test else ''}.csv", dtype=str).fillna("")
        val = manifest[manifest["split"] == "val"].sort_values(["group_id", "sample_id"])
        for group_id, group in val.groupby("group_id", sort=True):
            chosen = group.iloc[0]  # pre-model fixed lexicographic rule
            sample_id = str(chosen["sample_id"]); dataset = str(chosen["dataset"])
            selections.append({"fold": fold, "group_id": group_id, "sample_id": sample_id, "selection_rule": "lexicographically_first_sample_id", "seed": args.seed})
            gt = resolve(root, chosen.get("vessel_mask_path", ""))
            if gt and gt.is_file():
                mask = cv2.imread(str(gt), cv2.IMREAD_GRAYSCALE)
                ys, xs = np.where(mask > 0) if mask is not None else (np.array([]), np.array([]))
                cy, cx = (int(np.median(ys)), int(np.median(xs))) if len(ys) else (320, 320)
            else:
                cy, cx = 320, 320
            width = height = 160; x, y = max(0, cx - 80), max(0, cy - 80)
            common = {
                "NOISY": resolve(root, chosen["noisy_cache_path"]),
                "CLEAN": resolve(root, chosen["clean_cache_path"]),
                "DENOISED": resolve(root, chosen["denoised_cache_path"]),
                "GT_LAYER": resolve(root, chosen.get("layer_mask_path", "")),
                "GT_VESSEL": resolve(root, chosen.get("vessel_mask_path", "")),
            }
            columns: list[tuple[str, Path]] = [(key, value) for key, value in common.items() if value is not None]
            for arm in ARMS:
                prediction = runs / f"fold{fold}{fold_suffix}/{arm}_seed{args.seed}/final_validation/predictions/{dataset}"
                for name in suffixes:
                    columns.append((f"{arm.upper()}:{name}", prediction / f"{sample_id}_{name}.png"))
            tile_size = 144
            canvas = np.full((tile_size + 26, tile_size * len(columns), 3), 255, np.uint8)
            for index, (label, path) in enumerate(columns):
                tile, status = _tile(path, x, y, width, height, tile_size)
                canvas[26:, index * tile_size:(index + 1) * tile_size] = tile
                cv2.putText(canvas, label[:22], (index * tile_size + 2, 17), cv2.FONT_HERSHEY_SIMPLEX, .31, (0, 0, 0), 1)
                record = {"fold": fold, "seed": args.seed, "group_id": group_id, "sample_id": sample_id, "asset": label, "path": str(path), "status": status}
                if path.is_file():
                    record["sha256"] = sha256_file(path)
                else:
                    missing.append(record.copy())
                assets.append(record)
            target = png_dir / f"fold{fold}_{group_id}_{sample_id}.png"
            if target.exists() and not args.resume:
                raise FileExistsError(target)
            if not args.dry_run:
                ok, encoded = cv2.imencode(".png", canvas)
                if not ok: raise RuntimeError("Atlas encoding failed")
                target.write_bytes(encoded.tobytes())
            pages.append(f"<h3>fold {fold} / {html.escape(str(group_id))} / {html.escape(sample_id)}</h3><img loading='lazy' src='png/{html.escape(target.name)}'>")
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selections).to_csv(output / "atlas_selection.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(assets).to_csv(output / "atlas_asset_inventory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(missing).to_csv(output / "atlas_missing_assets.csv", index=False, encoding="utf-8-sig")
    (output / "index.html").write_text("<html><body>" + "\n".join(pages) + "</body></html>", encoding="utf-8")
    summary = {"status": "passed" if not missing else "passed_with_missing_assets", "positions": len(selections), "missing_assets": len(missing), "selection_uses_model_metrics": False, "forced_legacy_groups_present": {g: any(x["group_id"] == g for x in selections) for g in ("pku_0006", "pku_0012", "pku_0040")}, "test_assets_opened": 0}
    (output / "atlas_build_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
