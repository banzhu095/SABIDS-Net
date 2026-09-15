from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from . import METHOD_ORDER
from .image_io import decode_lossless, sha256_file


HISTORICAL_CANDIDATES = ("pku_0006_f26", "pku_0017_f26", "pku_0025_f26", "pku_0031_f26", "pku_0038_f23", "pku_0043_f26")


def find_image_manifest(input_root: str | Path) -> Path:
    root = Path(input_root).resolve()
    candidates = [root / "manifests" / "image_manifest.csv", root / "image_manifest.csv"]
    hits = [path for path in candidates if path.is_file()]
    if len(hits) != 1: raise FileNotFoundError(f"expected exactly one package image manifest, found {hits}")
    return hits[0]


def audit_local(input_root: str | Path, output_root: str | Path) -> dict[str, Any]:
    root, output = Path(input_root).resolve(), Path(output_root).resolve()
    audit = output / "audit"; audit.mkdir(parents=True, exist_ok=True)
    manifest_path = find_image_manifest(root)
    table = pd.read_csv(manifest_path, dtype={"sample_id": str, "position_id": str}, low_memory=False)
    key = [column for column in ("dataset", "split", "position_id", "sample_id", "asset_role", "method_id", "seed", "checkpoint_sha256") if column in table]
    issues = []
    duplicates = table[table.duplicated(key, keep=False)]
    for row in duplicates.to_dict("records"): issues.append({**row, "failure": "duplicate logical key"})
    inventory = []
    for row in table.itertuples(index=False):
        path = root / str(row.packaged_path)
        try:
            if not path.is_file(): raise FileNotFoundError(path)
            digest = sha256_file(path)
            if digest != str(row.sha256): raise ValueError("SHA256 mismatch")
            image = decode_lossless(path)
            if int(row.height) != image.shape[0] or int(row.width) != image.shape[1]:
                raise ValueError(f"manifest/image shape mismatch: {(row.height, row.width)} != {image.shape}")
            if hasattr(row, "dtype") and str(row.dtype) != str(image.dtype):
                raise ValueError(f"manifest/image dtype mismatch: {row.dtype} != {image.dtype}")
            if hasattr(row, "bit_depth") and int(row.bit_depth) != image.dtype.itemsize * 8:
                raise ValueError(f"manifest/image bit-depth mismatch: {row.bit_depth} != {image.dtype.itemsize * 8}")
            inventory.append({**row._asdict(), "absolute_path": str(path), "decoded": True, "actual_sha256": digest, "actual_height": image.shape[0], "actual_width": image.shape[1], "actual_dtype": str(image.dtype), "shape_matches": int(row.height) == image.shape[0] and int(row.width) == image.shape[1]})
        except Exception as exc:
            issues.append({**row._asdict(), "failure": f"{type(exc).__name__}: {exc}"})
    inventory_frame = pd.DataFrame(inventory)
    completeness = []
    expected = set(METHOD_ORDER[1:])
    for (position, sample), part in table.groupby(["position_id", "sample_id"]):
        methods = set(part.loc[part.asset_role == "method", "method_id"].astype(str))
        noisy = part[part.asset_role == "noisy"]
        reference = part[part.asset_role == "reference"]
        shapes = set(zip(part.height.astype(int), part.width.astype(int)))
        deep_primary_unique = all(
            len(part[(part.method_id == method) & (part.asset_role == "method")][[column for column in ("seed", "checkpoint_sha256") if column in part]].drop_duplicates()) <= 1
            for method in ("dncnn_paired", "nafnet_paired", "sabids_current", "tcfl_dncnn")
        )
        completeness.append({"position_id": position, "sample_id": sample, "methods_present": ";".join(sorted(methods)), "missing_methods": ";".join(sorted(expected - methods)), "noisy_unique": len(noisy) == 1, "reference_unique": len(reference) == 1, "shape_consistent": len(shapes) == 1, "deep_primary_seed_unique": deep_primary_unique, "complete": methods == expected and len(noisy) == 1 and len(reference) == 1 and len(shapes) == 1 and deep_primary_unique})
    completeness_frame = pd.DataFrame(completeness)
    zip_rows = []
    for archive in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(archive) as bundle:
                bad = bundle.testzip(); members = len(bundle.infolist())
                if bad is not None: raise ValueError(f"ZIP CRC failure at {bad}")
                embedded = pd.read_csv(bundle.open("IMAGE_MANIFEST.csv"), keep_default_na=False)
                payload_names = {str(value) for value in embedded.packaged_path}
                archived_payloads = {name for name in bundle.namelist() if name not in {"IMAGE_MANIFEST.csv", "README.md"}}
                if payload_names != archived_payloads:
                    raise ValueError(f"ZIP payload/manifest mismatch: manifest={len(payload_names)} archive={len(archived_payloads)}")
            zip_rows.append({"path": str(archive), "members": members, "crc_status": "passed" if bad is None else f"failed:{bad}", "sha256": sha256_file(archive)})
        except Exception as exc: issues.append({"path": str(archive), "failure": f"ZIP {type(exc).__name__}: {exc}"})
    inventory_frame.to_csv(audit / "local_image_inventory.csv", index=False)
    completeness_frame.to_csv(audit / "sample_completeness.csv", index=False)
    pd.DataFrame(issues).to_csv(audit / "missing_or_ambiguous_assets.csv", index=False)
    pd.DataFrame(zip_rows).to_csv(audit / "zip_inventory.csv", index=False)
    candidates = build_roi_candidates(root, table)
    candidate_path = output / "manifests" / "roi_candidate_samples.csv"; candidate_path.parent.mkdir(parents=True, exist_ok=True); candidates.to_csv(candidate_path, index=False)
    report = audit / "local_audit_report.md"
    report.write_text(f"# Local denoising package audit\n\n- Image records: {len(table)}\n- Decoded and hash-verified: {len(inventory_frame)}\n- Samples: {len(completeness_frame)}\n- Complete samples: {int(completeness_frame.complete.sum()) if not completeness_frame.empty else 0}\n- Issues: {len(issues)}\n- ZIP files checked: {len(zip_rows)}\n", encoding="utf-8")
    return {"status": "passed" if not issues and (completeness_frame.complete.all() if not completeness_frame.empty else False) else "incomplete", "manifest": str(manifest_path), "records": len(table), "samples": len(completeness_frame), "positions": table.position_id.nunique(), "issues": len(issues), "candidate_manifest": str(candidate_path)}


def build_roi_candidates(input_root: Path, table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sample in HISTORICAL_CANDIDATES:
        part = table[table.sample_id.astype(str) == sample]
        exists = not part.empty
        noisy = part[part.asset_role == "noisy"] if exists else pd.DataFrame()
        ref = part[part.asset_role == "reference"] if exists else pd.DataFrame()
        first = part.iloc[0] if exists else None
        rows.append({"dataset": str(first.dataset) if exists else "PKU37", "split": str(first.split) if exists else "test", "position_id": str(first.position_id) if exists else sample.rsplit("_f", 1)[0], "sample_id": sample, "noisy_path": str((input_root / noisy.iloc[0].packaged_path).resolve()) if len(noisy) == 1 else "", "reference_path": str((input_root / ref.iloc[0].packaged_path).resolve()) if len(ref) == 1 else "", "has_layer_label": False, "has_vessel_label": False, "selection_source": "historical_fixed_candidate_exact_match", "selection_reason": "Pre-existing candidate; not selected from denoising metrics", "exists": bool(exists and len(noisy) == 1 and len(ref) == 1)})
    return pd.DataFrame(rows)
