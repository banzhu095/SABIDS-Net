from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


TISSUES = ("vitreous", "retina", "choroid_vessel", "choroid_stroma", "custom")
ROI_COLUMNS = (
    "roi_id", "dataset", "split", "position_id", "sample_id", "tissue", "roi_size",
    "center_x", "center_y", "x0", "y0", "x1", "y1", "selection_image",
    "selection_reason", "reference_sha256", "noisy_sha256", "selected_at",
    "selection_phase", "locked", "locked_at", "registry_version",
)


def square_from_center(center_x: float, center_y: float, size: int, width: int, height: int) -> tuple[int, int, int, int]:
    if size not in {32, 48, 64}: raise ValueError("ROI size must be 32, 48, or 64 pixels")
    x0, y0 = int(round(center_x - size / 2)), int(round(center_y - size / 2))
    x1, y1 = x0 + size, y0 + size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(f"ROI would cross image boundary: {(x0, y0, x1, y1)} within {(width, height)}")
    return x0, y0, x1, y1


class ROIRegistry:
    def __init__(self, output_root: str | Path):
        self.output_root = Path(output_root).resolve()
        self.csv_path = self.output_root / "roi_registry.csv"
        self.json_path = self.output_root / "roi_registry.json"
        self.version_dir = self.output_root / "roi_registry_versions"
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.frame = pd.read_csv(self.csv_path, dtype={"sample_id": str, "position_id": str}) if self.csv_path.is_file() else pd.DataFrame(columns=ROI_COLUMNS)

    @property
    def locked(self) -> bool:
        return not self.frame.empty and self.frame.locked.astype(str).str.lower().isin({"true", "1"}).all()

    def _version(self) -> int:
        maximum = pd.to_numeric(self.frame.get("registry_version", pd.Series([0])), errors="coerce").max()
        frame_maximum = 0 if pd.isna(maximum) else int(maximum)
        saved_versions = []
        for path in self.version_dir.glob("roi_registry_v*.csv"):
            try: saved_versions.append(int(path.stem.rsplit("v", 1)[1]))
            except ValueError: continue
        return max([frame_maximum, *saved_versions], default=0) + 1

    def save(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        maximum = pd.to_numeric(self.frame.get("registry_version", pd.Series(dtype=float)), errors="coerce").max()
        version = self._version() if pd.isna(maximum) else int(maximum)
        for destination, writer in (
            (self.csv_path, lambda path: self.frame.to_csv(path, index=False)),
            (self.json_path, lambda path: Path(path).write_text(json.dumps(self.frame.where(pd.notna(self.frame), None).to_dict("records"), indent=2, ensure_ascii=False), encoding="utf-8")),
        ):
            fd, temp = tempfile.mkstemp(dir=self.output_root, prefix=f".{destination.name}.", suffix=".tmp"); os.close(fd)
            try: writer(temp); os.replace(temp, destination)
            finally: Path(temp).unlink(missing_ok=True)
        shutil.copy2(self.csv_path, self.version_dir / f"roi_registry_v{version:04d}.csv")

    def add(self, *, dataset: str, split: str, position_id: str, sample_id: str, tissue: str,
            roi_size: int, center_x: float, center_y: float, width: int, height: int,
            selection_image: str, selection_reason: str, reference_sha256: str, noisy_sha256: str) -> dict[str, Any]:
        if self.locked: raise RuntimeError("ROI registry is locked; use explicit unlock with a reason")
        if tissue not in TISSUES: raise ValueError(f"unknown tissue: {tissue}")
        x0, y0, x1, y1 = square_from_center(center_x, center_y, roi_size, width, height)
        version = self._version(); now = datetime.now(timezone.utc).isoformat()
        record = {"roi_id": f"{sample_id}_{tissue}_{len(self.frame)+1:03d}", "dataset": dataset, "split": split, "position_id": position_id, "sample_id": sample_id, "tissue": tissue, "roi_size": roi_size, "center_x": float(center_x), "center_y": float(center_y), "x0": x0, "y0": y0, "x1": x1, "y1": y1, "selection_image": selection_image, "selection_reason": selection_reason, "reference_sha256": reference_sha256, "noisy_sha256": noisy_sha256, "selected_at": now, "selection_phase": "blinded_selection", "locked": False, "locked_at": "", "registry_version": version}
        self.frame = pd.DataFrame([record], columns=ROI_COLUMNS) if self.frame.empty else pd.concat([self.frame, pd.DataFrame([record])], ignore_index=True)
        return record

    def delete(self, roi_id: str) -> None:
        if self.locked: raise RuntimeError("ROI registry is locked")
        self.frame = self.frame[self.frame.roi_id.astype(str) != str(roi_id)].reset_index(drop=True)

    def lock(self) -> None:
        if self.frame.empty: raise ValueError("cannot lock an empty ROI registry")
        now, version = datetime.now(timezone.utc).isoformat(), self._version()
        self.frame["locked"] = True; self.frame["locked_at"] = now; self.frame["registry_version"] = version
        self.save()

    def unlock(self, reason: str) -> None:
        if not reason.strip(): raise ValueError("unlock reason is required")
        if self.csv_path.is_file():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            shutil.copy2(self.csv_path, self.version_dir / f"roi_registry_locked_{stamp}.csv")
        version = self._version(); self.frame["locked"] = False; self.frame["locked_at"] = ""; self.frame["registry_version"] = version
        (self.version_dir / f"unlock_reason_v{version:04d}.txt").write_text(reason.strip() + "\n", encoding="utf-8")
        self.save()

    def reset(self, reason: str) -> None:
        """Start a blank working registry while retaining an auditable prior version."""
        if not reason.strip(): raise ValueError("reset reason is required")
        if self.locked: self.unlock(reason)
        version = self._version()
        if self.csv_path.is_file():
            shutil.copy2(self.csv_path, self.version_dir / f"roi_registry_before_reset_v{version:04d}.csv")
        (self.version_dir / f"reset_reason_v{version:04d}.txt").write_text(reason.strip() + "\n", encoding="utf-8")
        self.frame = pd.DataFrame(columns=ROI_COLUMNS)
        self.save()


def require_locked(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_id": str, "position_id": str})
    missing = set(ROI_COLUMNS) - set(frame.columns)
    if missing: raise ValueError(f"ROI registry missing columns: {sorted(missing)}")
    if frame.empty or not frame.locked.astype(str).str.lower().isin({"true", "1"}).all():
        raise RuntimeError("formal ROI evaluation requires a fully locked registry")
    return frame
