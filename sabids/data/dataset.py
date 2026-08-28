from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .io import read_gray, read_mask
from .transforms import JointOCTTransform


REQUIRED_COLUMNS = {
    "sample_id",
    "group_id",
    "dataset",
    "split",
    "image_path",
}


def _is_path(value: object) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return bool(str(value).strip())


class OCTManifestDataset(Dataset):
    """Manifest-based OCT dataset with same-location repeat sampling."""

    def __init__(
        self,
        manifest: str | Path,
        split: str,
        transform: JointOCTTransform,
        sample_repeat: bool = True,
        root: Optional[str | Path] = None,
        datasets: Optional[List[str]] = None,
        groups: Optional[List[str]] = None,
    ) -> None:
        self.manifest = Path(manifest).expanduser().resolve()
        self.root = Path(root).expanduser().resolve() if root else self.manifest.parent
        self.transform = transform
        self.sample_repeat = sample_repeat
        table = pd.read_csv(self.manifest, dtype=str).fillna("")
        missing = REQUIRED_COLUMNS - set(table.columns)
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        table = table[table["split"].astype(str) == str(split)].copy()
        if datasets:
            table = table[table["dataset"].isin(datasets)].copy()
        if groups:
            table = table[table["group_id"].isin([str(value) for value in groups])].copy()
        if table.empty:
            raise ValueError(f"No samples for split={split!r} in {self.manifest}")
        self.table = table.reset_index(drop=True)
        self.group_to_indices: Dict[str, List[int]] = {}
        for index, group_id in enumerate(self.table["group_id"].astype(str)):
            self.group_to_indices.setdefault(group_id, []).append(index)

    def __len__(self) -> int:
        return len(self.table)

    @property
    def groups(self) -> Dict[str, List[int]]:
        return self.group_to_indices

    def _resolve(self, value: object) -> Optional[Path]:
        if not _is_path(value):
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    def _load_optional(self, value: object, mask: bool = False) -> Optional[np.ndarray]:
        path = self._resolve(value)
        if path is None:
            return None
        return read_mask(path) if mask else read_gray(path)

    def _repeat_index(self, index: int, group_id: str) -> tuple[int, bool]:
        candidates = self.group_to_indices[group_id]
        if not self.sample_repeat or len(candidates) < 2:
            return index, False
        others = [candidate for candidate in candidates if candidate != index]
        return random.choice(others), True

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.table.iloc[index]
        group_id = str(row["group_id"])
        repeat_index, has_repeat = self._repeat_index(index, group_id)
        repeat_row = self.table.iloc[repeat_index]

        image = self._load_optional(row["image_path"])
        repeat = self._load_optional(repeat_row["image_path"])
        if image is None or repeat is None:
            raise RuntimeError(f"Missing required image for sample {row['sample_id']}")
        original_height, original_width = image.shape[-2:]

        clean = self._load_optional(row.get("clean_path", ""))
        layer = self._load_optional(row.get("layer_mask_path", ""), mask=True)
        vessel = self._load_optional(row.get("vessel_mask_path", ""), mask=True)
        has_clean = clean is not None
        has_layer = layer is not None
        has_vessel = vessel is not None
        label_valid = self._load_optional(
            row.get("label_valid_mask_path", ""), mask=True
        )
        if label_valid is None:
            label_valid = (
                np.ones_like(image, dtype=np.float32)
                if has_layer
                else np.zeros_like(image, dtype=np.float32)
            )
        vessel_valid = self._load_optional(
            row.get("vessel_valid_mask_path", ""), mask=True
        )
        if vessel_valid is None:
            vessel_valid = label_valid.copy() if has_vessel and has_layer else (
                np.ones_like(image, dtype=np.float32)
                if has_vessel
                else np.zeros_like(image, dtype=np.float32)
            )

        is_clean_only = str(row.get("is_clean", "0")).lower() in {"1", "true", "yes"}
        allow_strong = self.transform.training and (
            not self.transform.strong_private_only or not has_clean
        )
        transformed = self.transform(
            arrays={"image": image, "repeat": repeat, "clean": clean},
            masks={
                "layer_mask": layer,
                "vessel_mask": vessel,
                "label_valid_mask": label_valid,
                "vessel_valid_mask": vessel_valid,
                "valid_mask": np.ones_like(image, dtype=np.float32),
            },
            allow_strong=allow_strong,
        )

        shape = transformed["image"].shape
        zeros = torch.zeros(shape, dtype=torch.float32)
        output: Dict[str, object] = {
            "image": transformed["image"],
            "image_weak": transformed["image_weak"],
            "repeat": transformed["repeat"],
            "repeat_weak": transformed["repeat_weak"],
            "clean": transformed.get("clean", zeros.clone()),
            "layer_mask": transformed.get("layer_mask", zeros.clone()),
            "vessel_mask": transformed.get("vessel_mask", zeros.clone()),
            "label_valid_mask": transformed["label_valid_mask"],
            "vessel_valid_mask": transformed["vessel_valid_mask"],
            "valid_mask": transformed["valid_mask"],
            "has_clean": torch.tensor(has_clean, dtype=torch.bool),
            "has_layer": torch.tensor(has_layer, dtype=torch.bool),
            "has_vessel": torch.tensor(has_vessel, dtype=torch.bool),
            "has_repeat": torch.tensor(has_repeat, dtype=torch.bool),
            "is_clean": torch.tensor(is_clean_only, dtype=torch.bool),
            "sample_id": str(row["sample_id"]),
            "group_id": group_id,
            "patient_id": str(row.get("patient_id", group_id)),
            "dataset": str(row["dataset"]),
            "scan_protocol": str(row.get("scan_protocol", "unknown")),
            "source_split": str(row.get("original_split", row.get("split", "unknown"))),
            "input_role": str(row.get("input_role", "noisy_input")),
            "original_path": str(self._resolve(row["image_path"])),
            "clean_path": str(self._resolve(row.get("clean_path", "")) or ""),
            "layer_mask_path": str(
                self._resolve(row.get("layer_mask_path", "")) or ""
            ),
            "vessel_mask_path": str(
                self._resolve(row.get("vessel_mask_path", "")) or ""
            ),
            "label_valid_mask_path": str(
                self._resolve(row.get("label_valid_mask_path", "")) or ""
            ),
            "original_height": int(original_height),
            "original_width": int(original_width),
            "manifest_group_frames": int(len(self.group_to_indices[group_id])),
        }
        return output
