from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Iterator, List

import numpy as np
from torch.utils.data import Sampler

from .dataset import OCTManifestDataset


class GroupUniformSampler(Sampler[int]):
    """Sample anatomical groups uniformly, then sample one frame per group."""

    def __init__(
        self,
        dataset: OCTManifestDataset,
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset = dataset
        self.group_ids = sorted(dataset.groups)
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = []
        repeats = math.ceil(self.samples_per_epoch / len(self.group_ids))
        for _ in range(repeats):
            shuffled = rng.permutation(self.group_ids)
            for group_id in shuffled:
                indices = self.dataset.groups[str(group_id)]
                order.append(int(rng.choice(indices)))
                if len(order) >= self.samples_per_epoch:
                    return iter(order)
        return iter(order)

    def __len__(self) -> int:
        return self.samples_per_epoch


class SparseAnnotationSampler(Sampler[int]):
    """Patient-balanced sampling with a controlled manual-vessel fraction."""

    def __init__(
        self,
        dataset: OCTManifestDataset,
        vessel_fraction: float = 0.35,
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ) -> None:
        if not 0.0 <= vessel_fraction <= 1.0:
            raise ValueError("vessel_fraction must be between 0 and 1")
        self.dataset = dataset
        self.vessel_fraction = vessel_fraction
        self.samples_per_epoch = samples_per_epoch or len(dataset)
        self.seed = seed
        self.epoch = 0
        self.vessel_by_patient: Dict[str, List[int]] = defaultdict(list)
        self.other_by_patient: Dict[str, List[int]] = defaultdict(list)
        for index, row in dataset.table.iterrows():
            patient_id = str(row.get("patient_id", row["group_id"]))
            if str(row.get("vessel_mask_path", "")).strip():
                self.vessel_by_patient[patient_id].append(int(index))
            else:
                self.other_by_patient[patient_id].append(int(index))
        if not self.vessel_by_patient:
            raise ValueError(
                "SparseAnnotationSampler requires at least one vessel-labelled training row"
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _draw(
        rng: np.random.Generator,
        pool: Dict[str, List[int]],
    ) -> int:
        patients = sorted(pool)
        patient = patients[int(rng.integers(0, len(patients)))]
        indices = pool[patient]
        return int(indices[int(rng.integers(0, len(indices)))])

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        order = []
        for _ in range(self.samples_per_epoch):
            choose_vessel = (
                not self.other_by_patient
                or rng.random() < self.vessel_fraction
            )
            pool = self.vessel_by_patient if choose_vessel else self.other_by_patient
            order.append(self._draw(rng, pool))
        return iter(order)

    def __len__(self) -> int:
        return self.samples_per_epoch
