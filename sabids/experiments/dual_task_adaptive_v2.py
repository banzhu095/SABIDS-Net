from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from scipy.ndimage import label as connected_components
from torch.nn import functional as F

from .dual_task_adaptive import sha256_file

SCHEMA_VERSION = "dual-task-adaptive-v2"
EXPECTED_V1_SHA256 = "cb15653d26aa117f890e30dc4c73a7cb6b6fd96440c4f882b5c39c8a69ae6ab7"
REFERENCES = {
    "layer_dice": .95337, "layer_surface_dice": .77921,
    "vessel_dice": .74276, "vessel_roi_dice": .74551,
    "vessel_recall": .77886, "small_recall": .56379,
    "low_contrast_recall": .63901, "vessel_boundary_band_dice": .67584,
}


def vessel_protection_losses(strength: torch.Tensor, vessel: torch.Tensor,
                             layer: torch.Tensor, valid: torch.Tensor,
                             weak: torch.Tensor | None = None,
                             margin: float = .10) -> Dict[str, torch.Tensor]:
    strength, vessel, layer, valid = [x.float() for x in (strength, vessel, layer, valid)]
    vessel = vessel * valid; stroma = layer * (1.0 - vessel) * valid
    kernel = torch.ones((1, 1, 3, 3), device=vessel.device)
    dilated = (F.conv2d(vessel, kernel, padding=1) > 0).float()
    eroded = (F.conv2d(vessel, kernel, padding=1) >= 9).float()
    boundary = (dilated - eroded).clamp(0, 1) * valid
    weak = vessel if weak is None else weak.float() * vessel * valid
    zeros = strength.sum() * 0.0
    rank_terms = []
    for i in range(strength.shape[0]):
        if vessel[i].sum() > 0 and stroma[i].sum() > 0:
            vm = (strength[i] * vessel[i]).sum() / vessel[i].sum()
            sm = (strength[i] * stroma[i]).sum() / stroma[i].sum()
            rank_terms.append(F.relu(vm - sm + margin))
    rank = torch.stack(rank_terms).mean() if rank_terms else zeros
    boundary_loss = (strength * boundary).sum() / boundary.sum() if boundary.sum() > 0 else zeros
    weak_loss = (strength * weak).sum() / weak.sum() if weak.sum() > 0 else zeros
    return {"vessel_protect_rank": rank, "vessel_protect_boundary": boundary_loss,
            "vessel_protect_weak": weak_loss,
            "vessel_protect": rank + .5 * boundary_loss + .5 * weak_loss}


def frozen_weak_mask(vessel: torch.Tensor, layer: torch.Tensor, valid: torch.Tensor,
                     noisy: torch.Tensor, small_max: float,
                     low_contrast_max: float) -> torch.Tensor:
    """Build small-or-low masks from frozen train-derived thresholds."""
    output = torch.zeros_like(vessel)
    for index in range(vessel.shape[0]):
        v = (vessel[index, 0].detach().cpu().numpy() > .5) & (valid[index, 0].detach().cpu().numpy() > .5)
        l = (layer[index, 0].detach().cpu().numpy() > .5) & (valid[index, 0].detach().cpu().numpy() > .5)
        image = noisy[index, 0].detach().cpu().numpy()
        labels, count = connected_components(v, structure=np.ones((3, 3), dtype=np.uint8))
        stroma = l & ~v; selected = np.zeros_like(v)
        for component_id in range(1, count + 1):
            component = labels == component_id; area = int(component.sum())
            contrast = abs(float(image[component].mean()) - float(image[stroma].mean())) if component.any() and stroma.any() else np.inf
            if area <= small_max or contrast <= low_contrast_max: selected |= component
        output[index, 0] = torch.as_tensor(selected, device=output.device, dtype=output.dtype)
    return output


def checkpoint_eligibility(metrics: Dict[str, float], references: Dict[str, float] | None = None) -> Dict[str, Any]:
    ref = dict(REFERENCES if references is None else references)
    tolerances = {"layer_dice": 1e-4, "layer_surface_dice": 1e-4,
                  "vessel_dice": .002, "vessel_roi_dice": .002,
                  "vessel_recall": .005, "small_recall": .005,
                  "low_contrast_recall": .005, "vessel_boundary_band_dice": .005}
    failures = []
    for key, tolerance in tolerances.items():
        value = metrics.get(key)
        if value is None or not np.isfinite(float(value)) or float(value) < ref[key] - tolerance:
            failures.append(key)
    q = (0.35 * metrics.get("vessel_roi_dice", float("nan"))
         + 0.25 * metrics.get("vessel_dice", float("nan"))
         + 0.15 * metrics.get("vessel_recall", float("nan"))
         + 0.10 * metrics.get("vessel_boundary_band_dice", float("nan"))
         + 0.075 * metrics.get("small_recall", float("nan"))
         + 0.075 * metrics.get("low_contrast_recall", float("nan")))
    return {"eligible": not failures and np.isfinite(q), "failed_checks": failures,
            "q": float(q), "references": ref, "tolerances": tolerances}


def select_vessel_safe_epoch(rows: list[Dict[str, Any]]) -> int | None:
    eligible = [row for row in rows if bool(row.get("eligible")) and np.isfinite(float(row.get("q", np.nan)))]
    if not eligible: return None
    best = max(float(row["q"]) for row in eligible)
    return min(int(row["epoch"]) for row in eligible if float(row["q"]) == best)


def audit_v2_inputs(config: Dict[str, Any], root: str | Path) -> Dict[str, Any]:
    cfg = config.get("dual_task_adaptive_v2", {}); issues = []
    path = Path(str(cfg.get("v1_checkpoint", "")))
    path = path if path.is_absolute() else Path(root).resolve() / path
    if not path.is_file(): issues.append(f"Missing v1 checkpoint: {path}")
    elif sha256_file(path) != EXPECTED_V1_SHA256: issues.append("v1 best_joint SHA256 mismatch")
    else:
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if int(raw.get("epoch", -1)) + 1 != 17: issues.append("v1 checkpoint epoch is not 17")
    binding_value = cfg.get("evidence", {}).get("v1_binding")
    binding_path = Path(str(binding_value or ""))
    binding_path = binding_path if binding_path.is_absolute() else Path(root).resolve() / binding_path
    if not binding_path.is_file(): issues.append(f"Missing v1 checkpoint binding: {binding_path}")
    else:
        binding = json.loads(binding_path.read_text(encoding="utf-8-sig"))
        if binding.get("status") != "passed" or binding.get("checkpoint_sha256") != EXPECTED_V1_SHA256:
            issues.append("v1 checkpoint binding is invalid")
        if binding.get("selection_rule") != "best_validation_joint_soft_dice":
            issues.append("v1 checkpoint selection rule is not best_validation_joint_soft_dice")
        if int(binding.get("best_epoch", binding.get("checkpoint_epoch", -1))) != 17:
            issues.append("v1 checkpoint binding epoch is not 17")
    if int(config.get("seed", -1)) != 42: issues.append("V2 is preregistered for seed42 only")
    if float(cfg.get("vessel_strength_cap", -1)) != .5: issues.append("Vessel cap must be 0.5")
    if int(cfg.get("v1_best_epoch", -1)) != 17: issues.append("v1 best epoch must be 17")
    return {"schema_version": SCHEMA_VERSION, "status": "passed" if not issues else "blocked",
            "blocked_message": None if not issues else "BLOCKED: DUAL-TASK ADAPTIVE V2 INPUT EVIDENCE",
            "issues": issues, "test_assets_opened": 0}
