from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import development_rows, load_protocol_manifest
from .io import read_image, sha256_file
from .metrics import compute_metrics
from .methods.dncnn import DnCNN
from .methods.nafnet import NAFNet


class PositionBalancedPairs:
    def __init__(self, rows: pd.DataFrame, patch_size: int, seed: int, rotations: bool = False):
        if set(rows.dataset) != {"PKU37"} or set(rows.split) != {"train"}:
            raise ValueError("paired training accepts PKU37 train only")
        self.groups = {str(key): group.reset_index(drop=True) for key, group in rows.groupby("position_id")}
        self.positions = sorted(self.groups)
        self.patch_size = patch_size
        self.rng = np.random.default_rng(seed)
        self.rotations = rotations

    def sample(self) -> tuple[torch.Tensor, torch.Tensor, str]:
        position = self.positions[int(self.rng.integers(len(self.positions)))]
        rows = self.groups[position]
        row = rows.iloc[int(self.rng.integers(len(rows)))]
        noisy, _ = read_image(Path(row.image_path)); clean, _ = read_image(Path(row.clean_path))
        if noisy.shape != clean.shape:
            raise ValueError(f"pair shape mismatch: {row.sample_id}")
        patch = min(self.patch_size, *noisy.shape)
        y = int(self.rng.integers(noisy.shape[0] - patch + 1)); x = int(self.rng.integers(noisy.shape[1] - patch + 1))
        noisy, clean = noisy[y:y + patch, x:x + patch], clean[y:y + patch, x:x + patch]
        if self.rng.random() < 0.5:
            noisy, clean = noisy[:, ::-1], clean[:, ::-1]
        if self.rotations:
            k = int(self.rng.integers(4)); noisy, clean = np.rot90(noisy, k), np.rot90(clean, k)
        return torch.from_numpy(np.ascontiguousarray(noisy))[None], torch.from_numpy(np.ascontiguousarray(clean))[None], position


class PositionBalancedPairDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Index-deterministic position-balanced patches, safe with worker prefetch.

    Every sample derives its RNG from ``(seed, absolute_sample_index)``.  A
    resumed loader therefore produces the same patches as an uninterrupted
    loader even when worker scheduling or prefetch order differs.
    """

    def __init__(self, rows: pd.DataFrame, patch_size: int, seed: int, length: int,
                 start_index: int = 0, rotations: bool = False, cache_mode: str = "none"):
        if set(rows.dataset) != {"PKU37"} or set(rows.split) != {"train"}:
            raise ValueError("paired training accepts PKU37 train only")
        self.groups = {str(key): group.reset_index(drop=True) for key, group in rows.groupby("position_id")}
        self.positions = sorted(self.groups)
        self.patch_size = int(patch_size)
        self.seed = int(seed)
        self.length = int(length)
        self.start_index = int(start_index)
        self.rotations = bool(rotations)
        if cache_mode not in {"none", "uint8", "uint16"}:
            raise ValueError(f"unsupported cache mode: {cache_mode}")
        self.cache_mode = cache_mode
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def __len__(self) -> int:
        return self.length

    def _pair(self, row: Any) -> tuple[np.ndarray, np.ndarray]:
        key = str(row.sample_id)
        if key not in self._cache:
            noisy, _ = read_image(Path(row.image_path)); clean, _ = read_image(Path(row.clean_path))
            if noisy.shape != clean.shape:
                raise ValueError(f"pair shape mismatch: {row.sample_id}")
            if self.cache_mode != "none":
                dtype = np.uint8 if self.cache_mode == "uint8" else np.uint16
                scale = float(np.iinfo(dtype).max)
                self._cache[key] = (np.round(noisy * scale).astype(dtype), np.round(clean * scale).astype(dtype))
            else:
                return noisy, clean
        noisy, clean = self._cache[key]
        scale = float(np.iinfo(noisy.dtype).max)
        return noisy.astype(np.float32) / scale, clean.astype(np.float32) / scale

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        absolute_index = self.start_index + int(item)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, absolute_index]))
        position = self.positions[int(rng.integers(len(self.positions)))]
        rows = self.groups[position]
        row = rows.iloc[int(rng.integers(len(rows)))]
        noisy, clean = self._pair(row)
        patch = min(self.patch_size, *noisy.shape)
        y = int(rng.integers(noisy.shape[0] - patch + 1)); x = int(rng.integers(noisy.shape[1] - patch + 1))
        noisy, clean = noisy[y:y + patch, x:x + patch], clean[y:y + patch, x:x + patch]
        if rng.random() < 0.5:
            noisy, clean = noisy[:, ::-1], clean[:, ::-1]
        if self.rotations:
            k = int(rng.integers(4)); noisy, clean = np.rot90(noisy, k), np.rot90(clean, k)
        return torch.from_numpy(np.ascontiguousarray(noisy))[None], torch.from_numpy(np.ascontiguousarray(clean))[None], position


def build_model(method: str, config: dict[str, Any]) -> nn.Module:
    if method == "dncnn_paired":
        return DnCNN(int(config.get("depth", 17)), int(config.get("features", 64)))
    if method == "nafnet_paired":
        return NAFNet(int(config.get("width", 32)), tuple(config.get("enc_blocks", [2, 2, 4, 8])), int(config.get("middle_blocks", 12)), tuple(config.get("dec_blocks", [2, 2, 2, 2])))
    raise ValueError(method)


def checkpoint_selection_reason(selected_psnr: float, selected_ssim: float, candidate_psnr: float, candidate_ssim: float, tolerance: float) -> str | None:
    if candidate_psnr > selected_psnr + tolerance:
        return "higher_validation_position_macro_psnr"
    if abs(candidate_psnr - selected_psnr) <= tolerance and candidate_ssim > selected_ssim:
        return "validation_position_macro_ssim_tiebreak_within_psnr_tolerance"
    return None


def validation_can_select_checkpoint(scope: str) -> bool:
    return scope == "full_277_checkpoint_eligible"


def _validation(model: nn.Module, rows: pd.DataFrame, device: torch.device, limit: int | None = None) -> tuple[float, float]:
    model.eval(); values = []
    selected = rows if limit is None else rows.groupby("position_id", sort=True).head(limit)
    with torch.inference_mode():
        for row in selected.itertuples():
            noisy, _ = read_image(Path(row.image_path)); clean, _ = read_image(Path(row.clean_path))
            tensor = torch.from_numpy(noisy)[None, None].to(device)
            output = model(tensor)[0, 0].clamp(0, 1).cpu().numpy()
            metric = compute_metrics(noisy, clean, output)
            values.append({"position_id": row.position_id, "psnr": metric["psnr"], "ssim": metric["ssim"]})
    table = pd.DataFrame(values).groupby("position_id")[["psnr", "ssim"]].mean()
    return float(table.psnr.mean()), float(table.ssim.mean())


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)
    root, output = args.project_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    table = load_protocol_manifest(root, args.manifest)
    train_rows, val_rows = development_rows(table, "train"), development_rows(table, "val")
    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path): config[key] = str(value)
    if args.method == "nafnet_paired":
        config.update({"enc_blocks": args.enc_blocks, "dec_blocks": args.dec_blocks})
    device = torch.device(args.device)
    model = build_model(args.method, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.9 if args.method == "nafnet_paired" else 0.999), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_updates, eta_min=args.min_learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    start_update, best_psnr, best_ssim, best_update, highest_ssim, stale = 0, -float("inf"), -float("inf"), 0, -float("inf"), 0
    last_path = output / "last.pth"
    curve_path = output / "training_curves.csv"
    curves = pd.read_csv(curve_path).to_dict("records") if args.resume and curve_path.exists() else []
    loader_generator = torch.Generator().manual_seed(args.seed)
    prior_elapsed_seconds = 0.0
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        prior = state.get("config", {})
        immutable = ("method", "seed", "patch_size", "depth", "features", "width", "enc_blocks", "middle_blocks", "dec_blocks", "batch_size", "accumulation_steps", "learning_rate", "min_learning_rate", "max_updates", "scheduler_name", "cache_mode")
        changed = {key: (prior.get(key), config.get(key)) for key in immutable if prior.get(key) != config.get(key)}
        if changed:
            raise ValueError(f"resume configuration mismatch: {changed}")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        if "scheduler" not in state:
            raise ValueError("resume checkpoint lacks scheduler state")
        scheduler.load_state_dict(state["scheduler"])
        if "scaler" in state: scaler.load_state_dict(state["scaler"])
        start_update, best_psnr, best_ssim, stale = state["update"], state["best_psnr"], state["best_ssim"], state.get("stale", 0)
        best_update = int(state.get("best_update", state.get("update", 0)))
        highest_ssim = float(state.get("highest_ssim", max((row["val_position_macro_ssim"] for row in curves), default=-float("inf"))))
        prior_elapsed_seconds = float(state.get("training_elapsed_seconds", 0.0))
        if "python_random_state" in state: random.setstate(state["python_random_state"])
        if "numpy_random_state" in state: np.random.set_state(state["numpy_random_state"])
        if "torch_rng_state" in state: torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state_all"): torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        if "loader_generator_state" in state: loader_generator.set_state(state["loader_generator_state"])
    samples_per_update = args.batch_size * args.accumulation_steps
    dataset = PositionBalancedPairDataset(
        train_rows, args.patch_size, args.seed,
        length=max(args.max_updates - start_update, 0) * samples_per_update,
        start_index=start_update * samples_per_update,
        rotations=args.rotations, cache_mode=args.cache_mode,
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size, "shuffle": False, "num_workers": args.num_workers,
        "pin_memory": args.pin_memory, "generator": loader_generator,
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": args.persistent_workers, "prefetch_factor": args.prefetch_factor})
    loader = DataLoader(dataset, **loader_kwargs)
    batches = iter(loader)
    session_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    optimizer.zero_grad(set_to_none=True)
    for update in range(start_update + 1, args.max_updates + 1):
        model.train(); total_loss = 0.0; sampled = defaultdict(int)
        for _ in range(args.accumulation_steps):
            noisy, clean, positions = next(batches)
            for position in positions: sampled[str(position)] += 1
            noisy = noisy.to(device, non_blocking=args.pin_memory); clean = clean.to(device, non_blocking=args.pin_memory)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                output_image = model(noisy)
                if args.method == "dncnn_paired":
                    loss = F.mse_loss(noisy - output_image, noisy - clean)
                else:
                    mse = F.mse_loss(output_image, clean)
                    # Official NAFNet PSNRLoss minimizes 10*log10(MSE).
                    loss = 10.0 * torch.log10(mse.clamp_min(1e-8))
                loss = loss / args.accumulation_steps
            scaler.scale(loss).backward(); total_loss += float(loss.detach())
        scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        do_full = update % args.full_val_frequency == 0 or update == args.max_updates
        do_fast = update % args.fast_val_frequency == 0
        if do_fast or do_full:
            validation_scope = "full_277_checkpoint_eligible" if do_full else "fixed_fast_subset_not_checkpoint_eligible"
            limit = None if do_full else args.fast_validation_frames_per_position
            psnr, ssim = _validation(model, val_rows, device, limit)
            session_seconds = max(time.perf_counter() - session_started, 1e-9)
            curves.append({"method_id": args.method, "seed": args.seed, "optimizer_update": update, "train_loss": total_loss,
                           "learning_rate": scheduler.get_last_lr()[0], "validation_scope": validation_scope,
                           "checkpoint_eligible": bool(do_full), "val_position_macro_psnr": psnr, "val_position_macro_ssim": ssim,
                           "physical_batch_size": args.batch_size, "effective_batch_size": args.batch_size * args.accumulation_steps,
                           "session_patches_per_second": ((update - start_update) * samples_per_update) / session_seconds,
                           "gpu_peak_memory_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == "cuda" else 0.0})
            selection_reason = checkpoint_selection_reason(best_psnr, best_ssim, psnr, ssim, args.psnr_tolerance) if validation_can_select_checkpoint(validation_scope) else None
            improved = selection_reason is not None
            if improved:
                best_psnr, best_ssim, best_update, stale = psnr, ssim, update, 0
            elif do_full:
                stale += 1
            ssim_improved = do_full and ssim > highest_ssim
            if ssim_improved: highest_ssim = ssim
            state = {
                "architecture": args.method, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "update": update, "checkpoint_val_position_macro_psnr": psnr, "checkpoint_val_position_macro_ssim": ssim,
                "checkpoint_validation_scope": validation_scope,
                "best_psnr": best_psnr, "best_ssim": best_ssim, "best_update": best_update, "highest_ssim": highest_ssim, "stale": stale,
                "config": config, "manifest_sha256": sha256_file(args.manifest or root / "Manifests" / "manifest_denoise.csv"),
                "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "loader_generator_state": loader_generator.get_state(), "next_sample_index": update * samples_per_update,
                "seen_patches": update * samples_per_update,
                "training_elapsed_seconds": prior_elapsed_seconds + time.perf_counter() - session_started,
            }
            if improved:
                state["checkpoint_selection_reason"] = selection_reason
                torch.save(state, output / "best_psnr.pth")
            if ssim_improved:
                state["checkpoint_selection_reason"] = "highest_validation_position_macro_ssim"
                torch.save(state, output / "best_ssim.pth")
            state["checkpoint_selection_reason"] = "latest_training_state"
            torch.save(state, last_path)
            pd.DataFrame(curves).to_csv(curve_path, index=False)
            print(json.dumps(curves[-1]), flush=True)
            if do_full and stale >= args.early_stopping_patience:
                break
    inventory = []
    for checkpoint in output.glob("*.pth"):
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        inventory.append({"method_id": args.method, "seed": args.seed, "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size,
                          "optimizer_update": metadata.get("update"), "selected_best_update": metadata.get("best_update"),
                          "checkpoint_val_position_macro_psnr": metadata.get("checkpoint_val_position_macro_psnr"),
                          "checkpoint_val_position_macro_ssim": metadata.get("checkpoint_val_position_macro_ssim"),
                          "selected_val_position_macro_psnr": metadata.get("best_psnr"), "selected_val_position_macro_ssim": metadata.get("best_ssim"),
                          "seen_patches": metadata.get("seen_patches"), "training_elapsed_seconds": metadata.get("training_elapsed_seconds"),
                          "selection_reason": metadata.get("checkpoint_selection_reason", "legacy_checkpoint_without_reason")})
    pd.DataFrame(inventory).to_csv(output / "checkpoint_inventory.csv", index=False)
    return output


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(".")); p.add_argument("--manifest", type=Path)
    p.add_argument("--method", choices=["dncnn_paired", "nafnet_paired"], required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch-size", type=int, default=256); p.add_argument("--max-updates", type=int, default=100000)
    p.add_argument("--fast-val-frequency", type=int, default=1000); p.add_argument("--full-val-frequency", type=int, default=5000)
    p.add_argument("--fast-validation-frames-per-position", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=4); p.add_argument("--accumulation-steps", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4); p.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--cache-mode", choices=["none", "uint8", "uint16"], default="none")
    p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--scheduler-name", choices=["cosine"], default="cosine"); p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--early-stopping-patience", type=int, default=20); p.add_argument("--psnr-tolerance", type=float, default=1e-4)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--rotations", action="store_true"); p.add_argument("--resume", action="store_true"); p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--depth", type=int, default=17); p.add_argument("--features", type=int, default=64); p.add_argument("--width", type=int, default=32)
    p.add_argument("--enc-blocks", type=int, nargs="+", default=[2, 2, 4, 8]); p.add_argument("--middle-blocks", type=int, default=12); p.add_argument("--dec-blocks", type=int, nargs="+", default=[2, 2, 2, 2])
    return p


def main(argv: Sequence[str] | None = None) -> None:
    train(parser().parse_args(argv))


if __name__ == "__main__": main()
