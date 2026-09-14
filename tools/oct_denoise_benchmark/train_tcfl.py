from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .data import development_rows, load_protocol_manifest
from .io import read_image, sha256_file
from .metrics import compute_metrics
from .methods.deep_common import activate_cuda_device, cuda_max_memory_allocated, reset_cuda_peak_memory_stats
from .methods.tcfl_adapter import TCFLDiscriminator, TCFLGenerator
from .train_paired import checkpoint_selection_reason, cpu_rng_state


class TCFLUnpairedDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, str]]):
    """Index-deterministic PKU37-train-only A/B/C sampler.

    A and B are noisy frames from independently selected positions. C is a clean
    target independently selected from a third position whenever at least three
    positions are available. No paired noisy/clean row is returned together.
    """

    def __init__(self, rows: pd.DataFrame, patch_size: int, seed: int, length: int, start_index: int = 0):
        if set(rows.dataset) != {"PKU37"} or set(rows.split) != {"train"}:
            raise ValueError("TCFL training accepts PKU37 train only")
        self.groups = {str(key): group.reset_index(drop=True) for key, group in rows.groupby("position_id")}
        self.positions = sorted(self.groups)
        if len(self.positions) < 3:
            raise ValueError("TCFL independent A/B/C sampling requires at least three training positions")
        self.patch_size, self.seed, self.length, self.start_index = int(patch_size), int(seed), int(length), int(start_index)

    def __len__(self) -> int:
        return self.length

    @staticmethod
    def _crop(image: np.ndarray, patch: int, rng: np.random.Generator) -> np.ndarray:
        size = min(patch, *image.shape)
        y = int(rng.integers(image.shape[0] - size + 1)); x = int(rng.integers(image.shape[1] - size + 1))
        value = image[y:y + size, x:x + size]
        if size != patch:
            value = np.pad(value, ((0, patch - size), (0, patch - size)), mode="reflect")
        if rng.random() < 0.5:
            value = value[:, ::-1]
        return np.ascontiguousarray(value)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str, str]:
        absolute = self.start_index + int(item)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, absolute]))
        selected = rng.choice(len(self.positions), size=3, replace=False)
        pos_a, pos_b, pos_c = (self.positions[int(index)] for index in selected)
        row_a = self.groups[pos_a].iloc[int(rng.integers(len(self.groups[pos_a])))]
        row_b = self.groups[pos_b].iloc[int(rng.integers(len(self.groups[pos_b])))]
        row_c = self.groups[pos_c].iloc[int(rng.integers(len(self.groups[pos_c])))]
        noisy_a, _ = read_image(Path(row_a.image_path)); noisy_b, _ = read_image(Path(row_b.image_path)); clean_c, _ = read_image(Path(row_c.clean_path))
        tensors = [torch.from_numpy(self._crop(value, self.patch_size, rng))[None] for value in (noisy_a, noisy_b, clean_c)]
        return tensors[0], tensors[1], tensors[2], str(row_a.sample_id), str(row_b.sample_id), str(row_c.sample_id)


def tcfl_generator_loss(generator: TCFLGenerator, discriminator: TCFLDiscriminator, noisy_a: torch.Tensor,
                        noisy_b: torch.Tensor, clean_c: torch.Tensor, lambda_pixel: float = 6.0) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[torch.Tensor]]:
    """Faithful transcription of the six adversarial and four L1 TCFL terms."""
    noise_a, noise_b = generator(noisy_a), generator(noisy_b)
    clean_a1, clean_b1 = noisy_a - noise_a, noisy_b - noise_b
    clean_a2 = clean_a1 + noise_b - generator(clean_a1 + noise_b)
    clean_b2 = clean_b1 + noise_a - generator(clean_b1 + noise_a)
    clean_c1 = clean_c + noise_a - generator(clean_c + noise_a)
    clean_c2 = clean_c + noise_b - generator(clean_c + noise_b)
    generated = [clean_a1, clean_b1, clean_a2, clean_b2, clean_c1, clean_c2]
    adversarial_terms = []
    for value in generated:
        prediction = discriminator(value)
        adversarial_terms.append(F.mse_loss(prediction, torch.ones_like(prediction)))
    adversarial = sum(adversarial_terms)
    pixel = F.l1_loss(clean_a1, clean_a2) + F.l1_loss(clean_b1, clean_b2) + F.l1_loss(clean_c1, clean_c) + F.l1_loss(clean_c2, clean_c)
    return adversarial + lambda_pixel * pixel, {"adversarial": adversarial, "pixel": pixel}, generated


def tcfl_discriminator_loss(discriminator: TCFLDiscriminator, clean: torch.Tensor, generated: list[torch.Tensor]) -> torch.Tensor:
    prediction = discriminator(clean)
    real = F.mse_loss(prediction, torch.ones_like(prediction))
    fake_terms = []
    for value in generated:
        prediction = discriminator(value.detach())
        fake_terms.append(F.mse_loss(prediction, torch.zeros_like(prediction)))
    fake = sum(fake_terms)
    return real + 0.1 * fake


def _validation(generator: TCFLGenerator, rows: pd.DataFrame, device: torch.device, frames_per_position: int | None = None) -> tuple[float, float]:
    generator.eval(); values: list[dict[str, Any]] = []
    if frames_per_position is not None: rows = rows.groupby("position_id", sort=True).head(frames_per_position)
    with torch.inference_mode():
        for row in rows.itertuples():
            noisy, _ = read_image(Path(row.image_path)); clean, _ = read_image(Path(row.clean_path))
            tensor = torch.from_numpy(noisy)[None, None].to(device)
            output = (tensor - generator(tensor))[0, 0].clamp(0, 1).cpu().numpy()
            metric = compute_metrics(noisy, clean, output)
            values.append({"position_id": row.position_id, "psnr": metric["psnr"], "ssim": metric["ssim"]})
    grouped = pd.DataFrame(values).groupby("position_id")[["psnr", "ssim"]].mean()
    return float(grouped.psnr.mean()), float(grouped.ssim.mean())


def train(args: argparse.Namespace) -> Path:
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(args.deterministic, warn_only=True)
    root, output = args.project_root.resolve(), args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    table = load_protocol_manifest(root, args.manifest)
    train_rows, val_rows = development_rows(table, "train"), development_rows(table, "val")
    device = torch.device(args.device); cuda_index = activate_cuda_device(device) if device.type == "cuda" else None
    generator, discriminator = TCFLGenerator(num_layers=args.num_layers, features=args.features).to(device), TCFLDiscriminator().to(device)
    generator.apply(_official_weights_init); discriminator.apply(_official_weights_init)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    last_path, curve_path = output / "last.pth", output / "training_curves.csv"
    curves = pd.read_csv(curve_path).to_dict("records") if args.resume and curve_path.is_file() else []
    manifest_hash = sha256_file(args.manifest or root / "Manifests" / "manifest_denoise.csv")
    start_epoch, best_psnr, best_ssim, highest_ssim, elapsed = 0, -float("inf"), -float("inf"), -float("inf"), 0.0
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        immutable = ("seed", "epochs", "steps_per_epoch", "patch_size", "batch_size", "learning_rate", "beta1", "beta2", "lambda_pixel", "num_layers", "features")
        changed = {key: (state["config"].get(key), config.get(key)) for key in immutable if state["config"].get(key) != config.get(key)}
        if changed: raise ValueError(f"resume configuration mismatch: {changed}")
        if state.get("manifest_sha256") != manifest_hash: raise ValueError("resume manifest hash mismatch")
        generator.load_state_dict(state["generator"]); discriminator.load_state_dict(state["discriminator"])
        optimizer_g.load_state_dict(state["optimizer_g"]); optimizer_d.load_state_dict(state["optimizer_d"])
        start_epoch, best_psnr, best_ssim, highest_ssim, elapsed = int(state["epoch"]), float(state["best_psnr"]), float(state["best_ssim"]), float(state.get("highest_ssim", state["best_ssim"])), float(state.get("training_elapsed_seconds", 0))
        random.setstate(state["python_random_state"]); np.random.set_state(state["numpy_random_state"])
        torch.set_rng_state(cpu_rng_state(state["torch_rng_state"], "torch_rng_state"))
        if torch.cuda.is_available() and state.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all([cpu_rng_state(value, "cuda_rng_state_all") for value in state["cuda_rng_state_all"]])
    session_started = time.perf_counter()
    if cuda_index is not None: reset_cuda_peak_memory_stats()
    for epoch in range(start_epoch, args.epochs):
        dataset = TCFLUnpairedDataset(train_rows, args.patch_size, args.seed, args.steps_per_epoch * args.batch_size, epoch * args.steps_per_epoch * args.batch_size)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.pin_memory)
        generator.train(); discriminator.train(); totals = {"g": 0.0, "d": 0.0, "pixel": 0.0, "adversarial": 0.0}; audit_rows = []
        for batch_index, (a, b, c, sample_a, sample_b, sample_c) in enumerate(loader):
            a, b, c = a.to(device), b.to(device), c.to(device)
            optimizer_g.zero_grad(set_to_none=True)
            loss_g, parts, generated = tcfl_generator_loss(generator, discriminator, a, b, c, args.lambda_pixel)
            if not torch.isfinite(loss_g):
                _write_failure(output, args.seed, epoch + 1, batch_index, "non-finite generator loss")
                raise FloatingPointError("non-finite TCFL generator loss")
            loss_g.backward(); optimizer_g.step()
            optimizer_d.zero_grad(set_to_none=True)
            loss_d = tcfl_discriminator_loss(discriminator, c, generated)
            if not torch.isfinite(loss_d):
                _write_failure(output, args.seed, epoch + 1, batch_index, "non-finite discriminator loss")
                raise FloatingPointError("non-finite TCFL discriminator loss")
            loss_d.backward(); optimizer_d.step()
            totals["g"] += float(loss_g.detach()); totals["d"] += float(loss_d.detach()); totals["pixel"] += float(parts["pixel"].detach()); totals["adversarial"] += float(parts["adversarial"].detach())
            if args.audit_batches < 0 or batch_index < args.audit_batches:
                audit_rows.extend({"epoch": epoch + 1, "batch": batch_index, "sample_a": x, "sample_b": y, "sample_c": z} for x, y, z in zip(sample_a, sample_b, sample_c))
        psnr, ssim = _validation(generator, val_rows, device, args.validation_frames_per_position)
        reason = checkpoint_selection_reason(best_psnr, best_ssim, psnr, ssim, args.psnr_tolerance)
        if reason: best_psnr, best_ssim = psnr, ssim
        ssim_improved = ssim > highest_ssim
        if ssim_improved: highest_ssim = ssim
        scope = "full_277_checkpoint_eligible" if args.validation_frames_per_position is None else "smoke_subset_not_formal"
        state = {"architecture": "tcfl_dncnn", "generator": generator.state_dict(), "discriminator": discriminator.state_dict(), "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(), "scheduler": None, "scaler": None, "epoch": epoch + 1, "best_psnr": best_psnr, "best_ssim": best_ssim, "highest_ssim": highest_ssim, "checkpoint_val_position_macro_psnr": psnr, "checkpoint_val_position_macro_ssim": ssim, "checkpoint_validation_scope": scope, "config": config, "manifest_sha256": manifest_hash, "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(), "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [], "training_elapsed_seconds": elapsed + time.perf_counter() - session_started}
        torch.save(state, last_path)
        if reason: state["checkpoint_selection_reason"] = reason; torch.save(state, output / "best_psnr.pth")
        if ssim_improved: state["checkpoint_selection_reason"] = "highest_validation_position_macro_ssim"; torch.save(state, output / "best_ssim.pth")
        curve = {"method_id": "tcfl_dncnn", "seed": args.seed, "epoch": epoch + 1, "generator_loss": totals["g"] / len(loader), "discriminator_loss": totals["d"] / len(loader), "pixel_loss": totals["pixel"] / len(loader), "adversarial_loss": totals["adversarial"] / len(loader), "val_position_macro_psnr": psnr, "val_position_macro_ssim": ssim, "gpu_peak_memory_mb": cuda_max_memory_allocated() / 1024**2 if cuda_index is not None else 0.0}; curves.append(curve)
        pd.DataFrame(curves).to_csv(curve_path, index=False)
        if audit_rows: pd.DataFrame(audit_rows).to_csv(output / f"unpaired_sampling_epoch_{epoch + 1:03d}.csv", index=False)
        print(json.dumps(curve), flush=True)
    inventory = []
    for checkpoint in output.glob("*.pth"):
        metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
        inventory.append({"method_id": "tcfl_dncnn", "seed": args.seed, "checkpoint": str(checkpoint), "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size, "epoch": metadata.get("epoch"), "checkpoint_val_position_macro_psnr": metadata.get("checkpoint_val_position_macro_psnr"), "checkpoint_val_position_macro_ssim": metadata.get("checkpoint_val_position_macro_ssim"), "status": "formal_checkpoint"})
    pd.DataFrame(inventory).to_csv(output / "checkpoint_inventory.csv", index=False)
    return output


def _official_weights_init(module: torch.nn.Module) -> None:
    name = module.__class__.__name__
    if "Conv" in name and getattr(module, "weight", None) is not None: torch.nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif "BatchNorm2d" in name: torch.nn.init.normal_(module.weight.data, 1.0, 0.02); torch.nn.init.constant_(module.bias.data, 0.0)


def _write_failure(output: Path, seed: int, epoch: int, batch: int, error: str) -> None:
    value = {"method_id": "tcfl_dncnn", "seed": seed, "epoch": epoch, "batch": batch, "error": error, "torch_rng_state_sha256": __import__("hashlib").sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()}
    (output / "failure_context.json").write_text(json.dumps(value, indent=2), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, default=Path(".")); p.add_argument("--manifest", type=Path); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=100); p.add_argument("--steps-per-epoch", type=int, default=582); p.add_argument("--patch-size", type=int, default=640); p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5); p.add_argument("--beta1", type=float, default=0.5); p.add_argument("--beta2", type=float, default=0.999); p.add_argument("--lambda-pixel", type=float, default=6.0); p.add_argument("--num-layers", type=int, default=10); p.add_argument("--features", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=0); p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--audit-batches", type=int, default=-1, help="batches recorded per epoch; -1 records every sampled triplet"); p.add_argument("--validation-frames-per-position", type=int, help="smoke-only subset; omit for formal full validation"); p.add_argument("--psnr-tolerance", type=float, default=1e-4); p.add_argument("--resume", action="store_true"); p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True); return p


def main(argv: Sequence[str] | None = None) -> None: train(parser().parse_args(argv))


if __name__ == "__main__": main()
