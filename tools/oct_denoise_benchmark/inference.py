from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch

from .io import read_image, save_image, sha256_file
from .methods import AdapterContext, adapter_source_sha256, denoise
from .registry import load_yaml, stable_sha256
from .table_store import atomic_write_csv, merge_records


DEFAULT_EXTENSIONS = ".png,.tif,.tiff,.jpg,.jpeg,.bmp"


def _inputs(path: Path, recursive: bool, extensions: str) -> list[Path]:
    if path.is_file():
        return [path]
    allowed = {value.strip().lower() if value.strip().startswith(".") else "." + value.strip().lower() for value in extensions.split(",")}
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(item for item in iterator if item.is_file() and item.suffix.lower() in allowed)


def _preview(path: Path, images: list[tuple[str, np.ndarray]]) -> None:
    panels = []
    for name, image in images:
        panel = np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)
        panel = cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        panels.append(panel)
    target_height = min(panel.shape[0] for panel in panels)
    panels = [cv2.resize(panel, (round(panel.shape[1] * target_height / panel.shape[0]), target_height)) for panel in panels]
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".png", np.concatenate(panels, axis=1))
    if not ok:
        raise RuntimeError("preview encoding failed")
    data.tofile(str(path))


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    registry = load_yaml(args.registry)
    methods = list(registry.get("methods", {})) if args.method == "all" else [args.method]
    if not methods:
        raise ValueError("registry contains no methods")
    input_root = args.input.resolve()
    files = _inputs(input_root, args.recursive, args.extensions)
    if not files:
        raise FileNotFoundError(f"no supported images under {input_root}")
    rows = []
    failures = []
    context_cache: dict[str, AdapterContext] = {}
    manifest_path = args.output / "inference_manifest.csv" if input_root.is_dir() else args.output.parent / "inference_manifest.csv"
    failure_path = manifest_path.with_name("failures.csv")
    existing = pd.read_csv(manifest_path) if manifest_path.is_file() else pd.DataFrame()
    for source in files:
        source_hash = sha256_file(source)
        image, metadata = read_image(source)
        relative = source.relative_to(input_root) if input_root.is_dir() and args.preserve_relative_path else Path(source.name)
        comparisons = [("input", image)]
        for method in methods:
            if method not in registry.get("methods", {}):
                raise KeyError(f"method {method!r} missing from registry")
            entry = dict(registry["methods"][method])
            config = dict(entry.get("config", entry)); config["method_id"] = method
            config_hash = stable_sha256(config)
            source_code_hash = adapter_source_sha256(method)
            checkpoint = args.checkpoint or entry.get("checkpoint")
            if method in {"dncnn_paired", "nafnet_paired"} and not checkpoint:
                raise ValueError(f"{method} has no locked checkpoint; random weights are forbidden")
            checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
            checkpoint_hash = sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else ""
            context = context_cache.setdefault(method, AdapterContext(device=args.device, checkpoint=checkpoint_path, tile_size=args.tile_size, tile_overlap=args.tile_overlap, seed=int(entry.get("seed", 42))))
            if input_root.is_file():
                destination = args.output
            else:
                destination_root = args.output / method if args.method == "all" else args.output
                destination = destination_root / relative
            if destination.suffix.lower() not in {".png", ".tif", ".tiff"}:
                destination = destination.with_suffix(".png")
            if destination.exists() and not args.overwrite and not existing.empty:
                matched = existing[(existing.input.astype(str) == str(source)) & (existing.method_id == method)
                                   & (existing.source_sha256 == source_hash) & (existing.source_code_sha256 == source_code_hash) & (existing.config_sha256 == config_hash)
                                   & (existing.checkpoint_sha256.fillna("") == checkpoint_hash)]
                if not matched.empty and sha256_file(destination) == str(matched.iloc[-1].output_sha256):
                    rows.append(matched.iloc[-1].to_dict()); continue
                raise FileExistsError(f"existing output provenance does not match manifest; use --overwrite: {destination}")
            try:
                if args.device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.synchronize(torch.device(args.device))
                started = time.perf_counter()
                output = denoise(image, config, context)
                if args.device.startswith("cuda") and torch.cuda.is_available(): torch.cuda.synchronize(torch.device(args.device))
                elapsed = time.perf_counter() - started
                if output.shape != image.shape or not np.isfinite(output).all():
                    raise ValueError(f"invalid output shape/finite: {output.shape}, {bool(np.isfinite(output).all())}")
                save_image(destination, np.clip(output, 0, 1), metadata, args.preserve_bit_depth)
                comparisons.append((method, output))
                rows.append({"input": str(source), "method_id": method, "output": str(destination), "status": "success", "seconds": elapsed,
                             "height": int(output.shape[0]), "width": int(output.shape[1]), "bit_depth": metadata.get("bit_depth"),
                             "source_sha256": source_hash, "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash,
                             "source_code_sha256": source_code_hash, "output_sha256": sha256_file(destination)})
            except Exception as exc:
                failures.append({"input": str(source), "method_id": method, "output": str(destination), "source_sha256": source_hash,
                                 "config_sha256": config_hash, "checkpoint_sha256": checkpoint_hash,
                                 "source_code_sha256": source_code_hash,
                                 "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        if args.save_preview and len(comparisons) > 1:
            preview_root = args.output / "previews" if args.method == "all" else args.output / "previews"
            _preview(preview_root / relative.with_suffix(".png"), comparisons)
    if rows:
        if args.overwrite and manifest_path.is_file():
            prior = pd.read_csv(manifest_path)
            replaced = {(str(row["input"]), str(row["method_id"])) for row in rows}
            keep = ~prior.apply(lambda row: (str(row["input"]), str(row["method_id"])) in replaced, axis=1)
            atomic_write_csv(prior[keep], manifest_path, manifest_path.parent)
        merge_records(manifest_path, pd.DataFrame(rows), manifest_path.parent,
                      ["input", "method_id"], ["source_sha256", "source_code_sha256", "config_sha256", "checkpoint_sha256", "output_sha256"], replace_consistent=True)
    if failures:
        merge_records(failure_path, pd.DataFrame(failures), failure_path.parent,
                      ["input", "method_id", "source_sha256", "config_sha256", "checkpoint_sha256"], [], replace_consistent=True)
    log = args.output / "inference_log.json" if input_root.is_dir() else args.output.parent / "inference_log.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Protocol-locked OCT denoising inference")
    result.add_argument("--method", required=True)
    result.add_argument("--input", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--registry", type=Path, required=True)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--device", default="cpu")
    result.add_argument("--recursive", action="store_true")
    result.add_argument("--extensions", default=DEFAULT_EXTENSIONS)
    result.add_argument("--preserve-relative-path", action="store_true")
    result.add_argument("--preserve-bit-depth", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--save-preview", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    result.add_argument("--tile-size", type=int)
    result.add_argument("--tile-overlap", type=int, default=32)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    rows = run(args)
    print(json.dumps({"processed": len(rows), "success": sum(row["status"] == "success" for row in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
