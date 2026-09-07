from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .io import read_image, save_image
from .methods import AdapterContext, denoise
from .registry import load_yaml


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
    for source in files:
        image, metadata = read_image(source)
        relative = source.relative_to(input_root) if input_root.is_dir() and args.preserve_relative_path else Path(source.name)
        comparisons = [("input", image)]
        for method in methods:
            if method not in registry.get("methods", {}):
                raise KeyError(f"method {method!r} missing from registry")
            entry = dict(registry["methods"][method])
            config = dict(entry.get("config", entry)); config["method_id"] = method
            checkpoint = args.checkpoint or entry.get("checkpoint")
            if method in {"dncnn_paired", "nafnet_paired"} and not checkpoint:
                raise ValueError(f"{method} has no locked checkpoint; random weights are forbidden")
            checkpoint_path = Path(checkpoint).resolve() if checkpoint else None
            context = AdapterContext(device=args.device, checkpoint=checkpoint_path, tile_size=args.tile_size, tile_overlap=args.tile_overlap, seed=int(entry.get("seed", 42)))
            destination_root = args.output / method if args.method == "all" else args.output
            destination = destination_root / relative
            if destination.suffix.lower() not in {".png", ".tif", ".tiff"}:
                destination = destination.with_suffix(".png")
            if destination.exists() and not args.overwrite:
                rows.append({"input": str(source), "method_id": method, "output": str(destination), "status": "skipped_exists"})
                continue
            started = time.perf_counter()
            output = denoise(image, config, context)
            elapsed = time.perf_counter() - started
            save_image(destination, output, metadata, args.preserve_bit_depth)
            comparisons.append((method, output))
            rows.append({"input": str(source), "method_id": method, "output": str(destination), "status": "success", "seconds": elapsed, "shape": list(output.shape)})
        if args.save_preview and len(comparisons) > 1:
            preview_root = args.output / "previews" if args.method == "all" else args.output / "previews"
            _preview(preview_root / relative.with_suffix(".png"), comparisons)
    log = args.output / "inference_log.json"
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
