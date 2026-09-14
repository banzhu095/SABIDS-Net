from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import development_rows, load_protocol_manifest
from .io import read_image, save_image
from .metrics import compute_metrics
from .methods import AdapterContext, denoise


def run(root: Path, output: Path, sabids_checkpoint: Path, sabids_config: Path | None, tcfl_checkpoint: Path) -> None:
    root, output = root.resolve(), output.resolve(); output.mkdir(parents=True, exist_ok=True)
    table = load_protocol_manifest(root); records = []
    configs = {
        "sabids_current": ({"method_id": "sabids_current", **({"config_path": str(sabids_config.resolve())} if sabids_config else {})}, sabids_checkpoint),
        "tcfl_dncnn": ({"method_id": "tcfl_dncnn", "num_layers": 10, "features": 64}, tcfl_checkpoint),
    }
    for split in ("train", "val"):
        row = development_rows(table, split).sort_values(["position_id", "frame_id"]).iloc[0]
        noisy, metadata = read_image(Path(row.image_path)); clean, _ = read_image(Path(row.clean_path))
        for method, (config, checkpoint) in configs.items():
            context = AdapterContext(device="cuda:0" if torch.cuda.is_available() else "cpu", checkpoint=checkpoint)
            first = denoise(noisy, config, context); second = denoise(noisy, config, context)
            if first.shape != noisy.shape or first.dtype != np.float32 or not np.isfinite(first).all() or not np.array_equal(first, second):
                raise RuntimeError(f"{method}/{split} smoke contract failed")
            destination = save_image(output / split / method / f"{row.sample_id}.png", first, metadata, True)
            reopened, _ = read_image(destination); metric = compute_metrics(noisy, clean, reopened)
            records.append({"method_id": method, "split": split, "sample_id": row.sample_id, "shape": list(first.shape), "dtype": str(first.dtype), "minimum": float(first.min()), "maximum": float(first.max()), "deterministic": True, "output": str(destination), "psnr": metric["psnr"], "ssim": metric["ssim"], "padding": context.extras.get("last_padding")})
    (output / "smoke_results.json").write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--project-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--sabids-checkpoint", type=Path, required=True); p.add_argument("--sabids-config", type=Path); p.add_argument("--tcfl-checkpoint", type=Path, required=True); args = p.parse_args(); run(args.project_root, args.output, args.sabids_checkpoint, args.sabids_config, args.tcfl_checkpoint)


if __name__ == "__main__": main()
