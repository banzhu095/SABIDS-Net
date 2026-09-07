from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn

from .methods.dncnn import DnCNN
from .methods.nafnet import NAFNet


def profile(model: nn.Module, height: int = 640, width: int = 640, profile_size: int = 64) -> tuple[int, int]:
    operations = 0
    hooks = []
    def hook(module: nn.Conv2d, inputs, output):
        nonlocal operations
        batch, out_channels, out_height, out_width = output.shape
        kernel = module.kernel_size[0] * module.kernel_size[1]
        operations += int(batch * out_channels * out_height * out_width * (module.in_channels // module.groups) * kernel * 2)
    for module in model.modules():
        if isinstance(module, nn.Conv2d): hooks.append(module.register_forward_hook(hook))
    model.eval()
    with torch.inference_mode(): model(torch.zeros(1, 1, profile_size, profile_size))
    for item in hooks: item.remove()
    # Fully convolutional operations scale linearly with pixel count. Both the
    # profile canvas and formal 640x640 canvas are divisible by NAFNet's padder.
    operations = round(operations * (height * width) / (profile_size * profile_size))
    return sum(parameter.numel() for parameter in model.parameters()), operations


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--height", type=int, default=640); parser.add_argument("--width", type=int, default=640); args = parser.parse_args(argv)
    rows = []
    models = [
        ("dncnn_paired", "depth17-features64", DnCNN(17, 64)),
        ("nafnet_paired", "width32-enc1,1,1,28-middle1-dec1,1,1,1", NAFNet(32, (1, 1, 1, 28), 1, (1, 1, 1, 1))),
    ]
    for method, configuration, model in models:
        parameters, flops = profile(model, args.height, args.width)
        rows.append({"method_id": method, "configuration": configuration, "input_height": args.height, "input_width": args.width, "parameters": parameters, "flops": flops, "gflops": flops / 1e9, "status": "architecture_profile_untrained"})
    args.output.parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__": main()
