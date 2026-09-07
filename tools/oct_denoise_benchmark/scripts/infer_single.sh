#!/usr/bin/env bash
set -euo pipefail
run="$1"; method="$2"; input_image="$3"; output="$4"; device="${5:-cpu}"
python -m tools.oct_denoise_benchmark.inference --method "$method" --input "$input_image" --output "$output" --registry "$run/configs/inference_registry.yaml" --device "$device" --preserve-bit-depth
