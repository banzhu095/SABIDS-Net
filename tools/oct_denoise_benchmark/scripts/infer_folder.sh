#!/usr/bin/env bash
set -euo pipefail
run="$1"; method="$2"; input_folder="$3"; output="$4"; device="${5:-cpu}"
python -m tools.oct_denoise_benchmark.inference --method "$method" --input "$input_folder" --output "$output" --registry "$run/configs/inference_registry.yaml" --device "$device" --recursive --preserve-relative-path --preserve-bit-depth
