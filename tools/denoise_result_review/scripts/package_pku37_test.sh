#!/usr/bin/env bash
set -euo pipefail
root="${1:-/mnt/SABIDS-Net}"
output="${2:-$root/runs/denoise_review_packages}"
cd "$root"
python -m tools.denoise_result_review.cli package-test \
  --project-root "$root" --run-dir auto --dataset PKU37 --split test \
  --output-dir "$output" --primary-seeds-only --include-noisy --include-reference \
  --archive-by-position --archive-all --resume
