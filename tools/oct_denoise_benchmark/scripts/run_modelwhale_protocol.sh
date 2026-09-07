#!/usr/bin/env bash
set -euo pipefail
root="${1:-/mnt/SABIDS-Net}"
cd "$root"
run="${2:-$root/runs/denoise_benchmark_pku_protocol_$(date +%Y%m%d_%H%M%S)}"
python -m tools.oct_denoise_benchmark.cli init --project-root "$root" --run-dir "$run"
python -m tools.oct_denoise_benchmark.cli audit --project-root "$root" --run-dir "$run"
python -m tools.oct_denoise_benchmark.cli smoke --project-root "$root" --run-dir "$run"
python -m tools.oct_denoise_benchmark.cli calibrate --project-root "$root" --run-dir "$run"
for seed in 42 123 2026; do
  python -m tools.oct_denoise_benchmark.train_paired --project-root "$root" --method dncnn_paired --output "$run/checkpoints/dncnn_paired/seed_$seed" --device cuda:0 --seed "$seed" --patch-size 256 --max-updates 100000 --val-frequency 1000 --amp --resume
  python -m tools.oct_denoise_benchmark.train_paired --project-root "$root" --method nafnet_paired --output "$run/checkpoints/nafnet_paired/seed_$seed" --device cuda:0 --seed "$seed" --patch-size 256 --max-updates 100000 --val-frequency 1000 --width 32 --enc-blocks 1 1 1 28 --middle-blocks 1 --dec-blocks 1 1 1 1 --amp --resume
done
python -m tools.oct_denoise_benchmark.finalize_registry --project-root "$root" --run-dir "$run" --main-seed 42
python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$run" --methods all --splits test external_test --device cuda:0 --tile-size 512
python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$run" --methods all --splits train val --device cuda:0 --tile-size 512
python -m tools.oct_denoise_benchmark.package_light --project-root "$root" --run-dir "$run"
