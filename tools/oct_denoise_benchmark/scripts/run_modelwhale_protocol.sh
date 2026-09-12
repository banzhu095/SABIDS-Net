#!/usr/bin/env bash
set -euo pipefail

root="/mnt/SABIDS-Net"
run=""
track="full"
resume=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) root="$2"; shift 2 ;;
    --run-dir) run="$2"; shift 2 ;;
    --track) track="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$track" in preflight|deep|classical|merge|evaluate|package|full) ;; *) echo "Unknown --track: $track" >&2; exit 2 ;; esac

cd "$root"
[[ -n "$run" ]] || run="$root/runs/denoise_benchmark_pku_protocol_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run/logs" "$run/audit/tracks" "$run/tracks/deep" "$run/tracks/classical"
printf '%s\n' "$$" > "$run/logs/modelwhale_${track}.pid"
exec > >(tee -a "$run/logs/modelwhale_${track}.log") 2>&1

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  echo "Cannot locate conda initialization for myconda" >&2; exit 1
fi
conda activate myconda

commit="$(git rev-parse HEAD)"
tracked_dirty="$(git status --porcelain --untracked-files=no)"
if [[ -n "$tracked_dirty" ]]; then
  echo "Formal runs require committed tracked files. Current tracked changes:" >&2
  printf '%s\n' "$tracked_dirty" >&2
  echo "Commit/stash those changes, or restore the checkout to the intended benchmark commit." >&2
  exit 1
fi
# Runtime datasets, manifests, logs and server notes may legitimately be
# untracked. Still reject untracked source/config/test files because Python can
# import them even though they are absent from the recorded commit.
untracked_source="$(git ls-files --others --exclude-standard -- configs docs sabids tests tools '*.py' '*.sh' 'requirements*.txt')"
if [[ -n "$untracked_source" ]]; then
  echo "Formal runs refuse untracked source/config/test files:" >&2
  printf '%s\n' "$untracked_source" >&2
  exit 1
fi
signature="$({ printf '%s\n' "$commit"; sha256sum Manifests/manifest_denoise.csv configs/protocol.yaml configs/dncnn_paired.yaml configs/nafnet_paired.yaml requirements-denoise-benchmark.txt; } | sha256sum | awk '{print $1}')"
gpu_monitor_pid=""
stop_gpu_monitor() { if [[ -n "$gpu_monitor_pid" ]] && kill -0 "$gpu_monitor_pid" 2>/dev/null; then kill "$gpu_monitor_pid"; wait "$gpu_monitor_pid" 2>/dev/null || true; fi; }
trap stop_gpu_monitor EXIT

marker_valid() {
  local marker="$run/audit/tracks/$1.json"
  [[ -f "$marker" ]] && python - "$marker" "$signature" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("status") == "success" and value.get("signature") == sys.argv[2] else 1)
PY
}
mark() {
  python - "$run/audit/tracks/$1.json" "$1" "$signature" "$commit" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
path = sys.argv[1]
value = {"track": sys.argv[2], "status": "success", "signature": sys.argv[3], "git_commit": sys.argv[4], "completed_at_utc": datetime.now(timezone.utc).isoformat()}
fd, temp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".track-", suffix=".tmp")
with os.fdopen(fd, "w", encoding="utf-8") as stream: json.dump(value, stream, indent=2)
os.replace(temp, path)
PY
}
run_once() {
  local name="$1"; shift
  if [[ "$resume" -eq 1 ]] && marker_valid "$name"; then echo "[resume skip hash-matched] $name"; return; fi
  echo "[track start] $name $(date --iso-8601=seconds)"
  "$@"
  mark "$name"
  echo "[track complete] $name $(date --iso-8601=seconds)"
}

preflight() {
  git status --short --branch
  git diff --check
  python -m pytest -q
  python - "$root" <<'PY'
import sys, torch
from pathlib import Path
from tools.oct_denoise_benchmark.data import audit_protocol, load_protocol_manifest
root = Path(sys.argv[1])
audit = audit_protocol(load_protocol_manifest(root))
if not audit["passed"]: raise SystemExit(audit)
print({"cuda": torch.cuda.is_available(), "devices": torch.cuda.device_count(), "torch_cuda": torch.version.cuda, "audit": audit})
PY
  [[ -f "$run/configs/protocol.yaml" ]] || python -m tools.oct_denoise_benchmark.cli init --project-root "$root" --run-dir "$run"
}

train_one() {
  local method="$1" seed="$2" output="$3"; shift 3
  python -m tools.oct_denoise_benchmark.train_paired --project-root "$root" --method "$method" --output "$output" --device cuda:0 --seed "$seed" --patch-size 256 --batch-size 4 --accumulation-steps 1 --num-workers 4 --persistent-workers --pin-memory --prefetch-factor 2 --max-updates 100000 --fast-val-frequency 1000 --full-val-frequency 5000 --fast-validation-frames-per-position 3 --learning-rate 0.001 --min-learning-rate 0.000001 --amp --resume "$@"
}

deep() {
  python - <<'PY'
import torch
if not torch.cuda.is_available(): raise SystemExit("CUDA is required for the deep track")
print(torch.cuda.get_device_name(0))
PY
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
    nvidia-smi dmon -s pucm -d 5 -o DT > "$run/tracks/deep/gpu_dmon.log" & gpu_monitor_pid=$!
  fi
  for method in dncnn_paired nafnet_paired; do
    smoke="$run/tracks/deep/smoke/$method"
    extra=(); [[ "$method" == nafnet_paired ]] && extra=(--width 32 --enc-blocks 2 2 4 8 --middle-blocks 12 --dec-blocks 2 2 2 2)
    python -m tools.oct_denoise_benchmark.train_paired --project-root "$root" --method "$method" --output "$smoke" --device cuda:0 --seed 42 --patch-size 256 --batch-size 4 --max-updates 1000 --fast-val-frequency 1000 --full-val-frequency 1000 --num-workers 4 --amp --resume "${extra[@]}"
    python -m tools.oct_denoise_benchmark.train_paired --project-root "$root" --method "$method" --output "$smoke" --device cuda:0 --seed 42 --patch-size 256 --batch-size 4 --max-updates 1000 --fast-val-frequency 1000 --full-val-frequency 1000 --num-workers 4 --amp --resume "${extra[@]}"
  done
  for seed in 42 123 2026; do train_one dncnn_paired "$seed" "$run/checkpoints/dncnn_paired/seed_$seed"; done
  for seed in 42 123 2026; do train_one nafnet_paired "$seed" "$run/checkpoints/nafnet_paired/seed_$seed" --width 32 --enc-blocks 2 2 4 8 --middle-blocks 12 --dec-blocks 2 2 2 2; done
  stop_gpu_monitor; gpu_monitor_pid=""
}

classical() {
  local target="$run/tracks/classical"
  [[ -f "$target/configs/protocol.yaml" ]] || python -m tools.oct_denoise_benchmark.cli init --project-root "$root" --run-dir "$target"
  python -m tools.oct_denoise_benchmark.cli audit --project-root "$root" --run-dir "$target"
  python -m tools.oct_denoise_benchmark.cli calibrate --project-root "$root" --run-dir "$target" --methods bm3d_standard tv_chambolle nlm ksvd_self
}

merge() {
  [[ -f "$run/configs/protocol.yaml" ]] || python -m tools.oct_denoise_benchmark.cli init --project-root "$root" --run-dir "$run"
  [[ -f "$run/audit/data_split_audit.json" ]] || python -m tools.oct_denoise_benchmark.cli audit --project-root "$root" --run-dir "$run"
  python -m tools.oct_denoise_benchmark.merge_tracks --project-root "$root" --run-dir "$run" --classical-dir "$run/tracks/classical"
  python -m tools.oct_denoise_benchmark.finalize_registry --project-root "$root" --run-dir "$run" --main-seed 42
  python -m tools.oct_denoise_benchmark.model_complexity --output "$run/metrics/model_complexity.csv" --height 640 --width 640
}

evaluate() {
  if [[ ! -f "$run/audit/config_lock.json" ]]; then
    echo "evaluate requires a successful merge track; config_lock.json is missing" >&2
    exit 1
  fi
  python - "$run/audit/config_lock.json" <<'PY'
import json, sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
if lock.get("status") != "locked": raise SystemExit("sealed evaluation requires config_lock.status=locked")
PY
  python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$run" --methods all --splits test external_test --device cuda:0 --all-deep-seeds
  python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$run" --methods all --splits train val --device cuda:0 --all-deep-seeds
  python -m tools.oct_denoise_benchmark.downstream_manifest --project-root "$root" --run-dir "$run"
}

package() {
  if [[ ! -f "$run/audit/data_split_audit.json" || ! -f "$run/audit/config_lock.json" ]]; then
    echo "package requires successful merge and evaluate prerequisites" >&2
    exit 1
  fi
  python -m tools.oct_denoise_benchmark.package_light --project-root "$root" --run-dir "$run"
  if command -v node >/dev/null 2>&1 && [[ -d tools/oct_denoise_benchmark/node_modules ]]; then
    node tools/oct_denoise_benchmark/build_protocol_workbook.mjs "$run" "$run/benchmark_summary.xlsx" || echo "workbook skipped_optional: artifact tool unavailable or export failed"
  else
    echo "workbook skipped_optional: Node/artifact-tool unavailable"
  fi
}

case "$track" in
  preflight) run_once preflight preflight ;;
  deep) run_once deep deep ;;
  classical) run_once classical classical ;;
  merge) run_once merge merge ;;
  evaluate) run_once evaluate evaluate ;;
  package) run_once package package ;;
  full)
    run_once preflight preflight
    run_once deep deep
    run_once classical classical
    run_once merge merge
    run_once evaluate evaluate
    run_once package package
    ;;
esac
