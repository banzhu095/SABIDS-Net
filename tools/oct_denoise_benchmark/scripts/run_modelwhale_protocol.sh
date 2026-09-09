#!/usr/bin/env bash
set -euo pipefail

root="/mnt/SABIDS-Net"
run=""
resume=0
from_stage="preflight"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) root="$2"; shift 2 ;;
    --run-dir) run="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    --from-stage) from_stage="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$root"
if [[ -z "$run" ]]; then
  run="$root/runs/denoise_benchmark_pku_protocol_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$run/logs" "$run/audit/stages"
log="$run/logs/modelwhale_protocol.log"
pid_file="$run/logs/modelwhale_protocol.pid"
printf '%s\n' "$$" > "$pid_file"
exec > >(tee -a "$log") 2>&1

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "Cannot locate conda.sh for myconda" >&2; exit 1
fi
conda activate myconda

signature="$({ git rev-parse HEAD; git diff --binary HEAD; sha256sum Manifests/manifest_denoise.csv configs/protocol.yaml configs/dncnn_paired.yaml configs/nafnet_paired.yaml; } | sha256sum | awk '{print $1}')"
stages=(preflight audit tests calibrate dncnn nafnet lock sealed_eval development_outputs atlas workbook package)
from_index=-1
for index in "${!stages[@]}"; do [[ "${stages[$index]}" == "$from_stage" ]] && from_index="$index"; done
if [[ "$from_index" -lt 0 ]]; then echo "Unknown --from-stage: $from_stage" >&2; exit 2; fi

stage_marker() { printf '%s/audit/stages/%s.json' "$run" "$1"; }
stage_valid() {
  local marker; marker="$(stage_marker "$1")"
  [[ -f "$marker" ]] && python - "$marker" "$signature" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if value.get("status") == "success" and value.get("signature") == sys.argv[2] else 1)
PY
}
write_marker() {
  python - "$(stage_marker "$1")" "$1" "$signature" <<'PY'
import json, sys
from datetime import datetime, timezone
json.dump({"stage": sys.argv[2], "status": "success", "signature": sys.argv[3], "completed_at_utc": datetime.now(timezone.utc).isoformat()}, open(sys.argv[1], "w", encoding="utf-8"), indent=2)
PY
}
run_stage() {
  local name="$1"; shift; local index=-1
  for i in "${!stages[@]}"; do [[ "${stages[$i]}" == "$name" ]] && index="$i"; done
  if [[ "$index" -lt "$from_index" ]]; then
    if ! stage_valid "$name"; then echo "Required earlier stage is missing or hash-stale: $name" >&2; exit 1; fi
    echo "[skip before --from-stage] $name"; return
  fi
  if [[ "$resume" -eq 1 ]] && stage_valid "$name"; then echo "[resume skip hash-matched] $name"; return; fi
  echo "[stage start] $name $(date --iso-8601=seconds)"
  "$@"
  write_marker "$name"
  echo "[stage complete] $name $(date --iso-8601=seconds)"
}

preflight() {
  git status --short --branch
  git diff --check
  python - "$root" <<'PY'
import sys, torch
from pathlib import Path
from tools.oct_denoise_benchmark.data import audit_protocol, load_protocol_manifest
root = Path(sys.argv[1])
print({'torch_cuda_available': torch.cuda.is_available(), 'cuda_devices': torch.cuda.device_count(), 'torch_cuda': torch.version.cuda})
if not torch.cuda.is_available(): raise SystemExit('CUDA is required for the formal ModelWhale deep stages')
result = audit_protocol(load_protocol_manifest(root))
if not result['passed']: raise SystemExit(result)
print(result)
PY
  if [[ ! -f "$run/configs/protocol.yaml" ]]; then
    python -m tools.oct_denoise_benchmark.cli init --project-root "$root" --run-dir "$run"
  fi
}
train_dncnn() {
  for seed in 42 123 2026; do
    python -m tools.oct_denoise_benchmark.train_paired --project-root "$root" --method dncnn_paired --output "$run/checkpoints/dncnn_paired/seed_$seed" --device cuda:0 --seed "$seed" --patch-size 256 --max-updates 100000 --val-frequency 1000 --amp --resume
  done
}
train_nafnet() {
  for seed in 42 123 2026; do
    python -m tools.oct_denoise_benchmark.train_paired --project-root "$root" --method nafnet_paired --output "$run/checkpoints/nafnet_paired/seed_$seed" --device cuda:0 --seed "$seed" --patch-size 256 --max-updates 100000 --val-frequency 1000 --width 32 --enc-blocks 1 1 1 28 --middle-blocks 1 --dec-blocks 1 1 1 1 --amp --resume
  done
}
verify_lock() {
  python - "$run" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / 'audit' / 'config_lock.json'
value = json.load(open(p, encoding='utf-8'))
if value.get('status') != 'locked': raise SystemExit(f'lock gate failed: {value}')
print(value)
PY
}
sealed_evaluation() {
  verify_lock
  python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$run" --methods all --splits test external_test --device cuda:0 --tile-size 512 --all-deep-seeds
}
make_atlas() {
  python - "$root" "$run" <<'PY'
import sys
from pathlib import Path
from tools.oct_denoise_benchmark.package_light import materialize_fixed_atlas, select_fixed_atlas
root, run = Path(sys.argv[1]), Path(sys.argv[2])
selection = select_fixed_atlas(root, run)
status = materialize_fixed_atlas(root, run, selection)
if status.get('status') != 'materialized': raise SystemExit(status)
print(status)
PY
}
make_workbook() {
  if ! command -v node >/dev/null || [[ -z "${NODE_PATH:-}" ]]; then
    echo "Set NODE_PATH to a directory containing @oai/artifact-tool before the workbook stage" >&2; exit 1
  fi
  node tools/oct_denoise_benchmark/build_protocol_workbook.mjs "$run"
  node tools/oct_denoise_benchmark/validate_benchmark_workbook.mjs "$run/benchmark_summary.xlsx" "$run/reports/workbook_post_import_validation.json"
}

run_stage preflight preflight
run_stage audit python -m tools.oct_denoise_benchmark.cli audit --project-root "$root" --run-dir "$run"
run_stage tests python -m pytest -q
run_stage calibrate python -m tools.oct_denoise_benchmark.cli calibrate --project-root "$root" --run-dir "$run"
run_stage dncnn train_dncnn
run_stage nafnet train_nafnet
run_stage lock python -m tools.oct_denoise_benchmark.finalize_registry --project-root "$root" --run-dir "$run" --main-seed 42
run_stage sealed_eval sealed_evaluation
run_stage development_outputs python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$run" --methods all --splits train val --device cuda:0 --tile-size 512 --all-deep-seeds
run_stage atlas make_atlas
run_stage workbook make_workbook
run_stage package python -m tools.oct_denoise_benchmark.package_light --project-root "$root" --run-dir "$run"
rm -f "$pid_file"
echo "Completed. Run directory: $run"
