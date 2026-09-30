#!/usr/bin/env bash
set -Eeuo pipefail
ACTION="${1:-}"; shift || true
RUN_ID=""; DEVICE=cuda; OUTPUT=""; RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2;; --device) DEVICE="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;; --resume) RESUME=1; shift;;
    *) echo "Unknown argument: $1" >&2; exit 2;; esac
done
[[ -n "$ACTION" && -n "$RUN_ID" ]] || { echo "Usage: $0 {preflight|prepare|overfit|cuda-check|audit-cuda-check|pilot|evaluate|summary|atlas|package} --run-id ID" >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
REG="$ROOT/cache/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v2/$RUN_ID"
RUNBASE="$ROOT/runs/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v2/$RUN_ID"
REPORT="$ROOT/reports/adaptive_denoising/dual_task_adaptive_v2/$RUN_ID"
trap 'c=$?; echo "FAILED action=$ACTION run_id=$RUN_ID exit=$c; artifacts preserved" >&2; exit $c' ERR
prep(){ python tools/prepare_dual_task_adaptive_v2.py --project-root . --mode "$1" --run-id "$RUN_ID" --device "$DEVICE" ${2:-}; }
seed_evidence(){ mkdir -p "$1"; local src="$REG/training_input_inventory.json" dst="$1/training_asset_inventory_initial.json"; if [[ -e "$dst" ]]; then cmp -s "$src" "$dst" || { echo "BLOCKED: training inventory differs" >&2; exit 2; }; else cp "$src" "$dst"; fi; }
case "$ACTION" in
 preflight) python - <<'PY'
import shutil,torch
print('cuda_available:',torch.cuda.is_available()); print('cuda_devices:',torch.cuda.device_count()); print('free_bytes:',shutil.disk_usage('.').free)
PY
   prep preflight;;
 prepare) prep pilot;;
 overfit) prep overfit; seed_evidence "$RUNBASE/overfit_seed42"; python train.py --config "$REG/config_overfit_seed42.yaml" 2>&1 | tee "$REG/overfit.log"; python tools/audit_dual_task_adaptive_v2.py --engineering-check --run-dir "$RUNBASE/overfit_seed42" --output "$REG/overfit_audit.json";;
 cuda-check) prep cuda-check; seed_evidence "$RUNBASE/cuda_check_seed42"; python train.py --config "$REG/config_cuda-check_seed42.yaml" 2>&1 | tee "$REG/cuda_check.log";;
 audit-cuda-check) python tools/audit_dual_task_adaptive_v2.py --engineering-check --run-dir "$RUNBASE/cuda_check_seed42" --output "$REG/cuda_check_audit.json";;
 pilot)
   [[ -s "$REG/overfit_audit.json" ]] || { echo 'BLOCKED: run overfit audit first' >&2; exit 2; }
   [[ -s "$REG/cuda_check_audit.json" ]] || { echo 'BLOCKED: run CUDA audit first' >&2; exit 2; }
   python - "$REG/cuda_check_audit.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])); assert v['completed_epochs']==2
PY
   extra=""; [[ "$RESUME" -eq 1 ]] && extra="--resume"; prep pilot "$extra"
   seed_evidence "$RUNBASE/seed42"
   cfg="$REG/config_pilot_seed42.yaml"; [[ "$RESUME" -eq 1 ]] && cfg="$REG/config_pilot_resume_seed42.yaml"
   python train.py --config "$cfg" 2>&1 | tee "$REG/pilot.log"
   python tools/audit_dual_task_adaptive_v2.py --run-dir "$RUNBASE/seed42" --output "$REG/pilot_audit.json";;
 evaluate)
   [[ -s "$RUNBASE/seed42/best_vessel_safe.pth" ]] || { echo 'BLOCKED: NO VESSEL-SAFE CHECKPOINT' >&2; exit 3; }
   python tools/evaluate_dual_task_adaptive.py --config "$REG/config_pilot_seed42.yaml" --checkpoint "$RUNBASE/seed42/best_vessel_safe.pth" --output "$REPORT" --device "$DEVICE" --save-atlas;;
 summary) python -m json.tool "$REPORT/summary.json";;
 atlas) [[ -s "$REPORT/atlas_selection_manifest.csv" && -d "$REPORT/atlas" ]] && echo "PASS: $REPORT/atlas";;
 package) [[ "$OUTPUT" = /* ]] || { echo '--output must be absolute' >&2; exit 2; }; python tools/package_dual_task_adaptive_v2.py --project-root . --run-id "$RUN_ID" --output "$OUTPUT";;
 *) echo "Unknown action: $ACTION" >&2; exit 2;; esac
