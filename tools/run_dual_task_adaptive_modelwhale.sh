#!/usr/bin/env bash
set -Eeuo pipefail

ACTION="${1:-}"
shift || true
RUN_ID=""
DEVICE="cuda"
OUTPUT=""
RESUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --resume) RESUME=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$ACTION" || -z "$RUN_ID" ]]; then
  echo "Usage: bash tools/run_dual_task_adaptive_modelwhale.sh {audit|overfit|train-seed42|evaluate-best|export-atlas|summarize|package} --run-id ID [--device cuda] [--resume] [--output ABS_ZIP]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REGISTRY="$ROOT/cache/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v1/$RUN_ID"
RUN="$ROOT/runs/adaptive_denoising/pku37_binary_v3/dual_task_adaptive_v1/$RUN_ID/seed42"
REPORT="$ROOT/reports/adaptive_denoising/dual_task_adaptive_v1/$RUN_ID"

fail() {
  local code=$?
  echo "FAILED action=$ACTION run_id=$RUN_ID exit=$code" >&2
  echo "Artifacts were preserved; no run or report was overwritten." >&2
  exit "$code"
}
trap fail ERR

preflight() {
  python - <<'PY'
import shutil, torch
print("python/torch CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("free bytes:", shutil.disk_usage(".").free)
PY
  if [[ "$DEVICE" == cuda* ]]; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  fi
  python tools/prepare_dual_task_adaptive.py \
    --project-root . --mode preflight --run-id "$RUN_ID" --device "$DEVICE"
}

case "$ACTION" in
  audit|preflight)
    preflight
    ;;
  overfit)
    preflight
    python tools/prepare_dual_task_adaptive.py \
      --project-root . --mode overfit --run-id "$RUN_ID" --device "$DEVICE"
    python train.py --config "$REGISTRY/config_overfit_seed42.yaml" 2>&1 | tee "$REGISTRY/overfit.log"
    ;;
  train-seed42|formal)
    preflight
    RESUME_ARGS=()
    CONFIG="$REGISTRY/config_formal_seed42.yaml"
    if [[ "$RESUME" -eq 1 ]]; then
      RESUME_ARGS+=(--resume)
      CONFIG="$REGISTRY/config_formal_resume_seed42.yaml"
    fi
    TRAIN_LOG="$REGISTRY/train_seed42.log"
    [[ "$RESUME" -eq 1 ]] && TRAIN_LOG="$REGISTRY/train_seed42_resume.log"
    python tools/prepare_dual_task_adaptive.py \
      --project-root . --mode formal --run-id "$RUN_ID" --device "$DEVICE" "${RESUME_ARGS[@]}"
    python train.py --config "$CONFIG" 2>&1 | tee "$TRAIN_LOG"
    ;;
  evaluate-best|evaluate)
    preflight
    [[ -f "$RUN/dual_task_adaptive_training_metadata.json" ]] || { echo "BLOCKED: missing completed training audit" >&2; exit 2; }
    [[ -f "$RUN/best_joint.pth" ]] || { echo "BLOCKED: missing best_joint.pth" >&2; exit 2; }
    python - "$RUN/dual_task_adaptive_training_metadata.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "passed": raise SystemExit("BLOCKED: adaptive training audit failed")
PY
    python tools/evaluate_dual_task_adaptive.py \
      --config "$REGISTRY/config_formal_seed42.yaml" \
      --checkpoint "$RUN/best_joint.pth" \
      --output "$REPORT" --device "$DEVICE" --save-atlas 2>&1 | tee "$REGISTRY/evaluate_best.log"
    ;;
  export-atlas)
    preflight
    [[ -f "$REPORT/atlas_selection_manifest.csv" ]] || { echo "BLOCKED: run evaluate-best first" >&2; exit 2; }
    [[ -d "$REPORT/atlas" ]] || { echo "BLOCKED: atlas directory is missing" >&2; exit 2; }
    echo "PASS: fixed position atlas: $REPORT/atlas"
    ;;
  summarize)
    preflight
    for name in metrics_by_image.csv metrics_by_position.csv denoising_results.csv \
      segmentation_results.csv coarse_vs_fine_deltas.csv gate_metrics_by_image.csv \
      gate_metrics_by_position.csv gate_region_summary.csv gate_difference_summary.csv \
      RESULTS_TABLE.csv; do
      [[ -s "$REPORT/$name" ]] || { echo "BLOCKED: missing $REPORT/$name" >&2; exit 2; }
    done
    python - "$REPORT/summary.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("status") != "passed" or value.get("test_assets_opened") != 0:
    raise SystemExit("BLOCKED: evaluation summary failed")
print(json.dumps(value, indent=2))
PY
    echo "PASS: position-equal summary is complete"
    ;;
  package)
    preflight
    [[ -n "$OUTPUT" ]] || { echo "--output must be an absolute ZIP path" >&2; exit 2; }
    [[ "$OUTPUT" = /* ]] || { echo "--output must be absolute" >&2; exit 2; }
    python tools/package_dual_task_adaptive.py \
      --project-root . --run-id "$RUN_ID" --output "$OUTPUT"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
