#!/usr/bin/env bash
# Unique fail-closed ModelWhale entry for segmentation-guided adaptive v1.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_seg_guided_modelwhale.sh preflight      --run-id ID
  bash tools/run_seg_guided_modelwhale.sh intervene      --run-id ID
  bash tools/run_seg_guided_modelwhale.sh residual-audit --run-id ID
  bash tools/run_seg_guided_modelwhale.sh oracle         --run-id ID
  bash tools/run_seg_guided_modelwhale.sh cv-pilot       --run-id ID --folds 0 --seeds 42
  bash tools/run_seg_guided_modelwhale.sh cv-formal      --run-id ID --folds 0,1,2,3 --seeds 42,43,44
  bash tools/run_seg_guided_modelwhale.sh adaptive-pilot --run-id ID --folds 0 --seeds 42
  bash tools/run_seg_guided_modelwhale.sh summarize      --run-id ID
  bash tools/run_seg_guided_modelwhale.sh package        --run-id ID --output ABSOLUTE_ZIP

Required environment for real data:
  SABIDS_B3_RUN, SABIDS_FIXED_COMPONENT_INVENTORY,
  SABIDS_SEG_GUIDED_SOURCE_MANIFEST. Optional paths inherit dual-view defaults.
EOF
}

[[ $# -ge 1 ]] || { usage; exit 64; }
ACTION="$1"; shift
RUN_ID=""; FOLDS="0"; SEEDS="42"; OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?}"; shift 2 ;;
    --folds) FOLDS="${2:?}"; shift 2 ;;
    --seeds) SEEDS="${2:?}"; shift 2 ;;
    --output) OUTPUT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 64 ;;
  esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "BLOCKED: invalid --run-id" >&2; exit 64; }

PROJECT_ROOT="${SABIDS_PROJECT_ROOT:-/mnt/SABIDS-Net}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${SABIDS_PYTHON:-python}"
PROTOCOL_ID="${SABIDS_PROTOCOL_ID:-pku37_binary_v3}"
LOCK="${SABIDS_PROTOCOL_LOCK:-$PROJECT_ROOT/Manifests/$PROTOCOL_ID/active_protocol_lock.json}"
SPLIT="${SABIDS_SPLIT_CONTRACT:-$PROJECT_ROOT/configs/data/${PROTOCOL_ID}_split.yaml}"
B3_RUN="${SABIDS_B3_RUN:-}"
INVENTORY="${SABIDS_FIXED_COMPONENT_INVENTORY:-}"
SOURCE_MANIFEST="${SABIDS_SEG_GUIDED_SOURCE_MANIFEST:-}"
REPORT="$PROJECT_ROOT/reports/adaptive_denoising/seg_guided_adaptive_v1/$RUN_ID"
CV_PROTOCOL="$PROJECT_ROOT/cache/adaptive_denoising/$PROTOCOL_ID/seg_guided_adaptive_v1/$RUN_ID"

trap 'code=$?; echo "FAILED action=$ACTION run_id=$RUN_ID line=${BASH_LINENO[0]} exit=$code" >&2; echo "Artifacts preserved; no stage was silently skipped." >&2; exit $code' ERR

require_file() { [[ -s "$1" ]] || { echo "BLOCKED: missing $2: $1" >&2; return 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "BLOCKED: missing $2: $1" >&2; return 2; }; }

preflight() {
  require_file "$LOCK" "protocol lock"
  require_file "$SPLIT" "split contract"
  [[ -n "$B3_RUN" ]] || { echo "BLOCKED: DUAL-VIEW INPUT EVIDENCE (set SABIDS_B3_RUN)" >&2; return 2; }
  [[ -n "$INVENTORY" ]] || { echo "BLOCKED: DUAL-VIEW INPUT EVIDENCE (set SABIDS_FIXED_COMPONENT_INVENTORY)" >&2; return 2; }
  [[ -n "$SOURCE_MANIFEST" ]] || { echo "BLOCKED: DUAL-VIEW INPUT EVIDENCE (set SABIDS_SEG_GUIDED_SOURCE_MANIFEST)" >&2; return 2; }
  require_file "$B3_RUN/last.pth" "B3 last checkpoint"
  require_file "$B3_RUN/best.pth" "B3 best checkpoint"
  require_file "$B3_RUN/resolved_config.yaml" "B3 resolved config"
  require_file "$INVENTORY" "fixed component inventory"
  require_file "$SOURCE_MANIFEST" "development manifest"
  "$PYTHON_BIN" - "$LOCK" "$SPLIT" "$SOURCE_MANIFEST" "$B3_RUN/resolved_config.yaml" <<'PY'
import json, re, sys, shutil
from pathlib import Path
import pandas as pd, torch, yaml
lock, split, manifest, config = map(Path, sys.argv[1:])
table = pd.read_csv(manifest, dtype=str).fillna("")
if "split" not in table or not table.split.isin(["train", "val"]).all():
    raise SystemExit("BLOCKED: source manifest is not development-only")
lock_value = json.loads(lock.read_text(encoding="utf-8-sig"))
split_value = yaml.safe_load(split.read_text(encoding="utf-8")) or {}
sealed = (lock_value.get("sealed_test_positions") or lock_value.get("test_positions") or [])
sealed += split_value.get("test_positions", []) or []
canonical = lambda value: re.sub(r"[^a-z0-9]", "", str(value).lower())
sealed = {canonical(value) for value in sealed}
if table.group_id.map(canonical).isin(sealed).any():
    raise SystemExit("BLOCKED: sealed test group appears in development manifest")
cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
if cfg.get("dual_view", {}).get("arm") != "B3":
    raise SystemExit("BLOCKED: checkpoint run is not B3")
print("python/torch:", sys.version.split()[0], torch.__version__)
print("cuda:", torch.cuda.is_available(), torch.cuda.device_count())
print("disk_free_bytes:", shutil.disk_usage(".").free)
print("development_rows/groups:", len(table), table.group_id.nunique())
print("PASS: development-only evidence paths are structurally valid")
PY
  pgrep -af '(^|/)(train|evaluate)\.py' && { echo "BLOCKED: another SABIDS process is active" >&2; return 2; } || true
  mkdir -p "$REPORT"
  printf '{"status":"passed","test_assets_opened":0,"run_id":"%s"}\n' "$RUN_ID" > "$REPORT/preflight.json"
  echo "PASS: SEG-GUIDED preflight (sealed test not opened)"
}

ensure_cv_protocol() {
  if [[ ! -s "$CV_PROTOCOL/cv_registry.json" ]]; then
    "$PYTHON_BIN" tools/build_grouped_cv.py --project-root . --source-manifest "$SOURCE_MANIFEST" \
      --protocol-lock "$LOCK" --split-contract "$SPLIT" --output "$CV_PROTOCOL" \
      --run-id "$RUN_ID" --seeds 42 43 44
  fi
}

evaluate_cv_run() {
  local config="$1" run="$2" inventory="$3"
  for spec in "epoch012:validation_epoch012" "last:validation_last" "best:validation_best"; do
    local checkpoint="${spec%%:*}" folder="${spec##*:}"
    [[ -s "$run/$checkpoint.pth" ]] || { echo "BLOCKED: missing $run/$checkpoint.pth" >&2; return 2; }
    if [[ ! -s "$run/$folder/group_metrics.csv" ]]; then
      "$PYTHON_BIN" evaluate.py --config "$config" --checkpoint "$run/$checkpoint.pth" \
        --split val --output "$run/$folder" --tasks layer vessel --postprocess-modes p0 \
        --layer-threshold 0.5 --vessel-threshold 0.5 --no-restore-original-geometry \
        --fixed-component-inventory "$inventory" --capture-dual-diagnostics
    fi
  done
}

run_cv() {
  local mode="$1"
  IFS=',' read -r -a folds <<< "$FOLDS"; IFS=',' read -r -a seeds <<< "$SEEDS"
  ensure_cv_protocol
  for fold in "${folds[@]}"; do
    local inventory="$REPORT/cv/fold${fold}/fixed_component_inventory.json"
    mkdir -p "$(dirname "$inventory")"
    if [[ ! -s "$inventory" ]]; then
      "$PYTHON_BIN" tools/prepare_dual_view_strata.py --project-root . \
        --config "$CV_PROTOCOL/configs/fold${fold}_b0_seed${seeds[0]}.yaml" --output "$inventory"
    fi
    for seed in "${seeds[@]}"; do
      for arm in b0 b1 b3 b6 b3r b6r; do
        local config="$CV_PROTOCOL/configs/fold${fold}_${arm}_seed${seed}.yaml"
        local run="$PROJECT_ROOT/runs/adaptive_denoising/seg_guided_adaptive_v1/$RUN_ID/fold${fold}/${arm}_seed${seed}"
        require_file "$config" "CV config"
        if [[ ! -s "$run/last.pth" || ! -s "$run/best.pth" || ! -s "$run/epoch012.pth" ]]; then
          [[ ! -e "$run" ]] || { echo "BLOCKED: incomplete existing run $run" >&2; return 2; }
          "$PYTHON_BIN" train.py --config "$config"
        fi
        evaluate_cv_run "$config" "$run" "$inventory"
      done
    done
  done
  "$PYTHON_BIN" - "$REPORT/cv_${mode}_gate.json" "$PROJECT_ROOT/runs/adaptive_denoising/seg_guided_adaptive_v1/$RUN_ID" "$FOLDS" "$SEEDS" <<'PY'
import json, sys
from pathlib import Path
out, root, folds, seeds = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(","), sys.argv[4].split(",")
missing=[]
for f in folds:
  for s in seeds:
    for a in ("b0","b1","b3","b6","b3r","b6r"):
      run=root/f"fold{f}"/f"{a}_seed{s}"
      for name in ("epoch012.pth","last.pth","best.pth","validation_epoch012/group_metrics.csv","validation_last/group_metrics.csv","validation_best/group_metrics.csv"):
        if not (run/name).is_file(): missing.append(str(run/name))
value={"status":"passed" if not missing else "failed","formal_allowed":not missing,
       "missing":missing,"test_assets_opened":0}
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(value,indent=2)+"\n")
if missing: raise SystemExit("BLOCKED: CV engineering audit failed")
print(json.dumps(value,indent=2))
PY
}

case "$ACTION" in
  preflight) preflight ;;
  intervene)
    preflight
    "$PYTHON_BIN" tools/audit_b3_interventions.py --project-root . --b3-run "$B3_RUN" \
      --fixed-component-inventory "$INVENTORY" --output "$REPORT/interventions" --execute ;;
  residual-audit)
    preflight
    args=(); [[ -s "$REPORT/interventions/INTERVENTION_RESULTS.csv" ]] && args+=(--intervention-results "$REPORT/interventions/INTERVENTION_RESULTS.csv")
    "$PYTHON_BIN" tools/analyze_residual_structure.py --project-root . --manifest "$SOURCE_MANIFEST" \
      --fixed-component-inventory "$INVENTORY" --output "$REPORT/residual" "${args[@]}" ;;
  oracle)
    preflight
    "$PYTHON_BIN" tools/evaluate_spatial_oracle.py --project-root . --manifest "$SOURCE_MANIFEST" \
      --b3-run "$B3_RUN" --fixed-component-inventory "$INVENTORY" --output "$REPORT/oracle" --execute ;;
  cv-pilot)
    preflight
    [[ "$FOLDS" == "0" && "$SEEDS" == "42" ]] || { echo "BLOCKED: cv-pilot is preregistered as fold0 seed42" >&2; exit 2; }
    run_cv pilot ;;
  cv-formal)
    preflight
    require_file "$REPORT/cv_pilot_gate.json" "CV pilot gate"
    "$PYTHON_BIN" - "$REPORT/cv_pilot_gate.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1])); assert v.get("status")=="passed" and v.get("formal_allowed") is True and v.get("test_assets_opened")==0
PY
    run_cv formal ;;
  adaptive-pilot)
    preflight
    require_file "$REPORT/oracle/oracle_gate.json" "oracle gate"
    "$PYTHON_BIN" - "$REPORT/oracle/oracle_gate.json" <<'PY'
import json,sys
v=json.load(open(sys.argv[1]))
if v.get("status")!="passed" or v.get("adaptive_training_allowed") is not True:
  raise SystemExit("BLOCKED: NO SPATIAL DENOISING ORACLE UPPER BOUND")
PY
    [[ -n "${SABIDS_OOF_GUIDANCE_REGISTRY:-}" ]] || { echo "BLOCKED: oracle passed but cross-fitted S0 guidance registry is required" >&2; exit 2; }
    require_file "$SABIDS_OOF_GUIDANCE_REGISTRY" "OOF guidance registry"
    echo "BLOCKED: Stage-E training remains locked until its OOF registry/config audit is implemented; no deployable claim was made" >&2
    exit 2 ;;
  summarize)
    mapfile -t runs < <(find "$PROJECT_ROOT/runs/adaptive_denoising/seg_guided_adaptive_v1/$RUN_ID" -mindepth 2 -maxdepth 2 -type d -name '*_seed*' 2>/dev/null | sort)
    extra=(); [[ -d "$REPORT/interventions" ]] && extra+=(--intervention-report "$REPORT/interventions")
    [[ -d "$REPORT/residual" ]] && extra+=(--residual-report "$REPORT/residual")
    [[ -d "$REPORT/oracle" ]] && extra+=(--oracle-report "$REPORT/oracle")
    "$PYTHON_BIN" tools/summarize_seg_guided_cv.py --run-dirs "${runs[@]}" "${extra[@]}" --output "$REPORT/summary" ;;
  package)
    [[ "$OUTPUT" = /* ]] || { echo "BLOCKED: --output must be absolute" >&2; exit 64; }
    "$PYTHON_BIN" tools/package_seg_guided_for_gpt.py --roots "$REPORT" "$CV_PROTOCOL" --output "$OUTPUT" ;;
  *) usage; exit 64 ;;
esac
