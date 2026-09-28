#!/usr/bin/env bash
# Unique fail-closed ModelWhale entry for noisy+mild dual-view v1.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash tools/run_dual_view_modelwhale.sh preflight --run-id ID
  bash tools/run_dual_view_modelwhale.sh overfit   --run-id ID
  bash tools/run_dual_view_modelwhale.sh pilot     --run-id ID
  bash tools/run_dual_view_modelwhale.sh formal    --run-id ID --seeds 42,43,44
  bash tools/run_dual_view_modelwhale.sh summarize --run-id ID
  bash tools/run_dual_view_modelwhale.sh package   --run-id ID --output ABSOLUTE_ZIP_PATH
EOF
}

[[ $# -ge 1 ]] || { usage; exit 64; }
ACTION="$1"
shift
if [[ "$ACTION" == "-h" || "$ACTION" == "--help" ]]; then
  usage
  exit 0
fi
RUN_ID=""
SEEDS="42,43,44"
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id) RUN_ID="${2:?missing --run-id value}"; shift 2 ;;
    --seeds) SEEDS="${2:?missing --seeds value}"; shift 2 ;;
    --output) OUTPUT="${2:?missing --output value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 64 ;;
  esac
done
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || {
  echo "BLOCKED: --run-id must use letters, digits, dot, underscore or dash" >&2
  exit 64
}

PROJECT_ROOT="${SABIDS_PROJECT_ROOT:-/mnt/SABIDS-Net}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${SABIDS_PYTHON:-python}"
PROTOCOL_ID="${SABIDS_PROTOCOL_ID:-pku37_binary_v3}"
LOCK="${SABIDS_PROTOCOL_LOCK:-$PROJECT_ROOT/Manifests/$PROTOCOL_ID/active_protocol_lock.json}"
SPLIT="${SABIDS_SPLIT_CONTRACT:-$PROJECT_ROOT/configs/data/${PROTOCOL_ID}_split.yaml}"
D1_RUN="${SABIDS_D1_RUN:-$PROJECT_ROOT/runs/adaptive_denoising/$PROTOCOL_ID/d1_repro_fold0_seed42}"
D1_CHECKPOINT="${SABIDS_D1_CHECKPOINT:-$D1_RUN/best.pth}"
D1_INVENTORY="${SABIDS_D1_INVENTORY:-$D1_RUN/training_asset_inventory_initial.json}"
D1_BINDING="${SABIDS_D1_BINDING:-$D1_RUN/checkpoint_binding_best_d2_v1.json}"
D1_DOSE_REGISTRY="${SABIDS_D1_DOSE_REGISTRY:-$PROJECT_ROOT/cache/adaptive_denoising/$PROTOCOL_ID/dose_v1/preparations/pilot_s42_d1_best_sensitivity_v2/preparation_registry.json}"
CACHE_BASE="$PROJECT_ROOT/cache/adaptive_denoising/$PROTOCOL_ID/dual_view_v1"
RUN_BASE="$PROJECT_ROOT/runs/adaptive_denoising/$PROTOCOL_ID/dual_view_v1"
REPORT_BASE="$PROJECT_ROOT/reports/adaptive_denoising/dual_view_v1/$RUN_ID"

on_error() {
  local code=$?
  echo "FAILED action=$ACTION run_id=$RUN_ID line=${BASH_LINENO[0]} exit=$code" >&2
  echo "Artifacts were preserved; no arm was silently skipped." >&2
  exit "$code"
}
trap on_error ERR

evidence_args=(
  --project-root "$PROJECT_ROOT"
  --protocol-lock "$LOCK"
  --split-contract "$SPLIT"
  --d1-checkpoint "$D1_CHECKPOINT"
  --d1-checkpoint-binding "$D1_BINDING"
  --d1-training-asset-inventory "$D1_INVENTORY"
  --d1-dose-registry "$D1_DOSE_REGISTRY"
)

preflight() {
  local evidence_output="${1:-}"
  local evidence_tmp
  evidence_tmp="$(mktemp)"
  echo "=== DUAL-VIEW PREFLIGHT: $RUN_ID ==="
  "$PYTHON_BIN" - <<'PY'
import shutil, sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_0:", torch.cuda.get_device_name(0))
usage = shutil.disk_usage(".")
print("disk_free_bytes:", usage.free)
if not torch.cuda.is_available():
    raise SystemExit("BLOCKED: CUDA is unavailable")
if usage.free < 20 * 1024**3:
    raise SystemExit("BLOCKED: less than 20 GiB free")
PY
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
  if pgrep -af '(^|/)(train|evaluate)\.py' >/tmp/sabids_dual_view_processes.txt; then
    cat /tmp/sabids_dual_view_processes.txt >&2
    echo "BLOCKED: another SABIDS train/evaluate process is active" >&2
    rm -f /tmp/sabids_dual_view_processes.txt
    return 2
  fi
  rm -f /tmp/sabids_dual_view_processes.txt
  "$PYTHON_BIN" tools/prepare_dual_view_inputs.py \
    "${evidence_args[@]}" --mode audit | tee "$evidence_tmp"
  if [[ -n "$evidence_output" ]]; then
    if [[ -e "$evidence_output" ]]; then
      "$PYTHON_BIN" - "$evidence_output" "$evidence_tmp" <<'PY'
import json, sys
from pathlib import Path
left, right = (json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[1:])
if left != right:
    raise SystemExit("BLOCKED: existing preflight evidence differs")
print("PASS: reusing identical preflight evidence")
PY
      rm -f "$evidence_tmp"
    else
      mkdir -p "$(dirname "$evidence_output")"
      mv "$evidence_tmp" "$evidence_output"
    fi
  else
    rm -f "$evidence_tmp"
  fi
  echo "PASS: D1 alpha=0.25 evidence, protocol, split, GPU and space are usable"
  echo "PASS: preflight opened no test assets"
}

prepare_budget() {
  local budget="$1"
  local tag="$2"
  local preflight_output="$3"
  shift 3
  local seed_args=("$@")
  preflight "$preflight_output"
  "$PYTHON_BIN" tools/prepare_dual_view_inputs.py \
    "${evidence_args[@]}" \
    --mode prepare --budget "$budget" --tag "$tag" --seeds "${seed_args[@]}"
}

run_arm() {
  local tag="$1"
  local arm="$2"
  local seed="$3"
  local config="$CACHE_BASE/$tag/config_${arm,,}_seed${seed}.yaml"
  local run="$RUN_BASE/$tag/${arm,,}_seed${seed}"
  [[ -s "$config" ]] || { echo "BLOCKED: missing config $config" >&2; return 2; }
  if [[ -e "$run" ]]; then
    "$PYTHON_BIN" - "$config" "$run" <<'PY'
import json, sys
from pathlib import Path

import pandas as pd
import torch

from sabids.config import load_config
from sabids.experiments.protocol_lock import sha256_file

config_path, run = Path(sys.argv[1]), Path(sys.argv[2])
required = (
    "resolved_config.yaml", "history.csv", "last.pth", "best.pth",
    "dual_view_training_metadata.json",
)
missing = [name for name in required if not (run / name).is_file()]
if missing:
    raise SystemExit(f"BLOCKED: existing run is incomplete; missing {missing}: {run}")
expected, resolved = load_config(config_path), load_config(run / "resolved_config.yaml")
expected.pop("runtime", None)
resolved.pop("runtime", None)
if expected != resolved:
    raise SystemExit(f"BLOCKED: existing run config differs: {run}")
configured = int(expected["train"]["epochs"])
history = pd.read_csv(run / "history.csv")
if "epoch" not in history:
    raise SystemExit(f"BLOCKED: existing run history lacks epoch: {run}")
epochs = pd.to_numeric(history["epoch"], errors="raise").astype(int).tolist()
if epochs != list(range(1, configured + 1)):
    raise SystemExit(f"BLOCKED: existing run epoch sequence is incomplete: {run}")
checkpoint = torch.load(run / "last.pth", map_location="cpu", weights_only=False)
metadata = json.loads((run / "dual_view_training_metadata.json").read_text(encoding="utf-8"))
checks = {
    "checkpoint_epoch": int(checkpoint.get("epoch", -2)) + 1 == configured,
    "completed_epochs": int(metadata.get("completed_epochs", -1)) == configured,
    "optimizer_budget": metadata.get("completed_optimizer_steps") == metadata.get("expected_optimizer_steps"),
    "checkpoint_sha": metadata.get("primary_checkpoint_sha256") == sha256_file(run / "last.pth"),
    "frozen_unchanged": not metadata.get("changed_frozen_parameter_names"),
    "denoiser_unchanged": not metadata.get("frozen_denoiser_changed_parameter_names"),
    "test_assets_unopened": metadata.get("test_assets_opened") == 0,
}
if not all(checks.values()):
    raise SystemExit(f"BLOCKED: existing run failed completion audit {checks}: {run}")
print(f"PASS: reusing completed run {run}")
PY
    return 0
  fi
  mkdir -p "$run"
  "$PYTHON_BIN" train.py --config "$config" 2>&1 | tee "$run/train.log"
}

evaluation_complete() {
  local out="$1"
  local inventory="$2"
  "$PYTHON_BIN" - "$out" "$inventory" <<'PY'
import json, sys
from pathlib import Path

import pandas as pd

out, inventory_path = Path(sys.argv[1]), Path(sys.argv[2])
required = ("frame_metrics.csv", "group_metrics.csv", "component_metrics.csv", "summary.json")
missing = [name for name in required if not (out / name).is_file()]
if missing:
    raise SystemExit(f"BLOCKED: existing evaluation is incomplete; missing {missing}: {out}")
for name in ("frame_metrics.csv", "group_metrics.csv"):
    table = pd.read_csv(out / name)
    split_column = "source_split" if "source_split" in table else "split" if "split" in table else None
    if split_column is not None and not table[split_column].astype(str).eq("val").all():
        raise SystemExit(f"BLOCKED: non-validation rows in {out / name}")
summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
checks = {
    "inventory": summary.get("fixed_component_inventory_sha256") == inventory.get("inventory_sha256"),
    "test_assets_unopened": summary.get("test_assets_opened", 0) == 0,
}
if not all(checks.values()):
    raise SystemExit(f"BLOCKED: existing evaluation identity failed {checks}: {out}")
print(f"PASS: reusing complete validation evaluation {out}")
PY
}

audit_arm() {
  local report="$1"
  local run="$2"
  local arm="$3"
  local seed="$4"
  mkdir -p "$report/audits"
  local output="$report/audits/${arm,,}_seed${seed}.json"
  if [[ -e "$output" ]]; then
    "$PYTHON_BIN" - "$output" "$run" "$arm" "$seed" <<'PY'
import json, sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_run = str(Path(sys.argv[2]).resolve())
checks = {
    "status": value.get("status") == "passed",
    "run": value.get("run_dir") == expected_run,
    "arm": value.get("arm") == sys.argv[3],
    "seed": int(value.get("seed", -1)) == int(sys.argv[4]),
    "checks": bool(value.get("checks")) and all(value["checks"].values()),
    "test_assets_unopened": value.get("test_assets_opened") == 0,
}
if not all(checks.values()):
    raise SystemExit(f"BLOCKED: existing run audit failed identity checks {checks}")
print(f"PASS: reusing completed run audit {sys.argv[1]}")
PY
    return 0
  fi
  "$PYTHON_BIN" tools/audit_dual_view_run.py \
    --run-dir "$run" --output "$output" --device cuda
}

evaluate_arm() {
  local report="$1"
  local inventory="$2"
  local run="$3"
  local arm="$4"
  local seed="$5"
  local config="$run/resolved_config.yaml"
  for selection in last best; do
    local checkpoint="$run/${selection}.pth"
    local out="$run/validation_${selection}"
    [[ -s "$checkpoint" ]] || { echo "BLOCKED: missing $checkpoint" >&2; return 2; }
    if [[ -e "$out" ]]; then
      evaluation_complete "$out" "$inventory"
    else
      local prediction_flag=()
      [[ "$selection" == "last" ]] && prediction_flag=(--save-predictions)
      "$PYTHON_BIN" evaluate.py \
        --config "$config" --checkpoint "$checkpoint" --split val --output "$out" \
        --tasks layer vessel --postprocess-modes p0 --layer-threshold 0.5 \
        --vessel-threshold 0.5 --no-restore-original-geometry \
        --fixed-component-inventory "$inventory" "${prediction_flag[@]}"
    fi
  done
  if [[ "$arm" == "B3" ]]; then
    local c5="$run/validation_c5_last"
    if [[ -e "$c5" ]]; then
      evaluation_complete "$c5" "$inventory"
    else
      "$PYTHON_BIN" evaluate.py \
        --config "$config" --checkpoint "$run/last.pth" --split val --output "$c5" \
        --tasks layer vessel --postprocess-modes p0 --layer-threshold 0.5 \
        --vessel-threshold 0.5 --no-restore-original-geometry \
        --fixed-component-inventory "$inventory" --disable-dual-view-auxiliary \
        --save-predictions
    fi
  fi
  audit_arm "$report" "$run" "$arm" "$seed"
}

summarize_tag() {
  local tag="$1"
  local report="$2"
  shift 2
  local seeds=("$@")
  local summary="$report/summary"
  if [[ -e "$summary" ]]; then
    "$PYTHON_BIN" - "$summary" <<'PY'
import json, sys
from pathlib import Path

summary = Path(sys.argv[1])
required = (
    "summary.json", "paired_summary.csv", "metrics_by_seed_position.csv",
    "paired_position_differences.csv", "failure_case_index.csv",
)
missing = [name for name in required if not (summary / name).is_file()]
if missing:
    raise SystemExit(f"BLOCKED: existing summary is incomplete; missing {missing}: {summary}")
value = json.loads((summary / "summary.json").read_text(encoding="utf-8"))
if value.get("status") != "passed" or value.get("test_assets_opened") != 0:
    raise SystemExit(f"BLOCKED: existing summary failed identity checks: {summary}")
print(f"PASS: reusing complete summary {summary}")
PY
    return 0
  fi
  local runs=()
  for seed in "${seeds[@]}"; do
    for arm in b0 b1 b3 b6 c1; do
      local run="$RUN_BASE/$tag/${arm}_seed${seed}"
      [[ -s "$run/validation_last/group_metrics.csv" ]] || {
        echo "BLOCKED: incomplete validation $run" >&2
        return 2
      }
      runs+=("$run")
    done
  done
  mkdir -p "$report"
  "$PYTHON_BIN" tools/summarize_dual_view.py --run-dirs "${runs[@]}" --output "$summary"
}

run_matrix() {
  local budget="$1"
  local tag="$2"
  local report="$3"
  shift 3
  local seeds=("$@")
  mkdir -p "$report"
  prepare_budget "$budget" "$tag" "$report/preflight_evidence.json" "${seeds[@]}"
  local b0_config="$CACHE_BASE/$tag/config_b0_seed${seeds[0]}.yaml"
  local inventory="$report/fixed_component_inventory.json"
  if [[ -s "$inventory" ]]; then
    "$PYTHON_BIN" - "$inventory" <<'PY'
import json, sys
from pathlib import Path
from sabids.experiments.dose_response import stable_sha
value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
claimed = value.pop("inventory_sha256", None)
if claimed != stable_sha(value) or value.get("test_assets_opened") != 0:
    raise SystemExit("BLOCKED: existing fixed-component inventory is invalid")
print("PASS: reusing valid fixed-component inventory")
PY
  else
    "$PYTHON_BIN" tools/prepare_dual_view_strata.py \
      --project-root "$PROJECT_ROOT" --config "$b0_config" --output "$inventory"
  fi
  for seed in "${seeds[@]}"; do
    for arm in B0 B1 B3 B6 C1; do
      run_arm "$tag" "$arm" "$seed"
      evaluate_arm "$report" "$inventory" "$RUN_BASE/$tag/${arm,,}_seed${seed}" "$arm" "$seed"
    done
  done
  summarize_tag "$tag" "$report" "${seeds[@]}"
}

case "$ACTION" in
  preflight)
    preflight "$REPORT_BASE/preflight_evidence.json"
    ;;
  overfit)
    tag="${RUN_ID}_overfit"
    report="$REPORT_BASE/overfit"
    [[ ! -e "$report" ]] || { echo "BLOCKED: refusing existing report stage $report" >&2; exit 2; }
    mkdir -p "$report"
    prepare_budget overfit "$tag" "$report/preflight_evidence.json" 42
    for arm in B0 B1 B3 B6 C1; do
      run_arm "$tag" "$arm" 42
      audit_arm "$report" "$RUN_BASE/$tag/${arm,,}_seed42" "$arm" 42
    done
    echo "PASS: train-only overfit diagnostics completed (8 train + 4 train-held rows; no val/test selection)"
    ;;
  pilot)
    tag="${RUN_ID}_pilot"
    report="$REPORT_BASE/pilot"
    run_matrix pilot "$tag" "$report" 42
    if [[ -e "$report/gate.json" ]]; then
      "$PYTHON_BIN" - "$report/gate.json" <<'PY'
import json, sys
from pathlib import Path
gate = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if gate.get("test_assets_opened") != 0:
    raise SystemExit("BLOCKED: existing pilot gate has invalid test audit")
if gate.get("status") != "passed" or gate.get("formal_allowed") is not True:
    raise SystemExit("BLOCKED: existing preregistered pilot gate did not pass")
print("PASS: reusing passed preregistered pilot gate")
PY
    else
      "$PYTHON_BIN" tools/check_dual_view_gate.py \
        --paired-summary "$report/summary/paired_summary.csv" --output "$report/gate.json"
    fi
    echo "PASS: seed42 pilot and preregistered gate completed"
    ;;
  formal)
    [[ "$SEEDS" == "42,43,44" ]] || {
      echo "BLOCKED: formal requires exactly --seeds 42,43,44" >&2
      exit 2
    }
    gate="$REPORT_BASE/pilot/gate.json"
    "$PYTHON_BIN" - "$gate" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit("BLOCKED: missing preregistered pilot gate")
gate = json.loads(path.read_text(encoding="utf-8"))
if gate.get("formal_allowed") is not True or gate.get("status") != "passed":
    raise SystemExit("BLOCKED: seed42 pilot gate did not pass")
print("PASS: immutable pilot gate permits formal execution")
PY
    IFS=',' read -r -a seed_array <<< "$SEEDS"
    run_matrix full "${RUN_ID}_formal" "$REPORT_BASE/formal" "${seed_array[@]}"
    echo "PASS: formal seeds 42/43/44 completed"
    ;;
  summarize)
    if [[ -d "$RUN_BASE/${RUN_ID}_formal" ]]; then
      summarize_tag "${RUN_ID}_formal" "$REPORT_BASE/formal_resummary" 42 43 44
    elif [[ -d "$RUN_BASE/${RUN_ID}_pilot" ]]; then
      summarize_tag "${RUN_ID}_pilot" "$REPORT_BASE/pilot_resummary" 42
    else
      echo "BLOCKED: no pilot or formal runs for $RUN_ID" >&2
      exit 2
    fi
    ;;
  package)
    [[ -n "$OUTPUT" ]] || { echo "BLOCKED: package requires --output" >&2; exit 64; }
    [[ "$OUTPUT" = /* ]] || { echo "BLOCKED: --output must be an absolute path" >&2; exit 64; }
    "$PYTHON_BIN" tools/package_dual_view_for_gpt.py \
      --project-root "$PROJECT_ROOT" --run-id "$RUN_ID" --output "$OUTPUT"
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    usage
    exit 64
    ;;
esac
