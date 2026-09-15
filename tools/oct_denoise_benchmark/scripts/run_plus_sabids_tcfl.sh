#!/usr/bin/env bash
set -euo pipefail

root="/mnt/SABIDS-Net"
base=""
ext=""
track="full"
resume=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root) root="$2"; shift 2 ;;
    --base-run) base="$2"; shift 2 ;;
    --ext-run) ext="$2"; shift 2 ;;
    --track) track="$2"; shift 2 ;;
    --resume) resume=1; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$track" in init|smoke|train|lock|evaluate|merge|package|full) ;; *) echo "Unknown track: $track" >&2; exit 2 ;; esac
cd "$root"
[[ -n "$ext" ]] || ext="$root/runs/denoise_benchmark_plus_sabids_tcfl_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$ext/logs" "$ext/status"
exec > >(tee -a "$ext/logs/${track}.stdout.log") 2> >(tee -a "$ext/logs/${track}.stderr.log" >&2)

if command -v conda >/dev/null 2>&1; then eval "$(conda shell.bash hook)"; fi
conda activate myconda
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"

commit="$(git rev-parse HEAD)"
config_signature="$({ printf '%s\n' "$commit"; sha256sum Manifests/manifest_denoise.csv; } | sha256sum | awk '{print $1}')"
tracked_dirty="$(git status --porcelain --untracked-files=no)"
if [[ -n "$tracked_dirty" ]]; then
  echo "Formal extension tracks require committed tracked files:" >&2; printf '%s\n' "$tracked_dirty" >&2; exit 1
fi
untracked_source="$(git ls-files --others --exclude-standard -- configs docs sabids tests tools '*.py' '*.sh')"
if [[ -n "$untracked_source" ]]; then echo "Untracked source files are forbidden:" >&2; printf '%s\n' "$untracked_source" >&2; exit 1; fi

status_file="$ext/status/${track}.json"
write_status() {
  local state="$1" ended="${2:-}"
  python - "$status_file" "$track" "$state" "$$" "$commit" "$ext" "$base" "$ended" "$config_signature" <<'PY'
import json, os, sys, tempfile
from datetime import datetime, timezone
path, track, state, pid, commit, ext, base, ended, signature = sys.argv[1:]
old = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
resume = f'bash tools/oct_denoise_benchmark/scripts/run_plus_sabids_tcfl.sh --project-root "{os.getcwd()}" --ext-run "{ext}" --track {track} --resume'
value = {"stage": track, "status": state, "pid": int(pid), "git_commit": commit, "config_sha256": signature, "ext_run": ext, "base_run": base or old.get("base_run"), "started_at_utc": old.get("started_at_utc", datetime.now(timezone.utc).isoformat()), "ended_at_utc": datetime.now(timezone.utc).isoformat() if ended else None, "current_seed": None, "current_update": None, "current_dataset": None, "completed": None, "total": None, "eta": None, "resume_command": resume}
os.makedirs(os.path.dirname(path), exist_ok=True)
fd, temp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".status-", suffix=".tmp")
with os.fdopen(fd, "w", encoding="utf-8") as stream: json.dump(value, stream, indent=2)
os.replace(temp, path)
with open(os.path.join(os.path.dirname(path), f"{track}.job.json"), "w", encoding="utf-8") as stream:
    json.dump({"pid": int(pid), "stage": track, "git_commit": commit, "config_sha256": signature, "resume_command": resume, "stdout_log": os.path.join(ext, "logs", f"{track}.stdout.log"), "stderr_log": os.path.join(ext, "logs", f"{track}.stderr.log")}, stream, indent=2)
PY
}
write_status running
trap 'code=$?; if [[ $code -eq 0 ]]; then write_status success ended; else write_status failed ended; fi' EXIT
if [[ "$track" != "init" && -f "$ext/audit/extension_init.json" ]]; then
  python -m tools.oct_denoise_benchmark.extension_protocol verify-base --project-root "$root" --ext-run "$ext"
fi

do_init() {
  args=(init --project-root "$root" --ext-run "$ext")
  [[ -n "$base" ]] && args+=(--base-run "$base")
  python -m tools.oct_denoise_benchmark.extension_protocol "${args[@]}"
  python -m tools.oct_denoise_benchmark.extension_protocol resolve-sabids --project-root "$root" --ext-run "$ext"
}

do_smoke() {
  python -m compileall -q .
  python -m pytest -q
  python - "$ext/audit/tests_summary.json" <<'PY'
import json, sys
from datetime import datetime, timezone
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump({"scope": "compileall plus complete pytest suite", "status": "passed", "completed_at_utc": datetime.now(timezone.utc).isoformat()}, stream, indent=2)
PY
  smoke_tcfl="$ext/smoke/tcfl_dncnn"
  python -m tools.oct_denoise_benchmark.train_tcfl --project-root "$root" --output "$smoke_tcfl" --device cuda:0 --seed 42 --epochs 1 --steps-per-epoch 1 --patch-size 64 --batch-size 2 --validation-frames-per-position 1 --audit-batches -1 --resume
  if [[ -f "$ext/audit/sabids_selected.json" ]]; then
    readarray -t sabids_values < <(python - "$ext/audit/sabids_selected.json" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(value["checkpoint"])
print(value.get("config_path", ""))
PY
)
    smoke_sabids="${sabids_values[0]}"
    smoke_sabids_config="${sabids_values[1]}"
  else
    smoke_output="$ext/smoke/sabids_current"
    make_sabids_config 42 "$smoke_output" 1
    python - "$ext/configs/sabids_seed_42.yaml" <<'PY'
import sys, yaml
path = sys.argv[1]
value = yaml.safe_load(open(path, encoding="utf-8"))
value.setdefault("data", {})["samples_per_epoch"] = 2
value.setdefault("train", {})["batch_size"] = 1
value["train"]["num_workers"] = 0
with open(path, "w", encoding="utf-8") as stream:
    yaml.safe_dump(value, stream, sort_keys=False)
PY
    python train.py --config "$ext/configs/sabids_seed_42.yaml"
    smoke_sabids="$smoke_output/best.pth"
    smoke_sabids_config="$smoke_output/resolved_config.yaml"
  fi
  smoke_args=(--project-root "$root" --output "$ext/smoke/adapter_io" --sabids-checkpoint "$smoke_sabids" --tcfl-checkpoint "$smoke_tcfl/best_psnr.pth")
  [[ -n "$smoke_sabids_config" && "$smoke_sabids_config" != "embedded" ]] && smoke_args+=(--sabids-config "$smoke_sabids_config")
  python -m tools.oct_denoise_benchmark.smoke_extension "${smoke_args[@]}"
  python - "$ext/reports/smoke_test_report.md" "$ext/smoke/adapter_io/smoke_results.json" <<'PY'
from pathlib import Path
import json, sys
records = json.load(open(sys.argv[2], encoding="utf-8"))
text = "# Extension smoke report\n\nSynthetic contract tests and actual first PKU37 train/validation adapter checks passed. TCFL completed an actual one-batch forward/backward smoke. Shape, dtype, range, finite values, repeat determinism, output decode and common metrics passed. No test or Duke reference was read.\n\n```json\n" + json.dumps(records, indent=2) + "\n```\n"
Path(sys.argv[1]).write_text(text, encoding="utf-8")
PY
}

make_sabids_config() {
  local seed="$1" output="$2" smoke_epochs="${3:-}"
  python - "$root/configs/current/stage1_denoise_current.yaml" "$ext/configs/sabids_seed_${seed}.yaml" "$seed" "$output" "$smoke_epochs" <<'PY'
import sys, yaml
source, target, seed, output, epochs = sys.argv[1:]
value = {"_base_": source, "seed": int(seed), "deterministic": True, "data": {"train_datasets": ["PKU37"], "val_datasets": ["PKU37"], "test_datasets": ["PKU37"], "load_segmentation_labels": False}, "train": {"output_dir": output}}
if epochs: value["train"]["epochs"] = int(epochs)
with open(target, "w", encoding="utf-8") as stream: yaml.safe_dump(value, stream, sort_keys=False)
PY
}

do_train() {
  mkdir -p "$ext/tracks/tcfl_dncnn" "$ext/tracks/sabids_current"
  for seed in 42 123 2026; do
    python -m tools.oct_denoise_benchmark.train_tcfl --project-root "$root" --output "$ext/tracks/tcfl_dncnn/seed_$seed" --device cuda:0 --seed "$seed" --epochs 100 --steps-per-epoch 582 --patch-size 640 --batch-size 2 --learning-rate 0.00002 --beta1 0.5 --beta2 0.999 --lambda-pixel 6 --resume
  done
  if [[ ! -f "$ext/audit/sabids_selected.json" ]]; then
    for seed in 42 123 2026; do
      output="$ext/tracks/sabids_current/seed_$seed"
      make_sabids_config "$seed" "$output"
      if [[ -f "$output/last.pth" ]]; then
        python - "$ext/configs/sabids_seed_${seed}.yaml" "$output/last.pth" <<'PY'
import sys, yaml
path, checkpoint = sys.argv[1:]
value = yaml.safe_load(open(path, encoding="utf-8")); value.setdefault("train", {})["resume"] = checkpoint
yaml.safe_dump(value, open(path, "w", encoding="utf-8"), sort_keys=False)
PY
      fi
      python train.py --config "$ext/configs/sabids_seed_${seed}.yaml"
    done
  fi
}

do_lock() {
  args=(lock --project-root "$root" --ext-run "$ext")
  if [[ ! -f "$ext/audit/sabids_selected.json" ]]; then
    for seed in 42 123 2026; do args+=(--sabids-checkpoint "$ext/tracks/sabids_current/seed_$seed/best.pth"); done
    args+=(--sabids-config "$ext/configs/sabids_seed_42.yaml")
  fi
  python -m tools.oct_denoise_benchmark.extension_protocol "${args[@]}"
}

do_evaluate() {
  python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$ext" --registry "$ext/configs/inference_registry.yaml" --lock-file extension_config_lock.json --methods all --splits train --device cuda:0 --all-deep-seeds
  python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$ext" --registry "$ext/configs/inference_registry.yaml" --lock-file extension_config_lock.json --methods all --splits val --device cuda:0 --all-deep-seeds
  python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$ext" --registry "$ext/configs/inference_registry.yaml" --lock-file extension_config_lock.json --methods all --splits test --device cuda:0 --all-deep-seeds
  python -m tools.oct_denoise_benchmark.evaluate --project-root "$root" --run-dir "$ext" --registry "$ext/configs/inference_registry.yaml" --lock-file extension_config_lock.json --methods all --splits external_test --device cuda:0 --all-deep-seeds
}

do_merge() {
  python -m tools.oct_denoise_benchmark.extension_protocol merge --project-root "$root" --ext-run "$ext"
  python -m tools.oct_denoise_benchmark.downstream_manifest --project-root "$root" --run-dir "$ext"
}

do_package() {
  python -m tools.oct_denoise_benchmark.package_plus --project-root "$root" --ext-run "$ext"
}

case "$track" in
  init) do_init ;;
  smoke) do_smoke ;;
  train) do_train ;;
  lock) do_lock ;;
  evaluate) do_evaluate ;;
  merge) do_merge ;;
  package) do_package ;;
  full) do_init; do_smoke; do_train; do_lock; do_evaluate; do_merge; do_package ;;
esac
