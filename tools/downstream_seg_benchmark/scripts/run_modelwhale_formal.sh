#!/usr/bin/env bash
set -euo pipefail
ROOT=""
ARGS=("$@")
while (($#)); do
  case "$1" in
    --project-root) ROOT="$2"; shift 2 ;;
    *) shift ;;
  esac
done
[[ -n "$ROOT" && -f "$ROOT/tools/downstream_seg_benchmark/cli.py" ]] || { echo 'Valid --project-root required' >&2; exit 2; }
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
python -m tools.downstream_seg_benchmark.cli run-formal "${ARGS[@]}" --device cuda
