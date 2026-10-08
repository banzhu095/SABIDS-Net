#!/usr/bin/env bash
set -euo pipefail
ROOT=""; COMMAND=materialize-inputs; ARGS=()
while (($#)); do
  case "$1" in
    --project-root) ROOT="$2"; ARGS+=("$1" "$2"); shift 2 ;;
    --verify-only) COMMAND=verify-materialized-inputs; shift ;;
    --resume|--reinfer) ARGS+=("$1"); shift ;;
    --source-denoise-run|--run-dir|--methods) ARGS+=("$1" "$2"); shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$ROOT" ]] || { echo '--project-root required' >&2; exit 2; }
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m tools.downstream_seg_benchmark.cli "$COMMAND" "${ARGS[@]}" --device cuda
