#!/usr/bin/env bash
# Minimal DEVELOPMENT queue. No P0/P1/P4 or mechanism sweep is implicit.
set -euo pipefail
: "${SABIDS_ROOT:?Set actual project root}"
: "${LAC_BASE:?Set new/resumable persistent development run base}"
: "${OLD_CACHE:?Set verified historical FULL native cache}"
cd "$SABIDS_ROOT"
test -f "$OLD_CACHE/cache.json"
test -f tools/dispatch_learnable_ac_v1.py
mkdir -p "$LAC_BASE/logs"
# flock belongs to this scheduler only, no stale PID assumptions.
exec 9>"$LAC_BASE/.queue.lock"
flock -n 9 || { echo 'BLOCKED: queue already active'; exit 2; }
printf '%s\n' "$$" > "$LAC_BASE/logs/queue.pid"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LAC_BASE/logs/queue_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; printf "exit_code=%s time=%s\n" "$code" "$(date -Is)" > "$LAC_BASE/logs/queue_exit_${STAMP}.txt"' EXIT
EXTRA=()
if test "${LAC_WITH_P2:-0}" = 1; then EXTRA+=(--with-p2); fi
python -u tools/dispatch_learnable_ac_v1.py --mode run --base "$LAC_BASE" --input-cache "$OLD_CACHE" "${EXTRA[@]}"
