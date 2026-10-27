# Minimal seed42 DEVELOPMENT execution (2026-10-27)

Scope is **development / historical upstream exposure**, never formal. The
historical D25/teacher saw all5 current outer positions. Registration remains
the existing user_confirmation; no new registration proof is requested.

## Audit and run identity

- ac_r2 produces protected c, with base reliability ONCE, then z0=.5*m+.5*c.
  F5 changes only the74,056 controller parameters. D2 is an offline frozen
  preprocessor; S0 parameters/stats stay frozen, but F5 retains S0 input gradients.
- F0/adaptive_off use exactly P3 best. F1/F2/F5 use that same anchor, the same
  paired sampling/augmentation plan and20epochs. P3 budget60epochs is unchanged.
- F1 is a matched74,056-parameter low-rank logit refiner. Its image pathway does
  NOT backpropagate through S0, unlike F5; parameter equality and equal number
  of updates do not mean equal computation. Checks report both step/backward
  cost and full inference cost. Initial context gradients may be zero; later
  context gradients must start after coefficient-head updates.
- Existing MA0 is not accepted just by name: cache identity, coverage/uncertain
  treatment, source/study plan and candidate binding differ or are unproven.
  The dispatcher records reuse_audit.json and trains a new P3 unless its own
  complete, fully bound P3 already exists. Historical P0 is not mixed into the
  report. P2 must be a new matching run using this exact train-mild theta.
- Native NumPy and Torch r2 compare ALL20 fixed positions after preparation,
  on native float (both noisy and mild). No quantized-PNG reconstruction.
- No F3/F4/F6, seeds43/44, other folds or hyperparameter search is implicit.

## Commit/deployment boundary

Some required AC/data helper modules were untracked on the pre-existing Git
baseline. They are required import dependencies, not weights or new data.
Stage only reviewed task files and this minimal dependency closure. The shared
Segmenter encode/decode extraction is necessary and independently isolatable.
Do NOT stage unrelated task_separate_fusion/trainer changes, .gitignore,
mixed README/context/experiment-log edits or any runtime directory. Deployment
compares actual file hashes (and records benign platform line-ending differences
for unchanged dependencies), not just Git HEAD. Preserve existing cloud changes.

## New cloud terminal — prepare/check/run

The current verified endpoint supplied by the user is hz-4.matpool.com:29724.
Do not embed credentials in scripts. Hardware at first inspection: P10016GB,
GPU idle; runtime PyTorch2.1.2+cu121. Keep512/batch1/accumulation2/AMP=false.
An OOM is a resource failure: preserve logs, do not silently lower resolution.

```bash
cd /mnt/SABIDS-Net
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myconda
set -euo pipefail
export PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export SABIDS_ROOT=/mnt/SABIDS-Net
export LAC_BASE="$SABIDS_ROOT/runs/learnable_ac_v1/cloud_20261027_v1"
export OLD_CACHE="$SABIDS_ROOT/runs/denoise_then_ac_v1/native_fold0_seed42_20261020_v1/cache"
test -n "$LAC_BASE" && test -n "$OLD_CACHE"
test -f "$OLD_CACHE/cache.json" && test -f tools/dispatch_learnable_ac_v1.py
python tools/dispatch_learnable_ac_v1.py --mode status --base "$LAC_BASE"
ps -eo pid,ppid,lstart,args | grep -E '[t]rain.py|[d]ispatch_learnable|[r]un_learnable'
nvidia-smi
# Start ONLY if no same-run scheduler/child exists. flock prevents duplicate ownership.
mkdir -p "$LAC_BASE/logs"
nohup bash tools/run_learnable_ac_modelwhale.sh \
  > "$LAC_BASE/logs/launch.log" 2>&1 < /dev/null &
```

Order: prepare full cache -> independent native reference -> paired F1/F5 CUDA
check -> P3/resume60 -> best+last evaluation -> P3-bound F1/F5 CUDA check ->
F0 -> evaluate -> F1/resume20 -> evaluate -> F2/resume20 -> evaluate ->
F5/resume20 -> evaluate -> core summary, atlas and light ZIP. Core completion
does NOT wait for P2. To explicitly schedule supplemental P2 AFTER core output,
set `export LAC_WITH_P2=1` before launching. The default leaves it configured
but not started, so unknown extra GPU budget cannot delay core delivery.

## Read-only status (complete terminal variables, no PID-file assumptions)

```bash
cd /mnt/SABIDS-Net
source /root/miniconda3/etc/profile.d/conda.sh; conda activate myconda
export LAC_BASE=/mnt/SABIDS-Net/runs/learnable_ac_v1/cloud_20261027_v1
test -n "$LAC_BASE" && test -d "$LAC_BASE"
python tools/dispatch_learnable_ac_v1.py --mode status --base "$LAC_BASE"
tail -n 15 "$LAC_BASE/dispatch/events.jsonl"
ls "$LAC_BASE/training"/*/first_update.json 2>/dev/null || true
```

status includes actual /proc command/start ticks. Epoch JSON is a progress
snapshot, NOT proof of completion. Completion is checked from model payload,
optimizer/scheduler/RNG, expected steps, inner-best earliest tie and SHA; an
immutable checkpoint_selection_binding.json is then written per method.

## Full continuation / manual evaluation / results

Do not relaunch while the scheduler or its actual child is alive. If it failed,
inspect dispatch/failure_*.json and the referenced child log first. Fix only the
concrete failure, retain the last full checkpoint. Then the same dispatcher
skips verified completed methods and resumes partial ones from full state:

```bash
cd /mnt/SABIDS-Net
source /root/miniconda3/etc/profile.d/conda.sh; conda activate myconda
set -euo pipefail
export LAC_BASE=/mnt/SABIDS-Net/runs/learnable_ac_v1/cloud_20261027_v1
export OLD_CACHE=/mnt/SABIDS-Net/runs/denoise_then_ac_v1/native_fold0_seed42_20261020_v1/cache
test -n "$LAC_BASE" && test -f "$OLD_CACHE/cache.json"
python -u tools/dispatch_learnable_ac_v1.py --mode run --base "$LAC_BASE" --input-cache "$OLD_CACHE"
```

Manual commands after the relevant group completes (or CPU postprocessing):

```bash
cd /mnt/SABIDS-Net
source /root/miniconda3/etc/profile.d/conda.sh; conda activate myconda
set -euo pipefail
export BASE=/mnt/SABIDS-Net/runs/learnable_ac_v1/cloud_20261027_v1
test -n "$BASE" && test -f "$BASE/resolved_config.json"
C=(--cache "$BASE/cache" --run "$BASE/training" --config "$BASE/resolved_config.json" \
   --research-mode development --fold 0 --seed 42)
M=(P3 F0 F1 F2 F5)
python tools/run_learnable_ac_v1.py "${C[@]}" --mode evaluate-best --device cuda --methods "${M[@]}"
python tools/run_learnable_ac_v1.py "${C[@]}" --mode evaluate-last --device cuda --methods "${M[@]}"
STAMP=$(date +%Y%m%d_%H%M%S)
R="$BASE/report_core_manual_$STAMP"; A="$BASE/atlas_core_manual_$STAMP"
python tools/run_learnable_ac_v1.py "${C[@]}" --mode summarize --methods "${M[@]}" --output "$R"
python tools/run_learnable_ac_v1.py "${C[@]}" --mode atlas --device cuda --methods "${M[@]}" --output "$A"
python tools/run_learnable_ac_v1.py "${C[@]}" --mode package --report "$R" --atlas "$A" --output "$BASE/learnable_ac_core_manual_$STAMP.zip"
```

Primary comparisons F5-F0, F5-F1, F5-F2; Holm correction over available primary
Vessel Dice tests. P3-F0 is the identical anchor consistency check, not an
independent training replicate. Best/last separate, native P0 threshold.5,
CSI=lower_boundary_mae in native px, all5 outer positions including0001/0004.
No independent generalization/blanket positive-result claim is permitted.

Formal missing-source audit and exact conditional D1/teacher/D2 retraining
commands remain in LEARNABLE_AC_V1.md. Formal noisy-only P0/P2 uses --no-mild;
absence of a legal historical D2 does not block that separate path. Neither
upstream retraining nor formal CV is automatically launched in this queue.
