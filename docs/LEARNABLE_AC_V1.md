# learnable_ac_v1 — bounded task-specific AC (2026-10-26)

**2026-10-27 execution update:** follow `LEARNABLE_AC_MINIMAL_DEV.md` for the
authorized seed42 GPU queue, deployment checks and recovery commands. The broad
matrix examples below describe available research modes, not the current launch
scope. The earlier no-GPU statement records the original local-only milestone.

This is an opt-in experiment, **not evidence of efficacy**. No historical
trainer, AC implementation, label, cache or formal gate is modified. The user
currently has no GPU instance; only local synthetic CPU tests and local asset
inspection are possible. No long CUDA training is launched by this change.

## Audit and implemented scientific contract

- Existing input segmentation is `tools/downstream_seg_benchmark/model.py`:
  NAF stem/encoder shared by Layer/Vessel, separate task adapters, decoders and
  one-channel heads. **No layer boundary head or boundary loss exists in this
  path**. The historical full SABIDS model does have a boundary head; it is not
  this baseline. All new arms keep boundary auxiliary weight zero.
- D2 has a different model instance/weights from the segmentation encoder.
  `forward_denoise_only()['denoised']` is clipped restoration, not residual.
  D2 is frozen and offline. This is NOT joint end-to-end D2 optimization.
- `NativeStableAC` matches ac_r2's float64 reverse valid-neighbour trapezoidal
  integral on native complete valid depth; no padding pixels contribute.
  `N` is valid samples per A-line; denominator is `I + train_regularizer` in
  gray^p * depth-pixel units. Native depth is not GT-layer-relative depth.
- c is r2's `safe_output`, after linear train-window mapping, gain cap and
  base reliability. z0 mixes exactly once: `.5*m + .5*c`, computed in float64
  then float32. `m=clip(x-.25*(x-D2(x)),0,1)`. Control is evaluated AFTER common
  geometry, but analytical integration is always BEFORE common geometry.
- Fixed p=2, beta=.5, gamma=1, denominator, W, gain cap6, base reliability .2.
  Only train-mild images fit these statistics. One deterministic first frame
  per train position, equal position contribution. The SAME theta is applied
  to AC(noisy) and AC(mild). No per-image UI parameters, labels or clean fitting.
- Extra rho = valid * I/(I+regularizer) * (1-.5*high_clip). It limits EXTRA
  adjustment, does not modify c or reapply base reliability. It is not a dark
  pixel test. F4 sets extra rho=1 on valid pixels, retaining base protection.
- Three context scales (16/32/64), GroupNorm, four normalized Gaussian depth
  bases (centres 0,1/3,2/3,1; sigma .25). Head predicts low-resolution lateral
  coefficients, interpolated linearly; depth combination then bounded by tanh.
  `delta=.25` default; `.5` is explicit inner-only preregistered sensitivity,
  requiring a new config/cache/run. There is no multiplied zero scale.
- Detached coarse guidance comprises Layer/Vessel probabilities and binary
  entropy. No GT is a forward argument. No coarse-Layer vessel clipping.
- Refined task images `z_t=z0+rho*Delta_beta_t*(c-m)`, effective beta
  `.5+rho*Delta_beta_t`. F5/F1 do one coarse encoder pass plus two task encoder
  passes sharing S0 weights. S0 is eval/frozen, refined image forward is NOT
  inside no_grad during training. Extra encodes and both heads are profiled.
- F1 uses exactly the same context + coefficient network/parameter count as
  F5, but adds bounded low-rank logit residuals (4*adjustment), not image AC.
  This controls extra capacity/update budget; neither is assumed beneficial.

## Matrix (no implicit Cartesian-product training)

| Family | ID | Input / refinement |
|---|---|---|
| first input batch | P0 | noisy |
| | P1 | mild |
| | P2 | AC(noisy), same theta as P3 |
| | P3 | AC(mild), common S0 anchor |
| | P4 | monotone pooled train quantile LUT T(mild), no depth/GT/clean |
| optional | P5 | noisy + AC(mild), existing equal-capacity dual fusion |
| diagnostic | P6 | clean/reference, explicitly not deployable noisy inference |
| first refinement batch | F0 | frozen P3 anchor, no updates |
| | F1 | matched context logit refiner |
| | F2 | two global learned beta, zero raw coefficients; no extra spatial rho |
| | F5 | independent smooth Layer/Vessel control + rho |
| second mechanism batch | F3 | shared coefficient head |
| | F4 | no extra rho (base protection unchanged) |
| | F6 | probability/entropy channels zeroed, same context capacity |

All task heads zero-output initialized. Initial/off exactly equals anchor.
First-step context gradients may be zero; coefficient-head gradients start,
and context gradients must become nonzero after coefficient updates.

## Labels, supervision and selection

Frozen snapshot: `runs/ac_vessel_labelsv2/snapshot_20261016_v1`, identity
`8bb6d39bda627d7204d023f7b2db0b7a0a8a5cb5be247f5e7b8628a00203e5b7`.
20 confirmed positions, 929 repeated frames. Fold0 train12/548,
inner3/141, outer-development5/240. Position grouping because patient IDs are
unavailable. Same registered-coordinate `user_confirmation` remains valid;
no new registration or per-frame reviewer proof is requested.

Vessel positives outside old Layer remain (six reviewed pixels), Layer conflict
excluded rather than treating Vessel as negative. Supervision masks explicitly
intersect coverage and exclude uncertain for BOTH tasks, plus original validity
and geometric padding. No old run is reused merely by matching an arm name.
This uniform masking is a new namespace, not a rewritten historical protocol.

Layer masked BCE+Dice. Vessel BCE+Dice in GT Layer OR Vessel; outside-negative
BCE .5 excludes true Vessel positives. No restoration/RMAC/pseudo/UGBI, no
additional vessel/boundary losses. Controller depth second derivative squared,
lateral first derivative squared and amplitude squared penalties; raw/weighted
terms separately logged. Smooth and anchor lambda both .001.

Base60 / refinement20 epochs, batch1 accumulation2, AdamW5e-5, cosine fixed
budget, AMP=false, clip1. All arms use independent seeded position-uniform
sampling/flip plans and paired base initialization. F1/F2/F5 share EXACT P3 best
anchor and update budget. Epoch0 saved and eligible for earliest-tie selection.
Monitor ONLY inner position-equal Vessel soft Dice, strictly greater; both
tasks share one checkpoint. Best primary and last sensitivity never mixed.

## Development vs formal

Historical D25 task-best (20epoch pilot, selected epoch2) has SHA
`edc5b1987f189566666d1eb0b9f03701b9ff28409c2a30ad0d34c8eb5758e343`.
Historical D2/teacher trained on 0001/0004/0032/0041 and selected on0040:
ALL FIVE current outer positions were exposed. Local copied provenance is
historical evidence, not a fresh cloud checkpoint inspection. D1 initialization
lineage must also be inspected before any formal claim. Missing local full
D2 checkpoint is NOT substituted by synthetic or smoke weights.

`--research-mode development` retains explicit source disclosures.
`formal` requires supplied D1/teacher/D2 actual checkpoint/config SHAs,
actual embedded/resolved config equality, nonempty manifest train/selection,
no outer exposure and audited D2 binding + initialization lineage. Missing,
unknown or contradictory evidence fails closed. Existing historical formal
gates are unchanged. A formal S0 is trained from scratch only on the legal cache
train/inner. Its P3 cache/config/checkpoint identity is checked for refinement.

No legal mild/D2? `audit/prepare --no-mild` independently allows P0/P2, fits AC
on current fold train noisy, and BLOCKS P1/P3/refinement. This is explicitly a
different noisy-only study; do not mix it into mild method comparisons.
Observed positions are never rebranded as a new blind test set.

Formal target is four grouped folds, segmentation seeds42/43/44. Each fold
may reuse one legal D2 seed, clearly reporting the restricted upstream
randomness. Each seed/method/location must have exactly one OOF fold. Formal
execution is not ready until all provenance and fold-native assets exist.

## CLI and staged commands (Linux platform terminal)

The entry's actual `--help` exposes all modes below. First deploy this task's
source files without overwriting concurrently running training source. Do not
run training until GPU occupancy/status and audit have passed.

```bash
cd /mnt/SABIDS-Net
source /root/miniconda3/etc/profile.d/conda.sh
conda activate myconda
set -euo pipefail
export SABIDS_ROOT=/mnt/SABIDS-Net
export LAC_CACHE="$SABIDS_ROOT/runs/learnable_ac_v1/dev_fold0_v1/cache"
export LAC_RUN="$SABIDS_ROOT/runs/learnable_ac_v1/dev_fold0_v1/seed42"
export LAC_CONFIG="$SABIDS_ROOT/runs/learnable_ac_v1/configs_v1/development_fold0_seed42.json"
export OLD_CACHE="$SABIDS_ROOT/runs/denoise_then_ac_v1/native_fold0_seed42_20261020_v1/cache"
test -n "$LAC_CACHE" && test -n "$LAC_RUN" && test -n "$OLD_CACHE"
test -f "$OLD_CACHE/cache.json"
python tools/run_learnable_ac_v1.py --help
python tools/run_learnable_ac_v1.py --mode status --run "$LAC_RUN"
python tools/run_learnable_ac_v1.py --mode generate-configs \
  --run "$LAC_RUN" --output runs/learnable_ac_v1/configs_v1
test -f "$LAC_CONFIG"
python tools/run_learnable_ac_v1.py --mode audit --input-cache "$OLD_CACHE" \
  --run "$LAC_RUN" --output runs/learnable_ac_v1/dev_fold0_v1/audit.json
python tools/run_learnable_ac_v1.py --mode prepare --input-cache "$OLD_CACHE" \
  --run "$LAC_RUN" --output "$LAC_CACHE"
COMMON=(--cache "$LAC_CACHE" --run "$LAC_RUN" --config "$LAC_CONFIG" --fold 0 --seed 42)
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode preview \
  --output runs/learnable_ac_v1/dev_fold0_v1/preview_before_training || test "$?" -eq 2
# preview returns2 with an explicit PENDING TRAINING list when models absent.
# This is not a training/cache failure. Inspect its atlas_status.json.
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode cuda-check \
  --device cuda --output runs/learnable_ac_v1/dev_fold0_v1/check_cuda_512
```

Prepare produces a NEW cache, never reflags20-preview arrays as full cache.
It may take significant I/O/disk space; failed outputs are preserved. Use a new
output identity after correction. It refits declared train-mild statistics in
this namespace and records values; these need not equal the historical GPU
cache's float-rounded statistics. P2/P3 ALWAYS share the new theta.
It can also compute native mild directly from an actual audited D2, without
depending on a historical fold0 cache. This is the route for legal future
fold1–3 preprocessors; it freezes/evaluates D2, crops stride padding before AC,
and checks the denoiser state hash before/after preparation:

```bash
: "${FOLD_UPSTREAM_JSON:?actual fold-native upstream checkpoint records}"
: "${FOLD_CACHE:?new cache directory}"
: "${FOLD_INDEX:?0 to 3}"
test -f "$FOLD_UPSTREAM_JSON"
python tools/run_learnable_ac_v1.py --mode audit --research-mode formal \
  --fold "$FOLD_INDEX" --upstream "$FOLD_UPSTREAM_JSON"
python tools/run_learnable_ac_v1.py --mode prepare --research-mode formal \
  --fold "$FOLD_INDEX" --upstream "$FOLD_UPSTREAM_JSON" --device cuda \
  --output "$FOLD_CACHE"
```

`upstream_TEMPLATE.json` requires D1/teacher immutable initial-inventory SHAs as
well as checkpoint/config identity. A manifest path alone is not provenance.

Explicit serial training (not executed by implementation/test delivery):

```bash
# In a new terminal repeat variable/environment initialization above.
nohup bash tools/run_learnable_ac_modelwhale.sh \
  > "$LAC_RUN.launch.log" 2>&1 < /dev/null &
# Driver owns a flock, stops on failure; PID/exit record and tee logs persist
# under $LAC_RUN/logs. SSH disconnect does not rerun a job.
```

Manual equivalent, supports completed skip / partial full resume / new arms:

```bash
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode resume --device cuda \
  --methods P0 P1 P2 P3 P4 --queue
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode resume --device cuda \
  --methods F0 F1 F2 F5 --queue
# Optional SECOND batch, never automatically included:
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode resume --device cuda \
  --methods F3 F4 F6 --queue
```

Resume validates model, AdamW steps, scheduler, epoch/update budget, RNG, anchor,
cache, code and config. It does not infer completion from history row count or
best.pth existence. Best-ahead-of-last transaction disagreement stops for
inspection, never rewrites best to force continuation. No mid-epoch state is
claimed: an interrupted epoch replays from the last COMPLETE epoch boundary.
An immutable study-level plan also rejects different budgets, masks/cache,
seeds or optimizers silently introduced into another method in the same run.

```bash
METHODS=(P0 P1 P2 P3 P4 F0 F1 F2 F5)
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode evaluate-best --device cuda --methods "${METHODS[@]}"
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode evaluate-last --device cuda --methods "${METHODS[@]}"
STAMP=$(date +%Y%m%d_%H%M%S)
REPORT="$LAC_RUN/report_$STAMP"
ATLAS="$LAC_RUN/atlas_$STAMP"
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode summarize --output "$REPORT"
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode atlas --device cuda --output "$ATLAS"
python tools/run_learnable_ac_v1.py "${COMMON[@]}" --mode package \
  --report "$REPORT" --atlas "$ATLAS" --output "$LAC_RUN/learnable_ac_light_$STAMP.zip"
python tools/run_learnable_ac_v1.py --mode status --run "$LAC_RUN"
tail -n 20 "$LAC_RUN"/logs/queue_*.log
```

Multiple explicitly supplied `--runs` can be summarized after four legal folds
and three seeds, with `--methods` restricting the preregistered core. Incomplete
folds are never filled with zero or called complete CV. Development/formal
mixing is rejected. Best/last CSVs, frame/position/seed tables, OOF, paired gains,
position/patient bootstrap and primary Vessel-Dice Holm correction are separate.
CI averages seeds before clustering; small n is explicitly limited. The
requested-scope complete marker alone is not proof of complete formal CV.

## Upstream formal D1/teacher/D2 retraining

Do NOT reuse the historical all-outer-exposed D25 checkpoint in formal mode.
Generate fold-local source manifests and provenance for fresh D1, teacher and
D2 first, using current train/inner (not outer). The existing upstream tools
have their own evidence gates; these are not weakened here. Exact existing
entry points to inspect before binding newly created assets:

```bash
python tools/prepare_d2_teacher.py --help
python tools/prepare_d2_seed42.py --help
python train.py --help
```

`generate-configs` creates the downstream24 configs and an explicit
upstream_TEMPLATE.json. It does NOT invent legal upstream files or manufacture
a historical binding. Populate actual paths/SHAs only AFTER fold-local runs
are complete, then `audit --research-mode formal --upstream ACTUAL.json`.

`tools/prepare_learnable_ac_upstream.py` is a config-only adapter to the
existing evidence-gated D1/teacher/D2 system. It binds the **selected fold's**
manifest explicitly; the old D2 launcher retained a hardcoded legacy manifest
and therefore must not be assumed fold-native. D1 starts from scratch. Teacher
delegates to the unchanged formal teacher preparer (batch1 accumulation2 via
the new template). D2 keeps the unchanged D1 formal-preflight and teacher
evidence gates, produces a20epoch D25 pilot with task-best selection and actual
fold-native train/inner. It does not fabricate the required upstream lock,
label inventory, split contract or pre-optimizer checkpoint evidence.

Once those fold-specific assets really exist, these are exact CLI commands;
all variables deliberately fail closed if unset (no guessed binding filenames):

```bash
: "${FOLD_PROTOCOL_LOCK:?actual new fold lock}"
: "${FOLD_SPLIT_CONTRACT:?actual new fold split contract}"
: "${FOLD_UPSTREAM_DIR:?new isolated runs/learnable_ac_v1 upstream directory}"
test -f "$FOLD_PROTOCOL_LOCK" && test -f "$FOLD_SPLIT_CONTRACT"
python tools/prepare_learnable_ac_upstream.py --stage d1 \
  --protocol-lock "$FOLD_PROTOCOL_LOCK" --split-contract "$FOLD_SPLIT_CONTRACT" \
  --output "$FOLD_UPSTREAM_DIR/d1.yaml" --run-dir "$FOLD_UPSTREAM_DIR/d1_run"
# Explicit user-initiated training, never automatically run by preparation:
python train.py --config "$FOLD_UPSTREAM_DIR/d1.yaml"

# Bind the newly trained D1 best using the existing evidence/binding workflow.
# The binding is NOT replaced by last weights or an old D1 from another fold.
: "${FOLD_D1_CHECKPOINT:?actual completed best.pth}"
: "${FOLD_D1_INITIAL:?actual immutable pre-optimizer inventory}"
: "${FOLD_D1_BINDING:?actual audited best-checkpoint binding}"
test -f "$FOLD_D1_CHECKPOINT" && test -f "$FOLD_D1_INITIAL" && test -f "$FOLD_D1_BINDING"
python tools/prepare_learnable_ac_upstream.py --stage teacher \
  --protocol-lock "$FOLD_PROTOCOL_LOCK" --split-contract "$FOLD_SPLIT_CONTRACT" \
  --d1-checkpoint "$FOLD_D1_CHECKPOINT" --d1-initial-inventory "$FOLD_D1_INITIAL" \
  --d1-checkpoint-binding "$FOLD_D1_BINDING" \
  --output "$FOLD_UPSTREAM_DIR/teacher.yaml" --run-dir "$FOLD_UPSTREAM_DIR/teacher_run"
python train.py --config "$FOLD_UPSTREAM_DIR/teacher.yaml"

: "${FOLD_TEACHER_CHECKPOINT:?actual new teacher best}"
: "${FOLD_TEACHER_EVIDENCE:?actual teacher native binding}"
: "${FOLD_TRAIN_STRATA:?actual train-only D2 strata definition}"
test -f "$FOLD_TEACHER_CHECKPOINT" && test -f "$FOLD_TEACHER_EVIDENCE" && test -f "$FOLD_TRAIN_STRATA"
python tools/prepare_learnable_ac_upstream.py --stage d2 \
  --protocol-lock "$FOLD_PROTOCOL_LOCK" --split-contract "$FOLD_SPLIT_CONTRACT" \
  --d1-checkpoint "$FOLD_D1_CHECKPOINT" --d1-initial-inventory "$FOLD_D1_INITIAL" \
  --d1-checkpoint-binding "$FOLD_D1_BINDING" \
  --teacher-checkpoint "$FOLD_TEACHER_CHECKPOINT" --teacher-evidence "$FOLD_TEACHER_EVIDENCE" \
  --vessel-strata-definition "$FOLD_TRAIN_STRATA" \
  --output "$FOLD_UPSTREAM_DIR/d2.yaml" --run-dir "$FOLD_UPSTREAM_DIR/d2_run"
python train.py --config "$FOLD_UPSTREAM_DIR/d2.yaml"
```

This upstream adapter's config binding is CPU-tested; real gated preparation
and training require the currently missing assets and are NOT claimed passed.
It refuses existing config/run outputs. Legacy D2 does not support full-state
resume; do not advertise its weights-only restart as equivalent continuation.
The NEW segmentation/refinement engine does support full-state continuation.

## Interpretation, exports and cost

Primary F5-P3 and F5-F1; incremental AC P3-P1, before-denoise value P3-P2;
brightness P3-P4. Shared head/guidance mechanisms require F3/F6 retraining.
Negative results and all positions including0001/0004 remain. Gray PSNR is not
the AC optimization objective. Without brightness control, do not claim
mechanistic superiority to gray enhancement. Outer never chooses delta/epoch.

Atlas writes native float/probabilities/masks, beta/delta/rho separately, common
GT-Layer crops, frozen inputs and matched errors. Raw lateral coefficients are
not mislabeled as effective beta. Actual network tensors are saved in
`*_model_grid.npy`; dewarped visualizations are clearly `native_restored`.
Light ZIP excludes weights/optimizer/full cache/full probabilities; representative
floats only for0001/0004/0005. CRC and SHA receipt are generated. Synthetic ZIP
is explicitly ENGINEERING ONLY and cannot become a scientific primary table.

Budget: default5base*60 +3trained refinements*20 =360epochs,108000 updates when
600draws/epoch; F0 is zero updates. Historical A30 base cost about5.5h/60epochs
is only a rough reference, not a measurement of the new controllers. CUDA
memory/time remain unmeasured locally. `profiles.csv` records actual parameter
counts, encoder calls, cold full forward time, peak CUDA allocation and conv MACs
(not a fabricated total FLOPs number). D2/native AC offline cost is separate.
The actual default architecture has5,629,714 S0 parameters; F1/F5 each train
74,056 extra parameters (exact capacity match), F2 trains2; F0 trains0.
Tiny synthetic CPU profiles are not512 CUDA latency measurements.

Evaluation ROI Dice uses the fixed GT Layer, while training Vessel ROI is
Layer OR Vessel. Full Vessel metrics retain the six exterior positives. Depth
Recall bins are GT-Layer-relative shallow/middle/deep thirds on the native grid
(read-only evaluation, NOT controller depth). Exterior positives remain in
full Dice/Recall but have no GT-layer depth bin. Boundary/thickness errors
are native px with missing predicted boundaries penalized by image height;
annotation-incomplete columns are omitted, not fabricated. Difficulty thresholds
are train-native Q25 area/absolute local contrast from ORIGINAL noisy, never
redefined by each AC result. F1 is specifically the declared smooth low-rank
logit refiner, not a claim about every possible unconstrained refiner design.

Windows local CPU verification (new temp path, no old asset cleanup):

```powershell
$env:TEMP='D:/lacv1tmp'; $env:TMP='D:/lacv1tmp'
python -m pytest -q tests/test_learnable_ac_v1.py --basetemp D:/lacv1tmp/manual_test
python tools/run_learnable_ac_v1.py --mode cpu-smoke --threads 1 `
  --output runs/learnable_ac_v1/cpu_smoke_manual_v1
```

No GPU instance is currently rented. No current SSH download endpoint is
claimed. Once the platform supplies a real endpoint, use its displayed host
and port with `scp -P PORT root@HOST:ACTUAL_ZIP .`, then verify SHA256 with
`Get-FileHash`. Historical26047 is not assumed reachable.
