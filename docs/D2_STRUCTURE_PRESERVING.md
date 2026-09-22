# D2 task-structure-preserving denoising

D2 is an opt-in denoising protocol. Legacy configurations do not instantiate
the D2 loss, frozen teacher, component strata, checkpoint selectors, or D2
preflight. D2 keeps the D1 network architecture and initializes from an
explicit, evidence-bound D1 checkpoint.

The first registered comparison is D20/D22/D24/D25 at seed 42. D20 contains
Charbonnier, MS-SSIM, multiscale gradient and Laplacian reconstruction terms.
D22 adds vessel/stroma/outside ROI terms and a configurable vessel-boundary
band. D24 adds CNR plus a separate frozen segmentation teacher. D25 adds clean
identity (through the existing independent clean forward), residual amplitude,
and structure-leakage constraints. D21/D23 remain registered intermediate
ablations and are run only if the seed-42 decision rule asks for them.

Vessel size Q33/Q67 and noisy local-contrast Q25 are frozen from transformed
development-train labels only. The primary component endpoint is the fraction
of GT components with at least 25% predicted coverage. `group_metrics.csv` is
the anatomical-position table; frame metrics are never treated as independent
positions.

Every fixed-budget D2 run saves `best_pixel.pth`, selected by validation PSNR,
and `best_task_preserving.pth`, selected among checkpoints within 0.2 dB of the
best PSNR using the preregistered frozen-teacher preservation score; ties choose
the earlier epoch. It also saves `last.pth`. Each checkpoint is hash-bound to
the immutable training-start inventory and selection audit.

The seed-42 gate is fail-closed. A passed gate authorizes preparation of
seed-43/44 configs, not training. Sealed test assets remain unused until all
methods, checkpoints, doses, thresholds, and postprocessing are frozen.

## Cloud execution order

Run from `/mnt/SABIDS-Net`. Resolve the already locked split-contract path by
matching its SHA; do not guess its filename:

```bash
export SABIDS_ROOT=/mnt/SABIDS-Net
cd "$SABIDS_ROOT"
export SABIDS_LOCK=Manifests/pku37_binary_v3/active_protocol_lock.json
export SABIDS_SPLIT_CONTRACT="$(python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/mnt/SABIDS-Net'); lock=json.loads((root/'Manifests/pku37_binary_v3/active_protocol_lock.json').read_text())
matches=[]
for p in (root/'Manifests').rglob('*.yaml'):
    if hashlib.sha256(p.read_bytes()).hexdigest()==lock['split_contract_sha256']: matches.append(p)
if len(matches)!=1: raise SystemExit(f'BLOCKED split-contract matches={matches}')
print(matches[0])
PY
)"
export D1_RUN=runs/adaptive_denoising/pku37_binary_v3/d1_repro_fold0_seed42
```

Bind and audit D1 best, then freeze train-only vessel strata:

```bash
python tools/bind_training_checkpoint_evidence.py --project-root . \
  --initial-inventory "$D1_RUN/training_asset_inventory_initial.json" \
  --checkpoint "$D1_RUN/best.pth" --history "$D1_RUN/history.csv" \
  --resolved-config "$D1_RUN/resolved_config.yaml" \
  --run-metadata "$D1_RUN/run_metadata.json" --protocol-lock "$SABIDS_LOCK" \
  --split-contract "$SABIDS_SPLIT_CONTRACT" \
  --output "$D1_RUN/checkpoint_binding_best.json"

python tools/audit_adaptive_denoising_baseline.py --project-root . --mode formal \
  --denoiser-checkpoint "$D1_RUN/best.pth" --protocol-lock "$SABIDS_LOCK" \
  --split-contract "$SABIDS_SPLIT_CONTRACT" --selection-rule best_validation_psnr \
  --training-asset-inventory "$D1_RUN/training_asset_inventory_initial.json" \
  --checkpoint-binding "$D1_RUN/checkpoint_binding_best.json" \
  --output reports/adaptive_denoising/d1_best_preflight_$(date +%Y%m%d_%H%M%S)

python tools/prepare_vessel_strata.py --project-root . \
  --config configs/adaptive_denoising/d1_repro_pku37_v3_seed42.yaml \
  --ring-width 3 --output reports/adaptive_denoising/d2_runtime/vessel_strata_train_v1.json
```

The required D1-best sensitivity arms are generated as one matched preparation;
its alpha-zero arm is therefore a matched noisy baseline rather than a reused
historical result:

```bash
python tools/prepare_dose_response_inputs.py --project-root . --mode formal \
  --denoiser-checkpoint "$D1_RUN/best.pth" --protocol-lock "$SABIDS_LOCK" \
  --split-contract "$SABIDS_SPLIT_CONTRACT" --selection-rule best_validation_psnr \
  --training-asset-inventory "$D1_RUN/training_asset_inventory_initial.json" \
  --checkpoint-binding "$D1_RUN/checkpoint_binding_best.json" \
  --curves d1 --alphas 0 0.25 0.5 1.0 --seeds 42 --budget pilot \
  --tag d1_best_sensitivity_v1 --device cuda
```

Before D24/D25, locate one legal validation-selected segmentation teacher and
derive its evidence JSON. Do not guess these two source paths. The audit derives
the selection rule and train/validation cohort identity from the immutable
sources rather than trusting free-form labels:

```bash
export TEACHER=/absolute/path/to/legal_segmentation_checkpoint.pth
export TEACHER_CONFIG=/absolute/path/to/the_same_run/resolved_config.yaml
export TEACHER_EVIDENCE=reports/adaptive_denoising/d2_runtime/teacher_evidence_v1.json
python tools/audit_d2_teacher.py --project-root . \
  --checkpoint "$TEACHER" --resolved-config "$TEACHER_CONFIG" \
  --protocol-lock "$SABIDS_LOCK" --output "$TEACHER_EVIDENCE"
test $? -eq 0 || exit $?
export TEACHER_SELECTION_RULE="$(python -c 'import json,os; print(json.load(open(os.environ["TEACHER_EVIDENCE"]))["selection_rule"])')"
export TEACHER_TRAINING_DATA="$(python -c 'import json,os; print(json.load(open(os.environ["TEACHER_EVIDENCE"]))["training_data"])')"
export STRATA=reports/adaptive_denoising/d2_runtime/vessel_strata_train_v1.json
COMMON="--project-root . --d1-checkpoint $D1_RUN/best.pth --d1-initial-inventory $D1_RUN/training_asset_inventory_initial.json --d1-checkpoint-binding $D1_RUN/checkpoint_binding_best.json --protocol-lock $SABIDS_LOCK --split-contract $SABIDS_SPLIT_CONTRACT --vessel-strata-definition $STRATA --teacher-checkpoint $TEACHER --teacher-evidence $TEACHER_EVIDENCE --teacher-selection-rule $TEACHER_SELECTION_RULE --teacher-training-data $TEACHER_TRAINING_DATA --teacher-split development_train_val"
python tools/prepare_d2_seed42.py $COMMON --mode smoke --arms D20 D25 --tag smoke_v1
python train.py --config runs/adaptive_denoising/d2_v1_launch_configs/d20_smoke_smoke_v1_seed42.yaml
python train.py --config runs/adaptive_denoising/d2_v1_launch_configs/d25_smoke_smoke_v1_seed42.yaml

python tools/prepare_d2_seed42.py $COMMON --mode overfit --arms D25 --tag overfit_v1
python train.py --config runs/adaptive_denoising/d2_v1_launch_configs/d25_overfit_overfit_v1_seed42.yaml
python tools/audit_d2_overfit.py \
  --run-dir runs/adaptive_denoising/pku37_binary_v3/d2_v1/overfit_overfit_v1/d25_seed42 \
  --output reports/adaptive_denoising/d2_runtime/d25_overfit_audit_v1.json

python tools/prepare_d2_seed42.py $COMMON --mode pilot --arms D20 D22 D24 D25 --tag pilot_v1
for ARM in d20 d22 d24 d25; do
  python train.py --config "runs/adaptive_denoising/d2_v1_launch_configs/${ARM}_pilot_pilot_v1_seed42.yaml"
done
```

After pilot training, run denoising diagnostics on validation only. The frozen
train-derived strata are supplied solely to partition residual leakage into
small/low-contrast GT regions; component recall still belongs to the matched
segmentation evaluation:

```bash
for ARM in d20 d22 d24 d25; do
  RUN="runs/adaptive_denoising/pku37_binary_v3/d2_v1/pilot_pilot_v1/${ARM}_seed42"
  python evaluate.py --config "runs/adaptive_denoising/d2_v1_launch_configs/${ARM}_pilot_pilot_v1_seed42.yaml" \
    --checkpoint "$RUN/best_task_preserving.pth" --split val \
    --d2-checkpoint-kind d2_task \
    --d2-checkpoint-binding "$RUN/checkpoint_binding_best_task_preserving.json" \
    --output "$RUN/validation_results" --tasks denoise --postprocess-modes p0 \
    --vessel-strata-definition "$STRATA" \
    --no-restore-original-geometry --d2-diagnostics --evaluate-clean-identity \
    --save-predictions
done
```

Prepare D2-task dose/matched-input segmentation only after its checkpoint
binding exists. Run D2-pixel separately by replacing the three marked values;
separate registries prevent curves from different checkpoints being spliced:

```bash
export D2_RUN=runs/adaptive_denoising/pku37_binary_v3/d2_v1/pilot_pilot_v1/d25_seed42
python tools/prepare_dose_response_inputs.py --project-root . --mode formal \
  --denoiser-checkpoint "$D2_RUN/best_task_preserving.pth" \
  --d2-checkpoint-kind d2_task \
  --d2-checkpoint-binding "$D2_RUN/checkpoint_binding_best_task_preserving.json" \
  --protocol-lock "$SABIDS_LOCK" --split-contract "$SABIDS_SPLIT_CONTRACT" \
  --curves d2_task --alphas 0 0.25 0.5 0.75 1.0 1.25 \
  --seeds 42 --budget pilot --tag d2_task_pilot_v1 --device cuda
```

Every downstream segmentation evaluation must add
`--vessel-strata-definition "$STRATA" --tasks layer vessel
--postprocess-modes p0 --no-restore-original-geometry`. This creates
`component_metrics.csv`, `contrast_metrics.csv`, and position-level
`group_metrics.csv` without changing legacy evaluator defaults.

These commands create fresh paths and refuse overwrite. The following is a
config-generation draft only. It is rejected unless
`audit_d2_seed42_gate.py` has produced a passed gate; it does not launch
training:

```bash
export SEED42_GATE=reports/adaptive_denoising/d2_runtime/seed42_gate_v1.json
for SEED in 42 43 44; do
  python tools/prepare_d2_seed42.py $COMMON --mode full --arms D20 D22 D24 D25 \
    --seed "$SEED" --seed42-gate "$SEED42_GATE" --tag formal_v1
done
# Stop here and obtain confirmation before running any generated full config.
```
