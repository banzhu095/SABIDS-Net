# SABIDS-Net experiment log

This is the durable experiment ledger. Update it after an experiment finishes or
when a run is intentionally stopped. Do not rewrite old outcomes to match a new
hypothesis; append a new entry and link the relevant Git commit/config.

## 1. Logging standard

Every nontrivial run should record:

- date and short experiment ID;
- research question;
- Git commit (`git rev-parse --short HEAD`);
- machine, GPU, CUDA/PyTorch environment;
- exact command;
- resolved config path;
- manifest, fold, split unit, dataset report checksum or timestamp;
- input size, batch size, accumulation, seed, and epochs;
- initialization and whether training was fresh/resumed/reused;
- best epoch, monitor, and checkpoint;
- validation-selected thresholds;
- group-level metrics and number of independent groups;
- qualitative failure modes;
- conclusion and next decision.

Never describe `Reuse existing checkpoint` as a newly trained experiment.
Syntax checks, unit tests, and smoke tests demonstrate execution integrity, not
scientific effectiveness.

## 2. Current status summary

| Item | Status as of 2026-08-27 |
| --- | --- |
| Code line | v0.2.0; initial GitHub import commit observed on server: `a37f510` |
| Local/cloud Git workflow | connected; local pushes to GitHub, Juchiyun pulls read-only |
| Public data preparation | last report: Duke17=17 pairs, Duke28=28 pairs, PKU37=1,734 rows/37 positions, 13 labelled positions |
| Stage 1 | an earlier denoising checkpoint exists and may be reusable only after signature/size verification |
| Stage 2 | must be freshly retrained with v0.2 vessel losses/checkpoint monitor |
| Joint | old checkpoint is diagnostically useful but incompatible with the intended v0.2 retraining |
| Private Stage 5 | deferred until the public Stage 2/Joint baseline is stable |
| Main unresolved risk | vessel prediction collapsing toward the entire choroidal layer |

## 3. Historical diagnostic entry

### EXP-DIAG-OLD-JOINT — earlier Stage 2 versus reused Joint checkpoint

- Date reviewed: 2026-08-25
- Purpose: diagnose why vessel prediction resembled layer segmentation.
- Training status: the reported Joint command printed `Reuse existing
  checkpoint`; it did not retrain the model.
- Comparability warning: the earlier Stage 2 and Joint runs may have used
  different target sizes. Treat the numbers as diagnostic trends only.

| Metric | Stage 2 | Joint | Interpretation |
| --- | ---: | ---: | --- |
| Layer Dice | 0.8414 | 0.8045 | Joint degraded layer overlap |
| Layer Precision | 0.9639 | 0.7739 | layer expansion |
| Layer Recall | 0.7484 | 0.8394 | wider predicted layer |
| Vessel Dice | 0.6259 | 0.5205 | clear vessel degradation |
| Vessel Precision | 0.5425 | 0.3801 | many additional false positives |
| Vessel Recall | 0.7413 | 0.8260 | over-expansion increased recall |
| Predicted vessel fraction | 0.5604 | 0.7322 | severe overprediction |
| True vessel fraction | 0.4161 | 0.4152 | reference prevalence stable |

Decision: do not tune only the binary threshold or continue the old Joint
checkpoint. Retrain Stage 2 and Joint after the v0.2 loss/sampling/memory fixes.

Implemented response in v0.2:

- Tversky FP/FN weights changed to 0.6/0.4;
- added vessel BCE, stroma negative, and area-ratio terms;
- true-layer containment when labels exist;
- 35% vessel-labelled step sampling;
- RMAC weight reduced to 0.15;
- Joint LR reduced to `5e-5` and ramp extended to 30 epochs;
- checkpoint monitor changed to `vessel_soft_dice`;
- memory-safe repeat/clean teacher forwarding;
- validation threshold calibration and layer-vessel similarity diagnostics;
- incompatible checkpoint protection, `--resume`, `--skip-test`, and safe
  `--force` archiving.

## 4. Execution checks already observed

### CHECK-CPU-SMOKE

- Date: 2026-08-25
- Environment: Juchiyun `myconda`, CPU fallback.
- Outcome: tiny denoising smoke run completed an epoch; a log line reported
  `Epoch 001 monitor=31.28111`.
- Interpretation: the reduced pipeline could execute after dependency fixes.
  This is not a publishable PSNR result and should not be compared with the full
  model.

### CHECK-GIT-SYNC

- Date: 2026-08-27
- Server repository: `/mnt/SABIDS-Net`
- Observed commit before later edits: `a37f510 Initial import of SABIDS-Net
  v0.2.0`.
- Runtime ignored directories confirmed: `Data`, `Label`, `Manifests`, `runs`.
- Server worktree was clean and `git pull --ff-only origin main` reported up to
  date.
- File traversal observed 1,861 files under `Data` and 52 under `Label` after
  migration. `du -sh` reported zero because of the mounted filesystem's size
  accounting; file traversal, not that `du` value, confirmed accessibility.

## 5. Next controlled runs

### EXP-PUB-F0-S2-V02 — fresh public Stage 2

Research question: do the v0.2 supervised vessel constraints produce a
well-separated vessel probability map before Joint interaction?

Command:

```bash
cd /mnt/SABIDS-Net

python run_current_pipeline.py \
  --project-root /mnt/SABIDS-Net \
  --fold 0 \
  --stages segment \
  --device cuda \
  --batch-size 1 \
  --target-height 512 \
  --target-width 512 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --skip-test \
  --force
```

Go/no-go checks before Joint:

- vessel probability is not a near-copy of layer probability;
- `vessel_precision` is no longer dramatically below Recall;
- predicted vessel fraction approaches labelled prevalence;
- probability maps preserve multiple lumen-like regions rather than a solid
  layer fill;
- layer Dice, boundary MAE, and geometry remain acceptable;
- resolved config and Stage 1 initialization are compatible.

Result: pending.

### EXP-PUB-F0-JOINT-V02 — fresh public Joint

Run only after the Stage 2 gate passes.

```bash
python run_current_pipeline.py \
  --project-root /mnt/SABIDS-Net \
  --fold 0 \
  --stages joint \
  --device cuda \
  --batch-size 1 \
  --target-height 512 \
  --target-width 512 \
  --gradient-accumulation-steps 2 \
  --num-workers 4 \
  --skip-test \
  --force
```

If 512x512 still causes OOM, rerun both Stage 2 and Joint at 384x384 so their
comparison and initialization chain remain consistent. Do not mix a 384 Stage 2
checkpoint with a 512 Joint claim.

Result: pending.

### EXP-PUB-F0-THRESHOLD-TEST — calibration and locked test

Calibrate only on validation data:

```bash
python tools/calibrate_vessel_threshold.py \
  --config runs/current/stage4_joint_fold0/resolved_config.yaml \
  --checkpoint runs/current/stage4_joint_fold0/best.pth \
  --split val \
  --output runs/current/stage4_joint_fold0/threshold_calibration
```

After recording the selected threshold, apply it once to test:

```bash
python evaluate.py \
  --config runs/current/stage4_joint_fold0/resolved_config.yaml \
  --checkpoint runs/current/stage4_joint_fold0/best.pth \
  --split test \
  --vessel-threshold <VALIDATION_SELECTED_THRESHOLD> \
  --output runs/current/stage4_joint_fold0/test_results_calibrated \
  --save-predictions
```

Result: pending.

## 6. Planned ablation queue

Run each ablation with the same fold, input size, Stage 2 initialization, epoch
budget, data sampling, and validation policy as the full Joint model.

| Priority | Ablation | Main diagnosis |
| ---: | --- | --- |
| 1 | no RMAC | whether repeat consistency drives stable over-wide vessels |
| 2 | no UGBI | whether cross-task feature interaction causes Joint regression |
| 3 | denoise-to-seg only | whether restoration context helps segmentation |
| 4 | seg-to-denoise only | whether anatomy helps restoration without feedback |
| 5 | no uncertainty | value of uncertainty gates versus plain interaction |
| 6 | no area/stroma constraints | causal effect of v0.2 vessel safeguards |
| 7 | no pseudo-label | private sparse-label benefit/risk |
| 8 | vessel sampling 0.20/0.35/0.50 | sensitivity to labelled update fraction |

Do not launch the full ablation matrix until the full model and no-interaction
baseline can train reproducibly on fold 0.

## 7. Fold result table

Populate one row per independent fold after validation threshold selection.

| Model | Fold | Git commit | Input | Threshold | N groups | Layer Dice | Vessel Dice | Vessel Precision | Vessel Recall | Vessel area MAE | PSNR gain | Notes |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full v0.2 | 0 | pending | 512x512 | pending | pending | pending | pending | pending | pending | pending | pending | fresh retrain required |
| Full v0.2 | 1 | pending | 512x512 | pending | pending | pending | pending | pending | pending | pending | pending | |
| Full v0.2 | 2 | pending | 512x512 | pending | pending | pending | pending | pending | pending | pending | pending | |
| Full v0.2 | 3 | pending | 512x512 | pending | pending | pending | pending | pending | pending | pending | pending | |
| Full v0.2 | 4 | pending | 512x512 | pending | pending | pending | pending | pending | pending | pending | pending | |

## 8. Copyable experiment entry template

```markdown
### EXP-YYYYMMDD-NAME

- Question:
- Git commit:
- Machine/GPU:
- Python/PyTorch/CUDA:
- Manifest and dataset report:
- Split unit/fold/seed:
- Command:
- Resolved config:
- Initialization:
- Fresh/resume/reuse:
- Input/batch/accumulation:
- Epochs and best epoch:
- Monitor and best value:
- Validation-selected thresholds:
- Checkpoint:
- Independent validation/test groups:
- Group-level metrics:
- Frame-level metrics:
- Qualitative observations:
- Failure cases:
- Conclusion:
- Next action:
```

## 9. Resume and recovery notes

- `--resume` requires a compatible v0.2 `last.pth` and continues after the last
  complete epoch.
- `--force` starts the selected stage from its preceding stage checkpoint and
  archives recognized old artifacts under `archive_<timestamp>`.
- Old Joint checkpoints without a compatible resolved config must not be
  resumed.
- Never use `git reset --hard` or delete `runs/` to solve a training mismatch.
  Preserve the evidence, choose a new output directory or use safe archiving,
  and document the decision.

## 10. Code diagnosis: Stage 2 saturation and non-finite loss

### CODE-20260827-STAGE2-STABLE-STROMA

- Trigger: a fresh Stage 2 log stayed at
  `vessel_soft_dice=0.59735` through epochs 12--19 and printed `loss=nan`
  during epoch 20.
- Status: implementation correction completed locally; cloud retraining result
  pending. This entry is not an experimental performance claim.
- Confirmed code defect: the stroma negative term used a
  sigmoid/log/clamp expression whose gradient becomes zero after a positive
  vessel logit saturates sigmoid to one.
- Correction: use `softplus(vessel_logit)`, promote custom segmentation losses
  to FP32, run Stage 2 with LR `5e-5` and AMP disabled, fail fast on non-finite
  loss, and log collapse diagnostics during validation.
- Data audit: the 13 `Label/voc_jpg` references and PKU clean images had matching
  640x640 geometry and correlations approximately 0.991--0.995. A sampled
  global phase-shift audit of repeats was generally below two source pixels;
  this does not prove local vessel registration, but did not support gross
  global misregistration as the primary cause.
- Checkpoint rule: do not resume or reuse the affected Stage 2 checkpoint. The
  resolved loss now carries `definition_version: stable-stroma-fp32-v1` so the
  pipeline rejects the old objective.
- Required cloud outcome: pending fresh fold-0 Stage 2 run, probability-map
  review, and group-level validation diagnostics.

### CODE-20260827-STAGE2-E0-E3-DIAGNOSTICS

- Status: implementation and local CPU smoke verification completed; formal
  cloud training result pending. This is not a performance claim.
- Motivation: the numerically stable Stage 2 rerun reportedly peaked near
  epoch 8 and then degraded, suggesting optimization/representation drift in
  addition to the corrected non-finite loss.
- E0: epoch-0 and no-augmentation train/validation diagnostics, independent
  group counts, per-group CSVs, whole-layer/empty baselines, manifest SHA256,
  checkpoint lineage and a 2--4 training-group overfit mode.
- E1: freeze the complete Stage 1 upstream and denoising function, disable
  UGBI, enforce frozen modules in evaluation mode and fail if a fixed denoising
  probe drifts by more than `1e-6`.
- E2: open only detached denoising-to-segmentation UGBI receiver parameters;
  segmentation-to-denoising remains disabled and injection magnitudes/scales
  are logged.
- E3: optional ROI masked BCE+Dice vessel supervision with explicit unknown
  pixels; the outside containment constraint remains active.
- Outputs: `stage2_segment_safe_fold0`, `stage2_segment_d2s_fold0`, and
  `stage2_segment_roi_fold0`; these must not initialize Joint until the Stage 2
  gate is reviewed.
- Local verification: one real-manifest CPU smoke epoch of E1 completed and
  reported denoising probe maximum absolute drift `0.0`; this verifies the code
  path only, not model quality.
