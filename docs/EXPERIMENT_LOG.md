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

### CODE-20260827-STAGE2-E3B-OUTSIDE-BCE

- Evidence source: user-supplied analysis of complete E0/E1/E2/E3 history CSVs;
  no held-out test result was supplied.
- Observed decision: retain E1 as the main reference and E2 as interaction
  control. E3 preserved validation ROI Dice near `0.706` but full-image Dice
  fell to about `0.458`, with validation layer-exterior prediction fraction
  reported near `41.7%` of predicted vessel pixels.
- Implementation: new `roi_outside`/E3b variant adds only per-image FP32
  outside-GT-layer BCE(target=0) in logit space, default weight `0.5`; padding
  and unknown vessel pixels are excluded and empty outside regions are skipped.
- Logging: every active objective records raw, weight and weighted values. D2S
  records relative injection RMS, interaction/scale gradients, scale updates
  and same-checkpoint disable-D2S sensitivity.
- Diagnostics: full/ROI/oracle/soft-gate metrics, three predicted-layer error
  partitions, fixed-frame masks and error overlays; formulas are exported.
- Reproducibility: run metadata includes Stage 1 checkpoint SHA256,
  manifest/split hashes, effective group IDs, best epoch, thresholds and best
  checkpoint SHA256. Stage 1 train versus Stage 2 val/test overlap is fatal.
- Local verification: focused tests and a one-epoch real-manifest CPU E3b smoke
  passed; formal 512x512 cloud result remains pending.

### EXP-DIAG-20260827-E3B-NEW-PROTOCOL

- Evidence source: user-supplied 60-row E3b history CSV; best epoch 55 selected
  by `val_vessel_soft_dice`. No checkpoint, resolved config, per-group CSV or
  run metadata accompanied the history in this workspace.
- Reported protocol: 13 train-eval groups and 3 validation groups, versus 8/2
  in historical E1--E3. Therefore the apparent E3-to-E3b gain is not a valid
  single-factor outside-BCE estimate.
- Reported best metrics: validation vessel hard/soft Dice `0.73796/0.67618`,
  Precision/Recall `0.70429/0.77956`, ROI Dice `0.75658`, layer Dice `0.94573`
  and predicted-vessel outside-GT-layer fraction `0.04662`.
- Same-checkpoint diagnostics: predicted-layer soft gate hard Dice `0.75133`;
  disabling D2S reduced soft Dice to `0.60281`. The latter demonstrates model
  dependence, not an independently trained D2S benefit.
- Decision: retain E3b as candidate; train E3-current, E3b-no-D2S and
  E1-current from the exact same current protocol before Joint.

### CODE-20260827-STAGE2-CURRENT-CONTROLS

- Added isolated variants `roi_current`, `roi_outside_no_d2s` and
  `safe_current`, each with a new output directory. A focused test asserts that
  E3-current changes only outside weight, and E3b-no-D2S changes only D2S.
- Metadata now fingerprints label-file contents and records all split IDs/rows.
  `tools/compare_stage2_protocols.py` marks missing/different fingerprints as
  non-comparable.
- Evaluation rows now include source/reference paths, original/model geometry,
  manifest group frame count and evaluated frame IDs/counts. TP/FP/FN maps are
  exported both globally and inside the GT layer.
- Added boundary-band metrics and optional training-defined connected-component
  size recall. `tools/derive_vessel_component_thresholds.py` derives area
  tertiles from one deterministic training frame per labelled group.
- Threshold calibration supports separate `raw` and `soft_gate` modes and
  respects vessel-valid masks; selection remains validation-only.
- Local verification: all three new variants completed one real-manifest CPU
  smoke epoch. Their new metadata fingerprints were protocol-compatible with
  each other. Formal current 13/3/4 cloud runs remain pending.

### CODE-20260828-VALIDITY-AUDIT-AND-ANATOMICAL-POSTPROCESS

- Status: implementation and focused local verification completed; no new
  training or performance result is claimed.
- Protocol audit: new runs use `stage2-fingerprint-v2`, with ordered per-label
  raw-file and decoded-pixel SHA256, decoded shape/dtype/value counts, metadata
  version and inventory. Comparisons expose both values and distinguish
  `matched`, `different` and `unknown`; incomplete historical artifacts remain
  unknown and are not backfilled.
- Causal audit: resolved training configs are flattened and compared against
  explicit `--allowed-differences`; unlisted changes or declared-but-absent
  changes fail the single-factor check. Per-file output distinguishes binary
  serialization changes from decoded-pixel changes.
- V0 evaluation: explicit task selection supports denoise/layer/vessel in one
  frozen-checkpoint report. Independent evaluation restores the cropped canvas
  to source geometry before final metrics and exports metric coverage.
- Denoising diagnostics: noisy/denoised gain for PSNR, SSIM, RMSE, EPI and SNR;
  GT-layer ROI MSE/PSNR; vessel-versus-stroma CNR where both labels exist.
- Layer diagnostics: surface Dice, signed boundary/thickness bias, component and
  hole area, lower-boundary roughness and valid-column coverage in addition to
  Dice/IoU/HD95/ASSD/MAE.
- P0--P3: P0 is immutable raw output; P1 cleans only the layer main component
  and enclosed holes; P2 applies bounded quadratic smoothing only to the lower
  boundary; P3 strictly clips raw vessel by final predicted layer and
  vessel-valid pixels, recording removed TP/FP and empty outcomes. Vessel
  connected components are never pruned.
- Ignore semantics: public preparation now exports a shared label-valid mask;
  class 255 and allowed unknown classes are excluded from both layer and vessel
  supervision/evaluation instead of becoming background negatives.
- Verification: `python -m compileall -q .` and 24 direct tests passed (14
  existing plus 10 protocol/anatomical tests). A real-manifest one-epoch CPU
  E3b smoke run passed with the backward-compatible old manifest and emitted a
  V2 inventory; its self-audit returned identity/protocol/causal=`matched`.
  The local Python environment does not contain pytest, so the required
  complete `pytest -q` run remains for the cloud/CI environment.
# CODE-20260828-STAGE12-EXPORT

- Added a records-first Stage 1/2 registry and full non-test inference entry,
  fixed-threshold complete-validation evaluation, original-grid probability
  restoration, 16-bit denoised PNG and independent layer/vessel mask export.
- Reserved test and associated clean assets are excluded through manifest
  metadata by default. The sealed-test option is pure forward only and is not
  included in metrics, ranking, calibration or gallery outputs.
- Local dry-run found Data/Manifests but no current best checkpoints. It also
  found the local Stage 2 fold-0 manifest is the historical 8/2/3 protocol (100
  validation frames, 2 groups), not the recorded cloud 13/3/4 protocol; no new
  inference or performance result is claimed.
- Cloud resume exposed a legacy/incomplete float-cache schema during denoising
  drift aggregation (`denoised_clipped` absent). Resume now validates required
  NPZ keys/shapes, rewrites incomplete caches atomically, accepts documented
  legacy denoised keys for drift-only comparison, and records incompatible
  caches without aborting the completed inference archive.

# CODE-20260828-INTERACTION-FACTORIAL

- Implemented an acyclic, detached J00/J10/J01/J11 interaction path. S0 uses
  the trained final segmentation heads; the unsupervised historical per-scale
  probability heads are not used as reliable guidance.
- Added zero-initialized residual interaction scales, frozen-encoder `interaction`
  stage, explicit direction/detach aliases, independent data RNG, sampler-plan
  and post-load state fingerprints, optimizer duplication audit, per-scale
  injection/guidance diagnostics, mapping gradients/updates, elapsed time and
  CUDA peak memory logging.
- Added fixed-final validation evaluation, repeat-frame stability, reference
  edge error, clean-relative vessel/stroma CNR error, fixed prediction gallery,
  same-checkpoint perturbation diagnostics and group-first paired-gain reports.
- The interaction input recorder opens/hashes only train/validation label assets;
  reserved test assets are not opened and test is not evaluated or tuned.
- Local source checkpoint is absent, so the real-manifest B0/training smoke is
  intentionally blocked and no performance result is claimed. `compileall` and
  a synthetic forward/backward smoke passed. The local Python environment has
  PyTorch but no pytest package, so the complete test suite remains to run in
  the project conda/cloud environment.
- Cloud B0 exposed a registry-key collision: the audit invocation and CUDA B0
  invocation had legitimately different execution config hashes but shared
  `seed42_registry.json`. Registry schema v2 separated invocation modes; v3
  additionally uses content-addressed filenames so corrected execution configs
  coexist without mutating earlier records. Legacy files are left untouched.
- The first single-seed train reached validation but inherited Stage-2
  `monitor_denoise_drift=true`; this incorrectly rejected the intentional
  denoising-decoder update (`max_abs_diff=0.026565209`). The factorial common
  config now disables that inapplicable invariant, while Trainer also enables
  it only when the denoising decoder/head are actually frozen. Registry v3 uses
  content-addressed filenames, and `--resume-partial` safely accepts only an
  epoch0-only failure with no last checkpoint/history.
- Cloud atlas generation exposed an empty-crop failure when a requested
  clean/GT/diagnostic PNG was structurally absent: the placeholder was created
  at crop size and then sliced again with original-grid coordinates. Atlas
  loading now distinguishes missing/decode/OOB states, clamps valid original
  coordinates, never sends an empty image to OpenCV, prioritizes samples with
  B0 clean/layer/vessel availability, and exports an asset inventory plus a
  missing/invalid checklist.
  Rebuilding a partial atlas now requires `--archive-existing`, which moves the
  prior directory to a timestamped sibling instead of overwriting it.

# CODE-20260830-INTERACTION-SUPPRESSION-AND-INPUT-FACTORIAL

- Status: implementation and local code/synthetic verification completed; no
  cloud training or performance result is claimed.
- Interaction diagnosis now records signed/absolute/RMS residual scales,
  mapping RMS/update/gradient, source/transformed/receiver and injection RMS,
  gate distribution/saturation/entropy, guidance confidence, dataset and
  vessel-label strata, pre/post global clipping, clip coefficient and task/
  interaction gradient groups. Existing histories are never backfilled.
- Added fixed-checkpoint J10/J11 off/learned/target-RMS/shift/cross-position/
  self-capacity diagnostics and selected-position threshold, TP/FP/FN, stroma,
  boundary-band and original-area-derived small-vessel changes. These outputs
  are explicitly dependence/perturbation diagnostics, not retraining gains.
- Added `I_NOISY`/`I_DENOISED` using paired float32 noisy/D0 caches and a common
  pre-segmentation snapshot made directly from fold-specific Stage 1. The new
  `input_segment` stage trains the same shared encoder plus layer/vessel paths
  in both variants while freezing the internal denoising and UGBI paths.
- D0 cache generation is gated by checkpoint resolved-config, manifest/split
  fingerprints and train-versus-segmentation-val/test group overlap. Unknown
  identity or overlap blocks the experiment; test assets are not opened.
- Main I protocol uses the audited 60-epoch E3b budget, fixed final checkpoint,
  P0=0.5, restored original geometry and position-first paired aggregation.
  Artificial noise/blur/intensity remapping are off; only paired horizontal
  flip remains. Original-label-grid training components pre-register the small/
  medium/large area bins.
- Local verification: `python -m compileall -q sabids tools tests` passed and
  six new focused tests were executed directly with the project Python (float
  cache, selected input, parameter/loss isolation, paired config and D0 leakage
  gate). The environment lacks the pytest package, so full `pytest -q` remains
  required on cloud/CI. The real Stage 1 best and J run artifacts are absent
  locally, so real cache/train smoke and numeric scientific conclusions remain
  pending.
- A cloud legacy Stage 1 best checkpoint was found to contain no manifest,
  effective-split or train-group runtime fingerprints. The audit remains
  blocked by default. A compatibility path now accepts `run_metadata.json`
  only when its recorded best-checkpoint SHA256 exactly matches the inspected
  `best.pth` and all provenance fields are complete; mismatched, incomplete or
  merely path-matching sidecars remain blocked. Focused pass/mismatch tests
  passed locally.
- A fresh 512x512 Stage 1 attempt then exhausted a 24 GB GPU with physical
  batch 2 while retaining both noisy and clean-identity graphs. Stage 1 also
  unnecessarily evaluated frozen, unsupervised segmentation/UGBI branches.
  Fold-0 Stage 1 now uses physical batch 1 with two-step accumulation and a
  denoise-only forward; reconstruction, residual and clean identity supervision
  remain unchanged, while both interaction directions are explicitly disabled.
  Compact/full equivalence with interactions disabled and a two-graph backward
  smoke passed locally. Actual CUDA peak memory remains to verify on cloud.
- The next cloud attempt reached epoch 1 batch 6 (`duke28_ll8`) but the trainer
  aborted on a non-finite gradient after AMP unscale. Restoration reductions
  and identity are now explicitly FP32 while convolution remains autocast. The
  fold config starts GradScaler at 1024 instead of 65536; a detected AMP
  overflow now follows standard GradScaler behavior (skip mutation and reduce
  scale), with a hard failure after more than eight consecutive overflows.
  Non-AMP non-finite gradients still fail immediately. FP32-loss finite
  backward and resolved AMP-policy tests passed locally; cloud completion is
  pending.
