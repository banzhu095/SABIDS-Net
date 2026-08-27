# SABIDS-Net project instructions

This file is the stable entry point for Codex and other coding agents working on
SABIDS-Net. Read it before modifying code, configuration, data preparation, or
experiment documentation.

## Read first

Before making a material change, read the smallest relevant set of files:

1. `docs/PROJECT_CONTEXT.md` for the research question, implemented method, and
   current scientific priorities.
2. `docs/DATASET_PROTOCOL.md` for label semantics, manifests, split units, and
   leakage prevention.
3. `docs/EXPERIMENT_LOG.md` and `RESULT_ANALYSIS.md` for the latest confirmed
   run state and unresolved failures.
4. The resolved YAML or source file that actually controls the requested run.

The code and resolved configuration are authoritative for implemented behavior.
Project documents distinguish confirmed implementation from planned research;
do not silently present a planned module as already implemented.

## Project goal

SABIDS-Net jointly addresses OCT denoising, choroidal layer segmentation, and
choroidal vessel segmentation under sparse vessel annotation. Its current 2-D
B-scan method has two main interaction mechanisms:

- UGBI: uncertainty-gated bidirectional interaction between restoration and
  segmentation features.
- RMAC: repeat-scan multi-task anatomical consistency for registered noisy
  frames from the same anatomical position.

The immediate engineering priority is to obtain a reliable public Stage 2 and
Joint baseline without the previous full-layer vessel false-positive collapse.
Private SS-OCT adaptation comes after the public baseline is stable.

## Non-negotiable data rules

- Multiclass labels use `0=background`, `1=choroidal stroma/layer`,
  `2=choroidal vessel`, and optional `255=ignore`.
- Binary layer target is `(class == 1) OR (class == 2)`.
- Binary vessel target is `(class == 2)`; overlapping pixels must remain vessel
  in the multiclass export.
- All repeated PKU37 frames from one anatomical position share one `group_id`
  and must remain in one split.
- Split public repeated scans by `group_id`; use `patient_id` when multiple
  positions from the same patient can otherwise cross splits.
- Split private data by patient by default. Adjacent B-scans are different
  anatomical positions and must not be treated as RMAC repeats.
- Only registered repeat scans of the same anatomy may share `group_id` for
  RMAC.
- Never tune a threshold, choose a checkpoint, or make an architecture decision
  on the test split.
- Treat PKU37 repeat frames as correlated observations. Primary statistical
  summaries use independent groups/patients, not raw frame count.

If a requested change weakens any of these rules, stop and explain the leakage
or validity risk before implementing it.

## Current stage contract

| Stage | Purpose | Initialization | Main trainable path | Checkpoint monitor |
| --- | --- | --- | --- | --- |
| Stage 1 `denoise` | Paired denoising pretraining | none | shared encoder + denoising path | `psnr` |
| Stage 2 `segment` | Layer/vessel pretraining | Stage 1 best | segmentation paths; interaction frozen | `vessel_soft_dice` |
| Stage 4 `joint` | UGBI + RMAC joint learning | Stage 2 best | full joint model | `vessel_soft_dice` |
| Stage 5 `private_seg` | Sparse-label private adaptation | Stage 4 best | layer/vessel paths + denoise-to-seg interaction; encoder frozen by default | `vessel_soft_dice` |

Important invariants:

- Public main experiments use `512x512`; use `384x384` only if 512 still OOM.
  `256x256` is a debugging size, not the intended final vessel result.
- Private 12x12 mm images are approximately `500x1536`; the current conservative
  adaptation canvas is `320x960`, preserving aspect ratio through resize/pad.
- Joint training on a 24 GB GPU uses batch size 1, two-step gradient
  accumulation, and memory-safe auxiliary forwards.
- In memory-safe Joint, repeat and clean teacher outputs are stop-gradient and
  `loss.weights.identity` is zero.
- Stage 2 and Joint checkpoints made with old loss definitions or incompatible
  target sizes must not be resumed or reused.
- `--force` archives recognized old run artifacts. Do not replace that behavior
  with deletion.
- Stage 5 evaluation uses EMA weights when an EMA state is present.

## Current vessel safeguards

Preserve these unless the task explicitly studies their removal:

- Focal-Tversky false-positive/false-negative weights `0.6/0.4`.
- Vessel BCE contribution.
- Layer-interior stroma hard-negative loss.
- Per-image vessel fraction constraint.
- Containment uses ground-truth layer ROI when available and detaches predicted
  layer otherwise.
- Joint sampler targets 35% vessel-labelled optimization steps.
- RMAC total weight `0.15`, with vessel consistency weaker than layer
  consistency.
- Best checkpoint selection by continuous `vessel_soft_dice`.

Any loss change must also update the resolved config signature, README or
project context, and the experiment log entry describing why it changed.

## Repository and data safety

- `Data/`, `Label/`, `Manifests/`, `PreparedLabels/`, `runs/`, model weights,
  predictions, and patient data are local runtime artifacts and must not be
  committed.
- Preserve existing user data, labels, manifests, checkpoints, and training
  outputs. Never delete or overwrite them without explicit user authorization.
- The Windows repository is the editing source; GitHub carries code versions;
  the Juchiyun server normally only pulls code and runs experiments.
- Do not commit absolute patient-specific paths. Runtime examples may use the
  documented project roots, but reusable configs should prefer project-relative
  paths.
- Before a broad `git add`, check `git status --short` and ignored paths.

## Change workflow

For code changes:

1. Inspect the relevant data flow, configuration inheritance, and checkpoint
   compatibility path before editing.
2. Make the smallest coherent change; preserve unrelated user modifications.
3. Add or update a focused test for behavior changes where practical.
4. Run at least:

   ```bash
   python -m compileall -q .
   pytest -q
   ```

5. For training-path changes, run the isolated pipeline smoke test when the
   required manifests are available:

   ```bash
   python run_current_pipeline.py \
     --project-root . \
     --fold 0 \
     --stages denoise segment joint \
     --device cpu \
     --smoke-test \
     --force
   ```

6. Update documentation when CLI flags, data semantics, losses, stage behavior,
   evaluation, or recommended commands change.
7. Record actual experiments in `docs/EXPERIMENT_LOG.md`; include Git commit,
   resolved config, split/fold, checkpoint, hardware, and outcome.

Do not claim a training result from syntax checks or smoke tests. Do not replace
an unresolved scientific question with a guessed numeric result.

## Evaluation rules

- Calibrate the vessel threshold only on validation data, then freeze it for
  test evaluation.
- Report both frame-level compatibility metrics and group-level primary metrics.
- For vessels, inspect Dice, Precision, Recall, HD95/ASSD, vessel-area fraction
  error, predicted layer-vessel similarity, and layer-exterior vessel fraction.
- For layers, inspect Dice/IoU, HD95/ASSD, boundary MAE, and thickness MAE.
- For denoising, compare with the noisy input baseline and report PSNR, SSIM,
  RMSE, EPI, SNR, and CNR where defined.
- Qualitative review must include probability maps, binary masks, overlays, and
  failure cases. A high Recall with low Precision and excessive vessel area is
  a failure, even if Dice looks superficially acceptable.
- Use pixel units unless post-resize axial/lateral physical spacing has been
  verified.

## Scope boundaries

The current repository implements a 2-D B-scan framework. The following are
research extensions, not established current features: 2.5-D/3-D training,
en-face projection supervision, topology losses, projection-quality losses,
cross-device domain adaptation beyond the current conservative mechanisms, and
clinical biomarker statistics. Add them incrementally with isolated ablations;
do not mix them into the public baseline without a causal comparison.

