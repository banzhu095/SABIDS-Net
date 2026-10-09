# Unified downstream segmentation

This independent experiment measures information recoverability from denoised-only
images. Always label the SABIDS input **SABIDS-current Stage-1/D0 output**. It does
not evaluate Joint SABIDS, D→S, S→D or RMAC.

The segmenter reuses the existing NAF stem, encoder, task adapters and two
decoders. It instantiates no denoising or interaction modules. All parameters
are initialized once per segmentation seed, then copied to every method. The
fixed loss is masked layer BCE+Dice, vessel BCE+Dice inside the GT layer, and
outside-layer negative BCE weighted 0.5. Unlike the legacy composite Stage-2
objective, no auxiliary boundary head or containment term is added. This
uniform explicit objective follows the requested two-head experiment.

## Inputs and safety

The audit finds a unique `segmentation_primary_inputs.csv`, or accepts an explicit
`--manifest`. Conflicting protocols require explicit selection. It matches all
five keys: dataset, split, position, frame and sample. Only denoiser seed 42 is
accepted for deep methods. No filenames are paired by sorting. Native TIFFs can
have identical grayscale RGB channels; arbitrary color conversions are refused.

Public multiclass labels are authoritative: class 1|2 forms layer, class 2 forms
vessels, and 255 is excluded. Derived labels live inside the new run; originals
are unchanged and discrepancies are recorded. Source SHA256, decoded shape,
label identity, primary-seed identity and missing assets are audited. Raw noisy
and clean inputs are explicitly original/reference assets and need not have the
same encoded TIFF SHA as a separately exported identity-method output.

`audit` may inspect test images/labels for integrity, as requested. Training and
validation datasets reject test rows. Prediction/scoring starts only after every
participating arm completes the same update budget and all best checkpoints are
sealed. Repeated evaluation requires identical locked artifacts and reuses
completed per-arm metrics. No threshold tuning is performed; threshold is 0.5.

## Historical pilot compatibility (not used by the formal runner)

The default is four arms, seed 42, 15 epochs, 256-square, AMP, batch 4. A
train-only runtime/OOM probe chooses one common batch 4/2/1 before initialization
and plans are sealed. The ordered runtime reductions are three arms, then ten
epochs, then at most 20 equally spaced train frames per position. Validation and
test frames are never reduced. A CPU budget that still exceeds four hours fails
closed. A mid-training OOM stops the run rather than changing one arm's updates.

Optional DnCNN is included only if its assets exist and the conservative timing
projection places the core four within three hours and all five within the
four-hour budget. This is decided before test access to preserve the single
joint test-opening rule; it is not decided from test performance. Runtime
predictions are estimates, not guarantees. Epoch and ten-minute updates report
progress. An interrupted epoch is replayed deterministically from the last
complete epoch checkpoint.

Checkpoint selection is validation position-macro hard vessel ROI Dice at 0.5,
then Layer Dice, then earliest epoch. Common initialization, sampler order,
crop/augmentation plans and loader generator state are saved and hashed.

## Metrics and preset interpretation

Frame metrics aggregate to position, then segmentation seed, then method.
Three-position pilot outputs have no significance p-values. Binary boundary
Dice compares one-pixel eroded-mask edges within fully valid neighborhoods;
boundary errors are evaluation-grid pixels. A missing predicted layer column
receives a full-height boundary error. Empty/empty Dice equals one.

The requested categories overlap. Frozen precedence is: harmful/not-supportive,
supportive, mixed-direction ±0.005 inconclusive, remaining positive weak-support,
otherwise inconclusive. Harm means Precision falls by >0.01 or either outside-FP
fraction increases by >0.01. Support requires ROI Dice ≥0.005, ≥2/3 positions
positive, Recall positive, Precision decline ≤0.01, no harmful FP, and boundary
Dice nonnegative. These absolute proportions are percentage-point differences.

## Commands

Run from the checkout containing this module. Use a new directory for new data,
protocol or source versions. Existing denoising and Joint runs are never modified.

```bash
ROOT=/mnt/SABIDS-Net
PILOT="$ROOT/runs/downstream_seg_pilot_$(date +%Y%m%d_%H%M%S)"
# PRIMARY must be an existing, audited primary-input manifest on this server.
python -m tools.downstream_seg_benchmark.cli audit \
  --project-root "$ROOT" --manifest "$PRIMARY" --run-dir "$PILOT"
python -m tools.downstream_seg_benchmark.cli run-pilot \
  --project-root "$ROOT" --manifest "$PRIMARY" --run-dir "$PILOT" \
  --device cuda --num-workers 4 --resume
```

Individual commands: `build-plan`, `train`, `evaluate`, `build-atlas`,
`build-report`, `package-light`, each with `--run-dir` and `--resume` as needed.
`build-report` writes an honest blocked report when comparable results do not
exist. Local workbook creation uses artifact-tool; cloud environments without it
use explicitly logged openpyxl fallback (install `openpyxl`). ZIP packaging
excludes images, tensor caches and checkpoint binaries, includes their inventories,
and verifies ZIP CRC and every manifest entry SHA256.

Formal, after input completion (three labelled test positions are permitted with limited conclusions):

```bash
FORMAL="$ROOT/runs/downstream_seg_formal_$(date +%Y%m%d_%H%M%S)"
python -m tools.downstream_seg_benchmark.cli run-formal \
  --project-root "$ROOT" --manifest "$PRIMARY" --run-dir "$FORMAL" \
  --device cuda --num-workers 4 --resume
```

Formal fixes ten methods and segmentation seeds 42/123/2026, native 384 training
crops and 640 full-image evaluation. Three labelled test positions produce
`complete_limited_3_position_test` after all 30 arms finish. The cancelled pilot's
time reductions and decision categories are not used by this runner. Complete
pku_0025, pku_0038 and pku_0043 annotations before stronger paper-level claims.

## Formal protocol and execution

The historical `roi_outside_no_d2s` resolved config on the cloud has 60 epochs,
learning rate 5e-5 and weight decay 1e-4. Those budgets are retained. The isolated
two-head comparison uses a common cosine scheduler, AMP, no early stopping, full
validation every five epochs and fixed optimizer updates. Ties within 0.0001
ROI Dice use Layer Dice, then earliest epoch. P100 16GB defaults to batch 1;
no resizing of native training crops is performed. Three quarters of patches
are centred on a GT-layer pixel; one quarter are spatially uniform. All crop,
sampler and horizontal-flip plans are stored before training and reused across
methods for each segmentation seed.

Use the formal source already present on this server:

```bash
cd /mnt/SABIDS-Net
SOURCE=/mnt/SABIDS-Net/runs/denoise_benchmark_plus_sabids_tcfl_20260915_103144
RUN=/mnt/SABIDS-Net/runs/downstream_seg_formal_20261008_gpu
mkdir -p "$RUN/logs"
bash tools/downstream_seg_benchmark/scripts/materialize_formal_inputs.sh \
  --project-root /mnt/SABIDS-Net --source-denoise-run "$SOURCE" \
  --run-dir "$RUN" --resume
bash tools/downstream_seg_benchmark/scripts/materialize_formal_inputs.sh \
  --project-root /mnt/SABIDS-Net --run-dir "$RUN" --verify-only
nohup bash tools/downstream_seg_benchmark/scripts/run_modelwhale_formal.sh \
  --project-root /mnt/SABIDS-Net --source-denoise-run "$SOURCE" \
  --run-dir "$RUN" --batch-size 1 --patch-size 384 --num-workers 4 --resume \
  > "$RUN/logs/formal_launcher.log" 2>&1 &
```

Do not reuse this directory for changed source or protocol. Materialization
uses existing locked outputs first. `--reinfer` is an explicit recovery option
only for exact sealed primary checkpoint/config; historical encoded SHA conflicts
fail closed. It does not train a denoiser. Export verified labelled images with:

```bash
python -m tools.downstream_seg_benchmark.cli export-input-assets \
  --project-root /mnt/SABIDS-Net --run-dir "$RUN" \
  --output-dir "$RUN/exports"
```

Monitor with `pgrep -af downstream_seg_benchmark`, `nvidia-smi`,
`tail -n 30 "$RUN/logs/formal_launcher.log"` and:

```bash
python -m tools.downstream_seg_benchmark.cli status --run-dir "$RUN"
cat "$RUN/progress.json"
cat "$RUN/failures.csv"
```

Resume stages with the identical launcher command plus `--from-stage train`
(or lock/evaluate/atlas/report/package). To recover an individual arm:

```bash
python -m tools.downstream_seg_benchmark.cli train --run-dir "$RUN" \
  --device cuda --methods nafnet_paired --seg-seeds 123 --resume
```

Partial arm completion never opens test. All 30 arm completion markers, common
randomness hashes, equal update counts and best-checkpoint hashes are checked
before test scoring. An interrupted epoch is replayed from the last complete
epoch checkpoint. Changing sealed source, labels, inputs or protocol fails.

## Metrics, statistics and atlas

Metrics include Layer Dice/IoU, upper/lower boundary and thickness errors,
Layer/Vessel ASSD, Vessel Dice/ROI Dice, Precision/Recall/F1, one-pixel Boundary
Dice, ROI FP/FN, predicted-layer Vessel Dice and outside-GT-layer FP/fraction.
ASSD uses valid one-pixel boundaries, pixel units, zero for both empty and image
diagonal penalty for one empty. Small/medium/large vessel recall is pixel recall
within eight-connected GT components of area <64, 64–255, ≥256 native pixels;
an absent size bin is undefined, never zero.

Aggregation is frame → position within seed → seed mean per position → method.
Three positions, not frames or seeds, are the resampling units. Bootstrap uses
10000 clustered-position draws after seed averaging and is descriptive only;
no significance p-values are generated. Seed mean/std/min/max are retained.
The preset primary comparisons are SABIDS vs NAFNet, DnCNN, Noisy and BM3D.
Recall gains with lower Precision, more outside-layer FP or lower Boundary Dice
must not be described as structural superiority. Clean is an upper bound and
TCFL is unpaired. Descriptive method Spearman associations have no p-values.

Before training/scoring, atlas roles are fixed from Noisy+GT: median sorted
frame, most small GT components, weakest noisy gradient on the GT layer edge.
Ties use sample ID. Roles may coincide if criteria select the same frame;
all roles are retained. Every method uses the same crop, gamma 1, grayscale
[0,1], nearest interpolation, probability range [0,1] and threshold 0.5.

New labels must be audited in a separate directory, then evaluated only:

```bash
python -m tools.downstream_seg_benchmark.cli evaluate-incremental \
  --run-dir "$RUN" --incremental-audit-dir "$RUN/new_test_label_audit" \
  --output-dir "$RUN/expanded_test_evaluation" --device cuda --resume
```

The new audit must contain the unchanged old cohort plus newly labelled test
positions. Train/val and old test keys, images and label identities must be
identical. No checkpoint selection or training is performed. Original results
remain intact. Formal outputs include `benchmark_summary.xlsx`, the report,
metric/provenance CSVs, fixed atlas, training curves and a CRC/SHA-verified light
ZIP excluding full TIFFs, weights, original Data and tensor caches.

## Implementation log (2026-10-08)

Continuation preserved the unrelated dose-atlas changes. The previously
untracked pilot implementation was extended in place, adding materialization,
formal stages, full metrics, cluster bootstrap, preregistered atlas roles and
evaluate-only label expansion. CPU engineering smoke is not a scientific
result. Runtime experiment evidence, commit/hardware/config hashes and actual
completion status are recorded inside each new run, not in original runs.

## AMP recovery (2026-10-09)

The first formal launch failed with a nonfinite scaled gradient before an
epoch checkpoint was saved, on both P100 and A30. Three initial real patches
were finite on A30; that limited probe did not establish epoch stability.
The training engine now buffers each accumulation window and, upon a nonfinite
gradient, halves the AMP scale and replays the exact same samples, CPU/CUDA RNG
and model buffers. Only a successful finite clipped-gradient optimizer step
counts toward the fixed budget. At most 16 retries are allowed; nonfinite loss,
unscaled/non-AMP failures and persistent overflow still fail closed. Retry counts
and scales are saved in training curves; no method-specific numerical policy is
used. Loss, architecture, optimizer, scheduler and sampling remain unchanged.

A sealed old run must NOT have its source hash silently updated. For this
zero-checkpoint failure, recovery uses a separate run directory containing
byte-identical copied plans, shared initializations and audited asset manifests
(whose paths still point at original assets). Its new source seal and explicit
provenance record link to the old plan/hash/commit and preserve the old failure
log. Reject this migration if any epoch checkpoint, completed track, checkpoint
lock or test-open marker exists. Source/config/asset hashes remain enforced in
the recovered run; never use this route to migrate trained checkpoints.

Use `recover-unstarted-plan --source-run OLD --run-dir NEW` with explicit,
distinct paths. The destination must not exist. It copies source audit/logs
under `recovery_source_audit/`, checks all sealed asset/randomness hashes,
rejects any saved training history/weights or test/checkpoint seals, and writes
`recovery_provenance.json` linking both source hashes and the previous plan SHA.
Continue with `run_modelwhale_formal.sh --from-stage train --resume`, keeping the
original source-denoising run, batch, patch and worker settings.
