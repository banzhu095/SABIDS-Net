# Segmentation-guided adaptive denoising v1

## Scientific status

This protocol follows the completed negative `noisy+mild` seed-42 pilot.  That
pilot is hypothesis-generating only: B3 did not pass its preregistered gate and
the old threshold or conclusion is not changed.  The new protocol separates:

1. the historical three-position pilot;
2. a development-only 16-position grouped four-fold CV;
3. the sealed test, which is not opened by this implementation.

Code and CPU tests are not evidence that any intervention, oracle, residual
model or controller improves segmentation.  Long CUDA experiments must report
negative results unchanged.

## Audited definitions

- B0 uses noisy alone; B1 uses D1 mild alone; B3 uses paired noisy+mild; B6
  uses noisy+noisy; B3R adds the signed residual; B6R is the parameter-matched
  noisy+noisy+zero-residual control.  C1 is a same-split, different-position
  mild control.  C5 disables the auxiliary branch of an already trained B3.
- D1 mild is the registered `alpha=0.25` point between noisy (`alpha=0`) and
  strong (`alpha=1`).  The signed residual is always `noisy - mild`; it is
  recomputed after paired augmentation and is never independently normalized.
- The shared encoder is registered once.  Fusion at levels 3, 2 and 1 uses
  noisy-anchored residual injection.  Existing B3 concatenates noisy features,
  auxiliary features and their difference.  B3R additionally uses a light
  residual stem.  `gamma=0` makes the initialized output exactly the noisy
  main path; adapter gradients are therefore expected after gamma starts, not
  necessarily on the first optimization step.
- With the registered 512×512 input and one 2× downsample between encoder
  levels, fusion levels 1/2/3 correspond to 256×256, 128×128 and 64×64 feature
  grids respectively.
- The configured 512 model grid, fixed normalization and paired deterministic
  horizontal flip are shared by both views.  P0 at threshold 0.5 is fixed.
- Epoch 12 is the primary fixed endpoint; epoch 20 and best validation soft
  Dice are sensitivity analyses.  `fixed_checkpoint_epochs: [12]` writes an
  immutable `epoch012.pth`.

The cloud preflight requires an evidence-bound B3 run, fixed component
inventory, development source manifest, active protocol lock and split
contract.  Missing evidence produces `BLOCKED: DUAL-VIEW INPUT EVIDENCE`.

## Stages and gates

- A: T0--T6 replace only B3's auxiliary input under the same last/best
  checkpoint.  Wrong guides are same-split/different-group and repetitions are
  averaged inside sample and position.
- B: development-only signed-residual leakage audit and B3R/B6R.
- C: exactly 16 automatically discovered development positions, four folds,
  four validation positions per fold and seeds 42/43/44.  Sealed groups cause
  a fail-closed error.
- D: O0--O7 spatial-dose controls are explicitly labelled
  `ORACLE / USES VALIDATION GT / NOT A DEPLOYABLE PERFORMANCE ESTIMATE`.
  The implemented D1 same-checkpoint tool materializes validation rows only;
  it deliberately does not create an approximately 80 GiB full-development
  grid.  Oracle-aware retraining (D2) remains unimplemented until a fold-train
  combination-selection and streaming-cache protocol is separately locked.
- E: the convex noisy/mild/strong controller is implemented as a constrained
  mixer, but its training entry remains locked unless the oracle gate passes
  and a cross-fitted S0 guidance registry exists.  It never synthesizes pixels.

The independent statistical unit is anatomical position.  Frames are first
aggregated inside position, and seeds are training repeats rather than new
cases.  With only 16 development positions, confidence intervals remain
limited.

## ModelWhale entry

Use only `tools/run_seg_guided_modelwhale.sh`.  Set:

```bash
export SABIDS_B3_RUN=/absolute/path/to/completed/b3_seed42
export SABIDS_FIXED_COMPONENT_INVENTORY=/absolute/path/to/fixed_component_inventory.json
export SABIDS_SEG_GUIDED_SOURCE_MANIFEST=/absolute/path/to/development_manifest.csv
```

Then run `preflight`, `intervene`, `residual-audit`, `oracle`, `cv-pilot`, and
only after the recorded gates, `cv-formal` or `adaptive-pilot`.  The package
stage excludes checkpoints, NPY caches, raw data and test assets.
