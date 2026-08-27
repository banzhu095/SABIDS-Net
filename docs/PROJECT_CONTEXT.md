# SABIDS-Net project context

Last consolidated: 2026-08-27

## 1. Research objective

SABIDS-Net studies how OCT denoising, choroidal layer segmentation, and
choroidal vessel segmentation can assist one another when vessel boundaries are
low-contrast and only a small subset of scans has reliable vessel annotations.
The intended paper contribution is not simply a three-head network. It is a
sparse-annotation framework in which restoration and anatomy exchange only
reliable information, while repeated acquisitions of the same position provide
real noise-invariance supervision.

The main scientific questions are:

1. Does restoration improve layer and vessel segmentation without erasing dark
   vascular lumens or hallucinating structure?
2. Does segmentation provide anatomical constraints that improve denoising and
   preserve clinically meaningful boundaries?
3. Can repeat-scan consistency and conservative pseudo-labels make useful use
   of unlabelled vessel images without collapsing the vessel prediction into the
   whole choroidal layer?
4. Does the method generalize from public SD-OCT to private wide-field SS-OCT
   while preserving both restoration and segmentation performance?

## 2. Implemented method

### 2.1 Shared encoder and task-specific paths

The current network is a 2-D, fully convolutional B-scan model. A shared
four-scale NAF-style encoder uses channels `[32, 64, 128, 256]`. Task adapters
and decoders specialize the representation into:

- a denoising residual path;
- a binary choroidal layer path with a two-channel upper/lower boundary head;
- a binary choroidal vessel path.

The denoising output is residual-based:

```text
residual = 0.5 * tanh(residual_head)
denoised_raw = noisy - residual
denoised = clamp(denoised_raw, 0, 1)
```

The model does not hard-crop the vessel output with the predicted layer mask.
Instead, layer information is used through learned interactions and soft losses,
which preserves gradient flow and permits later analysis of layer-exterior
false positives.

### 2.2 UGBI

UGBI is implemented at encoder/decoder levels `[3, 2, 1]`, corresponding to
1/8, 1/4, and 1/2 resolution in the default four-level model. It supports:

- segmentation-to-denoising interaction;
- denoising-to-layer interaction;
- denoising-to-vessel interaction;
- uncertainty gating;
- zero-scale initialization of interaction residuals;
- temporary cross-task gradient detachment during Joint warm-up.

This design is intended to prevent unreliable early predictions from dominating
another task while still allowing learned bidirectional exchange later.

### 2.3 RMAC

RMAC uses two noisy scans from the same registered anatomical `group_id` and,
when available, their clean target. It constrains:

- denoised-image consistency;
- layer-probability consistency;
- vessel-probability consistency;
- deep anatomy-embedding consistency;
- consistency with a clean-image stop-gradient teacher.

The current memory-safe Joint path keeps the main frame's full computation graph
and evaluates repeat/clean teachers without retaining two additional full
backward graphs. Different repeat frames can still become the main frame across
training steps.

### 2.4 Sparse vessel supervision

Only a subset of public positions and an expected small subset of private scans
have manual vessel masks. Current safeguards against vessel over-expansion are:

- Dice, BCE, and Focal-Tversky vessel supervision;
- stronger false-positive than false-negative Tversky weighting (`0.6/0.4`);
- layer-interior non-vessel stroma negative loss;
- per-image vessel-area fraction matching;
- soft containment outside a true layer ROI when available;
- 35% vessel-labelled training-step sampling in Stage 4/5;
- weaker RMAC vessel consistency than layer consistency;
- EMA plus dark-lumen dual-source pseudo-labels during private adaptation.

The dark-lumen prior uses the darkest 5% inside the layer as a conservative
source, not as a complete vessel label. It should guide confident cores rather
than force every dark pixel to be a vessel.

## 3. Training stages

| Stage | Config | Role | Initialization |
| --- | --- | --- | --- |
| 1 | `configs/current/stage1_denoise_fold0.yaml` | Paired denoising | none |
| 2 | `configs/current/stage2_segment_fold0.yaml` | Public layer/vessel pretraining | Stage 1 `best.pth` |
| 4 | `configs/current/stage4_joint_fold0.yaml` | UGBI + RMAC public joint training | Stage 2 `best.pth` |
| 5 | `configs/current/stage5_private_seg_fold0.yaml` | Private sparse-label segmentation adaptation | Stage 4 `best.pth` |

Stage numbering retains the original research plan: Stage 3 is represented by
the initial cross-gradient-detached period inside Joint rather than a separate
current CLI stage.

Stage 5 defaults to conservative adaptation. It freezes the shared encoder and
denoising branch, disables segmentation-to-denoising interaction, and updates
the segmentation paths plus denoising-to-segmentation gates. An optional second
experiment may unfreeze only the deepest encoder level with
`private_train_encoder_levels: [3]`.

## 4. Current losses

The implemented objective is:

```text
L = λrec Lrec + λres Lresidual + λlayer Llayer + λvessel Lvessel
  + λstroma Lstroma + λarea Larea + λcontain Lcontain
  + λrmac LRMAC + λid Lidentity + λpseudo Lpseudo
```

Default global weights in `configs/base.yaml` are:

| Term | Weight | Active use |
| --- | ---: | --- |
| Reconstruction | 1.00 | Stage 1/Joint |
| Residual | 0.50 | Stage 1/Joint |
| Layer | 1.00 | Stage 2/Joint/Stage 5 |
| Vessel | 1.00 | Stage 2/Joint/Stage 5 when labelled |
| Stroma negatives | 0.25 | labelled layer/vessel samples |
| Vessel area | 0.20 | labelled layer/vessel samples |
| Containment | 0.10 | segmentation stages |
| RMAC | 0.15 | Joint, ramped |
| Clean identity | 0.05 base; 0 in memory-safe Joint | Stage 1 as configured |
| Pseudo-label | 0.50 | private adaptation, ramped |

Resolved YAML files saved beside checkpoints are the authoritative record for a
specific run; this table describes the current defaults only.

## 5. Current evidence and known failure

An earlier Joint checkpoint showed denoising improvement but severe vessel
false-positive expansion:

| Metric | Earlier Stage 2 | Earlier Joint |
| --- | ---: | ---: |
| Layer Dice | 0.8414 | 0.8045 |
| Vessel Dice | 0.6259 | 0.5205 |
| Vessel Precision | 0.5425 | 0.3801 |
| Vessel Recall | 0.7413 | 0.8260 |
| Predicted vessel fraction | 0.5604 | 0.7322 |
| True vessel fraction | 0.4161 | 0.4152 |

These numbers are diagnostic, not final paper results. The logged run reused an
existing checkpoint, and Stage 2/Joint may not have used exactly the same input
size. The pattern nevertheless clearly indicates whole-layer-like vessel
overprediction: Precision fell, Recall rose, and predicted area expanded.

Version 0.2 changed the loss, sampling, checkpoint monitor, learning rate, RMAC
strength, and memory behavior to address that failure. Therefore the old Stage
2 and Joint checkpoints are incompatible with the intended v0.2 experiment and
must not be resumed. See `RESULT_ANALYSIS.md` and `docs/EXPERIMENT_LOG.md`.

## 6. Immediate experiment sequence

The next decision gate is a fresh fold-0 public baseline:

1. Regenerate and audit manifests and binary masks.
2. Reuse Stage 1 only if its resolved configuration and target size are
   compatible.
3. Retrain Stage 2 at 512x512 and inspect validation probability maps.
4. Continue to Joint only if Stage 2 vessel predictions are distinct from the
   layer and vessel area is plausible.
5. Retrain Joint at 512x512 with batch 1 and accumulation 2.
6. Calibrate the vessel threshold on validation data only.
7. Evaluate the frozen checkpoint/threshold on test data.
8. Repeat all five folds before drawing a public segmentation conclusion.
9. Run the planned UGBI/RMAC/constraint ablations with identical folds and
   initialization.
10. Start private adaptation only after the public baseline and its failure
    criteria are stable.

## 7. Paper-level evidence plan

### Primary comparisons

- Denoising-only baseline, segmentation-only baseline, and independent cascade.
- Shared encoder without cross-task interaction.
- One-way segmentation-to-denoising and denoising-to-segmentation variants.
- Bidirectional interaction without uncertainty.
- Full UGBI.
- Full UGBI + RMAC.
- Full method with sparse vessel supervision and private adaptation.

### Required ablations

- no UGBI;
- no RMAC;
- each interaction direction alone;
- no uncertainty gating;
- no stroma/area vessel safeguards;
- no pseudo-labels;
- vessel-labelled sampling fraction sensitivity (`0.20/0.35/0.50`);
- RMAC repeat count sensitivity (`K=1/2/5/10/25/50`) where data allow;
- private encoder frozen versus deepest-level unfreezing.

### Reporting principles

- Public vessel segmentation: five-fold PKU37 group-level results with
  variability; do not treat repeated frames as independent samples.
- Duke17/Duke28: paired denoising external evaluation; do not claim vessel Dice
  without reliable vessel ground truth.
- Private SS-OCT: patient-level split, explicit scanner/protocol description,
  and separate results for scans with manual vessel labels.
- Report image restoration, layer geometry, vessel segmentation, and area/CVI-
  related errors together. A method is not clinically adequate based on PSNR or
  Dice alone.

## 8. Planned extensions, not current implementation

The following ideas remain future work or separate experiments:

- 2.5-D adjacent-slice input for private volumes;
- 3-D vessel continuity constraints;
- en-face vessel projection generation and projection-quality supervision;
- topology-aware loss after validating the 2-D lumen-label definition;
- bidirectional denoising/segmentation refinement beyond current UGBI;
- non-paired 51-line versus HD SS-OCT domain adaptation beyond the current noise
  bank synthesis;
- clinical biomarker analysis for MCT, CVV, CSV, CVI/CSVR, vessel density,
  skeleton density, diameter, and perimeter;
- disease-group statistics for DR severity and wide-field regional maps.

These extensions should be introduced one at a time after the v0.2 baseline,
with matching controls and without changing the held-out test set.

## 9. Working environments

- Local Windows project root: `E:\1-脉络膜\OCT降噪\SABIDS-Net`
- Juchiyun runtime root: `/mnt/SABIDS-Net`
- GitHub repository: `https://github.com/banzhu095/SABIDS-Net`
- Editing authority: local repository/Codex
- Training authority: Juchiyun GPU server
- Data, labels, manifests, checkpoints, and predictions remain outside Git
  tracking.

For every substantive experiment, record the exact Git commit returned by
`git rev-parse --short HEAD`; a version name such as v0.2 is not precise enough
for reproducibility.

