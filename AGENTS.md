# SABIDS-Net project instructions

## Project goal

SABIDS-Net jointly performs OCT denoising, choroidal layer segmentation,
and choroidal vessel segmentation.

## Dataset rules

- Duke17 and Duke28 provide noisy-clean pairs for denoising only.
- PKU37 provides repeated noisy frames and clean references.
- Only 13 PKU37 anatomical positions have layer and vessel labels.
- All repeated frames from the same PKU37 position must share one group_id.
- Frames from one group_id must never cross train/validation/test splits.
- Multiclass labels use:
  - 0 = background
  - 1 = choroidal stroma/layer
  - 2 = choroidal vessel
- Binary layer mask = class 1 or class 2.
- Binary vessel mask = class 2.

## Training rules

- Stage 1: denoising pretraining.
- Stage 2: layer and vessel segmentation.
- Stage 4: joint UGBI and RMAC training.
- Do not reuse checkpoints across incompatible target sizes or loss settings.
- Do not tune thresholds on the test split.
- Use validation soft Dice for checkpoint selection.
- Use group-level rather than frame-level statistical conclusions.

## Current known problems

- Previous Joint model overpredicted vessel regions.
- Vessel precision was 0.380 and recall was 0.826.
- Predicted vessel fraction was 0.732 versus ground truth 0.415.
- Final vessel experiments should use 512x512 or at least 384x384.
- 256x256 is for debugging only.

## Code requirements

- Preserve existing user data, labels, manifests, runs, and checkpoints.
- Never delete training outputs unless explicitly requested.
- Use memory-safe Joint forwarding.
- Run syntax checks and pytest after changes.
- Update README and RESULT_ANALYSIS.md when training behavior changes.