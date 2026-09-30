# Dual-task segmentation-guided adaptive denoising v1

## Frozen evidence and scope

This seed-42 experiment is exploratory and validation-only. It never opens the
sealed test groups. The coarse denoiser is the registered D2
`best_task_preserving` checkpoint (epoch 2, SHA256
`edc5b1987f189566666d1eb0b9f03701b9ff28409c2a30ad0d34c8eb5758e343`).
The coarse segmenter is the B3 best-validation-vessel-soft-Dice checkpoint
(epoch 11 of a completed 20-epoch run, earliest-tie rule, SHA256
`c26016890ab50b5a148873d9b868a4aa58a4228c98a82a50b7411027e134ae71`).
Both are frozen and are audited after training.

The validation set has only three anatomical positions (`pku_0006`,
`pku_0012`, `pku_0040`). Moreover, both anchor selection and this experiment
use these positions. Results are therefore hypothesis-generating and not an
unbiased deployable estimate. No causal or generalization claim is permitted.

## Model

`C0-Coarse` uses D2 residual dose alpha 0.25 followed by the frozen B3 coarse
segmenter. `C1-Adaptive` consumes only deployable inputs: noisy image, coarse
image, their residual, frozen coarse probabilities and entropy, and noisy
gradient magnitude. A shared quarter-resolution context encoder feeds separate
layer and vessel heads. Their strengths are bounded to `[0, 1.25]` and start at
1.0 and 0.5 respectively. The task images are
`x_task = clamp(x - alpha_task * (x - D2(x)), 0, 1)`.

The fine segmenter keeps a frozen noisy path, uses separate layer/vessel
auxiliary fusions, and adds zero-initialized residual corrections to frozen B3
logits. Thus initialization is exactly equivalent to C0 segmentation, while
multi-step training can update the controller, auxiliary fusions and correction
heads. Clean images and ground-truth labels never enter prediction or gate
conditioning.

The formal loss reuses the registered E3b-valid layer/vessel, outside-vessel and
containment objectives. It adds only gate TV (`0.001`) and a small paired-clean
reconstruction regularizer (`0.02`). Unknown and padding masks remain excluded
by the existing segmentation loss. Primary selection is the earliest maximum of
`0.5*val_layer_soft_dice + 0.5*val_vessel_soft_dice`, saved as
`best_joint.pth`; no separately selected task checkpoints are spliced.

## ModelWhale order

Copy `best_checkpoint_supplement_20260930_104533.tar.gz` into the repository's
`exports/` directory before preflight. Then run:

```bash
RUN_ID="dual_task_adaptive_v1_$(date +%Y%m%d_%H%M%S)"
bash tools/run_dual_task_adaptive_modelwhale.sh audit --run-id "$RUN_ID"
bash tools/run_dual_task_adaptive_modelwhale.sh overfit --run-id "${RUN_ID}_overfit"
bash tools/run_dual_task_adaptive_modelwhale.sh train-seed42 --run-id "$RUN_ID"
bash tools/run_dual_task_adaptive_modelwhale.sh evaluate-best --run-id "$RUN_ID"
bash tools/run_dual_task_adaptive_modelwhale.sh export-atlas --run-id "$RUN_ID"
bash tools/run_dual_task_adaptive_modelwhale.sh summarize --run-id "$RUN_ID"
bash tools/run_dual_task_adaptive_modelwhale.sh package --run-id "$RUN_ID" \
  --output "/mnt/SABIDS-Net/exports/${RUN_ID}_gpt_light.zip"
```

Overfit and formal use different run IDs so their immutable output namespaces
cannot collide. Preflight is metadata-only and reports
`BLOCKED: DUAL-TASK ADAPTIVE INPUT EVIDENCE` on any missing or changed anchor.
The overfit/formal preparation steps, not preflight, create deterministic
train/validation pixel and fixed-component inventories before optimization;
sealed test paths are never opened.

## Interpretation

The primary comparison is paired `C1-Adaptive - C0-Coarse`, first averaged by
anatomical position. Report segmentation, reconstruction, gate-distribution and
cost outcomes together. A segmentation gain accompanied by reconstruction or
weak-vessel harm is not an unqualified improvement. A null or negative result
must be retained without threshold tuning, P1/P2/P3 post-processing, or selective
sample choice.

The registered dual-view manifests store segmentation masks on the 512x512
model grid as float NPY caches while retaining the original 640x640 noisy TIFF.
Fixed component size and local-contrast strata are therefore explicitly defined
on `model_grid_px`: noisy images are area-resampled to the mask grid, masks use
nearest/discrete membership, and no fabricated physical or original-pixel scale
is reported.
