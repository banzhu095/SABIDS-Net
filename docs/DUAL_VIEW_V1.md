# Noisy-backed mild dual-view segmentation v1

This is an opt-in, validation-only causal experiment. It does not authorize
test access, formal CUDA training, or a claim that denoising itself improves
segmentation. Runtime caches, reports and checkpoints stay outside Git.

## Source audit: D0, D1 and D2

### D0

The implemented network is `SABIDSNet` in `sabids/models/sabids_net.py`.
`forward_denoise_only` uses the shared stem/encoder, the denoising adapters and
decoder, then predicts

```text
residual = residual_scale * tanh(residual_head)
denoised_raw = noisy - residual
denoised = clip(denoised_raw, 0, 1)
```

Thus the residual direction is `noisy-clean`, not `clean-noisy`. In the legacy
D0 objective (`SABIDSLoss._restoration`) the image term is

```text
Charbonnier(pred, clean; eps=1e-3)
+ 0.2 * three-level SSIM loss(window=7, C1=.01^2, C2=.03^2)
+ 0.1 * Haar wavelet L1(weights LL/LH/HL/HH=.25/1/1/1)
+ 0.1 * gradient L1 edge term
```

and the residual term is

```text
L1(predicted_residual, noisy-clean).
```

`configs/next_stage_v3/d1_d0_pku37_v3.yaml` applies global weights
reconstruction=1 and residual=.5. The current legacy D0/D1 restoration
reduction selects samples with clean targets but does **not** spatially mask
padding/unknown pixels inside those image losses; this is an audited fact, not
silently changed by dual-view v1. Segmentation supervision separately combines
spatial validity with label/vessel validity.

### D1

D1 uses the identical `SABIDSNet` inference network and residual direction.
Only the reconstruction objective changes (`restoration_mode=structure_d1`):

```text
1.0 * mean(sqrt((pred-clean)^2 + 1e-3^2))
+ .2 * mean_s[1-SSIM_s(pred,clean)]
+ .1 * mean_s mean((1 + beta*|grad(clean_s)|/max|grad(clean_s)|)
                     * (|dx(pred_s)-dx(clean_s)| + |dy(pred_s)-dy(clean_s)|))
+ .05 * mean_s L1(Laplacian(pred_s), Laplacian(clean_s)), s in {1,2,4}.
```

`structure_beta=2.0` acts only in the clean-edge weight of the multiscale
gradient term. The top-level residual weight is zero. The fresh formal
reproduction config is
`configs/adaptive_denoising/d1_repro_pku37_v3_seed42.yaml`; its checkpoint must
be accompanied by training-time asset evidence and its explicit checkpoint
binding. This implementation never infers those paths.

### D2

D2 also uses `SABIDSNet`; it is an independent D1-shaped denoiser, not a new
inference architecture. `D2StructureLoss` masks image supervision with spatial
validity and exposes modular weights. The registered arms in
`tools/prepare_d2_seed42.py` are:

| Arm | Additions to Char=1, MS-SSIM=.2, gradient=.1, Laplacian=.05 |
| --- | --- |
| D20 | none |
| D21 | vessel ROI=.5, stroma ROI=.1, outside ROI=.05 |
| D22 | D21 + boundary-gradient=.2 |
| D23 | D22 + vessel/stroma CNR error=.1 |
| D24 | D23 + frozen-teacher task=.2 + teacher consistency=.05 |
| D25 | D24 + structure leakage=.1 + residual amplitude=.01; clean identity=.05 |

There is no separate D2 loss named `edge`: edge preservation is represented by
the base gradient/Laplacian terms and the explicit vessel boundary-gradient
term. The top-level residual loss remains zero; D25's `.01` is the internal
`|noisy-prediction|` amplitude constraint. D24/D25 require a segmentation
teacher whose parameters have `requires_grad=false`, remain in `eval()`, and
are hashed before/after training. Recorded cloud work completed D20, D22, D24
and D25 seed-42 pilots; D21 and D23 are registered intermediates, not completed
primary comparisons.

D2 selection writes two different checkpoint identities:

- `best_pixel.pth`: maximum validation PSNR (fidelity);
- `best_task_preserving.pth`: best teacher-task preservation among epochs no
  worse than 0.2 dB from maximum PSNR.

Each has a `checkpoint_binding_*.json` containing checkpoint/source hashes,
epoch/global step, parameter audit and the original D1/teacher evidence chain.
D2 alpha=1 is reserved for a later experiment and is not mixed into phase 1.

On this local checkout the formal lock, split contract, D1 checkpoint/binding
and cloud dose registry are runtime assets and are absent. Consequently the
local formal audit must return exactly `BLOCKED: DUAL-VIEW INPUT EVIDENCE`.

## Implemented phase-1 contract

`NoisyMildDualViewSegmenter` is an independent segmentation model. It never
calls the denoising decoder. The noisy and mild images make two calls to the
same segmentation encoder parameters. At encoder levels 3/2/1 (1/8, 1/4,
1/2), it implements

```text
delta = A([N,M,N-M])
gate = sigmoid(G([N,M,abs(N-M)]))
fused = N + gamma*gate*delta
```

Every gamma starts at zero, so the initial dual output is bitwise equal to the
noisy-only path. Disabling the auxiliary input is the same noisy path. B6 uses
the exact B3 graph and parameters but supplies noisy as both views. C1 uses a
deterministic different-position mild mapping within each split. The training
history records gate distribution, gamma, feature/delta RMS, gamma/adapter/gate
gradient norms and parameter updates. Frozen denoiser parameters are audited
for zero updates.

The phase-1 matrix is B0 noisy, B1 mild, B3 noisy+mild, B6 noisy+noisy and C1
noisy+shuffled-mild. C5 is the B3 `last.pth` evaluated with
`--disable-dual-view-auxiliary`; it is not retrained. Strong, residual encoder,
D2, clean oracle, spatial-shift and blur/noise controls are deliberately not
implemented in phase 1.

Fixed vessel strata fit area thresholds from train GT and low-contrast from
train noisy only. Validation membership stores `sample_id+component_id`, the
component mask hash and its frozen bins. Evaluation rejects changed GT rather
than recomputing membership from a different arm input. Missing clean contrast
is `unknown`.

## Cloud commands (conditional on formal evidence)

Run from `/mnt/SABIDS-Net`. These variables are examples of the expected
runtime assets; do not replace a missing path with a historical checkpoint.

The supported ModelWhale interface is the single strict entry below. It calls
the real preparation, training, evaluation, audit, gate, summarization and ZIP
tools; it is not a command printer. Every run id is immutable and an existing
run/report/cache is rejected rather than overwritten.

```bash
cd /mnt/SABIDS-Net
export SABIDS_PROJECT_ROOT=/mnt/SABIDS-Net
# Override these only with the matching formally audited assets.
export SABIDS_PROTOCOL_LOCK=/mnt/SABIDS-Net/Manifests/pku37_binary_v3/active_protocol_lock.json
export SABIDS_SPLIT_CONTRACT=/mnt/SABIDS-Net/configs/data/pku37_binary_v3_split.yaml
export SABIDS_D1_RUN=/mnt/SABIDS-Net/runs/adaptive_denoising/pku37_binary_v3/d1_repro_fold0_seed42
export SABIDS_D1_DOSE_REGISTRY=/mnt/SABIDS-Net/cache/adaptive_denoising/pku37_binary_v3/dose_v1/preparations/pilot_s42_d1_best_sensitivity_v2/preparation_registry.json

RUN_ID="dual_view_v1_$(date +%Y%m%d_%H%M%S)"
bash tools/run_dual_view_modelwhale.sh preflight --run-id "$RUN_ID"
bash tools/run_dual_view_modelwhale.sh overfit   --run-id "$RUN_ID"
bash tools/run_dual_view_modelwhale.sh pilot     --run-id "$RUN_ID"
# Run only if pilot/gate.json says status=passed and formal_allowed=true.
bash tools/run_dual_view_modelwhale.sh formal    --run-id "$RUN_ID" --seeds 42,43,44
bash tools/run_dual_view_modelwhale.sh summarize --run-id "$RUN_ID"
bash tools/run_dual_view_modelwhale.sh package   --run-id "$RUN_ID" \
  --output "/mnt/SABIDS-Net/exports/GPT_light_${RUN_ID}.zip"
```

Overfit uses eight train rows for optimization and four train rows for its
diagnostic loader; it never uses validation/test for selection. Pilot is the
fixed 20-epoch seed-42 matrix. Formal is a fresh 60-epoch 42/43/44 matrix, so
the 20-epoch pilot is not mislabeled as formal seed 42. C5 is evaluated only
from B3 `last.pth`.

The pilot gate is source-frozen: B3 fixed-final vessel Dice must be strictly
higher than B0, B1 and B6; B3-B0 recall and boundary Dice may decline by at
most 0.01, and fixed-small/fixed-low-contrast recall@0.25 by at most 0.02;
B3-C1 and B3-C5 vessel Dice must be positive. A failed gate exits nonzero,
persists `gate.json`, and blocks formal. The package command can still archive
the failure, but automatically adds `_incomplete` to its filename. The three
validation positions are descriptive only and do not support population-level
inference.

The lower-level commands below document the implementation for debugging; the
script above is the authoritative execution order.

```bash
export LOCK=Manifests/pku37_binary_v3/active_protocol_lock.json
export SPLIT=configs/data/pku37_binary_v3_split.yaml
export D1_RUN=runs/adaptive_denoising/pku37_binary_v3/d1_repro_fold0_seed42
export D1_BINDING="$D1_RUN/checkpoint_binding_best_d2_v1.json"
export D1_REGISTRY=cache/adaptive_denoising/pku37_binary_v3/dose_v1/preparations/pilot_s42_d1_best_sensitivity_v2/preparation_registry.json

python tools/prepare_dual_view_inputs.py \
  --project-root . --mode audit \
  --protocol-lock "$LOCK" --split-contract "$SPLIT" \
  --d1-checkpoint "$D1_RUN/best.pth" \
  --d1-checkpoint-binding "$D1_BINDING" \
  --d1-training-asset-inventory "$D1_RUN/training_asset_inventory_initial.json" \
  --d1-dose-registry "$D1_REGISTRY"
```

If this is not `status: passed`, stop. For a CUDA overfit (4--8 samples, three
epochs), prepare a fresh tag, then run only the generated configs:

```bash
python tools/prepare_dual_view_inputs.py \
  --project-root . --mode prepare --budget overfit --seeds 42 --tag overfit_v1 \
  --protocol-lock "$LOCK" --split-contract "$SPLIT" \
  --d1-checkpoint "$D1_RUN/best.pth" --d1-checkpoint-binding "$D1_BINDING" \
  --d1-training-asset-inventory "$D1_RUN/training_asset_inventory_initial.json" \
  --d1-dose-registry "$D1_REGISTRY"

for CFG in cache/adaptive_denoising/pku37_binary_v3/dual_view_v1/overfit_v1/config_*.yaml; do
  python train.py --config "$CFG"
done
```

Require finite decreasing loss, completed budget, nonzero gamma updates and
later adapter updates in B3/B6/C1, and zero frozen changes. Then prepare the
seed-42 fixed 20-epoch pilot:

```bash
python tools/prepare_dual_view_inputs.py \
  --project-root . --mode prepare --budget pilot --seeds 42 --tag pilot_v1 \
  --protocol-lock "$LOCK" --split-contract "$SPLIT" \
  --d1-checkpoint "$D1_RUN/best.pth" --d1-checkpoint-binding "$D1_BINDING" \
  --d1-training-asset-inventory "$D1_RUN/training_asset_inventory_initial.json" \
  --d1-dose-registry "$D1_REGISTRY"

for CFG in cache/adaptive_denoising/pku37_binary_v3/dual_view_v1/pilot_v1/config_*.yaml; do
  python train.py --config "$CFG"
done
```

Freeze validation component membership from the generated B0 config:

```bash
python tools/prepare_dual_view_strata.py \
  --project-root . \
  --config cache/adaptive_denoising/pku37_binary_v3/dual_view_v1/pilot_v1/config_b0_seed42.yaml \
  --output cache/adaptive_denoising/pku37_binary_v3/dual_view_v1/pilot_v1/fixed_components.json
```

Evaluate every run's `last.pth` and `best.pth` on complete validation only,
P0/0.5, without original-geometry restoration:

```bash
export DV_ROOT=runs/adaptive_denoising/pku37_binary_v3/dual_view_v1/pilot_v1
export FIXED=cache/adaptive_denoising/pku37_binary_v3/dual_view_v1/pilot_v1/fixed_components.json
for RUN in "$DV_ROOT"/{b0,b1,b3,b6,c1}_seed42; do
  python evaluate.py --config "$RUN/resolved_config.yaml" --checkpoint "$RUN/last.pth" \
    --split val --output "$RUN/validation_last" --tasks layer vessel \
    --postprocess-modes p0 --layer-threshold 0.5 --vessel-threshold 0.5 \
    --no-restore-original-geometry --fixed-component-inventory "$FIXED"
  python evaluate.py --config "$RUN/resolved_config.yaml" --checkpoint "$RUN/best.pth" \
    --split val --output "$RUN/validation_best" --tasks layer vessel \
    --postprocess-modes p0 --layer-threshold 0.5 --vessel-threshold 0.5 \
    --no-restore-original-geometry --fixed-component-inventory "$FIXED"
done

python evaluate.py --config "$DV_ROOT/b3_seed42/resolved_config.yaml" \
  --checkpoint "$DV_ROOT/b3_seed42/last.pth" --split val \
  --output "$DV_ROOT/b3_seed42/validation_c5_last" --tasks layer vessel \
  --postprocess-modes p0 --layer-threshold 0.5 --vessel-threshold 0.5 \
  --no-restore-original-geometry --fixed-component-inventory "$FIXED" \
  --disable-dual-view-auxiliary
```

Audit and summarize:

```bash
for RUN in "$DV_ROOT"/{b0,b1,b3,b6,c1}_seed42; do
  python tools/audit_dual_view_run.py --run-dir "$RUN" \
    --output "$RUN/dual_view_audit.json" --device cuda
done

python tools/summarize_dual_view.py \
  --run-dirs "$DV_ROOT"/{b0,b1,b3,b6,c1}_seed42 \
  --output reports/adaptive_denoising/dual_view/pilot_seed42_v1
```

Stop unless the source-frozen tolerances above all pass. Only after that gate,
run the unique entry's `formal --seeds 42,43,44`; it creates one common
initialization checkpoint per seed. Test remains sealed.

Local non-scientific smoke:

```bash
python tools/smoke_dual_view.py \
  --project-root . \
  --output runs/adaptive_denoising/synthetic_dual_view_smoke_$(date +%Y%m%d_%H%M%S)
```

The allowed conclusion is generated from fixed-final paired validation
results. If B1 is not better than B0 but B3 passes all content controls, the
strongest permitted wording is: “轻度降噪视图与noisy图像具有互补信息，作为辅助视图促进分割。”
