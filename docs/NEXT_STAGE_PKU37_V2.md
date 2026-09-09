# PKU37 v2 next-stage experiments

This development protocol is validation-only. Threshold is fixed at 0.5 and
network output is raw P0. Duke17/28 are never training or checkpoint-selection
datasets. Sealed test assets are not opened by protocol refresh, training,
reporting, or packaging.

## Mandatory gate

Do not regenerate or rename the protocol while D1 is active. After D1 stops,
extract its protocol facts with `tools/audit_active_d1.py --write-lock`. All
subsequent launch, evaluation, report and package commands require the resulting
`Manifests/<protocol_id>/active_protocol_lock.json`. A missing fact or a D1-seed
disagreement is a hard blocker. The lock records sealed-test IDs as metadata;
the tools do not open those assets.

`initialization_audit.json` contains a seed-dependent sampler schedule digest.
New runs expose it as `sampler_plan_sha256`; its historical
`data_plan_sha256` alias is not protocol identity and is never used to construct
the active lock. The authoritative protocol digest is read from the resolved
run configuration (including its embedded active lock). Candidate evidence is
written to `runs/reports/d1_active_protocol_audit/`. If unrelated historical D1
runs are present, pass their exact intended replacements with `--run-dirs`;
explicit selection does not bypass any consistency check.

## Implemented scope

- D0 and residual D1 configurations use PKU37 only. D1 combines Charbonnier,
  MS-SSIM, clean-edge-weighted multiscale gradient, and multiscale Laplacian
  losses. Raw D1 components are emitted in the ordinary loss dictionary/history.
- Strong decoder interaction supports per-sample RMS-normalized residual
  injection and a 20% epoch ramp. Sources remain detached in confirm configs.
- DS/SD/ALT are one continuous training job per arm. Model, AdamW, global-step
  cosine scheduler, AMP scaler, RNG and phase state are checkpointed together;
  frozen task decoders are placed in eval mode.
- Input probes train a neutral segmentation model against noisy, D0, D1 or
  aligned clean inputs. D0/D1 are prepared as external float caches and are not
  loaded into the segmentation encoder.
- Shuffle and receiver-capacity controls use the same UGBI parameterization.
  Shuffle maps are fixed, within split, cross-position derangements.
- Report and lightweight tar tools exclude test-named paths, checkpoints,
  arrays, Data, Label, and caches.

## Required ordering

1. Finish/audit D0 and D1 and create the active protocol lock.
2. Create neutral order/input anchors; prepare D0/D1 input caches.
3. Run input probes and the three order arms.
4. Evaluate fixed-final order checkpoints and build the order report. Only a
   complete three-seed report writes `interaction_anchor_selection.json`.
5. Run seven rho pilots and build their report/strength lock.
6. Run J00/J10/J01/J11, then the predeclared shuffle/self-adapter controls.

The launcher is fail-closed: it does not overwrite existing runs, requires an
explicit protocol lock, preserves stale lock files for inspection, and blocks a
new full GPU suite while another project training process is active unless the
user explicitly passes `--allow-concurrent-training`.

External Duke evaluation still requires an explicit audited development
manifest and never infers a split.

Smoke checks are engineering checks and must not be reported as performance.
