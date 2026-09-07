# PKU37 v2 next-stage experiments

This development protocol is validation-only. Threshold is fixed at 0.5 and
network output is raw P0. Duke17/28 are never training or checkpoint-selection
datasets. Sealed test assets are not opened by protocol refresh, training,
reporting, or packaging.

## Mandatory gate

Run `python tools/refresh_pku37_binary_protocol.py --project-root . --write`.
Training is permitted only when `Manifests/pku37_binary_v2/protocol_audit.json`
reports `status: passed`. A disagreement among existing sealed splits is a hard
blocker and must be resolved by a human; the tool never invents a split.

## Implemented scope

- D0 and residual D1 configurations use PKU37 only. D1 combines Charbonnier,
  MS-SSIM, clean-edge-weighted multiscale gradient, and multiscale Laplacian
  losses. Raw D1 components are emitted in the ordinary loss dictionary/history.
- Strong decoder interaction supports per-sample RMS-normalized residual
  injection and a 20% epoch ramp. Sources remain detached in confirm configs.
- Report and lightweight tar tools exclude test-named paths, checkpoints,
  arrays, Data, Label, and caches.

## Deliberate blockers

The ordered DS/SD/ALT YAML contracts are present, but execution is refused by
the suite runner until one continuous optimizer/global-step state machine is
implemented. Running three independent `train.py` jobs would not be the stated
experiment. External Duke evaluation likewise requires an explicit audited
development manifest and never infers a split.

Smoke checks are engineering checks and must not be reported as performance.
