# Dual-task adaptive v2

This opt-in seed-42 protocol freezes the complete bound v1 model and therefore
preserves its Layer path. Only a private Vessel controller, three noisy-backed
additive feature increments (1/8, 1/4 and 1/2), and an additive vessel-logit
correction are trainable. Vessel strength is `0.5*sigmoid(z)` and starts at
0.25. Feature and logit scales start at zero, so the initial and explicit
`vessel_adaptive_off` predictions equal the bound coarse Vessel logits.

The training-only protection term is
`rank + 0.5*boundary + 0.5*weak`, with margin 0.10 and total weight 0.05.
Reconstruction weight is zero. Small/low-contrast thresholds are frozen from
development-train labels/images; validation labels are used only for evaluation.

The formal checkpoint is `best_vessel_safe.pth`. Every eligibility requirement
is applied before the preregistered Q score and earliest-epoch tie break. If no
epoch is eligible the run is reported as `BLOCKED: NO VESSEL-SAFE CHECKPOINT`;
`best_unconstrained.pth` is diagnostic and is never promoted. V1 remains an
unchanged negative control. This implementation does not establish efficacy;
only the new seed-42 validation pilot can do so.
