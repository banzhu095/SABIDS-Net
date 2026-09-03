# I-NOISY / I-DENOISED / I-CLEAN fixed-final visualization

- Seed: 42
- Checkpoint: `last.pth`; required completed epoch: 60
- Threshold: 0.5; postprocess: P0
- Selection: lexicographically first valid sample ID per outer-validation position
- CLEAN is evaluated once per anatomical position and is not duplicated into repeat frames.
- Layer overlay: RGB(0,255,0), alpha=0.23. Vessel overlay: RGB(255,0,0), alpha=0.40.
- Probabilities are restored to original geometry with bilinear interpolation; masks/GT use nearest-neighbor restoration in the evaluator.
- Test denylist was applied before any image read. Test assets opened: 0.
- Inspect `audit/AUDIT_SUMMARY.md`: existing input-segment training fine-tuned the shared encoder despite the inherited freeze flag.
