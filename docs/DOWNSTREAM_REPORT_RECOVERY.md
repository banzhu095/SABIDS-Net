# Report-only recovery of completed downstream experiments

The 2026-10-13 report fix accepts equivalent duplicate denoising source copies
only after checking dataset, split, method, sample, seed and every source column.
Deep denoiser inputs remain primary seed 42. Numeric round-trip tolerance is
`rtol=1e-12, atol=1e-15`; identities and missingness are exact. Values are never
averaged across sources. Different metrics still fail closed with the key and
column in the exception. The result writes `denoise_join_status.json`.

For an already completed formal experiment, resume **only** report/package:

```bash
python -m tools.downstream_seg_benchmark.cli run-formal \
  --project-root /mnt/SABIDS-Net \
  --source-denoise-run "$SOURCE_DENOISE_RUN" \
  --run-dir "$RUN" --batch-size 1 --patch-size 384 --num-workers 4 \
  --from-stage report --resume
```

All arguments must match the original sealed request. Recovery verifies the
successful stage output hashes before any refresh. To intentionally rebuild a
successful report and its package, add `--refresh-report`; this option is rejected
for training or test stages and copies the previous report/workbook receipts.
Recovery verifies the
sealed cohort, labels, shared randomness files, and successful train, lock,
evaluate and atlas stage output hashes. It retains the original training source
hash and input signature, while recording the new postprocessing source hash
under `postprocessing_recovery/<UTC timestamp>/provenance.json`. It never calls
training, checkpoint selection, test inference or atlas generation. Failed
stage markers, error records and any prior report/workbook are copied to that
recovery directory before replacement. Training/evaluation source checks remain
strict and cannot use this reporting-only recovery path.

The report, workbook, cross-metric join and lightweight ZIP are regenerated.
Old failure evidence stays in the recovery folder; root failure status becomes
resolved only when packaging is reached. The live package stage marker is not
embedded while it still says running. Final `stages/package.json` and
`package_verification.json` remain external SHA/CRC receipts. ZIP contents have
an embedded per-file SHA256 manifest, omit model weights and full predictions,
and include metric tables, fixed atlas images, training curves, configs,
checkpoint inventory, source snapshot and recovery provenance.
