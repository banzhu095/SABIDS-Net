# OCT denoising result review

This tool discovers locked denoising runs, summarizes method provenance, packages lossless PKU37 test assets on Linux, and performs blinded fixed-ROI review on Windows. It never trains a model or selects a checkpoint from ROI/test results.

## Install

```bash
python -m pip install -r tools/denoise_result_review/requirements.txt
```

## Discover and summarize

```bash
python -m tools.denoise_result_review.cli discover --project-root /mnt/SABIDS-Net
python -m tools.denoise_result_review.cli summarize --project-root /mnt/SABIDS-Net --run-dir auto
```

`auto` scans `runs/` and accepts a formal result only when required tables, registry, lock, denoised manifest, PKU37 test rows, all nine method identities, and unique logical keys are present. Missing SABIDS-current or TCFL-DnCNN remains `missing/not_completed`.

## Package PKU37 test on Juchiyun

Run a size-only check first, then the resumable package:

```bash
python -m tools.denoise_result_review.cli package-test --project-root /mnt/SABIDS-Net --run-dir auto --dataset PKU37 --split test --output-dir /mnt/SABIDS-Net/runs/denoise_review_packages --primary-seeds-only --include-noisy --include-reference --archive-by-position --archive-all --dry-run
bash tools/denoise_result_review/scripts/package_pku37_test.sh
```

Progress and outputs are in `runs/denoise_review_packages/PKU37_test_primary/`. Download the ZIP for each position, or the total ZIP after its CRC status is `passed` in `manifests/archive_inventory.csv`. Re-run with `--resume`; matching files and valid archives are retained. A hash conflict stops that asset. Use `--positions pku_0006,pku_0017`, `--samples pku_0006_f26`, or `--methods bm3d_standard,dncnn_paired`. Use `--include-all-seeds` only for a separate seed-sensitivity package.

`auto` selects the newest package-ready run: all eight required audit/config/metric files, real PKU37 `test` rows, and no duplicate logical keys. Missing methods remain `missing/not_completed` and produce `completed_with_missing_methods`; rerun with `--resume` after SABIDS/TCFL arrive. `--allow-incomplete-run` is an explicit development-smoke escape hatch for a structurally incomplete or non-test run; it does not relabel `val` as `test`, and its outputs are never confirmatory.

When SABIDS/TCFL finishes later, rerun the same command against the newly discovered complete run. Already identical package files are not overwritten.

## Windows extraction and audit

Extract to:

```text
E:\1-脉络膜\OCT降噪\SABIDS-Net\Downloaded_Denoise_Review\PKU37_test_primary
```

Then run:

```powershell
python -m tools.denoise_result_review.cli audit-local `
  --input-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\Downloaded_Denoise_Review\PKU37_test_primary" `
  --output-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\ROI_Denoise_Analysis"
```

The audit verifies manifest uniqueness, SHA256, decoding, dtype, bit depth, shapes, method completeness, primary seeds, and ZIP CRC. It creates `manifests/roi_candidate_samples.csv` by exact historical candidate IDs only.

## Blinded ROI selection and locking

```powershell
python -m tools.denoise_result_review.cli select-roi `
  --input-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\Downloaded_Denoise_Review\PKU37_test_primary" `
  --selection-list "E:\1-脉络膜\OCT降噪\SABIDS-Net\ROI_Denoise_Analysis\manifests\roi_candidate_samples.csv" `
  --output-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\ROI_Denoise_Analysis" --roi-size 48
```

Before locking, the GUI shows only noisy and reference. Keys: `1` vitreous, `2` retina, `3` choroid vessel, `4` choroid stroma, `5` custom, `S` save, `D` delete last ROI, `N/P` navigate, `L` lock, `Q` save and quit. Clicking chooses the square center in original image coordinates. Edge-crossing ROIs are rejected rather than shifted.

To revise locked coordinates, run `select-roi --unlock-rois --unlock-reason "reason"`; the prior registry and reason are versioned.

## Evaluate, create panels, Excel, and report

```powershell
python -m tools.denoise_result_review.cli evaluate-roi `
  --input-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\Downloaded_Denoise_Review\PKU37_test_primary" `
  --roi-registry "E:\1-脉络膜\OCT降噪\SABIDS-Net\ROI_Denoise_Analysis\roi_registry.csv" `
  --output-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\ROI_Denoise_Analysis" --require-locked-rois --primary-seeds-only --resume

python -m tools.denoise_result_review.cli build-report `
  --project-root . --run-dir auto `
  --input-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\Downloaded_Denoise_Review\PKU37_test_primary" `
  --output-root "E:\1-脉络膜\OCT降噪\SABIDS-Net\ROI_Denoise_Analysis"
```

The one-command equivalent is `tools/denoise_result_review/scripts/run_roi_review_windows.ps1`. It stops after the GUI if ROIs are not locked. Noninteractive recovery uses `evaluate-roi --resume` followed by `build-report`.

Quantitative reads use dtype-fixed `[0,1]` conversion (`uint8/255`, `uint16/65535`, float/manifest range). There is no per-image min-max, histogram equalization, CLAHE, gamma, resize, or JPEG metric path. Adding a new method requires a unique `method_id`, registry entry, per-image logical keys, and literature row; then add it to `METHOD_ORDER` and rerun discovery/tests.
