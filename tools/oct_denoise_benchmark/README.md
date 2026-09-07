# Classical OCT denoising benchmark reproduction

Run from the SABIDS-Net repository root. Python 3.9+, MATLAB R2024a (only for auditing the protected packages), and the packages imported by `tools/oct_denoise_benchmark/adapters.py` are required. Sealed test is excluded by default.

## One-command run

```powershell
python -m tools.oct_denoise_benchmark.benchmark all `
  --project-root . `
  --run-dir "E:\1-脉络膜\OCT降噪\SABIDS-Net\SABIDS-Net\runs\denoise_classical_benchmark_20260902_224920" `
  --methods noisy_identity bm3d nlm_speckle wavelet tv gaussian msbtd ascibp
```

The equivalent auditable stages are:

```powershell
$run = "E:\1-脉络膜\OCT降噪\SABIDS-Net\SABIDS-Net\runs\denoise_classical_benchmark_20260902_224920"
$methods = @("noisy_identity", "bm3d", "nlm_speckle", "wavelet", "tv", "gaussian", "msbtd", "ascibp")
python -m tools.oct_denoise_benchmark.benchmark audit --project-root . --run-dir $run --methods $methods
python -m tools.oct_denoise_benchmark.benchmark smoke --project-root . --run-dir $run --methods $methods
python -m tools.oct_denoise_benchmark.benchmark calibrate --project-root . --run-dir $run --methods $methods --workers 6
python -m tools.oct_denoise_benchmark.benchmark run --project-root . --run-dir $run --methods $methods --workers 6
python -m tools.oct_denoise_benchmark.benchmark summarize --project-root . --run-dir $run
```

Resume full inference with the `run` command and the same run directory. Completed images are skipped only when the locked method configuration and adapter source hash match. A changed adapter intentionally receives a new source/config hash and is not mixed with the previous output.

The optional `--include-sealed-test` switch exists for an explicitly authorized final test run. Do not use it for development, calibration, method selection or report iteration.

## Workbook

The Excel summary is built with `@oai/artifact-tool` after the CSV summaries and acceptance checks exist:

```powershell
$node = "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
$modules = "C:\Users\ASUS\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
if (-not (Test-Path tools\oct_denoise_benchmark\node_modules)) {
  New-Item -ItemType Junction -Path tools\oct_denoise_benchmark\node_modules -Target $modules | Out-Null
}
& $node tools\oct_denoise_benchmark\build_benchmark_workbook.mjs "E:\1-脉络膜\OCT降噪\SABIDS-Net\SABIDS-Net\runs\denoise_classical_benchmark_20260902_224920" "E:\1-脉络膜\OCT降噪\SABIDS-Net\SABIDS-Net\runs\denoise_classical_benchmark_20260902_224920\benchmark_summary.xlsx"
$builderExit = $LASTEXITCODE
& $node tools\oct_denoise_benchmark\validate_benchmark_workbook.mjs "E:\1-脉络膜\OCT降噪\SABIDS-Net\SABIDS-Net\runs\denoise_classical_benchmark_20260902_224920\benchmark_summary.xlsx" "E:\1-脉络膜\OCT降噪\SABIDS-Net\SABIDS-Net\runs\denoise_classical_benchmark_20260902_224920\reports\workbook_post_import_validation.json"
if ($LASTEXITCODE -ne 0) { throw "Workbook round-trip validation failed." }
if ($builderExit -ne 0) { Write-Warning "The Windows artifact-tool process reported a cleanup-stage exit after export; the saved workbook passed independent round-trip validation." }
```

On the bundled Windows runtime, artifact-tool may report `0xC0000409` during
native process cleanup after the workbook and previews have already been
written. The separate import/inspect validator is therefore the authoritative
success gate; it must exit zero.

Re-run `summarize` after any completed inference extension; it deterministically rebuilds position/dataset/overall aggregates, 10,000-sample bootstrap intervals (`seed=42`), the fixed atlas, report, acceptance table and asset inventory.
