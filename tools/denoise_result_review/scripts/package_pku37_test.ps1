param(
  [string]$ProjectRoot = (Resolve-Path ".").Path,
  [string]$OutputDir = ""
)
if (-not $OutputDir) { $OutputDir = Join-Path $ProjectRoot "runs\denoise_review_packages" }
python -m tools.denoise_result_review.cli package-test `
  --project-root "$ProjectRoot" --run-dir auto --dataset PKU37 --split test `
  --output-dir "$OutputDir" --primary-seeds-only --include-noisy --include-reference `
  --archive-by-position --archive-all --resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
