param(
  [string]$ProjectRoot = (Resolve-Path ".").Path,
  [string]$InputRoot = "E:\1-脉络膜\OCT降噪\SABIDS-Net\Downloaded_Denoise_Review\PKU37_test_primary",
  [string]$OutputRoot = "E:\1-脉络膜\OCT降噪\SABIDS-Net\ROI_Denoise_Analysis",
  [ValidateSet(32,48,64)][int]$RoiSize = 48
)
Set-Location -LiteralPath $ProjectRoot
python -m tools.denoise_result_review.cli run-all-local `
  --project-root "$ProjectRoot" --run-dir auto --input-root "$InputRoot" `
  --output-root "$OutputRoot" --roi-size $RoiSize --primary-seeds-only --resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
