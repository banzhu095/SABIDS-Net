param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [string]$OutputPath
)

$ErrorActionPreference = "Stop"

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$runRoot = (Resolve-Path -LiteralPath $RunDir).Path
$packageRoot = Join-Path $runRoot "packages"
New-Item -ItemType Directory -Path $packageRoot -Force | Out-Null

if ([string]::IsNullOrWhiteSpace($OutputPath)) {
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputPath = Join-Path $packageRoot "SABIDS_classical_benchmark_GPT_light_$stamp.zip"
}
$outputFull = [System.IO.Path]::GetFullPath($OutputPath)
if (Test-Path -LiteralPath $outputFull) {
    throw "Refusing to overwrite existing package: $outputFull"
}

$staging = Join-Path $packageRoot ("_gpt_light_staging_" + [guid]::NewGuid().ToString("N"))
$stagingFull = [System.IO.Path]::GetFullPath($staging)
$packageRootFull = [System.IO.Path]::GetFullPath($packageRoot)
$requiredPrefix = $packageRootFull.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $stagingFull.StartsWith($requiredPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Unsafe staging path: $stagingFull"
}
New-Item -ItemType Directory -Path $stagingFull | Out-Null

function Copy-PackageFile {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$DestinationRelative
    )
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "Required package input is missing: $Source"
    }
    $destination = Join-Path $stagingFull $DestinationRelative
    New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $destination
}

try {
    $runRelativeFiles = @(
        "benchmark_summary.xlsx",
        "failures.csv",
        "asset_inventory.csv",
        "configs\locked_method_configs.yaml",
        "audit\acceptance_checks.csv",
        "audit\adapter_contract_tests.csv",
        "audit\dataset_inventory.csv",
        "audit\environment_versions.csv",
        "audit\fixed_atlas_selection.csv",
        "audit\method_inventory.csv",
        "audit\method_inventory.md",
        "audit\missing_or_ambiguous_pairs.csv",
        "audit\paired_manifest.csv",
        "audit\pairing_audit.csv",
        "audit\split_audit.md",
        "metrics\bootstrap_confidence_intervals.csv",
        "metrics\overall_metrics.csv",
        "metrics\paired_method_differences.csv",
        "metrics\parameter_search_results.csv",
        "metrics\per_dataset_metrics.csv",
        "metrics\per_image_metrics.csv",
        "metrics\per_position_metrics.csv",
        "metrics\runtime_summary.csv",
        "metrics\selected_parameters.csv",
        "metrics\small_scale_validation.csv",
        "metrics\smoke_test_metrics.csv",
        "logs\calibration_failures.csv",
        "logs\resource_observation.md",
        "logs\smoke_failures.csv",
        "reports\benchmark_report.md",
        "reports\smoke_test_report.md",
        "reports\workbook_post_import_validation.json",
        "reports\workbook_verification.json"
    )
    foreach ($relative in $runRelativeFiles) {
        Copy-PackageFile -Source (Join-Path $runRoot $relative) -DestinationRelative (Join-Path "experiment" $relative)
    }

    $repoRelativeFiles = @(
        "docs\EXPERIMENT_LOG.md",
        "tests\test_oct_denoise_benchmark.py",
        "tools\oct_denoise_benchmark\README.md",
        "tools\oct_denoise_benchmark\adapters.py",
        "tools\oct_denoise_benchmark\benchmark.py",
        "tools\oct_denoise_benchmark\default_config.yaml",
        "tools\oct_denoise_benchmark\method_inventory.py",
        "tools\oct_denoise_benchmark\build_benchmark_workbook.mjs",
        "tools\oct_denoise_benchmark\validate_benchmark_workbook.mjs",
        "tools\oct_denoise_benchmark\build_gpt_light_package.ps1"
    )
    foreach ($relative in $repoRelativeFiles) {
        Copy-PackageFile -Source (Join-Path $projectRoot $relative) -DestinationRelative (Join-Path "reproduction" $relative)
    }

    $forbiddenExtensions = @(".tif", ".tiff", ".png", ".jpg", ".jpeg", ".pth", ".pt", ".ckpt", ".onnx")
    $stagedFiles = Get-ChildItem -LiteralPath $stagingFull -Recurse -File
    $forbidden = $stagedFiles | Where-Object { $forbiddenExtensions -contains $_.Extension.ToLowerInvariant() }
    if ($forbidden) {
        throw "Forbidden image/model assets entered the package: $($forbidden.FullName -join ', ')"
    }

    $manifestRows = foreach ($file in $stagedFiles | Sort-Object FullName) {
        [pscustomobject]@{
            relative_path = [System.IO.Path]::GetRelativePath($stagingFull, $file.FullName).Replace("\", "/")
            size_bytes = $file.Length
            sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    }
    $manifestPath = Join-Path $stagingFull "PACKAGE_MANIFEST.csv"
    $manifestRows | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8

    Compress-Archive -Path (Join-Path $stagingFull "*") -DestinationPath $outputFull -CompressionLevel Optimal

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($outputFull)
    try {
        $entryNames = @($archive.Entries | ForEach-Object { $_.FullName })
        $badEntries = @($entryNames | Where-Object {
            $extension = [System.IO.Path]::GetExtension($_).ToLowerInvariant()
            $forbiddenExtensions -contains $extension
        })
        if ($badEntries.Count -gt 0) {
            throw "Forbidden entries found after packaging: $($badEntries -join ', ')"
        }
        foreach ($entry in $archive.Entries | Where-Object { -not [string]::IsNullOrEmpty($_.Name) }) {
            $stream = $entry.Open()
            try {
                $buffer = New-Object byte[] 65536
                while ($stream.Read($buffer, 0, $buffer.Length) -gt 0) { }
            }
            finally {
                $stream.Dispose()
            }
        }
        $entryCount = $archive.Entries.Count
    }
    finally {
        $archive.Dispose()
    }

    $zipHash = (Get-FileHash -LiteralPath $outputFull -Algorithm SHA256).Hash.ToLowerInvariant()
    [pscustomobject]@{
        output_path = $outputFull
        size_bytes = (Get-Item -LiteralPath $outputFull).Length
        entry_count = $entryCount
        sha256 = $zipHash
        excluded_full_outputs = $true
        excluded_raw_clean_images = $true
        excluded_sealed_test_assets = $true
    } | ConvertTo-Json -Depth 3
}
finally {
    if (Test-Path -LiteralPath $stagingFull) {
        Remove-Item -LiteralPath $stagingFull -Recurse -Force
    }
}
