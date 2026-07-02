# astra_eval.ps1 — Quick launcher cho ASTRA (Windows PowerShell)
# Chạy Baseline + Full ASTRA cho model được chọn

param(
    [ValidateSet("2B", "4B", "8B")]
    [string]$Model = "2B",
    [string]$Split = "test",
    [string]$OutputDir = "outputs\astra",
    [string]$Device = "cuda",
    [int]$NPerms = 3,
    [int]$MaxSamples = 0
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
Set-Location $ProjectDir

$VARIANTS = @("baseline", "ASTRA_full")

$ModelName = switch ($Model) {
    "2B" { "Qwen3-VL-2B" }
    "4B" { "Qwen3-VL-4B" }
    "8B" { "Qwen3-VL-8B" }
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  ASTRA Evaluation — $ModelName" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Split:   $Split"
Write-Host "Output:  $OutputDir\$Model"
Write-Host "Device:  $Device"
Write-Host "========================================" -ForegroundColor Cyan

foreach ($variant in $VARIANTS) {
    Write-Host ""
    Write-Host "[$variant]" -ForegroundColor Yellow

    $outDir = "$OutputDir\$Model\$variant"
    $outFile = "$outDir\results.jsonl"
    New-Item -ItemType Directory -Force -Path $outDir | Out-Null

    $args = @(
        "eval",
        "--model", $ModelName,
        "--split", $Split,
        "--output", $outFile,
        "--device", $Device,
        "--n-perms", $NPerms
    )
    if ($variant -eq "baseline") {
        $args += "--baseline"
    }
    if ($MaxSamples -gt 0) {
        $args += @("--max-samples", $MaxSamples)
    }

    python main.py $args
}

Write-Host ""
Write-Host "[Compare] Summary..." -ForegroundColor Yellow
python main.py compare --results-dir "$OutputDir\$Model" --save "$OutputDir\$Model\summary.json"

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Done! Results in: $OutputDir\$Model" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
