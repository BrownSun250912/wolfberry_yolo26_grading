[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$YoloCommand = (Join-Path $PSScriptRoot '..\.venv\Scripts\yolo.exe'),
    [int]$ImageSize = 640,
    [int]$Batch = 8,
    [string]$Device = '0',
    [string]$RunName = 'final_test'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$data = Join-Path $projectRoot 'data\split_v1\wolfberry_seg.yaml'
$reports = Join-Path $projectRoot 'outputs\reports'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'environment\ultralytics_config'
[void](New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR)
$Model = (Resolve-Path -LiteralPath $Model).Path

if (-not (Test-Path -LiteralPath $YoloCommand -PathType Leaf)) { throw "Missing YOLO executable: $YoloCommand" }

Write-Warning 'This command evaluates the frozen test set. Do not use its result for iterative tuning.'
& $YoloCommand segment val "model=$Model" "data=$data" split=test `
    "imgsz=$ImageSize" "batch=$Batch" "device=$Device" save_json=True plots=True `
    "project=$reports" "name=$RunName"

if ($LASTEXITCODE -ne 0) { throw "YOLO test evaluation failed with exit code $LASTEXITCODE" }
