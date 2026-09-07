[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$YoloCommand = '',
    [int]$ImageSize = 640,
    [int]$Batch = 8,
    [string]$Device = '0',
    [string]$RunName = 'yolo26n_seg_grading4_baseline_seed20260906_test'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([string]::IsNullOrWhiteSpace($YoloCommand)) {
    $YoloCommand = Join-Path $projectRoot '.venv\Scripts\yolo.exe'
}
$data = Join-Path $projectRoot 'data\grading_four_class_v3\data.yaml'
$reports = Join-Path $projectRoot 'outputs\reports'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'environment\ultralytics_config'
[void](New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR)

if (-not (Test-Path -LiteralPath $YoloCommand -PathType Leaf)) { throw "Missing YOLO executable: $YoloCommand" }
if (-not (Test-Path -LiteralPath $data -PathType Leaf)) { throw "Missing dataset YAML: $data" }
$Model = (Resolve-Path -LiteralPath $Model).Path

Write-Warning 'Evaluating the frozen test split. Do not use this result for iterative tuning.'
& $YoloCommand segment val "model=$Model" "data=$data" split=test `
    "imgsz=$ImageSize" "batch=$Batch" "device=$Device" save_json=True plots=True `
    "project=$reports" "name=$RunName"

if ($LASTEXITCODE -ne 0) { throw "YOLO grading test evaluation failed with exit code $LASTEXITCODE" }
