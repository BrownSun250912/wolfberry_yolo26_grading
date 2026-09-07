[CmdletBinding()]
param(
    [string]$YoloCommand = '',
    [int]$Epochs = 100,
    [int]$ImageSize = 640,
    [int]$Batch = 8,
    [string]$Device = '0',
    [int]$Seed = 20260906,
    [int]$Patience = 30,
    [string]$RunName = 'yolo26n_seg_grading4_baseline_seed20260906'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if ([string]::IsNullOrWhiteSpace($YoloCommand)) {
    $YoloCommand = Join-Path $projectRoot '.venv\Scripts\yolo.exe'
}
$model = Join-Path $projectRoot 'models\pretrained\yolo26n-seg.pt'
$data = Join-Path $projectRoot 'data\grading_four_class_v3\data.yaml'
$runs = Join-Path $projectRoot 'models\experiments'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'environment\ultralytics_config'
[void](New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR)

if (-not (Test-Path -LiteralPath $model -PathType Leaf)) { throw "Missing pretrained model: $model" }
if (-not (Test-Path -LiteralPath $data -PathType Leaf)) { throw "Missing dataset YAML: $data" }
if (-not (Test-Path -LiteralPath $YoloCommand -PathType Leaf)) { throw "Missing YOLO executable: $YoloCommand" }

& $YoloCommand segment train "model=$model" "data=$data" "epochs=$Epochs" `
    "imgsz=$ImageSize" "batch=$Batch" "device=$Device" "seed=$Seed" `
    "patience=$Patience" workers=8 deterministic=True end2end=True cache=False `
    "project=$runs" "name=$RunName"

if ($LASTEXITCODE -ne 0) { throw "YOLO grading baseline training failed with exit code $LASTEXITCODE" }
