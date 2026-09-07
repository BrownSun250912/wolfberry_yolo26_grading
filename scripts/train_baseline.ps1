[CmdletBinding()]
param(
    [string]$YoloCommand = (Join-Path $PSScriptRoot '..\.venv\Scripts\yolo.exe'),
    [int]$Epochs = 100,
    [int]$ImageSize = 640,
    [int]$Batch = 8,
    [string]$Device = '0',
    [int]$Seed = 20260823,
    [string]$RunName = 'yolo26n_seg_baseline_seed20260823'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$model = Join-Path $projectRoot 'models\pretrained\yolo26n-seg.pt'
$data = Join-Path $projectRoot 'data\split_v1\wolfberry_seg.yaml'
$runs = Join-Path $projectRoot 'models\experiments'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'environment\ultralytics_config'
[void](New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR)

if (-not (Test-Path -LiteralPath $model -PathType Leaf)) { throw "Missing pretrained model: $model" }
if (-not (Test-Path -LiteralPath $data -PathType Leaf)) { throw "Missing dataset YAML: $data" }
if (-not (Test-Path -LiteralPath $YoloCommand -PathType Leaf)) { throw "Missing YOLO executable: $YoloCommand" }

& $YoloCommand segment train "model=$model" "data=$data" "epochs=$Epochs" `
    "imgsz=$ImageSize" "batch=$Batch" "device=$Device" "seed=$Seed" `
    deterministic=True end2end=True "project=$runs" "name=$RunName"

if ($LASTEXITCODE -ne 0) { throw "YOLO training failed with exit code $LASTEXITCODE" }
