$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'environment\ultralytics_config'
[void](New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR)

& (Join-Path $projectRoot '.venv\Scripts\Activate.ps1')
Write-Host "Activated: $projectRoot\.venv"
python --version
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

