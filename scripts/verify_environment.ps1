$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$yolo = Join-Path $projectRoot '.venv\Scripts\yolo.exe'
$env:YOLO_CONFIG_DIR = Join-Path $projectRoot 'environment\ultralytics_config'
[void](New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw "Missing Python: $python" }
if (-not (Test-Path -LiteralPath $yolo -PathType Leaf)) { throw "Missing YOLO CLI: $yolo" }

& $python -u -c @'
import cv2
import pandas
import sklearn
import skimage
import torch
import torchvision
import ultralytics

assert torch.cuda.is_available(), "CUDA is not available"
x = torch.randn((512, 512), device="cuda")
y = x @ x
torch.cuda.synchronize()

print("Python:       ", __import__("sys").version.split()[0])
print("PyTorch:      ", torch.__version__)
print("Torchvision:  ", torchvision.__version__)
print("CUDA runtime: ", torch.version.cuda)
print("GPU:          ", torch.cuda.get_device_name(0))
print("Capability:   ", torch.cuda.get_device_capability(0))
print("Ultralytics:  ", ultralytics.__version__)
print("OpenCV:       ", cv2.__version__)
print("pandas:       ", pandas.__version__)
print("scikit-learn: ", sklearn.__version__)
print("scikit-image: ", skimage.__version__)
print("GPU tensor:   ", tuple(y.shape), "OK")
'@

if ($LASTEXITCODE -ne 0) { throw 'Environment verification failed.' }

