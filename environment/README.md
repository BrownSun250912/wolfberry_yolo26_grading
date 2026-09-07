# 环境说明

环境已配置在项目根目录 `.venv`。旧 `instance_seg_demo/.venv` 不再使用。

`requirements_previous.txt` 是旧成功训练环境的版本快照，仅用于追溯；`requirements.in` 是本次
直接依赖，`requirements-lock.txt` 是实际安装后的完整版本锁。

本次环境使用 Python 3.12、PyTorch 2.13 + CUDA 13.0 和 Ultralytics 8.4.117。CUDA 运行库已包含
在 PyTorch wheel 内，不需要单独安装完整 CUDA Toolkit。

在项目根目录激活环境：

```powershell
. .\scripts\activate.ps1
```

随时可重新验证环境：

```powershell
.\scripts\verify_environment.ps1
```

训练和评估脚本使用 `.venv\Scripts\yolo.exe` 的绝对项目路径，因此不激活环境也可以运行。
