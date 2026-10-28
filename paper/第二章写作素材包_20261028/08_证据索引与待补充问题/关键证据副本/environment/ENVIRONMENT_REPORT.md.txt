# 环境验收记录

验收日期：2026-08-23

## 硬件与核心版本

- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，8151 MiB
- NVIDIA 驱动：591.86（驱动报告 CUDA 13.1）
- Python：3.12.13
- PyTorch：2.13.0+cu130
- Torchvision：0.28.0+cu130
- Ultralytics：8.4.117
- OpenCV：5.0.0
- pandas：3.0.5
- scikit-learn：1.9.0
- scikit-image：0.26.0

## 已通过的检查

- `pip check`：无依赖冲突；
- `torch.cuda.is_available()`：True；
- CUDA 运行时：13.0；
- GPU 计算能力：12.0；
- GPU 512×512 矩阵乘法：通过；
- YOLO26n-seg 权重加载：通过；
- train/val 图片和 YOLO segmentation 标签加载：通过；
- 1 个 epoch、1% train 数据、GPU 反向传播冒烟训练：通过；
- 旧最佳模型在 GPU 上单图实例分割：通过，检测到 5 个实例；
- 单图推理时间：约 7.5 ms（不含预处理和后处理，仅作环境验收，不作为正式基准）。

冒烟训练已通过；其临时输出在项目整理时删除，因为这些权重和指标不是正式实验结果。

## 数据质量提示

冒烟扫描发现 YOLO 标签中有完全相同外接框的重复实例。全量只读核验共找到 35 个，分布在
train 26 个、val 5 个、test 4 个。Ultralytics 会自动移除完全重复框，但部分同一果实的两条
轮廓略有差异，不一定被自动识别。因此在正式基线训练前需要发布清洗后的新标签版本，不能把
冒烟指标作为基线结果。
