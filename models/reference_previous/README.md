# 历史参考模型

`best_previous_1285.pt` 是旧 1028/257 train/val 实验的最佳权重，可立即用于推理、导出 mask
和开发 OpenCV 特征流程。

它不能作为新 `split_v1` 的无偏正式 test 模型：旧训练/验证数据与新 test 存在交叉，且旧 val
参与过最佳轮次选择。新实验必须从 `models/pretrained/yolo26n-seg.pt` 重新训练。

