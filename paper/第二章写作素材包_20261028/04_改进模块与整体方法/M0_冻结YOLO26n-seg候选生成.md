# M0：冻结YOLO26n-seg候选生成

| 核验项 | 材料 |
|---|---|
| 身份/最终状态 | 原有，最终主方案 |
| 具体问题与设计动机 | 提供整图多果定位、四类初判和实例掩膜；最终不通过改骨干实现分级改进。 |
| 输入/来源/维度 | 一张BGR uint8 H×W×3图像，内部转换/letterbox；imgsz640，test缓存实际384×640。 |
| 内部结构及步骤 | 检测器eval/FP32，conf=.001、max_det=300、retina_masks=True、iou=.7；top-k公共输出的box/class/conf与多边形依次保留。输出多边形栅格化为原生图尺寸M_i。 |
| 输出与去向 | N行boxes(N×4)、baseline(N)、raw confidence(N)、原掩膜列表N；可选同次forward的soft logits、32通道prototype和candidate index。 |
| 训练目标与损失 | 历史200轮四类实例分割；后续所有模块训练与最终推理期间完全冻结。 |
| 阈值、选择与范围 | 正式训练best16，权重与sha在第03目录；阈值由冻结协议统一，非按单张test调节。 |
| 异常/缺失回退 | 无预测返回空数组；候选/box/mask行数或索引谱系不一致抛错，不猜测匹配。 |
| 权重/兼容依赖 | `models\experiments\yolo26n_seg_offlineaug_e200_mask50_seed20260915\weights\best.pt` |

实现证据：[E026] `scripts/attribute_local_runtime_20261025.py`:L59（`self.options=dict`）；源文件逐字节副本见第08索引。设计依据是项目中的错误审计和成对控制，不把未验证的收益写成事实。通用算法名不等于本文原创；具体文献引文应由作者从已共享文献库核对原文后绑定，本包没有替作者补造引用。

模块连接可据第07的同名Mermaid图绘制；下游均按原candidate索引对齐。主方案M0→M1/M2→M3→M4→M5；M6是可选输出分支，不能并入“所有最终输出均细化”的描述。
