# M6：同次forward的prototype边界细化与双mask分工

| 核验项 | 材料 |
|---|---|
| 身份/最终状态 | 可选分割变体；主方案不启用 |
| 具体问题与设计动机 | 设计上只在边界带修正分割轮廓，同时避免把细化mask反馈给原分级链改变其特征。 |
| 输入/来源/维度 | 只细化raw≥.25且非空mask；ROI按原mask外接框pad=max(12,ceil(.25长边))裁剪，直接resize160²（不是R1的letterbox）；RGB/255不做ImageNet归一化。coarse mask1×160²，soft logit1×160²（clip±8/8），prototype32×40²（tanh(P/4)后resize160²）；soft/proto缓存FP16后头计算转FP32；原生mask与原生边界带。 |
| 内部结构及步骤 | 36通道(RGB3+soft1+prototype32)→1×1Conv36→32SiLU→16SiLU→1；δ=6tanh输出，最后层零初始化。原生band=dilate9(M)−erode9(M)，半径4px；δ双线性回原ROI，仅band内叠加原mask±2基准logit并>0二值化。 |
| 输出与去向 | output_masks细化轮廓用于分割评价/输出；grading_masks保持原mask用于R1/几何/排序；候选和keep索引相同。 |
| 训练目标与损失 | 4186条IoU≥.5的train匹配预测，3349/837按图像角色；GPU FP32 AdamW .001/wd.0001 batch32 max30 patience8；损失加权pixel BCE(1+4band)+Dice(smoothing1)；held原生IoU最高、相对面积误差较小择优，best27，完成30。 |
| 阈值、选择与范围 | hard/soft×无/有prototype，3种子25/26/27共12 fits；家族held平均IoU提高≥.001且面积误差不劣于+.005才合格；固定25。可选身份在test前冻结，不能按test提升为主方案。 |
| 异常/缺失回退 | 同次forward capture索引/公共结果不一致直接抛错；若不启用细化返回原mask。代码未提供细化失败静默回退原mask的统一try/except，不能假称有。 |
| 权重/兼容依赖 | `models\experiments\attribute_local_20261025\prototype_fits\soft_prototype_seed20261025\best.pt` |

实现证据：[E031] `scripts/attribute_local_proto_fit_20261025.py`:L9（`class PointHead`）；源文件逐字节副本见第08索引。设计依据是项目中的错误审计和成对控制，不把未验证的收益写成事实。通用算法名不等于本文原创；具体文献引文应由作者从已共享文献库核对原文后绑定，本包没有替作者补造引用。

模块连接可据第07的同名Mermaid图绘制；下游均按原candidate索引对齐。主方案M0→M1/M2→M3→M4→M5；M6是可选输出分支，不能并入“所有最终输出均细化”的描述。
