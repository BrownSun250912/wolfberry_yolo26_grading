# 写作证据台账

写作日期：2026-10-25。用户授权：根据历次真实实验撰写中英文完整初稿。本目录为新写作产物，不修改任何封存实验、权重、数据或测试协议。采用 paper-writer；Introduction 按其路由采用 intro-drafter 的内容契约。研究定位沿用 CONTRIBUTION_POSITIONING_20261025.md。

## 本地证据

以下路径均相对于 。

| ID | 来源 | 级别 | 支持 | 不支持 | 使用位置 | 风险 |
|---|---|---|---|---|---|---|
| E01 | docs/experiments/refine_module_20261022/实验报告.md；outputs/reports/refine_module_20261022/完整流程固定种子对照.csv | L1 | 原R1、学习残差、直接规则、规则锚定残差的固定种子结果；同阶段耗时 | 跨阶段毫秒横比；规则残差通过原实用门槛 | 方法、表3、补充结果 | 反复开发；原规则残差未过内部MAE门槛 |
| E02 | docs/experiments/direction_experiments_20261023/实验结果报告.md | L1 | 历史边界细化与完整分级不兼容；光照压力结果 | 真实水流或外部鲁棒性 | 结果、讨论 | 合成干预 |
| E03 | docs/experiments/joint_pipeline_20261024/实验报告.md | L1 | 真正NMS；同候选四类输出与邻域证据的区别；Direct Rule强对照；RR22 | 从IoU框簇推断独立物理果；全类别分数等于概率后验 | 方法、表2、讨论 | 阶段内排序/选择曾适应性修改 |
| E04 | docs/experiments/attribute_local_20261025/实验报告.md | L1 | 150拟合、排序信息对照、输出细化、来源审计、负结果 | 完整系统独立来源泛化；全局最优；显著AP50增益 | 全文 | 重复开发集；原缺陷排序为追加机制对照 |
| E05 | outputs/reports/attribute_local_20261025/headline_results.csv；selected_rank_study_metrics.csv；selected_rank_intervals.csv | L1 | 固定25完整流程数字和逐项配对区间 | 将AP等同分级准确率；最新排序新增33个正确分级 | 表2、4 | CI未校正选择或多重比较 |
| E06 | outputs/reports/attribute_local_20261025/native_geometry_summary.csv；prototype_val_metrics.csv；historical_boundary_control.csv | L1 | 原图几何、历史强对照、输出支路与重新评分区别 | prototype所有指标优于历史二值模型 | 表5 | 固定匹配几何不含漏检 |
| E07 | outputs/reports/attribute_local_20261025/rank_latency_summary.csv；runtime_environment.json | L1 | 同一设备、conf=.001、24图×3、解码/加载除外的实测耗时 | 极简引擎速度；历史conf=.25速度直接横比 | 表6 | 局部安装ops已有内存补丁 |
| E08 | outputs/reports/attribute_local_20261025/FROZEN_CANDIDATES_AND_TEST_PROTOCOL.json | L1 | 当前锁定方案、阈值、权重、未来终测方案 | 当前方案已在test得到结果；test从未被看过 | 方法、局限 | 早期基线曾评估test |
| E09 | docs/YOLO26_GRADING_BASELINE_AND_PAPER_PLAN.md | L1 | 历史129测试图及早期基线使用test的事实 | 该100轮模型与当前200轮冻结检测器相同 | 数据、局限 | 历史数据版本，不混用测试指标 |
| E10 | docs/FEATURE_DEFINITIONS.md；outputs/grading_standard_manual_v4_final/grading_standard_config.json | L1 | 18项特征定义、像素单位、分级公式与权重、标签构建过程 | 毫米尺寸、营养品质、行业标准符合性 | 材料、方法 | 部分等级来自规则，不是独立专家质量金标准 |
| E11 | scripts/joint_pipeline_policy_20261024.py；refine_score_residual_ablation.py；refine_rule_anchored_residual.py | L1 | 门控、边界/果梗融合、真实NMS、残差公式与训练参数 | 门控融合全条件安全 | 方法 | 保留等级0由原检测类别决定的实际实现 |
| E12 | scripts/joint_pipeline_rank_fit_20261024.py；attribute_local_rank_20261025.py | L1 | 22输入、385参数、独立多IoU目标、训练范围、class0保留raw分数 | 完整PointRend；概率已校准 | 方法 | 零通道相同分配参数不等于相同有效容量 |
| E13 | docs/YOLO26_OFFLINEAUG_E200_FROZEN_PROTOCOL_20260918.md | L1 | 原检测器训练预算、end2end=True、增强与选模 | 此处旧val数字直接替代当前valfix重放 | 方法 | 固定检测器单训练种子；数据版本已变 |

## 文献边界

正文参考文献逐条以本次检索的出版方、CVF/PMLR/NeurIPS、作者机构或官方软件文档为依据；只在已读取全文/本地原始实现时写具体方法细节。摘要级来源用于研究方向概述，不移植外部性能数字。

核验前已发现并更正的两项：Mask Scoring R-CNN实际页码6409–6418，旧综述6473–6482不采用；foods15162869实际题名为 Detection of Kernel-Level Spoilage Adulteration in Dried Goji Berries Using Zero-Shot Learning and Computer Vision，旧综述改写题名不采用。独立引用核验另存 CITATION_VERIFICATION.md。

## 贡献—证据对应

| 贡献 | 方法位置 | 证据位置 | 证据边界 |
|---|---|---|---|
| 异质规则下的分层分级和规则锚定残差 | 3.1–3.3 | 表2、3；E01/E03/E05 | 整体收益来自累计级联；规则锚定残差本身未通过原MAE实用门槛 |
| 原缺陷证据复用的质量排序 | 3.4 | 表4；E04/E05/E12 | 追加探索性机制对照；相对RR22 AP50区间跨零；分级不变 |
| 保留原分级输入的输出掩膜细化 | 3.5 | 表5；E04/E06 | 当前输出设计兼容；不等于原评分器适配新掩膜；历史模型部分分割指标更好 |

## 未完成事项的处置

当前完整方法无新test结果：正文明确只报告开发结果，不填造测试表。相机镜头、拍摄距离、独立实物/采集批次、作者单位、资助和开放许可均不臆造，写作交付说明单独列明。当前稿件为完整正文初稿，不宣称可直接投稿。
