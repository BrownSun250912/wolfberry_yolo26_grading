# 图表与原始证据对应

项目根目录为 。两份正文采用相同编号。本文件是写作追溯索引，不属于投稿正文。

| 图表 | 证据来源（项目根目录下） | 取数及解释规则 |
|---|---|---|
| 表1 | data/grading_four_class_v3_valfix_20261006/audit/r1_multitask_targets.csv | 按split、image_id、final_class_id计数；原多任务目标仅train/val；新增test行来自原v3 label_manifest及final_test_preflight核验；内部773/193来自最新报告 |
| 表2，当前七行参考/候选 | outputs/reports/attribute_local_20261025/headline_results.csv | 固定25；历史18维固定24；AP×100保留两位；correct/allGT分母518 |
| 表2，Original R1/learned residual/direct rule三行及direct rule边界行 | docs/experiments/joint_pipeline_20261024/实验报告.md，第3节“正常图完整对照” | 同等真实NMS，全518GT；不混入前阶段不同实现耗时 |
| 表3 | docs/experiments/refine_module_20261022/实验报告.md，最新固定种子表；outputs/reports/refine_module_20261022/完整流程固定种子对照.csv | 评分MAE仅448有目标实例；历史24图×5同轮耗时 |
| 表4 | outputs/reports/attribute_local_20261025/selected_rank_study_metrics.csv；attribute_rank_control_metrics.csv；selected_rank_intervals.csv | 固定seed25与全部seed25/26/27；附加信息与原始18维分开 |
| 表5 | outputs/reports/attribute_local_20261025/fixed_candidate_metrics.csv；native_geometry_summary.csv；historical_boundary_control.csv | AP为重匹配完整流程；原图几何固定500匹配；不能据类别不变保证匹配不变 |
| 表6 | outputs/reports/attribute_local_20261025/rank_latency_summary.csv；runtime_environment.json | 原始pool .001，24×3；含全流程实际计算，排除加载/解码 |
| 表7 | docs/experiments/attribute_local_20261025/实验报告.md，“原缺陷排序在全部扰动上”；outputs/reports/attribute_local_20261025/rank_stress_metrics.csv | 固定25、原缺陷排序；正确/全GT统一除518；各条件定位数不同；十条件原表保留 |
| 表A1 | docs/FEATURE_DEFINITIONS.md | 18项可解释描述特征，不等于六维grader几何或22维rank输入 |
| 表A2 | scripts/joint_pipeline_rank_fit_20261024.py:inputs；scripts/attribute_local_rank_20261025.py:features | 按实际输入顺序；最后一通道原缺陷版本为零 |
| 表A3 | outputs/reports/attribute_local_20261025/headline_results.csv | class0–3_Mask_AP50/AP95乘100；不合格逐类AP与操作风险分开 |
| 图1 | scripts/joint_pipeline_policy_20261024.py；attribute_local_runtime_20261025.py；attribute_local_rank_20261025.py | 从已实现计算图整理的Mermaid流程，不是水流装置或实物照片 |
| 图2 | outputs/reports/attribute_local_20261025/figures/08_rank_information_controls.png | 原图复用；数据源与图注见同目录captions.md |
| 图3 | outputs/reports/attribute_local_20261025/figures/05_geometry_sensitivity.png | 原图复用；固定500几何队列、全部边界宽度 |
| 图4 | outputs/reports/attribute_local_20261025/figures/10_qualitative_best_and_worst.png | 原图复用；GT选例只用于说明，不称为代表性频率 |
| 图5 | outputs/reports/attribute_local_20261025/figures/07_runtime.png | 原图复用；严格重放与前缀复用分列 |

## 验证范围

SOURCE_SNAPSHOT.json保存主要CSV及train/val目标文件的SHA-256。DRAFT_VALIDATION.json核对当前主表舍入、计数、引用集合、中英文数值表格一致性、图像路径、公式定界与占位符；它不等价于独立重跑全部实验或最终测试。旧阶段补充对照与方法表述另经过原报告、脚本阅读核对。

本次没有改动或重封装先前实验的复核压缩包。正文中图像仍链接到已有项目产物，若迁移到其他电脑，应同时迁移原图并更新链接。


## 2026-10-27最终测试补充

| 图表/正文 | 证据来源（项目根目录下） | 取数及解释规则 |
|---|---|---|
| 表8、双语摘要/结论 | outputs/reports/final_test_preflight_20261027/test_replay/metrics.csv | 固定五组，129图576GT，定位560；AP×100；输出细化正确480而非481 |
| 表9、差值区间 | outputs/reports/final_test_preflight_20261027/test_analysis/paired_contrasts.csv；analysis_plan.json | 2000次整图配对、来源分层，seed20261027；逐项95%描述区间 |
| 表A4 | outputs/reports/final_test_preflight_20261027/test_replay/metrics.csv | class0–3_Mask_AP50/AP95；百分数 |
| 标签来源、test数量 | outputs/reports/final_test_preflight_20261027/data_preflight.json；FROZEN_INPUTS.json | 302双审、51裁决、223规则；25不合格 |
| 输出细化负结果 | outputs/reports/final_test_preflight_20261027/test_analysis/refined_matching_changes.json | wb_000911__i001匹配行1→2，标签与排序未改；误拒37→38 |

working/DRAFT_VALIDATION.json与SOURCE_SNAPSHOT.json为2026-10-25历史记录；更新稿应查看working/FINAL_TEST_DRAFT_VALIDATION_20261027.json，旧记录不重新包装为当前验证。
