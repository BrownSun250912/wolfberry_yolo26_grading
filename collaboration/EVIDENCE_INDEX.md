# 图表与结果证据索引

所有链接均相对仓库。正文以中文/英文Markdown为主；下列历史实验只作为开发证据，不冒充最终test。更细的对应见[原始图表来源说明](../paper/manuscript_20261025/TABLE_AND_FIGURE_SOURCES.md)。

## 最终测试

| 用途 | 依据 |
|---|---|
| 表8、摘要与结论最终数字 | [指标CSV](../outputs/reports/final_test_preflight_20261027/test_replay/metrics.csv) · [最终报告](../outputs/reports/final_test_preflight_20261027/最终测试报告.md) |
| 表9、差值区间 | [配对区间](../outputs/reports/final_test_preflight_20261027/test_analysis/paired_contrasts.csv) · [分析方案](../outputs/reports/final_test_preflight_20261027/test_analysis/analysis_plan.json) |
| 表A4、逐类结果 | [同一原始指标CSV](../outputs/reports/final_test_preflight_20261027/test_replay/metrics.csv)中的class0–3字段 |
| 终测错误与混淆矩阵 | [按实例结果](../outputs/reports/final_test_preflight_20261027/test_replay/details.json) · [细化匹配变化](../outputs/reports/final_test_preflight_20261027/test_analysis/refined_matching_changes.json) |
| 标签来源与固定版本 | [数据核验](../outputs/reports/final_test_preflight_20261027/data_preflight.json) · [冻结输入](../outputs/reports/final_test_preflight_20261027/FROZEN_INPUTS.json) |
| 本地原稿数字核对 | [2026-10-27验证记录](../paper/manuscript_20261025/working/FINAL_TEST_DRAFT_VALIDATION_20261027.json)；搬迁后的哈希另见发布清单 |

## 开发实验与图表

| 正文材料 | 依据 |
|---|---|
| 表2、表A3 | [开发主指标CSV](../outputs/reports/attribute_local_20261025/headline_results.csv) |
| 表3，早期评分/速度 | [固定种子对照](../outputs/reports/refine_module_20261022/完整流程固定种子对照.csv) |
| 表4，信息来源与区间 | [排序研究](../outputs/reports/attribute_local_20261025/selected_rank_study_metrics.csv) · [区间](../outputs/reports/attribute_local_20261025/selected_rank_intervals.csv) |
| 表5，掩膜兼容性 | [固定候选](../outputs/reports/attribute_local_20261025/fixed_candidate_metrics.csv) · [原图几何](../outputs/reports/attribute_local_20261025/native_geometry_summary.csv) |
| 表6，严格计时 | [计时CSV](../outputs/reports/attribute_local_20261025/rank_latency_summary.csv) |
| 表7，压力测试 | [全部条件CSV](../outputs/reports/attribute_local_20261025/rank_stress_metrics.csv) |
| 文献、写作边界 | [引用核验](../paper/manuscript_20261025/working/CITATION_VERIFICATION.md) · [历史综述](../paper/literature_review_20261023/LITERATURE_REVIEW.md) |

## 图表原文件

以下十组图均为开发实验产物。每组含PNG、SVG和PDF；现稿引用其中四组，保留其余图供作者整理补充材料。

| 图源 | PNG | 可编辑SVG | PDF |
|---|---|---|---|
| 01_paired_main_results | [PNG](../outputs/reports/attribute_local_20261025/figures/01_paired_main_results.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/01_paired_main_results.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/01_paired_main_results.pdf) |
| 02_attribute_factorial | [PNG](../outputs/reports/attribute_local_20261025/figures/02_attribute_factorial.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/02_attribute_factorial.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/02_attribute_factorial.pdf) |
| 03_boundary_dose | [PNG](../outputs/reports/attribute_local_20261025/figures/03_boundary_dose.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/03_boundary_dose.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/03_boundary_dose.pdf) |
| 04_stress_all_conditions | [PNG](../outputs/reports/attribute_local_20261025/figures/04_stress_all_conditions.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/04_stress_all_conditions.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/04_stress_all_conditions.pdf) |
| 05_geometry_sensitivity | [PNG](../outputs/reports/attribute_local_20261025/figures/05_geometry_sensitivity.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/05_geometry_sensitivity.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/05_geometry_sensitivity.pdf) |
| 06_defect_probability | [PNG](../outputs/reports/attribute_local_20261025/figures/06_defect_probability.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/06_defect_probability.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/06_defect_probability.pdf) |
| 07_runtime | [PNG](../outputs/reports/attribute_local_20261025/figures/07_runtime.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/07_runtime.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/07_runtime.pdf) |
| 08_rank_information_controls | [PNG](../outputs/reports/attribute_local_20261025/figures/08_rank_information_controls.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/08_rank_information_controls.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/08_rank_information_controls.pdf) |
| 09_shape_score_tradeoff | [PNG](../outputs/reports/attribute_local_20261025/figures/09_shape_score_tradeoff.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/09_shape_score_tradeoff.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/09_shape_score_tradeoff.pdf) |
| 10_qualitative_best_and_worst | [PNG](../outputs/reports/attribute_local_20261025/figures/10_qualitative_best_and_worst.png) | [SVG](../outputs/reports/attribute_local_20261025/figures/10_qualitative_best_and_worst.svg) | [PDF](../outputs/reports/attribute_local_20261025/figures/10_qualitative_best_and_worst.pdf) |

[图注与源数据](../outputs/reports/attribute_local_20261025/figures/CAPTIONS.md) · [图源清单](../outputs/reports/attribute_local_20261025/figures/figure_manifest.json)
