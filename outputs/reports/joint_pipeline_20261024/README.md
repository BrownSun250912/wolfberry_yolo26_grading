# 2026-10-24 联合流水线实验交付

本轮126次新头拟合已完成：排序器54、第一版分支选择器36、空间增强12、跨来源嵌套审计24。仅train/val；没有新检测器训练，没有访问test，没有替换原生产入口。当前数据反复用于开发，所有区间均为探索性逐项区间。

## 先看这些文件

- [完整实验报告](../../../docs/experiments/joint_pipeline_20261024/实验报告.md)：方案、完整主表、10条件结果、负结果、统计、计时与限制。
- [主结果CSV](report_main_table.csv)、[10条件CSV](report_stress_table.csv)：由原始表自动汇总，未按val最优种子重选。
- [交付审计](delivery_audit.json)：126份权重哈希、原始文件未变、10个检查用例、行数及最终代码哈希。
- [缓存指标重放](delivery_metric_replay.csv)：26配置重算与原始记录一致；这是同实现重放，不是独立AP算法验证。
- [环境](reproduction_environment.json)、[复现清单](reproduction_manifest.json)、[轻量复现包](../../../collaboration/LOCAL_ONLY_ARTIFACTS.md#item-56ba5ce18633)。压缩包依赖原项目代码、基础权重与train/val缓存，不能脱离项目运行。
- [图表/文档/压缩包检查](artifact_qa_status.json)：来源哈希、图片编码及dpi、SVG/PDF、文档本地链接、ZIP CRC。该检查保存在包外，避免自引用归档校验；图表均已人工逐幅查看。
- [推理CLI示例检查](inference_examples_status.json)：两张真实val图已执行，结果及overlay在`inference_examples/`。

## 研究候选的状态

|候选|正常图正确/定位|状态|
|---|---:|---|
|既有Selected22+原rank+同等真实去重|442/500|公平参考；AP50 85.7643%|
|Selected22边界+果梗/邻域|446/500|保守研究起点；AP50 85.7695%，额外AP收益区间跨零|
|Rule-anchored22边界+果梗/邻域|449/500|较高点值；约95.90ms/图；无默认不合格恢复|
|第一版学习选择器，固定seed24|447/500|部分干预改善；仅前景偏色215/484，保留研究分支|
|空间增强选择器，固定seed24|445/500|未过正常图至少446保护线；仅前景偏色347/484，不提升为主模型|
|rr22加权不合格恢复|452/500|新增不合格误放风险，不设为默认|

所有表中的“正确/定位”都在真实删除预测后重新匹配；主CSV另有正确/全部518GT。NMS包含raw高置信优先选择，不是所有领域通用的标准协议。不要将历史54ms计时与本轮统一计时直接比较。

## 运行与复现

以下命令在PowerShell中直接执行，使用项目现有`.venv`。它们不会选择test，不需要安装pytest。运行时公共模块自动把Ultralytics配置放到本实验输出目录。

```powershell
# 7个有实际故障意义的回归检查
& '.venv\Scripts\python.exe' -X utf8 'tests\test_joint_pipeline_20261024.py'

# 两个真实val图的CLI示例；不覆盖历史模型
& '.venv\Scripts\python.exe' -X utf8 'scripts\joint_pipeline_examples_20261024.py'

# 指标重放、126份权重核验、回归检查及轻量复现包；会更新本轮交付文件
& '.venv\Scripts\python.exe' -X utf8 'scripts\joint_pipeline_delivery_20261024.py'

# 只从现成CSV重生成报告与9幅图
& '.venv\Scripts\python.exe' -X utf8 'scripts\joint_pipeline_report_20261024.py'
& '.venv\Scripts\python.exe' -X utf8 'scripts\joint_pipeline_figures_20261024.py'
```

对自己的图像运行示例。`--image`换成实际文件，文件路径不能含`test`目录名，以避免本阶段误触封存评估；该路径保护只是辅助，不能替代数据来源管理。

```powershell
& '.venv\Scripts\python.exe' -X utf8 'scripts\joint_pipeline_inference_20261024.py' --image 'F:\path\your_image.jpg' --source selected22 --policy boundary_stem
```

可选`--source rule_anchored22`，或`--source selected22 --policy arbitration`使用第一版内部选定选择器。默认邻域证据，可用`--evidence raw_same_candidate`或`top1`作研究对照。显式`--gate-checkpoint`才能切换空间候选；其未通过正常保护线，不能称推荐部署模型。模型在GPU上运行，输入解码失败会报错。

输出JSON保留低阈值AP候选、原detector conf、最终用于排序的confidence、评分、果梗输出、决策来源和冲突复核标记；PNG仅画raw≥.25的框。`selector_score`未校准，复核标记没有人工纠错效果保证；JSON未导出掩膜多边形，Python API返回`polygons`。旧rank对新类别未重新校准，因此入口拒绝不受验证的排序/策略组合，主研究决策使用raw置信。

## 数据与脚本索引

|问题|主要证据|入口|
|---|---|---|
|原输出/真实去重公平对照|`p0_baseline_matrix.csv`，78行|`joint_pipeline_p0_20261024.py`|
|同次前向四类捕获|模型目录`raw_classes_train_manifest.json`、`raw_classes_val_manifest.json`，1082图|`joint_pipeline_capture_20261024.py train/val`|
|边界/果梗/不合格恢复分离|`p2_decision_matrix.csv`，792行|`joint_pipeline_decision_matrix_20261024.py`|
|7条件真实图像重推理|`p3_stress_matrix.csv`，210行|`joint_pipeline_stress_20261024.py`|
|54次新决策排序器|`rank_fit_ledger.csv`、`rank_fit_metrics.csv`、`rank_internal_selections.json`|`joint_pipeline_rank_fit_20261024.py`|
|36次第一版选择器|`arbitration_fit_ledger.csv`、`arbitration_fixed_candidate.csv`|`joint_pipeline_arbitration_fit_20261024.py`及development脚本|
|12次空间增强|`spatial_arbitration_fit_ledger.csv`、`spatial_arbitration_fixed_candidate.csv`，10条件|`joint_pipeline_spatial_arbitration_20261024.py`及development脚本|
|24次跨来源新头审计|`source_nested_fit_ledger.csv`、`source_nested_metrics.csv`，168行|`joint_pipeline_source_nested_20261024.py`|
|不合格共识策略|只使用`consensus_v2_*`；v1失效标记保留|`joint_pipeline_consensus_20261024.py`|
|证据负对照|`evidence_control_summary.csv`及逐次表；540次确定性评估|`joint_pipeline_evidence_controls_20261024.py`|
|对象合并与操作定义|`object_grouping_metrics.csv`，120行；`object_operational_audit.csv`|`joint_pipeline_object_grouping_20261024.py`、`joint_pipeline_object_review_20261024.py`|
|所有输出的歧义与复核残余|`output_risk_metrics.csv`，47条件/方法×3固定规则，141行|`joint_pipeline_output_risk_20261024.py`；额外3例检查在`tests/test_joint_pipeline_output_risk_20261024.py`|
|配对区间|`bootstrap_intervals.csv`，72行；`spatial_bootstrap_intervals.csv`，144行|`joint_pipeline_bootstrap_20261024.py`、`joint_pipeline_spatial_bootstrap_20261024.py`|
|相同raw置信/原rank对照|`confidence_contrast_intervals.csv`，6对比×6端点|`joint_pipeline_confidence_contrasts_20261024.py`|
|划分图像粗筛|`split_screen_status.json`、`split_nearest_pairs.csv`、固定前6对拼图；跨train/val字节重复0对，不能证明物理果实独立|`joint_pipeline_split_screen_20261024.py`|
|空预测实际API探针|`runtime_edge_status.json`，6配置×3灰度合成帧=18例|`joint_pipeline_runtime_edges_20261024.py`|
|实际运行一致性|`native_parity_status.json`，812例；`stress_mask_parity.json`，812例|`joint_pipeline_native_parity_20261024.py`、`joint_pipeline_verify_masks_20261024.py`|
|最终同协议成本|`runtime_final_runs.csv`，960行；`runtime_final_paired_intervals.csv`|`joint_pipeline_final_benchmark_20261024.py`|

原始拟合/推理矩阵脚本受本次5h窗口和`quota_latest.json`保护，到期后不再启动新拟合；不要删除保护文件冒充同一次实验。未来续跑应登记新预算和新实验阶段。只做指标重放、图表与报告的交付脚本不依赖旧窗口继续有效。

`joint_pipeline_artifact_qa_20261024.py`在交付打包后核查全部本地链接与ZIP完整性；`artifact_qa_status.json`留在包外，不作为包内自我验证。若重生成图表/报告，应再运行delivery脚本打包，然后运行artifact_qa脚本。

最终推理与policy源码哈希必须与`runtime_final_status.json`相同，否则原计时不能自动套到修改版。复现包已包含本轮脚本及两份固定选择器，基础检测器/评分器/果梗权重与完整候选缓存留在原项目。完整参数、分割ID、选点方法与初始/最终权重哈希保存在模型目录各plan/ledger里。

## 明确保留的失败记录

`consensus_v1_invalidated.json`记录创建class0赋值遗漏及修复。`high_scope_full_status.json`、`high_scope_scoring_only_failure.json`、`batch_aligned_status.json`记录编码加速未过预设容差。它们不参与最终计时或生产替换。不要把文件存在等同于实现通过。

图表为本轮原CSV直接生成，PNG 300dpi及可编辑SVG/PDF；每图有来源哈希。当前未指定期刊，不认证具体期刊格式。所有负结果、少数类误放、正常图保护失败与未训练组合失效均在报告中保留。
