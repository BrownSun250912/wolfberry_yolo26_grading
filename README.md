# 枸杞实例分割与外观分级：论文协作仓库

更新：2026-10-28。当前阶段：模型与最终测试已冻结，进入共同改稿和投稿准备。

## 从这里开始

| 需要做的事 | 入口 |
|---|---|
| 新加入作者先看 | [论文交接与协作说明](collaboration/START_HERE.md) |
| 改中文正文 | [中文Markdown稿](paper/manuscript_20261025/MANUSCRIPT_ZH.md) |
| 改英文正文 | [英文Markdown稿](paper/manuscript_20261025/MANUSCRIPT_EN.md) |
| 阅读和批注 | [中文Word](paper/manuscript_20261025/MANUSCRIPT_ZH.docx) · [中文PDF](paper/manuscript_20261025/MANUSCRIPT_ZH.pdf) · [英文PDF](paper/manuscript_20261025/MANUSCRIPT_EN.pdf) |
| 核对论文主结果 | [最终测试报告](outputs/reports/final_test_preflight_20261027/最终测试报告.md) · [原始指标CSV](outputs/reports/final_test_preflight_20261027/test_replay/metrics.csv) |
| 查表格、图的依据 | [图表证据索引](collaboration/EVIDENCE_INDEX.md) · [图表来源说明](paper/manuscript_20261025/TABLE_AND_FIGURE_SOURCES.md) |
| 分配修改工作 | [修改任务清单](collaboration/TASKS.md) · [提交规则](CONTRIBUTING.md) |
| 查方法实现与复现范围 | [方法代码说明](collaboration/METHODS_AND_REPRODUCIBILITY.md) |

## 当前最终结果：test，不是val

129张图像、576个GT，五组方案均定位560个实例；所有方法固定后统一测试，未根据test换模型或调阈值。

| 固定方案 | Mask AP50 | Mask AP50–95 | 分级正确／已定位 |
|---|---:|---:|---:|
| 原YOLO26（本轮冻结检测器） | 78.04% | 65.84% | 456/560 |
| 原YOLO26＋同类NMS | 78.89% | 66.46% | 456/560 |
| RR22分层参考 | 79.75% | 67.16% | 481/560 |
| 原缺陷证据排序主方案 | **80.72%** | **68.03%** | **481/560** |
| 主方案＋输出掩膜细化 | **80.72%** | **69.12%** | 480/560 |

主方案相对原YOLO提高2.68个百分点AP50，配对95%描述性区间[1.43,4.08]；相对RR22提高0.97个百分点，区间[0.27,1.70]。开发集83.48%→87.10%作为开发证据单列，不写成test成绩。9月100轮基线的历史82.1%来自另一训练运行及评价记录，不能与本表直接相减作公平对照。

## 结果解释必须保留的内容

- test标签为302个双审一致、51个人工裁决、223个冻结规则沿用；不合格GT仅25个。
- 主方案仍有2/25匹配不合格漏放、37个匹配合格误拒；新增漏放为0不等于绝对漏放为0。
- 输出细化提高严格IoU平均AP，但引起一个重匹配错误，分级正确数481→480。
- 早期基线曾使用test；来源编号及无字节完全重复图像不证明实物果/采集批次独立。
- 当前目标是候选外观分级规则的视觉复现，没有毫米标定或商业分选可靠性认证。

## 材料范围

本次同步当前双语稿、已有Word/PDF阅读稿、论文图表及其数据、终测与开发报告、标签版本说明、文献调研与引用核验，以及冻结方法源码审阅快照。正文链接已改为仓库相对路径，下载ZIP后也能定位。

已有训练代码和历史标签继续保留。最新训练RGB图像、模型权重、训练运行缓存及第三方论文全文不在本次交接范围；方法代码快照用于审阅，不代表clone后即可重跑完整模型。原始实验JSON中可能保留本机路径与原始哈希，发布副本的校验值见[发布清单](collaboration/PUBLISH_MANIFEST.json)。Word/PDF是当前阅读快照，后续以Markdown为协作主版本。

历史入口：[9月README原文](docs/history/README_before_paper_handoff_20261028.txt) · [9月HANDOFF原文](docs/history/HANDOFF_before_paper_handoff_20261028.txt) · [原始数据交接](DATASET.md)。
