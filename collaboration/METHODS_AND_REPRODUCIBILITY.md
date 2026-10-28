# 方法源码与复现范围

[research_code/frozen_20261027](../research_code/frozen_20261027/)收录最终val重放实际加载的项目源码及最终测试统计脚本，供核对方法、排序输入、NMS、掩膜和评价口径。源文件保持原字节，来源及SHA-256见[发布清单](PUBLISH_MANIFEST.json)。安装环境的Ultralytics原生掩膜代码哈希保留于环境和冻结记录，当前没有重新分发整个Python环境。

这个目录是源码审阅快照，不是完整可执行发行包：原源码的ROOT、权重和数据路径对应实验目录；新模型权重、RGB图像及逐图PT缓存没有包含。已有根scripts与models是历史基线交接，不能假定它们已等价于当前完整级联。作者若需要完整新机器复现实验，应单独移交这些依赖并重新做运行验证。

写作与数值检查不需要GPU：最终指标在[test_replay/metrics.csv](../outputs/reports/final_test_preflight_20261027/test_replay/metrics.csv)，按实例结果和混淆矩阵在[details.json](../outputs/reports/final_test_preflight_20261027/test_replay/details.json)。[test_analysis](../outputs/reports/final_test_preflight_20261027/test_analysis/)包含2000次重采样结果、区间、分析方案与匹配改变说明。开发集主表、计时、压力和原图几何数据在[证据索引](EVIDENCE_INDEX.md)。

本次搬迁仅改Markdown路径和增加协作说明，未重跑模型、修改实验数字、权重或标签。旧核验JSON中的源码/文件哈希是原实验时点的哈希；Markdown搬迁后的发布哈希另记。Word/PDF只作已核对终测数字的阅读快照，未重新排版或重新认证布局。
