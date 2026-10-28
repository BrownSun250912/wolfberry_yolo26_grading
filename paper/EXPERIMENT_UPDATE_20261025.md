# 2026-10-25 实验进展

已完成150次新增拟合，全部结果保留；未运行test，原YOLO权重和生产入口未替换。

[完整实验报告](../docs/experiments/attribute_local_20261025/实验报告.md)

[固定终测方案与模型hash](../outputs/reports/attribute_local_20261025/FROZEN_CANDIDATES_AND_TEST_PROTOCOL.json)

[复核包](../collaboration/LOCAL_ONLY_ARTIFACTS.md#item-5aa566207337)

固定seed25：原YOLO26n-seg AP50 83.4817%；前阶段RR22 86.7046%；本轮原缺陷证据排序87.1013%，AP50–95 73.2874%，449/500。加可选输出细化后AP50仍87.1013%，AP50–95 74.2348%，449/500。新增AP50对RR22的配对95%区间跨零，仍需最终固定测试与独立条件验证。
