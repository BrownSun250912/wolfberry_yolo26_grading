# 数据集核验记录

构建前已确认：

- `new` 中有 1285 张 PNG 和 1285 个同名 X-AnyLabeling/LabelMe JSON；
- `labels` 中有 1285 个同名 YOLO segmentation TXT；
- PNG、JSON、TXT 三者配对缺失数均为 0；
- 类别 ID 为 `0`，JSON 类别名为 `wolfberry`；
- 原编号范围为 1–1300，缺少 15 个编号：
  `145, 408, 766–775, 1070, 1174, 1265`；
- TXT 每行是 `class_id x1 y1 x2 y2 ...` 的归一化实例多边形，不是普通检测框；
- JSON 的 `imageData` 为 null，重命名副本时需要同步更新 `imagePath`；其中 367 个历史 JSON
  的 `imagePath` 带有 `..\新建文件夹\` 相对路径前缀，但文件名仍正确，构建后统一规范化。

新命名不会填补或伪造原始编号。`wb_000001`～`wb_001285` 只是连续工作 ID，原编号永久保留在
`data/manifests/dataset_manifest.csv` 中。

旧目录 `instance_seg_demo` 已有一个 1028/257 的 train/val 训练结果，但没有带真值的独立 test。
新工作区将建立 1028/128/129 的 train/val/test，并冻结 test。

构建后复核结果：

- 三类文件均为 1285 个，split 内 PNG/JSON/TXT 基名完全一致；
- 连续工作 ID `1–1285` 无缺失、无重复；
- 新 JSON 的 `imagePath` 全部与新 PNG 文件名一致；
- train/val/test 图片数为 1028/128/129；实例总数为 4593/577/582；
- 三套数据的平均实例数为 4.47/4.51/4.51；
- 原始编号四个区间在三个 split 中均有覆盖；
- 7 个以上实例的少见图片在 train/val/test 中为 8/1/1；
- 所有复制图片与源图片的字节大小一致，最终完整性校验错误数为 0。

这里的“完整性错误数为 0”表示文件配对、命名、路径、坐标范围和复制结果一致，不代表所有
人工标注在语义上都正确。

## 冒烟训练发现的重复实例

2026-08-23 使用 Ultralytics 数据加载器进行训练冒烟检查后，又对三个 split 做了全量只读扫描：

- 完全相同外接框的重复实例共 35 个；
- train：26 个；val：5 个；test：4 个；
- val 中已确认的示例包括 `wb_000214`、`wb_000421`、`wb_000480`；
- QA 叠加图保存在 `outputs/reports/duplicate_label_qa`。

Ultralytics 会在加载时忽略完全重复的外接框，但 JSON/TXT 仍不一致于“每果一个实例”的标注
规范，而且还有轮廓略有差异的同果重复标注可能不会被自动去重。正式训练前应先人工核对并发布
清洗版数据（建议命名为 `split_v2`），保留 `split_v1` 和修订日志，不要静默覆盖。
