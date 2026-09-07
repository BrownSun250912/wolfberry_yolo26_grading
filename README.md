# 枸杞分割、单果特征与分级标准项目

> 交接入口：新接手开发者请先阅读 [`HANDOFF.md`](HANDOFF.md)；RGB 图片不保存在 GitHub 中，获取和放置方式见 [`DATASET.md`](DATASET.md)。

## 最新入口（2026-09-06）

三人果梗复核和测试集审核完成后，已制作四等级实例分割数据集：[`data/grading_four_class_v3/README.md`](data/grading_four_class_v3/README.md)。
当前训练集966张、验证集116张、独立测试集129张，共1211张图片、5371个实例；配置为 `data/grading_four_class_v3/data.yaml`。
类别总数为：不合格451、一级1186、二级1456、三级2278。全部标签已通过逐文件校验和 Ultralytics train/val/test 分割加载检查，加载丢失数为0。
另有74张训练/验证图片因原始轮廓数量或JSON/TXT不一致而暂缓，不会影响当前版本训练。
已完成 YOLO26n 四等级实例分割基线训练和冻结测试集首次评估：测试集 Mask mAP50 为0.821、Mask mAP50–95 为0.678。完整记录见 `docs/YOLO26_GRADING_BASELINE_AND_PAPER_PLAN.md`。
下文部分内容是此前分割与特征阶段的历史记录；当前等级数据制作结果以上述新入口为准。

## 当前状态

- 已完成 YOLO26 枸杞实例分割基线训练。
- 已对 train/val 共 5,108 个单果实例完成去白边、Mask 保存和 18 项可解释特征提取。
- test 集未参与阈值选择，现已完成独立审核并加入最终数据集，仅用于模型评价。
- 已完成2278颗单果人工腐烂复检：腐烂447颗、正常1831颗。
- 已按冻结的尺寸/形态阈值与缺梗三级规则生成四等级标签。
- 已完成 YOLO26n 四等级实例分割基线100轮训练和独立测试集评估。

## 当前使用入口

| 工作内容 | 入口 |
| --- | --- |
| 最终人工腐烂标签 | `review/rot_review_v1/rot_review_strict_completed_2026-09-02.xlsx` |
| 四等级训练/验证/测试数据集 | `data/grading_four_class_v3/` |
| 最新分级清单与审计 | `outputs/grading_standard_post_review_v5/` |
| 三人审核结果与测试裁决 | `review/three_person_audit_v1/04_审核结果汇总/` |
| 全量单果和 Mask | `outputs/features_frozen_v1/` |
| 最新腐烂排除与健康候选池 | `outputs/strict_rot_exclusion_v2/` |
| 最佳分割模型 | `models/experiments/yolo26n_seg_baseline_seed20260823/weights/best.pt` |
| 四等级基线最佳模型 | `models/experiments/yolo26n_seg_grading4_baseline_seed20260906/weights/best.pt` |
| 四等级基线测试报告 | `outputs/reports/yolo26n_seg_grading4_baseline_seed20260906_test/` |
| 后续实验与论文路线 | `docs/YOLO26_GRADING_BASELINE_AND_PAPER_PLAN.md` |
| 数据集和原始标签 | `data/split_v1/` |
| 特征定义 | `docs/FEATURE_DEFINITIONS.md` |

## 目录结构

```text
wolfberry_yolo26_cv/
├── data/
│   ├── split_v1/                   # train/val/test图像、YOLO标签和LabelMe JSON
│   ├── manifests/                  # 文件重命名和划分追溯表
│   └── reviews/                    # 已完成人工审核的稳定数据
├── models/
│   ├── pretrained/                 # 通用预训练分割权重
│   ├── reference_previous/         # 历史参考模型
│   └── experiments/                # 正式基线模型及训练记录
├── outputs/
│   ├── features_frozen_v1/         # 5,108个单果、Mask、轮廓和特征
│   ├── feature_analysis_frozen_v1/ # 全量特征统计和图表
│   ├── grade_candidate_audit_v1/   # 240颗人工审核记录
│   ├── rot_triage_full_v1/         # 腐烂位置与连通区域检测结果
│   └── rot_screening_tiers_v2/     # 疑似排除集与分级分析候选集
├── review/
│   └── rot_review_v1/              # 当前人工腐烂复检包
├── configs/                        # 正式配置和腐烂筛查配置
├── scripts/                        # 数据、训练、特征和筛查脚本
├── docs/                           # 数据审计、特征定义和路线说明
├── environment/                    # 环境版本与依赖锁
└── .venv/                          # 已配置的Python/GPU运行环境
```

## 当前推荐流程

1. 打开 `review/rot_review_v1/rot_review.csv`，逐张查看同目录 `images/` 中的单果图。
2. 在 `rot` 列填写 `YES`、`NO` 或 `UNCERTAIN`，不要修改 `instance_id` 和 `split`。
3. 完成人工复检后，只排除确认 `rot=YES` 的实例；误检且正常的实例重新加入健康果。
4. 使用剩余健康果分析尺寸、形态和颜色分布，提出一级、二级、三级候选标准。
5. 人工确认标准后生成等级标签，再训练和改进后续等级识别模型。

## 重要约束

- 高召回“腐烂疑似排除”只是人工复检入口，不是腐烂真值。
- `LOW_EVIDENCE` 不表示绝对没有腐烂。
- 当前没有可靠的 px/mm 标定，因此尺寸单位仍为 px，面积单位为 px²。
- 不应直接把疑似腐烂候选写入 YOLO 等级标签。
