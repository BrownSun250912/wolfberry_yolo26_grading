# 项目交接说明

更新日期：2026-09-07

## 1. 当前研究目标

项目先使用实例分割定位每颗枸杞，再依据人工确认的腐烂情况、尺寸/形态规则和“合格果缺梗归三级”规则构造四等级实例分割数据集，最后训练并改进 YOLO26 四等级实例分割模型。

四个类别为：

| class_id | 类别 |
| ---: | --- |
| 0 | `unqualified`（不合格） |
| 1 | `grade_1`（一级） |
| 2 | `grade_2`（二级） |
| 3 | `grade_3`（三级） |

注意：四等级标签是当前研究标准的结果，不是国家标准；论文中必须完整说明腐烂人工复核、尺寸边界和缺梗覆盖规则。

## 2. 已冻结的数据版本

正式数据入口为 `data/grading_four_class_v3/`：

| 划分 | 图像数 | 实例数 |
| --- | ---: | ---: |
| train | 966 | 4,277 |
| val | 116 | 518 |
| test | 129 | 576 |
| 合计 | 1,211 | 5,371 |

类别实例数：不合格 451、一级 1,186、二级 1,456、三级 2,278。

`test` 是冻结测试集，只用于最终模型比较，不得据此反复调整阈值、训练超参数或选择模型。

Git 仓库包含该数据集的 YOLO 标签、LabelMe JSON、审计文件和配置，但不包含 RGB 图片。图片须按 `DATASET.md` 单独交接，并放回对应的 `images/train`、`images/val`、`images/test` 目录。

## 3. 已完成基线

四等级 YOLO26n 实例分割基线：

- 训练轮次：100
- 随机种子：20260906
- 最佳权重：`models/experiments/yolo26n_seg_grading4_baseline_seed20260906/weights/best.pt`
- 权重 SHA256：`0A6DE467B53FCF5E6F65AE0DF6CB5196590BB5C25C2E218EB14452CDF5D28623`
- 测试集 Mask mAP50：0.821
- 测试集 Mask mAP50-95：0.678

测试集分类型结果：

| 类别 | Precision | Recall | Mask mAP50-95 |
| --- | ---: | ---: | ---: |
| unqualified | 0.355 | 0.960 | 0.522 |
| grade_1 | 0.816 | 0.905 | 0.769 |
| grade_2 | 0.791 | 0.748 | 0.645 |
| grade_3 | 0.956 | 0.740 | 0.776 |

最明显的问题是不合格类别召回率高但精确率低，因此后续改进不能只报告总 mAP，必须同时报告各类别 Precision、Recall、F1、混淆矩阵以及 Mask mAP。

完整实验说明见 `docs/YOLO26_GRADING_BASELINE_AND_PAPER_PLAN.md`，测试输出见 `outputs/reports/yolo26n_seg_grading4_baseline_seed20260906_test/`。

## 4. 新电脑复现步骤

以下命令在项目根目录的 PowerShell 中执行。

1. 创建并激活环境：

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   ```

2. 根据新电脑的显卡驱动安装匹配的 PyTorch。原实验环境使用 `torch==2.13.0+cu130` 与 `torchvision==0.28.0+cu130`，完整版本记录见 `environment/requirements-lock.txt`。

3. 安装项目依赖：

   ```powershell
   python -m pip install -r environment/requirements.in
   ```

4. 解压单独交接的数据包，把图片放入 `data/grading_four_class_v3/images/{train,val,test}`，并执行：

   ```powershell
   .\scripts\verify_environment.ps1
   python .\scripts\verify_four_class_dataset.py
   ```

5. 复现训练：

   ```powershell
   .\scripts\train_grading_baseline.ps1
   ```

6. 只有在模型和超参数冻结后，才运行测试集评价：

   ```powershell
   .\scripts\evaluate_grading_test.ps1 -Model .\models\experiments\新实验名\weights\best.pt
   ```

## 5. 后续开发优先级

1. 先复现当前基线，确认数据加载数量和指标误差处于可接受范围。
2. 只使用 train/val 做改进和消融实验，固定随机种子及训练预算。
3. 优先改善不合格类别的误报，并检查等级相邻类别之间的混淆。
4. 每个改进点至少保留：假设、配置、代码版本、训练曲线、验证集指标和推理速度。
5. 确定最终方案后仅评估一次冻结 test，并与同等训练条件的 YOLO26n 基线比较。
6. 论文同时报告总体指标、各类别指标、参数量、FLOPs、推理速度和消融实验。

## 6. 关键入口

| 内容 | 路径 |
| --- | --- |
| 根说明 | `README.md` |
| 数据包说明 | `DATASET.md` |
| 四等级数据说明 | `data/grading_four_class_v3/README.md` |
| 四等级标签生成审计 | `outputs/grading_standard_post_review_v5/` |
| 训练脚本 | `scripts/train_grading_baseline.ps1` |
| 测试脚本 | `scripts/evaluate_grading_test.ps1` |
| 论文与改进路线 | `docs/YOLO26_GRADING_BASELINE_AND_PAPER_PLAN.md` |
| 特征定义 | `docs/FEATURE_DEFINITIONS.md` |
| 环境锁定 | `environment/requirements-lock.txt` |

## 7. 数据治理约束

- 不得修改 `instance_id`、原图编号或冻结的 train/val/test 归属而不留下新版本和审计表。
- 不得用测试集挑选模型、阈值或超参数。
- 不得将颜色异常候选直接当作腐烂真值；当前不合格标签以已完成人工复核的数据为依据。
- 当前没有可靠的 px/mm 标定，尺寸只能报告 px，面积只能报告 px²。
- 不要提交 `.venv`、原始 RGB 图像、批量 ROI/Mask、缓存或本机绝对路径。
