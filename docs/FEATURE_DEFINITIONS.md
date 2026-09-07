# 单果可解释特征定义

本项目当前只生成可解释特征和“疑似损伤区域”，不生成一级、二级、三级或损伤果真值标签。

## 数据粒度和来源

`features_pilot.csv` 每行对应一个预测实例。`image_id` 来自数据清单；`instance_id` 按实例质心
从上到下、从左到右稳定排序，格式为 `image_id__iNNN`。`mask_source=prediction`，模型路径和
SHA-256随每行保存。

Mask、Crop和轮廓均保存ROI副本；`roi_x1/y1/x2/y2` 和原图坐标轮廓可将结果映射回原图。
所有计算在原始分辨率Mask上完成。程序同时保存 YOLO 原始 Mask、去白边后的 geometry mask、
实际颜色统计 Mask 和损伤有效区域 Mask。

geometry mask 只在 YOLO Mask 内部最多 8 px 的边界带中删除低饱和度、较亮且 Lab 色度较低的
像素。三个条件必须同时满足，避免把正常红色果体直接删除；删除比例超过配置安全上限时会回退，
并记录状态。当前尺寸和形态特征均根据 geometry mask 计算。

## 18项核心特征

| 字段 | 定义 | 当前单位 |
|---|---|---|
| major_axis | 椭圆拟合长轴；失败时使用最小旋转矩形长边 | px |
| minor_axis | 椭圆拟合短轴；失败时使用最小旋转矩形短边 | px |
| area | 分析Mask非零像素数 | px² |
| aspect_ratio | major_axis / minor_axis | 无量纲 |
| L_mean/L_std | 非高光果体像素的CIELAB L*均值/标准差 | L*: 0–100 |
| a_mean/a_std | 非高光果体像素的CIELAB a*均值/标准差 | 近似-128–127 |
| b_mean/b_std | 非高光果体像素的CIELAB b*均值/标准差 | 近似-128–127 |
| S_mean | 非高光果体像素HSV饱和度均值 | 0–1 |
| V_mean | 非高光果体像素HSV明度均值 | 0–1 |
| circularity | 4π×轮廓面积/perimeter² | 无量纲 |
| solidity | 轮廓面积/凸包面积 | 无量纲 |
| eccentricity | sqrt(1-(minor_axis/major_axis)²) | 无量纲 |
| dark_ratio | 暗区候选面积/非高光有效果体面积 | 0–1 |
| abnormal_color_ratio | Lab a*b*稳健异常色面积/非高光有效果体面积 | 0–1 |
| max_damage_ratio | 暗区与异常色并集的最大连通域面积/果体面积 | 0–1 |

## 高光和疑似损伤

高光、暗区、异常色、形态学处理及最小连通域阈值全部位于
`configs/feature_extraction_pilot.yaml`。这些参数只用于试运行和生成疑似区域，不是最终分级阈值。

`damage_candidates`编码：85=暗区、170=异常色、255=二者重叠。高光从颜色和损伤候选统计中
排除，但不改变尺寸和形态特征。损伤检测还会排除 geometry mask 的可配置边缘带，减少白边和
轮廓插值误报。

v4 另外保存三个辅助候选比例，它们不属于18项核心特征：`highlight_ratio`、
`calyx_candidate_ratio` 和 `edge_abnormal_ratio`。果蒂候选必须同时位于椭圆长轴端部、靠近轮廓，
且 Lab a* 明显低于红色果体。边缘异常使用与内部损伤相同的暗度/异常色判据，但单独保存，不进入
内部三个损伤比例。高光使用固定 HSV 条件和实例内部相对亮度/饱和度条件的并集。

颜色统计前会按配置轻微腐蚀 geometry mask 边界。暗区候选同时要求满足绝对 L* 上限和实例
内部稳健低亮度离群条件，避免把正常偏暗果体整体当作损伤。异常色使用 a*、b* 相对本实例
中位颜色的稳健标准化距离。候选区域经过开闭运算，并删除过小连通域。`dark_ratio` 和
`abnormal_color_ratio` 的分母为损伤有效区域像素数；`max_damage_ratio` 为暗区与异常色并集的
最大连通域面积除以 geometry mask 面积。`area`保持 geometry mask 像素面积，圆度和实心度则
使用彼此一致的连续轮廓面积口径。

## 标定接口

配置中的 `calibration.px_per_mm` 当前为null，因此毫米字段为空。未来给出同一成像平面的
px/mm后：长轴和短轴除以px/mm，面积除以(px/mm)²。若存在明显透视或横纵像素尺度不同，
必须改用二维标定或单应性校正，不能使用单一标量。

## 质量字段

`shape_success`、`color_success`、`damage_success`分别表示三个模块是否成功；单个实例异常不会
中止整张图。不可计算值保留为空，并在`error_reason`记录原因。`warning_reason`记录多连通域、
小组件清除等非致命情况。
