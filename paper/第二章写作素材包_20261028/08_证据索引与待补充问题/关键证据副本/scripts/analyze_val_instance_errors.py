"""Run reproducible instance-level error analysis on the frozen validation split.

The script deliberately never reads the test split.  It performs class-agnostic
one-to-one mask matching first, then separates localization failures from grade
classification errors.  Human review is not required; generated gallery images
are optional sanity-check material only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/grading_four_class_v3"
DEFAULT_WEIGHTS = (
    ROOT
    / "models/experiments/yolo26n_seg_offlineaug_e200_mask50_seed20260915/weights/best.pt"
)
DEFAULT_OUTPUT = ROOT / "outputs/reports/yolo26n_offlineaug_e200_val_error_analysis"
CLASS_NAMES = ("unqualified", "grade_1", "grade_2", "grade_3")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SIZE_LABELS = ("Q1_smallest", "Q2_small", "Q3_large", "Q4_largest")


@dataclass
class Instance:
    class_id: int
    mask: np.ndarray
    confidence: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--inference-conf",
        type=float,
        default=0.05,
        help="Low threshold retained for confidence diagnostics before the core threshold is applied.",
    )
    parser.add_argument(
        "--analysis-conf",
        type=float,
        default=0.25,
        help="Fixed threshold used for the primary instance-level error analysis.",
    )
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument(
        "--agnostic-nms",
        action="store_true",
        help="Suppress overlapping predictions across classes; off by default to reproduce the frozen run.",
    )
    parser.add_argument(
        "--mask-dedup-iou",
        type=float,
        default=None,
        help="Optional confidence-first, class-agnostic mask deduplication threshold.",
    )
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--candidate-overlap-iou", type=float, default=0.10)
    parser.add_argument("--gallery-per-category", type=int, default=6)
    parser.add_argument("--max-images", type=int, default=None, help="Smoke-test only; omit for the full val split.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean_or_zero(values: Iterable[float]) -> float:
    items = list(values)
    return float(np.mean(items)) if items else 0.0


def polygon_to_mask(points: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if points.size == 0:
        return mask
    polygon = np.rint(points).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [polygon], 1)
    return mask.astype(bool)


def read_yolo_instances(label_path: Path, height: int, width: int) -> list[Instance]:
    instances: list[Instance] = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split()
        if len(parts) < 7 or len(parts[1:]) % 2:
            raise ValueError(f"Invalid YOLO segmentation row: {label_path}:{line_number}")
        class_id = int(parts[0])
        if not 0 <= class_id < len(CLASS_NAMES):
            raise ValueError(f"Unexpected class {class_id}: {label_path}:{line_number}")
        points = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        points *= np.asarray([width, height], dtype=np.float32)
        mask = polygon_to_mask(points, height, width)
        if not np.any(mask):
            raise ValueError(f"Empty ground-truth mask: {label_path}:{line_number}")
        instances.append(Instance(class_id=class_id, mask=mask))
    return instances


def mask_iou_matrix(left: list[Instance], right: list[Instance]) -> np.ndarray:
    matrix = np.zeros((len(left), len(right)), dtype=np.float32)
    if not left or not right:
        return matrix
    left_areas = [int(np.count_nonzero(item.mask)) for item in left]
    right_areas = [int(np.count_nonzero(item.mask)) for item in right]
    for left_index, left_item in enumerate(left):
        for right_index, right_item in enumerate(right):
            intersection = int(np.count_nonzero(left_item.mask & right_item.mask))
            union = left_areas[left_index] + right_areas[right_index] - intersection
            matrix[left_index, right_index] = intersection / union if union else 0.0
    return matrix


def deduplicate_predictions(
    predictions: list[Instance], threshold: float
) -> tuple[list[Instance], list[Instance]]:
    """Keep the highest-confidence prediction from each near-identical mask group."""
    if not 0 < threshold <= 1:
        raise ValueError("mask deduplication IoU must be in (0, 1]")
    ordered = sorted(predictions, key=lambda item: float(item.confidence), reverse=True)
    kept: list[Instance] = []
    removed: list[Instance] = []
    for candidate in ordered:
        overlaps = mask_iou_matrix([candidate], kept)
        if kept and bool(np.any(overlaps[0] >= threshold)):
            removed.append(candidate)
        else:
            kept.append(candidate)
    return kept, removed


def optimal_match(iou_matrix: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    """Maximize valid-pair cardinality first and total IoU second."""
    if iou_matrix.size == 0:
        return []
    valid = iou_matrix >= threshold
    scores = np.where(valid, 1.0 + iou_matrix, 0.0)
    left_indices, right_indices = linear_sum_assignment(-scores)
    return [
        (int(left), int(right), float(iou_matrix[left, right]))
        for left, right in zip(left_indices, right_indices, strict=True)
        if valid[left, right]
    ]


def geometry(mask: np.ndarray) -> dict[str, float]:
    area = float(np.count_nonzero(mask))
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area_px": area, "long_axis_px": 0.0, "short_axis_px": 0.0, "equiv_diameter_px": 0.0}
    points = np.concatenate(contours, axis=0)
    _, (axis_a, axis_b), _ = cv2.minAreaRect(points)
    return {
        "area_px": area,
        "long_axis_px": float(max(axis_a, axis_b)),
        "short_axis_px": float(min(axis_a, axis_b)),
        "equiv_diameter_px": float(math.sqrt(4.0 * area / math.pi)),
    }


def size_boundaries(area_fractions: list[float]) -> tuple[float, float, float]:
    if not area_fractions:
        return (0.0, 0.0, 0.0)
    values = np.asarray(area_fractions, dtype=float)
    return tuple(float(value) for value in np.quantile(values, [0.25, 0.50, 0.75]))


def size_bin(value: float, boundaries: tuple[float, float, float]) -> str:
    return SIZE_LABELS[int(np.searchsorted(np.asarray(boundaries), value, side="left"))]


def class_metrics_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    gt_rows = [row for row in rows if row["record_type"] == "ground_truth"]
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_gt = [row for row in gt_rows if row["gt_class_id"] == class_id]
        correct = sum(row["error_type"] == "correct" for row in class_gt)
        localized = sum(bool(row["localization_matched"]) for row in class_gt)
        wrong_out = sum(row["error_type"] == "class_confusion" for row in class_gt)
        missed = sum(row["error_type"] == "missed" for row in class_gt)
        predicted_as_class = [
            row
            for row in rows
            if row["pred_class_id"] == class_id and row["record_type"] in {"ground_truth", "prediction_only"}
        ]
        false_positive = len(predicted_as_class) - correct
        false_negative = len(class_gt) - correct
        precision = safe_div(correct, correct + false_positive)
        recall = safe_div(correct, correct + false_negative)
        metrics.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "gt_count": len(class_gt),
                "prediction_count": len(predicted_as_class),
                "tp_correct_class_iou50": correct,
                "fp": false_positive,
                "fn": false_negative,
                "precision": precision,
                "recall": recall,
                "f1": safe_div(2 * precision * recall, precision + recall),
                "localization_recall_class_agnostic": safe_div(localized, len(class_gt)),
                "classification_accuracy_when_localized": safe_div(correct, localized),
                "class_confusion_count": wrong_out,
                "missed_count": missed,
                "mean_mask_iou_when_localized": mean_or_zero(
                    float(row["mask_iou"]) for row in class_gt if row["localization_matched"]
                ),
            }
        )
    return metrics


def decide_p2(size_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {row["size_bin"]: row for row in size_rows}
    smallest = by_name.get("Q1_smallest")
    larger_rows = [by_name[name] for name in SIZE_LABELS[1:] if name in by_name]
    if not smallest or not larger_rows:
        return {
            "decision": "insufficient_evidence",
            "evidence_strength": "insufficient",
            "reason": "尺寸分层不足，无法判断 P2 小目标分支。",
        }

    larger_n = sum(int(row["n_gt"]) for row in larger_rows)
    larger_localized = sum(int(row["localized_count"]) for row in larger_rows)
    larger_iou_sum = sum(float(row["mask_iou_sum_all_gt"]) for row in larger_rows)
    larger_missed = sum(int(row["missed_count"]) for row in larger_rows)
    small_n = int(smallest["n_gt"])
    small_localization = float(smallest["localization_recall"])
    larger_localization = safe_div(larger_localized, larger_n)
    small_iou = float(smallest["mean_mask_iou_all_gt"])
    larger_iou = safe_div(larger_iou_sum, larger_n)
    small_miss_rate = safe_div(int(smallest["missed_count"]), small_n)
    larger_miss_rate = safe_div(larger_missed, larger_n)
    localization_gap = larger_localization - small_localization
    iou_gap = larger_iou - small_iou
    miss_rate_gap = small_miss_rate - larger_miss_rate
    small_localization_errors = int(smallest["missed_count"])
    small_class_errors = int(smallest["class_confusion_count"])

    supports = (
        localization_gap >= 0.10
        and (iou_gap >= 0.05 or miss_rate_gap >= 0.08)
        and small_localization_errors >= small_class_errors
    )
    if supports:
        strength = "moderate" if localization_gap < 0.15 or iou_gap < 0.08 else "strong"
        reason = (
            "最小尺寸四分位的类别无关定位召回和全体 GT Mask IoU 均低于较大目标，"
            "且漏检不少于等级混淆；这与增加高分辨率 P2 检测头的作用机制一致。"
        )
        decision = "support_p2_single_change_experiment"
    elif small_class_errors > small_localization_errors and localization_gap < 0.10:
        strength = "moderate"
        reason = (
            "最小目标的主要问题是已定位实例的等级混淆，而不是尺寸相关漏检；"
            "P2 不应作为首选结构改动。"
        )
        decision = "do_not_prioritize_p2"
    else:
        strength = "weak"
        reason = (
            "自动证据未显示足够稳定的尺寸相关定位退化；"
            "当前不足以仅凭验证集错误分析优先采用 P2。"
        )
        decision = "p2_not_yet_supported"
    return {
        "decision": decision,
        "evidence_strength": strength,
        "reason": reason,
        "smallest_localization_recall": small_localization,
        "larger_localization_recall": larger_localization,
        "localization_recall_gap": localization_gap,
        "smallest_mean_mask_iou_all_gt": small_iou,
        "larger_mean_mask_iou_all_gt": larger_iou,
        "mask_iou_gap": iou_gap,
        "smallest_miss_rate": small_miss_rate,
        "larger_miss_rate": larger_miss_rate,
        "miss_rate_gap": miss_rate_gap,
        "smallest_missed_count": small_localization_errors,
        "smallest_class_confusion_count": small_class_errors,
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def contours(mask: np.ndarray) -> list[np.ndarray]:
    found, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return found


def annotated_image(
    image: np.ndarray,
    gt_instances: list[Instance],
    predictions: list[Instance],
    matches: list[tuple[int, int, float]],
) -> np.ndarray:
    canvas = image.copy()
    matched_gt = {gt for gt, _, _ in matches}
    matched_pred = {pred for _, pred, _ in matches}
    match_by_gt = {gt: (pred, overlap) for gt, pred, overlap in matches}
    for gt_index, item in enumerate(gt_instances):
        color = (0, 200, 0) if gt_index in matched_gt else (0, 0, 255)
        cv2.drawContours(canvas, contours(item.mask), -1, color, 2, cv2.LINE_AA)
        points = np.column_stack(np.nonzero(item.mask))
        if len(points):
            y, x = points.min(axis=0)
            suffix = "MISS"
            if gt_index in match_by_gt:
                pred_index, overlap = match_by_gt[gt_index]
                pred = predictions[pred_index]
                suffix = f"{CLASS_NAMES[pred.class_id]} IoU={overlap:.2f}"
            cv2.putText(
                canvas,
                f"GT {CLASS_NAMES[item.class_id]} | {suffix}",
                (int(x), max(18, int(y) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    for pred_index, item in enumerate(predictions):
        color = (255, 200, 0) if pred_index in matched_pred else (255, 0, 255)
        cv2.drawContours(canvas, contours(item.mask), -1, color, 2, cv2.LINE_AA)
        if pred_index not in matched_pred:
            points = np.column_stack(np.nonzero(item.mask))
            if len(points):
                y, x = points.min(axis=0)
                cv2.putText(
                    canvas,
                    f"FP {CLASS_NAMES[item.class_id]} {item.confidence:.2f}",
                    (int(x), min(canvas.shape[0] - 6, int(y) + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    color,
                    1,
                    cv2.LINE_AA,
                )
    banner = np.full((42, canvas.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(
        banner,
        "GT: green=matched red=miss | Prediction: cyan=matched magenta=unmatched",
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, canvas])


def update_gallery(
    gallery: dict[str, list[tuple[float, str, np.ndarray]]],
    category: str,
    score: float,
    image_id: str,
    rendered: np.ndarray,
    limit: int,
) -> None:
    items = gallery[category]
    items.append((float(score), image_id, rendered.copy()))
    items.sort(key=lambda item: (-item[0], item[1]))
    del items[limit:]


def plot_confusion(matrix: np.ndarray, output: Path) -> None:
    labels = list(CLASS_NAMES) + ["missed"]
    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Ground-truth class")
    ax.set_title("Validation instance confusion at Mask IoU >= 0.50")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(int(matrix[row, column])), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_size_metrics(size_rows: list[dict[str, Any]], output_recall: Path, output_iou: Path) -> None:
    labels = [row["size_bin"] for row in size_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(x - 0.19, [row["localization_recall"] for row in size_rows], width=0.38, label="Localization recall")
    ax.bar(x + 0.19, [row["correct_detection_recall"] for row in size_rows], width=0.38, label="Correct class recall")
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Recall by ground-truth size quartile")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_recall, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.bar(x, [row["mean_mask_iou_all_gt"] for row in size_rows], color="#4C78A8", label="All GT (miss=0)")
    ax.plot(
        x,
        [row["mean_mask_iou_when_localized"] for row in size_rows],
        marker="o",
        color="#E45756",
        label="Localized pairs only",
    )
    ax.set_xticks(x, labels, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mask IoU")
    ax.set_title("Mask overlap by ground-truth size quartile")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_iou, dpi=180)
    plt.close(fig)


def plot_confidence(confidence_rows: list[dict[str, Any]], output: Path) -> None:
    labels = [row["confidence_bin"] for row in confidence_rows]
    correct = np.asarray([row["correct_predictions"] for row in confidence_rows], dtype=float)
    confusion = np.asarray([row["class_confusion_predictions"] for row in confidence_rows], dtype=float)
    extra = np.asarray([row["false_positive_predictions"] for row in confidence_rows], dtype=float)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 5.4))
    ax.bar(x, correct, label="Correct", color="#54A24B")
    ax.bar(x, confusion, bottom=correct, label="Class confusion", color="#F2CF5B")
    ax.bar(x, extra, bottom=correct + confusion, label="Unmatched prediction", color="#E45756")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Prediction count")
    ax.set_title("Prediction outcome by confidence")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def build_report(summary: dict[str, Any], class_rows: list[dict[str, Any]], size_rows: list[dict[str, Any]]) -> str:
    overall = summary["overall"]
    decision = summary["p2_assessment"]
    category_counts = summary["error_category_counts"]
    class_lines = [
        (
            f"| {row['class_name']} | {row['gt_count']} | {row['precision']:.3f} | "
            f"{row['recall']:.3f} | {row['f1']:.3f} | "
            f"{row['localization_recall_class_agnostic']:.3f} | "
            f"{row['classification_accuracy_when_localized']:.3f} |"
        )
        for row in class_rows
    ]
    size_lines = [
        (
            f"| {row['size_bin']} | {row['n_gt']} | {100 * row['area_fraction_median']:.3f}% | "
            f"{row['localization_recall']:.3f} | {row['correct_detection_recall']:.3f} | "
            f"{row['classification_accuracy_when_localized']:.3f} | {row['mean_mask_iou_all_gt']:.3f} |"
        )
        for row in size_rows
    ]
    top_confusions = summary["top_class_confusions"]
    confusion_text = "、".join(
        f"{item['gt_class']}→{item['pred_class']}：{item['count']}" for item in top_confusions
    ) or "无"
    return "\n".join(
        [
            "# 冻结模型原始验证集自动错误分析",
            "",
            f"_生成时间：{summary['generated_at']}。本报告仅分析原始 val；test 未读取。_",
            "",
            "## 结论先行",
            "",
            f"- 在固定置信度 {summary['protocol']['analysis_confidence']:.2f}、类别无关 Mask IoU "
            f"≥ {summary['protocol']['match_iou']:.2f} 下：{overall['gt_instances']} 个 GT 中，"
            f"{overall['localized_instances']} 个被定位，{overall['correct_instances']} 个同时分级正确；"
            f"漏检 {category_counts.get('missed', 0)}，等级混淆 {category_counts.get('class_confusion', 0)}，"
            f"未匹配预测 {category_counts.get('false_positive', 0)}。",
            f"- P2 判断：`{decision['decision']}`，证据强度 `{decision['evidence_strength']}`。{decision['reason']}",
            f"- 最常见跨级错误：{confusion_text}。",
            "- 这些是自动匹配诊断，不是人工确认的标签错误；错误图库仅供可选抽查，不构成交付门槛。",
            "",
            "## 固定分析协议",
            "",
            f"- 权重：`{summary['inputs']['weights']}`（SHA-256 `{summary['inputs']['weights_sha256']}`）。",
            f"- 数据：`{summary['inputs']['dataset']}/images/val`，共 {overall['images']} 张；未访问 test。",
            f"- 推理：imgsz={summary['protocol']['imgsz']}，NMS IoU={summary['protocol']['nms_iou']:.2f}；"
            f"agnostic_nms={summary['protocol']['agnostic_nms']}；"
            f"mask_dedup_iou={summary['protocol']['mask_dedup_iou']}；"
            f"先保留 confidence ≥ {summary['protocol']['inference_confidence']:.2f} 的预测用于置信度诊断，"
            f"主分析固定使用 confidence ≥ {summary['protocol']['analysis_confidence']:.2f}。",
            "- 匹配：先完全忽略类别，以 Mask IoU 最大化的一对一分配确定是否定位成功；之后才判断类别是否正确。",
            "",
            "## 逐类结果",
            "",
            "| 类别 | GT | Precision | Recall | F1 | 类别无关定位召回 | 已定位后的分类准确率 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *class_lines,
            "",
            "![类别混淆矩阵](confusion_matrix.png)",
            "",
            "## 尺寸分层",
            "",
            "尺寸四分位按每个 GT Mask 占本图面积的比例划分，避免图像分辨率差异造成尺度假象。",
            "",
            "| 尺寸组 | GT | 面积占比中位数 | 定位召回 | 正确类别召回 | 已定位分类准确率 | 全体 GT 平均 Mask IoU |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            *size_lines,
            "",
            "![尺寸与召回](size_vs_recall.png)",
            "",
            "![尺寸与 Mask IoU](size_vs_mask_iou.png)",
            "",
            "## 置信度与自动候选",
            "",
            "![置信度与结果](confidence_vs_error.png)",
            "",
            f"- 重复预测候选：{summary['candidate_counts']['duplicate_prediction_candidate']}；"
            f"合并候选：{summary['candidate_counts']['merge_candidate']}；"
            f"拆分候选：{summary['candidate_counts']['split_candidate']}。",
            "- 候选由几何重叠规则触发，只表示值得查看的模式，不等同于人工确认。",
            "",
            "## 可复现产物",
            "",
            "- `val_instance_errors.csv`：每个 GT 及未匹配预测的明细。",
            "- `val_image_summary.csv`：逐图守恒统计。",
            "- `class_metrics.csv`、`class_confusion_matrix.csv`：逐类指标与混淆。",
            "- `size_stratified_metrics.csv`、`confidence_stratified_metrics.csv`：尺寸与置信度诊断。",
            "- `error_category_counts.csv`、`analysis_summary.json`：汇总和机器可读结论。",
            "- `error_gallery/`：自动挑选的代表性错误图，仅供可选 sanity check。",
            "",
            "## 解释边界",
            "",
            "- 已验证：在上述固定阈值和当前 val 上的计数、匹配、尺寸分层及混淆模式。",
            "- 较可能：P2 判断只说明错误模式是否符合高分辨率小目标分支的预期作用机制，不证明改动后一定提升。",
            "- 未解决：标签本身是否存在少量错标、val 与部署场景的分布差异、独立 test 泛化；这些均未由本分析声称解决。",
            "",
        ]
    )


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    images_dir = dataset / "images/val"
    labels_dir = dataset / "labels/val"

    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Validation split missing under {dataset}")
    if "test" in {part.lower() for part in images_dir.parts}:
        raise ValueError("Refusing to analyze a test path")
    if not 0 <= args.inference_conf <= args.analysis_conf <= 1:
        raise ValueError("Expected 0 <= inference-conf <= analysis-conf <= 1")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gallery_dir = output / "error_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"No validation images found in {images_dir}")
    missing_labels = [path.stem for path in image_paths if not (labels_dir / f"{path.stem}.txt").is_file()]
    if missing_labels:
        raise FileNotFoundError(f"Missing validation labels: {missing_labels[:5]}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    image_records: list[dict[str, Any]] = []
    gt_records: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    confidence_prediction_records: list[dict[str, Any]] = []
    gallery: dict[str, list[tuple[float, str, np.ndarray]]] = defaultdict(list)
    candidate_counts: Counter[str] = Counter()
    postprocess_counts: Counter[str] = Counter()

    for image_number, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(image_path)
        height, width = image.shape[:2]
        gt_instances = read_yolo_instances(labels_dir / f"{image_path.stem}.txt", height, width)
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.inference_conf,
            iou=args.nms_iou,
            agnostic_nms=args.agnostic_nms,
            max_det=300,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )[0]
        all_predictions: list[Instance] = []
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            polygons = result.masks.xy
            if not (len(classes) == len(confidences) == len(polygons)):
                raise ValueError(f"Prediction output length mismatch: {image_path}")
            for class_id, confidence, polygon in zip(classes, confidences, polygons, strict=True):
                if not 0 <= int(class_id) < len(CLASS_NAMES):
                    raise ValueError(f"Unexpected prediction class {class_id}: {image_path}")
                mask = polygon_to_mask(np.asarray(polygon), height, width)
                if np.any(mask):
                    all_predictions.append(
                        Instance(class_id=int(class_id), confidence=float(confidence), mask=mask)
                    )
        if args.mask_dedup_iou is not None:
            all_predictions, removed_predictions = deduplicate_predictions(
                all_predictions, args.mask_dedup_iou
            )
            postprocess_counts["mask_duplicate_predictions_removed"] += len(removed_predictions)

        all_iou_matrix = mask_iou_matrix(gt_instances, all_predictions)
        all_matches = optimal_match(all_iou_matrix, args.match_iou)
        all_match_by_pred = {pred: (gt, overlap) for gt, pred, overlap in all_matches}
        for pred_index, pred in enumerate(all_predictions):
            gt_index, overlap = all_match_by_pred.get(pred_index, (None, 0.0))
            gt = gt_instances[gt_index] if gt_index is not None else None
            outcome = (
                "correct"
                if gt is not None and gt.class_id == pred.class_id
                else "class_confusion"
                if gt is not None
                else "false_positive"
            )
            confidence_prediction_records.append(
                {
                    "image_id": image_path.stem,
                    "confidence": pred.confidence,
                    "outcome": outcome,
                }
            )

        predictions = [item for item in all_predictions if float(item.confidence) >= args.analysis_conf]
        iou_matrix = mask_iou_matrix(gt_instances, predictions)
        matches = optimal_match(iou_matrix, args.match_iou)
        match_by_gt = {gt: (pred, overlap) for gt, pred, overlap in matches}
        match_by_pred = {pred: (gt, overlap) for gt, pred, overlap in matches}
        rendered: np.ndarray | None = None

        merge_predictions = {
            pred_index
            for pred_index in range(len(predictions))
            if int(np.count_nonzero(iou_matrix[:, pred_index] >= args.candidate_overlap_iou)) >= 2
        }
        split_gt = {
            gt_index
            for gt_index in range(len(gt_instances))
            if int(np.count_nonzero(iou_matrix[gt_index, :] >= args.candidate_overlap_iou)) >= 2
        }
        duplicate_predictions = {
            pred_index
            for pred_index in range(len(predictions))
            if pred_index not in match_by_pred and np.any(iou_matrix[:, pred_index] >= args.match_iou)
        }
        candidate_counts["merge_candidate"] += len(merge_predictions)
        candidate_counts["split_candidate"] += len(split_gt)
        candidate_counts["duplicate_prediction_candidate"] += len(duplicate_predictions)

        image_gt_records: list[dict[str, Any]] = []
        for gt_index, gt in enumerate(gt_instances):
            geom = geometry(gt.mask)
            matched = gt_index in match_by_gt
            pred_index, overlap = match_by_gt.get(gt_index, (None, 0.0))
            pred = predictions[pred_index] if pred_index is not None else None
            if not matched:
                error_type = "missed"
            elif pred.class_id == gt.class_id:
                error_type = "correct"
            else:
                error_type = "class_confusion"
            max_iou = float(np.max(iou_matrix[gt_index])) if predictions else 0.0
            record = {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "record_type": "ground_truth",
                "gt_index": gt_index,
                "pred_index": pred_index,
                "gt_class_id": gt.class_id,
                "gt_class_name": CLASS_NAMES[gt.class_id],
                "pred_class_id": pred.class_id if pred is not None else None,
                "pred_class_name": CLASS_NAMES[pred.class_id] if pred is not None else "",
                "confidence": pred.confidence if pred is not None else None,
                "mask_iou": overlap,
                "max_iou_any_prediction": max_iou,
                "localization_matched": matched,
                "class_correct": bool(matched and pred.class_id == gt.class_id),
                "error_type": error_type,
                "gt_area_px": geom["area_px"],
                "gt_area_fraction": geom["area_px"] / (height * width),
                "gt_long_axis_px": geom["long_axis_px"],
                "gt_short_axis_px": geom["short_axis_px"],
                "gt_equiv_diameter_px": geom["equiv_diameter_px"],
                "gt_size_bin": "",
                "merge_candidate": bool(pred_index in merge_predictions) if pred_index is not None else False,
                "split_candidate": gt_index in split_gt,
                "duplicate_prediction_candidate": False,
            }
            gt_records.append(record)
            image_gt_records.append(record)

        for pred_index, pred in enumerate(predictions):
            matched = pred_index in match_by_pred
            gt_index, overlap = match_by_pred.get(pred_index, (None, 0.0))
            gt = gt_instances[gt_index] if gt_index is not None else None
            outcome = (
                "correct"
                if matched and gt.class_id == pred.class_id
                else "class_confusion"
                if matched
                else "false_positive"
            )
            prediction_records.append(
                {
                    "image_id": image_path.stem,
                    "pred_index": pred_index,
                    "pred_class_id": pred.class_id,
                    "pred_class_name": CLASS_NAMES[pred.class_id],
                    "confidence": pred.confidence,
                    "matched_gt_index": gt_index,
                    "matched_gt_class_id": gt.class_id if gt is not None else None,
                    "mask_iou": overlap,
                    "outcome": outcome,
                    "merge_candidate": pred_index in merge_predictions,
                    "duplicate_prediction_candidate": pred_index in duplicate_predictions,
                }
            )
            if not matched:
                max_iou = float(np.max(iou_matrix[:, pred_index])) if gt_instances else 0.0
                gt_records.append(
                    {
                        "image_id": image_path.stem,
                        "image_path": str(image_path),
                        "record_type": "prediction_only",
                        "gt_index": None,
                        "pred_index": pred_index,
                        "gt_class_id": None,
                        "gt_class_name": "",
                        "pred_class_id": pred.class_id,
                        "pred_class_name": CLASS_NAMES[pred.class_id],
                        "confidence": pred.confidence,
                        "mask_iou": 0.0,
                        "max_iou_any_prediction": max_iou,
                        "localization_matched": False,
                        "class_correct": False,
                        "error_type": "false_positive",
                        "gt_area_px": None,
                        "gt_area_fraction": None,
                        "gt_long_axis_px": None,
                        "gt_short_axis_px": None,
                        "gt_equiv_diameter_px": None,
                        "gt_size_bin": "",
                        "merge_candidate": pred_index in merge_predictions,
                        "split_candidate": False,
                        "duplicate_prediction_candidate": pred_index in duplicate_predictions,
                    }
                )

        error_types = Counter(row["error_type"] for row in image_gt_records)
        unmatched_predictions = len(predictions) - len(matches)
        image_records.append(
            {
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "width": width,
                "height": height,
                "gt_instances": len(gt_instances),
                "predictions_at_inference_conf": len(all_predictions),
                "predictions_at_analysis_conf": len(predictions),
                "localized_gt": len(matches),
                "correct": error_types["correct"],
                "class_confusion": error_types["class_confusion"],
                "missed": error_types["missed"],
                "false_positive": unmatched_predictions,
                "merge_candidates": len(merge_predictions),
                "split_candidates": len(split_gt),
                "duplicate_prediction_candidates": len(duplicate_predictions),
                "conservation_ok": len(matches) + error_types["missed"] == len(gt_instances)
                and len(matches) + unmatched_predictions == len(predictions),
            }
        )
        if not image_records[-1]["conservation_ok"]:
            raise AssertionError(f"Instance accounting failed: {image_path.stem}")

        gallery_categories: list[tuple[str, float]] = []
        if error_types["missed"]:
            gallery_categories.append(("missed", float(error_types["missed"])))
        if error_types["class_confusion"]:
            gallery_categories.append(("class_confusion", float(error_types["class_confusion"])))
        if unmatched_predictions:
            fp_conf = max(
                (float(pred.confidence) for index, pred in enumerate(predictions) if index not in match_by_pred),
                default=0.0,
            )
            gallery_categories.append(("false_positive", float(unmatched_predictions) + fp_conf))
        low_iou = [overlap for _, _, overlap in matches if overlap < 0.65]
        if low_iou:
            gallery_categories.append(("low_iou", 1.0 - min(low_iou)))
        if gallery_categories:
            rendered = annotated_image(image, gt_instances, predictions, matches)
            for category, score in gallery_categories:
                update_gallery(
                    gallery,
                    category,
                    score,
                    image_path.stem,
                    rendered,
                    args.gallery_per_category,
                )

        print(f"[{image_number:03d}/{len(image_paths):03d}] {image_path.stem}", flush=True)

    ground_truth_rows = [row for row in gt_records if row["record_type"] == "ground_truth"]
    boundaries = size_boundaries([float(row["gt_area_fraction"]) for row in ground_truth_rows])
    for row in ground_truth_rows:
        row["gt_size_bin"] = size_bin(float(row["gt_area_fraction"]), boundaries)

    class_rows = class_metrics_from_rows(gt_records)
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES) + 1), dtype=int)
    for row in ground_truth_rows:
        column = int(row["pred_class_id"]) if row["localization_matched"] else len(CLASS_NAMES)
        confusion[int(row["gt_class_id"]), column] += 1

    confusion_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        record: dict[str, Any] = {"gt_class": class_name}
        record.update({f"pred_{name}": int(confusion[class_id, index]) for index, name in enumerate(CLASS_NAMES)})
        record["missed"] = int(confusion[class_id, -1])
        confusion_rows.append(record)

    size_rows: list[dict[str, Any]] = []
    for label in SIZE_LABELS:
        group = [row for row in ground_truth_rows if row["gt_size_bin"] == label]
        localized = [row for row in group if row["localization_matched"]]
        correct = [row for row in group if row["error_type"] == "correct"]
        iou_sum_all = sum(float(row["mask_iou"]) for row in group)
        areas = [float(row["gt_area_fraction"]) for row in group]
        size_rows.append(
            {
                "size_bin": label,
                "n_gt": len(group),
                "area_fraction_min": min(areas) if areas else 0.0,
                "area_fraction_median": float(np.median(areas)) if areas else 0.0,
                "area_fraction_max": max(areas) if areas else 0.0,
                "localized_count": len(localized),
                "correct_count": len(correct),
                "missed_count": sum(row["error_type"] == "missed" for row in group),
                "class_confusion_count": sum(row["error_type"] == "class_confusion" for row in group),
                "localization_recall": safe_div(len(localized), len(group)),
                "correct_detection_recall": safe_div(len(correct), len(group)),
                "classification_accuracy_when_localized": safe_div(len(correct), len(localized)),
                "mean_mask_iou_when_localized": mean_or_zero(float(row["mask_iou"]) for row in localized),
                "mask_iou_sum_all_gt": iou_sum_all,
                "mean_mask_iou_all_gt": safe_div(iou_sum_all, len(group)),
            }
        )

    confidence_edges = [args.inference_conf, 0.10, 0.25, 0.40, 0.55, 0.70, 0.85, 1.000001]
    confidence_edges = sorted(set(edge for edge in confidence_edges if edge >= args.inference_conf))
    confidence_rows: list[dict[str, Any]] = []
    # Confidence outcomes use the low-threshold prediction set and its own one-to-one matching.
    for lower, upper in zip(confidence_edges[:-1], confidence_edges[1:], strict=True):
        selected = [
            row for row in confidence_prediction_records if lower <= float(row["confidence"]) < upper
        ]
        counts = Counter(row["outcome"] for row in selected)
        confidence_rows.append(
            {
                "confidence_bin": f"[{lower:.2f}, {min(upper, 1.0):.2f}{']' if upper > 1 else ')'}",
                "prediction_count": len(selected),
                "correct_predictions": counts["correct"],
                "class_confusion_predictions": counts["class_confusion"],
                "false_positive_predictions": counts["false_positive"],
                "precision_like_correct_fraction": safe_div(counts["correct"], len(selected)),
            }
        )

    category_counts = Counter(row["error_type"] for row in gt_records)
    error_rows = [{"error_category": key, "count": category_counts.get(key, 0)} for key in (
        "correct", "class_confusion", "missed", "false_positive"
    )]
    class_confusions = Counter(
        (row["gt_class_name"], row["pred_class_name"])
        for row in ground_truth_rows
        if row["error_type"] == "class_confusion"
    )
    top_confusions = [
        {"gt_class": pair[0], "pred_class": pair[1], "count": count}
        for pair, count in class_confusions.most_common()
    ]
    p2_assessment = decide_p2(size_rows)
    overall_correct = category_counts["correct"]
    overall_localized = overall_correct + category_counts["class_confusion"]
    prediction_count = len(prediction_records)
    overall_precision = safe_div(overall_correct, prediction_count)
    overall_recall = safe_div(overall_correct, len(ground_truth_rows))
    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "inputs": {
            "dataset": str(dataset),
            "weights": str(weights),
            "weights_sha256": sha256(weights),
            "split": "val",
            "test_accessed": False,
        },
        "protocol": {
            "imgsz": args.imgsz,
            "device": str(args.device),
            "inference_confidence": args.inference_conf,
            "analysis_confidence": args.analysis_conf,
            "nms_iou": args.nms_iou,
            "agnostic_nms": bool(args.agnostic_nms),
            "mask_dedup_iou": args.mask_dedup_iou,
            "match_iou": args.match_iou,
            "matching": "class_agnostic_max_cardinality_then_iou_hungarian",
            "candidate_overlap_iou": args.candidate_overlap_iou,
            "size_definition": "ground_truth_mask_area_divided_by_image_area_quartiles",
        },
        "overall": {
            "images": len(image_paths),
            "gt_instances": len(ground_truth_rows),
            "predictions_at_analysis_conf": prediction_count,
            "localized_instances": overall_localized,
            "correct_instances": overall_correct,
            "localization_recall": safe_div(overall_localized, len(ground_truth_rows)),
            "classification_accuracy_when_localized": safe_div(overall_correct, overall_localized),
            "micro_precision_correct_class_iou50": overall_precision,
            "micro_recall_correct_class_iou50": overall_recall,
            "micro_f1_correct_class_iou50": safe_div(
                2 * overall_precision * overall_recall, overall_precision + overall_recall
            ),
            "mean_mask_iou_when_localized": mean_or_zero(
                float(row["mask_iou"]) for row in ground_truth_rows if row["localization_matched"]
            ),
        },
        "error_category_counts": dict(category_counts),
        "candidate_counts": {key: int(candidate_counts[key]) for key in (
            "duplicate_prediction_candidate", "merge_candidate", "split_candidate"
        )},
        "postprocess_counts": {
            "mask_duplicate_predictions_removed": int(
                postprocess_counts["mask_duplicate_predictions_removed"]
            )
        },
        "size_quartile_boundaries_area_fraction": list(boundaries),
        "top_class_confusions": top_confusions,
        "p2_assessment": p2_assessment,
        "interpretation_limits": {
            "automatic_not_human_confirmed": True,
            "gallery_optional": True,
            "label_changes_made": False,
            "causal_claim_about_p2": False,
        },
    }

    instance_fields = [
        "image_id", "image_path", "record_type", "gt_index", "pred_index", "gt_class_id", "gt_class_name",
        "pred_class_id", "pred_class_name", "confidence", "mask_iou", "max_iou_any_prediction",
        "localization_matched", "class_correct", "error_type", "gt_area_px", "gt_area_fraction",
        "gt_long_axis_px", "gt_short_axis_px", "gt_equiv_diameter_px", "gt_size_bin", "merge_candidate",
        "split_candidate", "duplicate_prediction_candidate",
    ]
    write_csv(output / "val_instance_errors.csv", gt_records, instance_fields)
    write_csv(output / "val_image_summary.csv", image_records)
    write_csv(output / "class_metrics.csv", class_rows)
    write_csv(output / "class_confusion_matrix.csv", confusion_rows)
    write_csv(output / "size_stratified_metrics.csv", size_rows)
    write_csv(output / "confidence_stratified_metrics.csv", confidence_rows)
    write_csv(output / "error_category_counts.csv", error_rows)
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    plot_confusion(confusion, output / "confusion_matrix.png")
    plot_size_metrics(size_rows, output / "size_vs_recall.png", output / "size_vs_mask_iou.png")
    plot_confidence(confidence_rows, output / "confidence_vs_error.png")
    for category, items in sorted(gallery.items()):
        category_dir = gallery_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for rank, (_, image_id, rendered) in enumerate(items, start=1):
            target = category_dir / f"{rank:02d}_{image_id}.jpg"
            if not cv2.imwrite(str(target), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise IOError(target)

    (output / "VAL_ERROR_ANALYSIS.md").write_text(
        build_report(summary, class_rows, size_rows), encoding="utf-8"
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps(summary["p2_assessment"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
