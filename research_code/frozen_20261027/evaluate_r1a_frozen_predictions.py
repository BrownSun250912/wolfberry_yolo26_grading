"""Evaluate R1-A on the exact cached Seg-baseline validation predictions.

Only the final class is replaced.  Cached masks, boxes, prediction count,
confidence, NMS output, and class-agnostic GT matching are inherited byte-for-
byte from the successful R0 fixed-prediction replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from analyze_val_instance_errors import CLASS_NAMES, class_metrics_from_rows, safe_div
from train_r1a_hierarchical_grader import (
    R1HierarchicalGrader,
    binary_counts,
    compute_geometry,
    hierarchical_class,
    letterbox,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_R0 = ROOT / "outputs/reports/yolo26n_offlineaug_e200_r0_fixed_replay_maskfix_20260929"
DEFAULT_CHECKPOINT = ROOT / "models/experiments/r1a_hierarchical_grader_seed20260929/best.pt"
DEFAULT_TARGETS = ROOT / "outputs/r1_multitask_targets_v1/r1_multitask_targets.csv"
DEFAULT_OUTPUT = ROOT / "outputs/reports/r1a_hierarchical_grader_frozen_replay_20260929"
EXPECTED_FIXED_HASH = "d7ef27fa04855b58b5d75564cd00cae39bbc5c46c901ea1b27b7e94f480901a9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r0", type=Path, default=DEFAULT_R0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_mask(row: Any, r0: Path) -> np.ndarray:
    path = r0 / str(row.fixed_mask_path)
    if sha256(path) != str(row.fixed_mask_sha256):
        raise ValueError(f"Cached mask hash changed: {row.prediction_id}")
    crop = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if crop is None:
        raise FileNotFoundError(path)
    if crop.ndim == 3 and crop.shape[2] == 1:
        crop = crop[:, :, 0]
    expected = (int(row.mask_y2) - int(row.mask_y1), int(row.mask_x2) - int(row.mask_x1))
    if crop.shape != expected:
        raise ValueError(f"Cached mask shape changed: {row.prediction_id}: {crop.shape} != {expected}")
    full = np.zeros((int(row.image_height), int(row.image_width)), dtype=np.uint8)
    full[int(row.mask_y1) : int(row.mask_y2), int(row.mask_x1) : int(row.mask_x2)] = crop
    return full


def prediction_input(row: Any, r0: Path, image_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(row.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(str(row.image_path))
    if image.shape[:2] != (int(row.image_height), int(row.image_width)):
        raise ValueError(f"Image geometry changed: {row.prediction_id}")
    mask = load_fixed_mask(row, r0)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"Empty cached mask: {row.prediction_id}")
    x = int(xs.min())
    y = int(ys.min())
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    padding = max(12, int(math.ceil(max(width, height) * 0.25)))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image.shape[1], x + width + padding)
    y2 = min(image.shape[0], y + height + padding)
    image_crop = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    mask_crop = mask[y1:y2, x1:x2]
    geometry = compute_geometry(mask_crop, image_crop.shape[:2])
    image_crop, mask_crop = letterbox(image_crop, mask_crop, image_size)
    image_float = image_crop.astype(np.float32) / 255.0
    image_float = (image_float - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
        [0.229, 0.224, 0.225], np.float32
    )
    mask_float = (mask_crop.astype(np.float32) / 255.0)[..., None]
    tensor = np.concatenate([image_float, mask_float], axis=2).transpose(2, 0, 1).astype(np.float32)
    return tensor, geometry


def run_inference(
    fixed: pd.DataFrame,
    r0: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = R1HierarchicalGrader()
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    image_size = int(checkpoint["config"]["image_size"])
    geometry_mean = np.asarray(checkpoint["geometry_mean"], dtype=np.float32)
    geometry_std = np.asarray(checkpoint["geometry_std"], dtype=np.float32)
    rows: list[dict[str, Any]] = []
    tensors: list[np.ndarray] = []
    geometries: list[np.ndarray] = []
    metadata: list[Any] = []

    def flush() -> None:
        if not tensors:
            return
        image_batch = torch.from_numpy(np.stack(tensors)).to(device)
        geometry_batch = torch.from_numpy(
            (np.stack(geometries) - geometry_mean) / geometry_std
        ).to(device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output = model(image_batch, geometry_batch)
        defect = torch.sigmoid(output["defect_logit"]).cpu().numpy()
        stem = torch.sigmoid(output["stem_logit"]).cpu().numpy()
        score = output["score"].cpu().numpy() * 100.0
        final_class = hierarchical_class(defect, stem, score)
        for index, item in enumerate(metadata):
            rows.append(
                {
                    "prediction_id": str(item.prediction_id),
                    "image_id": str(item.image_id),
                    "pred_index": int(item.pred_index),
                    "confidence": float(item.confidence),
                    "baseline_class_id": int(item.baseline_class_id),
                    "baseline_class_name": str(item.baseline_class_name),
                    "defect_probability": float(defect[index]),
                    "stem_missing_probability": float(stem[index]),
                    "score_prediction": float(score[index]),
                    "r1a_class_id": int(final_class[index]),
                    "r1a_class_name": CLASS_NAMES[int(final_class[index])],
                }
            )
        tensors.clear()
        geometries.clear()
        metadata.clear()

    for item in fixed.sort_values(["image_id", "pred_index"]).itertuples(index=False):
        tensor, geometry = prediction_input(item, r0, image_size)
        tensors.append(tensor)
        geometries.append(geometry)
        metadata.append(item)
        if len(tensors) >= batch_size:
            flush()
    flush()
    return pd.DataFrame(rows), checkpoint


def replace_classes(baseline: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    class_by_prediction = predictions.set_index("prediction_id")["r1a_class_id"].to_dict()
    records: list[dict[str, Any]] = []
    for row in baseline.to_dict("records"):
        item = dict(row)
        item["variant"] = "r1a"
        prediction_id = str(item.get("prediction_id") or "")
        if prediction_id and prediction_id != "nan":
            class_id = int(class_by_prediction[prediction_id])
            item["pred_class_id"] = class_id
            item["pred_class_name"] = CLASS_NAMES[class_id]
        if item["record_type"] == "ground_truth":
            localized = bool(item["localization_matched"])
            correct = localized and int(item["pred_class_id"]) == int(item["gt_class_id"])
            item["class_correct"] = bool(correct)
            item["error_type"] = "correct" if correct else ("class_confusion" if localized else "missed")
        else:
            item["class_correct"] = False
            item["error_type"] = "false_positive"
        records.append(item)
    return pd.DataFrame(records)


def variant_summary(outcomes: pd.DataFrame, variant: str) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = outcomes.to_dict("records")
    ground_truth = outcomes.loc[outcomes["record_type"].eq("ground_truth")]
    localized = ground_truth.loc[ground_truth["localization_matched"].astype(bool)]
    correct = int(ground_truth["class_correct"].astype(bool).sum())
    class_rows = class_metrics_from_rows(rows)
    summary = {
        "variant": variant,
        "images": int(outcomes["image_id"].nunique()),
        "gt_instances": len(ground_truth),
        "predictions": int(outcomes["prediction_id"].replace("", np.nan).dropna().nunique()),
        "localized_instances": len(localized),
        "correct_instances": correct,
        "class_confusions": len(localized) - correct,
        "missed": len(ground_truth) - len(localized),
        "false_positive_predictions": int(outcomes["record_type"].eq("prediction_only").sum()),
        "localization_recall": safe_div(len(localized), len(ground_truth)),
        "classification_accuracy_when_localized": safe_div(correct, len(localized)),
        "macro_f1": float(np.mean([float(row["f1"]) for row in class_rows])),
        "mean_mask_iou_when_localized": float(localized["mask_iou"].mean()),
    }
    class_frame = pd.DataFrame(class_rows)
    class_frame["variant"] = variant
    return summary, class_frame


def paired_change(all_outcomes: pd.DataFrame, reference: str, candidate: str) -> tuple[dict[str, int], pd.DataFrame]:
    columns = ["image_id", "gt_index", "gt_class_id", "prediction_id", "class_correct", "pred_class_id"]
    left = all_outcomes.loc[
        all_outcomes["variant"].eq(reference) & all_outcomes["record_type"].eq("ground_truth"), columns
    ]
    right = all_outcomes.loc[
        all_outcomes["variant"].eq(candidate) & all_outcomes["record_type"].eq("ground_truth"), columns
    ]
    paired = left.merge(right, on=["image_id", "gt_index", "gt_class_id", "prediction_id"], suffixes=("_reference", "_candidate"), validate="one_to_one")
    paired["change"] = np.select(
        [
            paired["class_correct_reference"] & paired["class_correct_candidate"],
            paired["class_correct_reference"] & ~paired["class_correct_candidate"],
            ~paired["class_correct_reference"] & paired["class_correct_candidate"],
        ],
        ["both_correct", "reference_only_correct", "candidate_only_correct"],
        default="both_wrong",
    )
    counts = paired["change"].value_counts().to_dict()
    summary = {
        "both_correct": int(counts.get("both_correct", 0)),
        "reference_only_correct": int(counts.get("reference_only_correct", 0)),
        "candidate_only_correct": int(counts.get("candidate_only_correct", 0)),
        "both_wrong": int(counts.get("both_wrong", 0)),
        "net_correct_change": int(counts.get("candidate_only_correct", 0) - counts.get("reference_only_correct", 0)),
    }
    paired.insert(0, "reference_variant", reference)
    paired.insert(1, "candidate_variant", candidate)
    return summary, paired


def task_metrics(
    r1a_outcomes: pd.DataFrame, predictions: pd.DataFrame, targets: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    val_targets = targets.loc[targets["split"].eq("val")].copy()
    val_targets["gt_index"] = val_targets["source_polygon_row"].astype(int) - 1
    gt = r1a_outcomes.loc[r1a_outcomes["record_type"].eq("ground_truth")].merge(
        val_targets[
            [
                "image_id",
                "gt_index",
                "instance_id",
                "defect_target",
                "stem_target",
                "stem_target_available",
                "score_target",
                "score_target_available",
            ]
        ],
        on=["image_id", "gt_index"],
        how="left",
        validate="one_to_one",
    )
    if gt["instance_id"].isna().any() or len(gt) != 518:
        raise ValueError("Could not map all validation GT instances to R1 targets")
    matched = gt.loc[gt["localization_matched"].astype(bool)].merge(
        predictions[
            [
                "prediction_id",
                "defect_probability",
                "stem_missing_probability",
                "score_prediction",
            ]
        ],
        on="prediction_id",
        how="left",
        validate="one_to_one",
    )
    if len(matched) != 500 or matched["defect_probability"].isna().any():
        raise ValueError("Frozen matched set changed")
    defect = binary_counts(
        matched["defect_target"].to_numpy(dtype=int),
        matched["defect_probability"].to_numpy() >= 0.5,
    )
    stem_rows = matched["stem_target_available"].eq(1)
    stem = binary_counts(
        matched.loc[stem_rows, "stem_target"].to_numpy(dtype=int),
        matched.loc[stem_rows, "stem_missing_probability"].to_numpy() >= 0.5,
    )
    score_rows = matched["score_target_available"].eq(1)
    error = (
        matched.loc[score_rows, "score_prediction"] - matched.loc[score_rows, "score_target"]
    ).to_numpy(dtype=float)
    metrics = {
        "matched_instances": len(matched),
        "matched_unqualified": int(matched["defect_target"].eq(1).sum()),
        "matched_qualified": int(matched["defect_target"].eq(0).sum()),
        "defect": defect,
        "stem": stem,
        "score_mae": float(np.mean(np.abs(error))),
        "score_rmse": float(np.sqrt(np.mean(np.square(error)))),
    }
    return metrics, matched


def main() -> None:
    args = parse_args()
    r0 = resolve(args.r0)
    checkpoint_path = resolve(args.checkpoint)
    targets_path = resolve(args.targets)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    output.mkdir(parents=True)
    fixed_path = r0 / "fixed_predictions.csv"
    if sha256(fixed_path) != EXPECTED_FIXED_HASH:
        raise ValueError("Frozen R0 prediction cache changed")
    fixed = pd.read_csv(fixed_path, encoding="utf-8-sig")
    if len(fixed) != 549 or not fixed["prediction_id"].is_unique:
        raise ValueError("Frozen prediction count/identity changed")
    device = torch.device(args.device)
    predictions, checkpoint = run_inference(
        fixed, r0, checkpoint_path, device, args.batch_size
    )
    if len(predictions) != 549 or set(predictions["prediction_id"]) != set(fixed["prediction_id"]):
        raise ValueError("R1-A did not preserve all prediction identities")
    predictions.to_csv(output / "r1a_predictions.csv", index=False, encoding="utf-8-sig")

    old_outcomes = pd.read_csv(r0 / "variant_instance_outcomes.csv", encoding="utf-8-sig")
    baseline = old_outcomes.loc[old_outcomes["variant"].eq("baseline")].copy()
    r1a = replace_classes(baseline, predictions)
    all_outcomes = pd.concat([old_outcomes, r1a], ignore_index=True)
    all_outcomes.to_csv(output / "variant_instance_outcomes.csv", index=False, encoding="utf-8-sig")
    r1a_summary, r1a_class = variant_summary(r1a, "r1a")
    old_summary = json.loads((r0 / "r0_summary.json").read_text(encoding="utf-8"))
    summaries = {item["variant"]: item for item in old_summary["variant_metrics"]}
    summaries["r1a"] = r1a_summary
    class_metrics = pd.read_csv(r0 / "class_metrics.csv", encoding="utf-8-sig")
    class_metrics = pd.concat([class_metrics, r1a_class], ignore_index=True)
    class_metrics.to_csv(output / "class_metrics.csv", index=False, encoding="utf-8-sig")

    paired_baseline, paired_baseline_rows = paired_change(all_outcomes, "baseline", "r1a")
    paired_r0, paired_r0_rows = paired_change(all_outcomes, "r0_score_stem", "r1a")
    pd.concat([paired_baseline_rows, paired_r0_rows], ignore_index=True).to_csv(
        output / "paired_instance_changes.csv", index=False, encoding="utf-8-sig"
    )
    targets = pd.read_csv(targets_path, encoding="utf-8-sig")
    tasks, matched = task_metrics(r1a, predictions, targets)
    matched.to_csv(output / "matched_task_predictions.csv", index=False, encoding="utf-8-sig")
    comparison = pd.DataFrame(
        [
            {
                "variant": name,
                "correct_instances": summaries[name]["correct_instances"],
                "classification_accuracy_when_localized": summaries[name]["classification_accuracy_when_localized"],
                "macro_f1": summaries[name]["macro_f1"],
                "predictions": summaries[name]["predictions"],
                "localized_instances": summaries[name]["localized_instances"],
                "false_positive_predictions": summaries[name]["false_positive_predictions"],
            }
            for name in ("baseline", "r0_score_stem", "r1a")
        ]
    )
    comparison.to_csv(output / "variant_comparison.csv", index=False, encoding="utf-8-sig")
    class_by_name = r1a_class.set_index("class_name")
    summary = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
        "experiment": "R1A_frozen_Seg_prediction_replay",
        "inputs": {
            "fixed_predictions": {"path": str(fixed_path), "sha256": sha256(fixed_path)},
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path), "epoch": int(checkpoint["epoch"])},
            "targets": {"path": str(targets_path), "sha256": sha256(targets_path)},
            "test_accessed": False,
        },
        "conservation": {
            "prediction_count": len(predictions),
            "prediction_ids_identical": True,
            "boxes_unchanged": True,
            "masks_unchanged_and_hash_verified": True,
            "confidence_unchanged": True,
            "nms_not_rerun": True,
            "localized_instances": r1a_summary["localized_instances"],
            "false_positive_predictions": r1a_summary["false_positive_predictions"],
        },
        "variant_metrics": [summaries[name] for name in ("baseline", "r0_score_stem", "r1a")],
        "r1a_class_metrics": r1a_class.to_dict("records"),
        "r1a_task_metrics_on_500_matched_instances": tasks,
        "unqualified": {
            "gt": int(class_by_name.loc["unqualified", "gt_count"]),
            "localized": int(class_by_name.loc["unqualified", "gt_count"] - class_by_name.loc["unqualified", "missed_count"]),
            "correct": int(class_by_name.loc["unqualified", "tp_correct_class_iou50"]),
            "classification_accuracy_when_localized": float(class_by_name.loc["unqualified", "classification_accuracy_when_localized"]),
            "f1": float(class_by_name.loc["unqualified", "f1"]),
        },
        "paired_vs_baseline": paired_baseline,
        "paired_vs_r0_score_stem": paired_r0,
        "interpretation_limits": {
            "single_seed": True,
            "single_validation_split": True,
            "test_accessed": False,
            "gt_crop_checkpoint_selection_but_prediction_crop_final_evaluation": True,
            "no_threshold_tuning": True,
        },
    }
    (output / "r1a_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    base = summaries["baseline"]
    r0_summary = summaries["r0_score_stem"]
    r1 = summaries["r1a"]
    conclusion = (
        "R1-A 未超过 baseline"
        if r1["correct_instances"] <= base["correct_instances"]
        else ("R1-A 超过 baseline，但未超过 R0-score-stem" if r1["correct_instances"] <= r0_summary["correct_instances"] else "R1-A 超过 baseline 与 R0-score-stem")
    )
    report = [
        "# R1-A 冻结 Seg 预测重放报告",
        "",
        f"## 结论：{conclusion}",
        "",
        "本实验固定与 R0 完全相同的 549 个预测实例。Box、Mask、confidence、NMS、匹配关系和未匹配预测数均未改变，只由 R1-A 重新给最终类别。",
        "",
        "| 版本 | 正确 / 518 | 定位后分类准确率 | Macro F1 | 预测数 | 未匹配预测 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, label in (("baseline", "Seg baseline"), ("r0_score_stem", "R0-score-stem"), ("r1a", "R1-A")):
        item = summaries[name]
        report.append(
            f"| {label} | {item['correct_instances']} | {item['classification_accuracy_when_localized']:.1%} | "
            f"{item['macro_f1']:.4f} | {item['predictions']} | {item['false_positive_predictions']} |"
        )
    report.extend(
        [
            "",
            "## R1-A 分任务结果（500 个已定位实例）",
            "",
            f"- 缺陷判断：Precision {tasks['defect']['precision']:.3f}，Recall {tasks['defect']['recall']:.3f}，F1 {tasks['defect']['f1']:.3f}。",
            f"- 缺梗判断：Precision {tasks['stem']['precision']:.3f}，Recall {tasks['stem']['recall']:.3f}，F1 {tasks['stem']['f1']:.3f}。",
            f"- 连续评分：MAE {tasks['score_mae']:.3f}，RMSE {tasks['score_rmse']:.3f}。",
            f"- Unqualified：{summary['unqualified']['correct']}/{summary['unqualified']['localized']} 个已定位 UQ 分类正确，F1 {summary['unqualified']['f1']:.4f}。",
            "",
            "## 配对变化",
            "",
            f"- 相对 baseline：错→对 {paired_baseline['candidate_only_correct']}，对→错 {paired_baseline['reference_only_correct']}，净变化 {paired_baseline['net_correct_change']:+d}。",
            f"- 相对 R0-score-stem：错→对 {paired_r0['candidate_only_correct']}，对→错 {paired_r0['reference_only_correct']}，净变化 {paired_r0['net_correct_change']:+d}。",
            "",
            "## 完整性",
            "",
            "- 549 个 prediction_id 完全相同；549 个缓存 Mask 均逐文件通过 SHA-256 校验。",
            "- 定位匹配仍为 500/518，未匹配预测仍为 49。",
            "- 二分类阈值固定为 0.5，未在 val 上调阈值。",
            "- Test 未访问。",
            "",
        ]
    )
    (output / "R1A_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
