"""Run R0 partial rule replay on a frozen set of baseline predictions.

The experiment intentionally preserves the baseline inference path.  The model
is run once with the frozen confidence and NMS settings, and every final
prediction (box, mask, confidence, and original class) is cached.  All R0
variants reload that cache and may change only the final class label.

The test split is never read.  The first two R0 variants are deliberately
partial rule replays: baseline unqualified decisions are retained while the
qualified grade is replayed from the frozen score rule, optionally followed by
the automatic missing-stem rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml

from analyze_val_instance_errors import (
    CLASS_NAMES,
    IMAGE_SUFFIXES,
    Instance,
    class_metrics_from_rows,
    mask_iou_matrix,
    optimal_match,
    polygon_to_mask,
    read_yolo_instances,
    safe_div,
)
from apply_stem_rule import STATUS_MISSING, detect_stem
from extract_features import process_instance


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/grading_four_class_v3"
DEFAULT_WEIGHTS = (
    ROOT
    / "models/experiments/yolo26n_seg_offlineaug_e200_mask50_seed20260915/weights/best.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/reports/yolo26n_offlineaug_e200_r0_fixed_replay_20260929"
)
DEFAULT_FEATURE_CONFIG = ROOT / "configs/feature_extraction_frozen_v1_val.yaml"
DEFAULT_STEM_CONFIG = ROOT / "configs/stem_detection_v1.yaml"
DEFAULT_MANIFEST = ROOT / "data/manifests/dataset_manifest.csv"
DEFAULT_REFERENCE_FEATURES = ROOT / "outputs/strict_rot_exclusion_v2/qualified_features.csv"
DEFAULT_REFERENCE_MODE = "recover_from_candidate_ecdf"
DEFAULT_REFERENCE_SIZE = 2649
DEFAULT_GRADING_CONFIG = (
    ROOT / "outputs/grading_standard_manual_v4_final/grading_standard_config.json"
)
DEFAULT_OLD_CANDIDATES = (
    ROOT / "outputs/grading_standard_manual_v4_final/grading_labels_candidate_v3.csv"
)
DEFAULT_EXPECTED_SUMMARY = (
    ROOT
    / "outputs/reports/yolo26n_offlineaug_e200_val_error_analysis/analysis_summary.json"
)
GLOBAL_SCORE_FEATURES = ("area", "minor_axis", "solidity", "aspect_ratio", "circularity")
VARIANTS = ("baseline", "r0_score", "r0_score_stem")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-config", type=Path, default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--stem-config", type=Path, default=DEFAULT_STEM_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--reference-features", type=Path, default=DEFAULT_REFERENCE_FEATURES)
    parser.add_argument(
        "--reference-mode",
        choices=("recover_from_candidate_ecdf", "reference_features"),
        default=DEFAULT_REFERENCE_MODE,
    )
    parser.add_argument("--reference-size", type=int, default=DEFAULT_REFERENCE_SIZE)
    parser.add_argument("--grading-config", type=Path, default=DEFAULT_GRADING_CONFIG)
    parser.add_argument("--old-candidates", type=Path, default=DEFAULT_OLD_CANDIDATES)
    parser.add_argument("--expected-summary", type=Path, default=DEFAULT_EXPECTED_SUMMARY)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--inference-conf", type=float, default=0.05)
    parser.add_argument("--analysis-conf", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--max-images", type=int, default=None, help="Smoke test only.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def save_mask_crop(mask: np.ndarray, output: Path) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"Cannot cache empty mask: {output}")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    output.parent.mkdir(parents=True, exist_ok=True)
    crop = mask[y1:y2, x1:x2].astype(np.uint8) * 255
    if not cv2.imwrite(str(output), crop):
        raise IOError(f"Could not write cached mask: {output}")
    return x1, y1, x2, y2


def load_mask_crop(row: pd.Series, output: Path) -> np.ndarray:
    crop_path = output / str(row["fixed_mask_path"])
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if crop is None:
        raise FileNotFoundError(crop_path)
    # Ultralytics patches cv2.imread and preserves a singleton channel for
    # grayscale images. Normalize that equivalent HxWx1 representation to HxW.
    if crop.ndim == 3 and crop.shape[2] == 1:
        crop = crop[:, :, 0]
    height, width = int(row["image_height"]), int(row["image_width"])
    x1, y1, x2, y2 = [int(row[name]) for name in ("mask_x1", "mask_y1", "mask_x2", "mask_y2")]
    expected = (y2 - y1, x2 - x1)
    if crop.shape != expected:
        raise ValueError(
            f"Cached mask shape mismatch for {row['prediction_id']}: {crop.shape} != {expected}"
        )
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = crop > 0
    return mask


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    if not ref.size:
        raise ValueError("Empty scoring reference")
    values = np.asarray(values, dtype=float)
    left = np.searchsorted(ref, values, side="left")
    right = np.searchsorted(ref, values, side="right")
    return np.clip((left + right) / (2.0 * ref.size), 0.0, 1.0)


def build_references(reference_features: pd.DataFrame) -> dict[str, np.ndarray]:
    train = reference_features.loc[reference_features["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("No train rows in reference feature table")
    references: dict[str, np.ndarray] = {}
    for feature in GLOBAL_SCORE_FEATURES:
        if feature not in train:
            raise KeyError(f"Missing reference feature: {feature}")
        values = train[feature].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Reference feature contains non-finite values: {feature}")
        references[feature] = np.sort(values)
    return references


def recover_references_from_candidate_ecdf(
    old_candidates: pd.DataFrame,
    reference_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], pd.DataFrame]:
    """Recover the missing frozen ECDF values from stored raw-value/percentile anchors."""
    if reference_size <= 0:
        raise ValueError("reference_size must be positive")

    references: dict[str, np.ndarray] = {}
    feature_audit: dict[str, dict[str, Any]] = {}
    recovered_rows: list[dict[str, Any]] = []
    tolerance = 1e-12

    for feature in GLOBAL_SCORE_FEATURES:
        percentile_column = f"pct_{feature}"
        missing = [
            column
            for column in (feature, percentile_column)
            if column not in old_candidates.columns
        ]
        if missing:
            raise KeyError(f"Candidate ECDF anchors missing columns: {missing}")

        anchors = old_candidates[[feature, percentile_column]].copy()
        anchors[feature] = pd.to_numeric(anchors[feature], errors="coerce")
        anchors[percentile_column] = pd.to_numeric(
            anchors[percentile_column], errors="coerce"
        )
        if not np.isfinite(anchors.to_numpy(dtype=float)).all():
            raise ValueError(f"Candidate ECDF anchors contain non-finite values: {feature}")
        if not anchors[percentile_column].between(0.0, 1.0).all():
            raise ValueError(f"Candidate ECDF percentiles are outside [0, 1]: {feature}")

        grouped = (
            anchors.groupby(feature, sort=True, as_index=False)[percentile_column]
            .agg(["min", "max"])
            .reset_index()
        )
        max_tie_spread = float((grouped["max"] - grouped["min"]).abs().max())
        if max_tie_spread > tolerance:
            raise ValueError(
                f"Inconsistent stored percentile for tied {feature} values: "
                f"max_spread={max_tie_spread}"
            )

        previous_count = 0
        values: list[float] = []
        max_grid_residual = 0.0
        for raw_value, percentile in zip(
            grouped[feature].to_numpy(dtype=float),
            grouped["min"].to_numpy(dtype=float),
            strict=True,
        ):
            grid_value = 2.0 * reference_size * percentile
            rounded_grid_value = int(round(grid_value))
            max_grid_residual = max(
                max_grid_residual, abs(grid_value - rounded_grid_value)
            )
            multiplicity = rounded_grid_value - 2 * previous_count
            if multiplicity < 0:
                raise ValueError(
                    f"Candidate ECDF anchors are not monotone for {feature}: "
                    f"value={raw_value}, multiplicity={multiplicity}"
                )
            if previous_count + multiplicity > reference_size:
                raise ValueError(
                    f"Recovered {feature} reference exceeds reference_size={reference_size}"
                )
            if multiplicity:
                values.extend([float(raw_value)] * multiplicity)
                previous_count += multiplicity

        if max_grid_residual > 1e-8:
            raise ValueError(
                f"Stored {feature} percentiles do not fit reference_size={reference_size}: "
                f"max_grid_residual={max_grid_residual}"
            )
        if previous_count != reference_size:
            raise ValueError(
                f"Recovered {feature} reference has {previous_count} values; "
                f"expected {reference_size}"
            )

        reference = np.asarray(values, dtype=float)
        replayed_percentile = empirical_percentile(
            reference, anchors[feature].to_numpy(dtype=float)
        )
        stored_percentile = anchors[percentile_column].to_numpy(dtype=float)
        max_percentile_difference = float(
            np.max(np.abs(replayed_percentile - stored_percentile))
        )
        percentile_mismatches = int(
            np.count_nonzero(
                np.abs(replayed_percentile - stored_percentile) > tolerance
            )
        )
        if percentile_mismatches:
            raise ValueError(
                f"Recovered {feature} reference does not reproduce stored percentiles: "
                f"max_difference={max_percentile_difference}, "
                f"mismatches={percentile_mismatches}"
            )

        references[feature] = reference
        feature_audit[feature] = {
            "reference_values": int(reference.size),
            "unique_reference_values": int(np.unique(reference).size),
            "max_grid_residual": max_grid_residual,
            "max_tied_percentile_spread": max_tie_spread,
            "max_percentile_difference": max_percentile_difference,
            "percentile_mismatches": percentile_mismatches,
        }
        recovered_rows.extend(
            {
                "reference_scope": "recovered_frozen_global_train",
                "feature": feature,
                "rank": rank,
                "n": reference_size,
                "value": float(value),
            }
            for rank, value in enumerate(reference, start=1)
        )

    audit = {
        "mode": "recover_from_candidate_ecdf",
        "reference_size": reference_size,
        "anchor_rows": int(len(old_candidates)),
        "features": feature_audit,
        "status": "exact_percentile_reproduction",
    }
    return references, audit, pd.DataFrame(recovered_rows)


def apply_frozen_score(
    frame: pd.DataFrame,
    references: dict[str, np.ndarray],
    grading_config: dict[str, Any],
) -> pd.DataFrame:
    scored = frame.copy()
    finite = np.ones(len(scored), dtype=bool)
    for feature in GLOBAL_SCORE_FEATURES:
        values = pd.to_numeric(scored[feature], errors="coerce").to_numpy(dtype=float)
        finite &= np.isfinite(values)
        scored[f"pct_{feature}"] = empirical_percentile(references[feature], values)

    scored["aspect_typicality"] = np.clip(
        1.0 - 2.0 * np.abs(scored["pct_aspect_ratio"] - 0.5), 0.0, 1.0
    )
    scored["shape_score"] = 100.0 * (
        0.50 * scored["pct_solidity"]
        + 0.30 * scored["aspect_typicality"]
        + 0.20 * scored["pct_circularity"]
    )
    weights = grading_config["score_formula"]["weights"]
    scored["rule_score"] = (
        100.0
        * (
            float(weights["pct_area"]) * scored["pct_area"]
            + float(weights["pct_minor_axis"]) * scored["pct_minor_axis"]
        )
        + float(weights["shape_score"]) * scored["shape_score"]
    )
    low = float(grading_config["thresholds"]["grade3_to_2"])
    high = float(grading_config["thresholds"]["grade2_to_1"])
    scored["rule_grade"] = np.where(
        scored["rule_score"].ge(high),
        1,
        np.where(scored["rule_score"].lt(low), 3, 2),
    ).astype(int)
    scored["score_available"] = finite
    scored.loc[~finite, "rule_score"] = np.nan
    scored.loc[~finite, "rule_grade"] = pd.NA
    return scored


def validate_reconstructed_scoring(
    old_candidates: pd.DataFrame,
    references: dict[str, np.ndarray],
    grading_config: dict[str, Any],
) -> dict[str, Any]:
    replayed = apply_frozen_score(old_candidates, references, grading_config)
    stored_score = old_candidates["manual_calibrated_score"].to_numpy(dtype=float)
    replayed_score = replayed["rule_score"].to_numpy(dtype=float)
    max_difference = float(np.max(np.abs(stored_score - replayed_score)))
    grade_mismatches = int(
        np.count_nonzero(
            old_candidates["automatic_base_grade_v3"].to_numpy(dtype=int)
            != replayed["rule_grade"].to_numpy(dtype=int)
        )
    )
    if max_difference > 1e-10 or grade_mismatches:
        raise ValueError(
            "Reconstructed frozen score reference failed validation: "
            f"max_score_difference={max_difference}, grade_mismatches={grade_mismatches}"
        )
    return {
        "rows": int(len(old_candidates)),
        "max_score_difference": max_difference,
        "automatic_base_grade_mismatches": grade_mismatches,
        "status": "exact_reproduction",
    }


def load_manifest_index(path: Path) -> dict[str, dict[str, Any]]:
    manifest = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    required = {"new_stem", "split", "original_number", "original_stem", "source_group"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise KeyError(f"Manifest missing columns: {missing}")
    if manifest["new_stem"].duplicated().any():
        raise ValueError("Manifest new_stem must be unique")
    return {str(row["new_stem"]): row.to_dict() for _, row in manifest.iterrows()}


def validate_protocol(args: argparse.Namespace, expected: dict[str, Any], weights: Path) -> None:
    protocol = expected["protocol"]
    comparisons = {
        "imgsz": (int(args.imgsz), int(protocol["imgsz"])),
        "inference_conf": (float(args.inference_conf), float(protocol["inference_confidence"])),
        "analysis_conf": (float(args.analysis_conf), float(protocol["analysis_confidence"])),
        "nms_iou": (float(args.nms_iou), float(protocol["nms_iou"])),
        "match_iou": (float(args.match_iou), float(protocol["match_iou"])),
    }
    mismatches = {
        name: {"actual": actual, "expected": wanted}
        for name, (actual, wanted) in comparisons.items()
        if not math.isclose(actual, wanted, rel_tol=0.0, abs_tol=1e-12)
    }
    if bool(protocol["agnostic_nms"]):
        mismatches["agnostic_nms"] = {"actual": False, "expected": True}
    if protocol["mask_dedup_iou"] is not None:
        mismatches["mask_dedup_iou"] = {"actual": None, "expected": protocol["mask_dedup_iou"]}
    expected_hash = str(expected["inputs"]["weights_sha256"])
    actual_hash = sha256(weights)
    if actual_hash != expected_hash:
        mismatches["weights_sha256"] = {"actual": actual_hash, "expected": expected_hash}
    if mismatches:
        raise ValueError(f"R0 protocol differs from frozen baseline: {json.dumps(mismatches)}")


def evaluate_variant(
    variant: str,
    predictions: pd.DataFrame,
    images_dir: Path,
    labels_dir: Path,
    output: Path,
    match_iou: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    class_column = f"{variant}_class_id"
    records: list[dict[str, Any]] = []
    confusion: Counter[tuple[int, int]] = Counter()
    matched_details: list[dict[str, Any]] = []
    images = 0
    gt_count = 0
    prediction_count = 0
    localized = 0
    correct = 0
    ious: list[float] = []

    for image_id, group in predictions.groupby("image_id", sort=True):
        image_path = images_dir / f"{image_id}.png"
        if not image_path.is_file():
            candidates = sorted(images_dir.glob(f"{image_id}.*"))
            if len(candidates) != 1:
                raise FileNotFoundError(f"Could not resolve validation image: {image_id}")
            image_path = candidates[0]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(image_path)
        height, width = image.shape[:2]
        gt_instances = read_yolo_instances(labels_dir / f"{image_id}.txt", height, width)
        ordered = group.sort_values("pred_index")
        pred_rows = [row for _, row in ordered.iterrows()]
        pred_instances = [
            Instance(
                class_id=int(row[class_column]),
                confidence=float(row["confidence"]),
                mask=load_mask_crop(row, output),
            )
            for row in pred_rows
        ]
        iou_matrix = mask_iou_matrix(gt_instances, pred_instances)
        matches = optimal_match(iou_matrix, match_iou)
        match_by_gt = {gt: (pred, overlap) for gt, pred, overlap in matches}
        match_by_pred = {pred: (gt, overlap) for gt, pred, overlap in matches}

        images += 1
        gt_count += len(gt_instances)
        prediction_count += len(pred_instances)
        localized += len(matches)
        ious.extend(overlap for _, _, overlap in matches)

        for gt_index, gt in enumerate(gt_instances):
            pred_index, overlap = match_by_gt.get(gt_index, (None, 0.0))
            pred = pred_instances[pred_index] if pred_index is not None else None
            pred_row = pred_rows[pred_index] if pred_index is not None else None
            is_correct = pred is not None and pred.class_id == gt.class_id
            correct += int(is_correct)
            if pred is None:
                error_type = "missed"
            elif is_correct:
                error_type = "correct"
            else:
                error_type = "class_confusion"
            if pred is not None:
                confusion[(gt.class_id, pred.class_id)] += 1
            record = {
                "variant": variant,
                "image_id": image_id,
                "record_type": "ground_truth",
                "gt_index": gt_index,
                "prediction_id": pred_row["prediction_id"] if pred_row is not None else "",
                "gt_class_id": gt.class_id,
                "gt_class_name": CLASS_NAMES[gt.class_id],
                "pred_class_id": pred.class_id if pred is not None else None,
                "pred_class_name": CLASS_NAMES[pred.class_id] if pred is not None else "",
                "confidence": pred.confidence if pred is not None else None,
                "mask_iou": overlap,
                "localization_matched": pred is not None,
                "class_correct": bool(is_correct),
                "error_type": error_type,
            }
            records.append(record)
            matched_details.append(record.copy())

        for pred_index, pred in enumerate(pred_instances):
            if pred_index in match_by_pred:
                continue
            pred_row = pred_rows[pred_index]
            records.append(
                {
                    "variant": variant,
                    "image_id": image_id,
                    "record_type": "prediction_only",
                    "gt_index": None,
                    "prediction_id": pred_row["prediction_id"],
                    "gt_class_id": None,
                    "gt_class_name": "",
                    "pred_class_id": pred.class_id,
                    "pred_class_name": CLASS_NAMES[pred.class_id],
                    "confidence": pred.confidence,
                    "mask_iou": 0.0,
                    "localization_matched": False,
                    "class_correct": False,
                    "error_type": "false_positive",
                }
            )

    class_rows = class_metrics_from_rows(records)
    macro_f1 = float(np.mean([float(row["f1"]) for row in class_rows]))
    summary = {
        "variant": variant,
        "images": images,
        "gt_instances": gt_count,
        "predictions": prediction_count,
        "localized_instances": localized,
        "correct_instances": correct,
        "class_confusions": localized - correct,
        "missed": gt_count - localized,
        "false_positive_predictions": prediction_count - localized,
        "localization_recall": safe_div(localized, gt_count),
        "classification_accuracy_when_localized": safe_div(correct, localized),
        "micro_precision_correct_class_iou50": safe_div(correct, prediction_count),
        "micro_recall_correct_class_iou50": safe_div(correct, gt_count),
        "micro_f1_correct_class_iou50": safe_div(
            2 * safe_div(correct, prediction_count) * safe_div(correct, gt_count),
            safe_div(correct, prediction_count) + safe_div(correct, gt_count),
        ),
        "macro_f1": macro_f1,
        "mean_mask_iou_when_localized": float(np.mean(ious)) if ious else 0.0,
    }
    confusion_rows = [
        {
            "variant": variant,
            "gt_class_id": gt,
            "gt_class_name": CLASS_NAMES[gt],
            "pred_class_id": pred,
            "pred_class_name": CLASS_NAMES[pred],
            "count": count,
        }
        for (gt, pred), count in sorted(confusion.items())
    ]
    for row in class_rows:
        row["variant"] = variant
    return summary, class_rows, records, confusion_rows


def paired_summary(outcomes: pd.DataFrame, variant: str) -> dict[str, int]:
    gt = outcomes.loc[outcomes["record_type"].eq("ground_truth")]
    baseline = gt.loc[gt["variant"].eq("baseline"), ["image_id", "gt_index", "class_correct"]]
    candidate = gt.loc[gt["variant"].eq(variant), ["image_id", "gt_index", "class_correct"]]
    paired = baseline.merge(candidate, on=["image_id", "gt_index"], suffixes=("_baseline", "_variant"))
    both = paired["class_correct_baseline"] & paired["class_correct_variant"]
    baseline_only = paired["class_correct_baseline"] & ~paired["class_correct_variant"]
    variant_only = ~paired["class_correct_baseline"] & paired["class_correct_variant"]
    neither = ~paired["class_correct_baseline"] & ~paired["class_correct_variant"]
    return {
        "both_correct": int(both.sum()),
        "baseline_only_correct": int(baseline_only.sum()),
        "variant_only_correct": int(variant_only.sum()),
        "both_wrong": int(neither.sum()),
        "net_correct_change": int(variant_only.sum() - baseline_only.sum()),
    }


def build_report(summary: dict[str, Any]) -> str:
    metrics = {row["variant"]: row for row in summary["variant_metrics"]}
    paired = summary["paired_vs_baseline"]
    lines = [
        "# R0 固定预测实例规则重放",
        "",
        "## 结论",
        "",
        summary["conclusion"],
        "",
        "## 冻结口径",
        "",
        "- 使用当前 200 轮 Seg baseline 的原始推理路径：confidence/NMS/one-to-many head 均不改变。",
        "- 首次推理后缓存最终 Box、预测 Mask、confidence 与原类别；所有变体仅修改类别。",
        "- `baseline`、`r0_score`、`r0_score_stem` 的 prediction_id 集合完全相同。",
        "- `r0_score` 与 `r0_score_stem` 都保留 baseline 的 UQ/qualified 二元决定，因此只是 partial rule replay。",
        "- 自动果梗状态为 UNCERTAIN 时保留评分等级，不进行人工复检。",
        "- 只读取 val；test_accessed=false。",
        "",
        "## 同实例结果",
        "",
        "| 变体 | 预测数 | 定位 | 正确 | 分级混淆 | 漏检 | 未匹配预测 | 定位后准确率 | Macro F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = {
        "baseline": "Baseline 四分类 Head",
        "r0_score": "R0-score：UQ 不变 + 评分重放",
        "r0_score_stem": "R0-score-stem：再加自动缺梗规则",
    }
    for variant in VARIANTS:
        row = metrics[variant]
        lines.append(
            f"| {labels[variant]} | {row['predictions']} | {row['localized_instances']} | "
            f"{row['correct_instances']} | {row['class_confusions']} | {row['missed']} | "
            f"{row['false_positive_predictions']} | {row['classification_accuracy_when_localized']:.4f} | "
            f"{row['macro_f1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 与 Baseline 的逐 GT 配对变化",
            "",
            "| 变体 | 两者都对 | 仅 Baseline 对 | 仅变体对 | 两者都错 | 净正确数变化 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant in ("r0_score", "r0_score_stem"):
        row = paired[variant]
        lines.append(
            f"| {labels[variant]} | {row['both_correct']} | {row['baseline_only_correct']} | "
            f"{row['variant_only_correct']} | {row['both_wrong']} | {row['net_correct_change']:+d} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这不是三任务神经网络，也没有训练新的评分、果梗或缺陷 Head。",
            "- UQ 判断仍来自原四分类 Head；本实验不能证明一个新 Defect Head 的效果。",
            "- `4791/4795=99.92%` 只属于标签生成规则一致性检查，不属于本报告的模型性能。",
            "- 果梗规则是自动候选规则；UNCERTAIN 不会被静默改成三级。",
            "- 结果只适用于单一 val 划分和当前权重，不使用 test 做选择。",
            "",
            "## 审计",
            "",
            f"- 固定预测缓存：`fixed_predictions.csv`，SHA-256 `{summary['artifacts']['fixed_predictions_sha256']}`。",
            f"- 权重 SHA-256：`{summary['inputs']['weights_sha256']}`。",
            f"- 冻结评分重建：{summary['scoring_reconstruction']['status']}，最大数值差 "
            f"{summary['scoring_reconstruction']['max_score_difference']:.3e}。",
            f"- 自动果梗状态：{json.dumps(summary['stem_status_counts'], ensure_ascii=False)}。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    dataset = resolve(args.dataset)
    weights = resolve(args.weights)
    output = resolve(args.output)
    feature_config_path = resolve(args.feature_config)
    stem_config_path = resolve(args.stem_config)
    manifest_path = resolve(args.manifest)
    reference_features_path = resolve(args.reference_features)
    grading_config_path = resolve(args.grading_config)
    old_candidates_path = resolve(args.old_candidates)
    expected_summary_path = resolve(args.expected_summary)
    images_dir = dataset / "images/val"
    labels_dir = dataset / "labels/val"

    if "test" in {part.lower() for part in images_dir.parts}:
        raise ValueError("Refusing to run R0 on a test path")
    required_paths = (
        weights,
        feature_config_path,
        stem_config_path,
        manifest_path,
        grading_config_path,
        old_candidates_path,
        expected_summary_path,
    )
    if args.reference_mode == "reference_features":
        required_paths = (*required_paths, reference_features_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing R0 inputs: {missing_paths}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    expected = json.loads(expected_summary_path.read_text(encoding="utf-8"))
    validate_protocol(args, expected, weights)
    manifest_index = load_manifest_index(manifest_path)
    grading_config = json.loads(grading_config_path.read_text(encoding="utf-8"))
    feature_config = yaml.safe_load(feature_config_path.read_text(encoding="utf-8"))
    stem_config = yaml.safe_load(stem_config_path.read_text(encoding="utf-8"))
    weight_hash = sha256(weights)

    old_candidates = pd.read_csv(old_candidates_path, encoding="utf-8-sig")
    recovered_reference_path: Path | None = None
    if args.reference_mode == "recover_from_candidate_ecdf":
        references, reference_recovery, recovered_reference_frame = (
            recover_references_from_candidate_ecdf(
                old_candidates, args.reference_size
            )
        )
        recovered_reference_path = output / "recovered_scoring_reference_values.csv"
        write_csv(recovered_reference_path, recovered_reference_frame)
    else:
        reference_features = pd.read_csv(reference_features_path, encoding="utf-8-sig")
        references = build_references(reference_features)
        reference_recovery = {
            "mode": "reference_features",
            "reference_size": {
                feature: int(values.size) for feature, values in references.items()
            },
            "status": "loaded_from_reference_features",
        }
    scoring_reconstruction = validate_reconstructed_scoring(
        old_candidates, references, grading_config
    )
    scoring_reconstruction["reference_recovery"] = reference_recovery

    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise ValueError(f"No val images in {images_dir}")

    fixed_mask_dir = output / "fixed_predictions/masks"
    feature_workspace = output / "replay_workspace"
    feature_root = feature_workspace / "outputs/features_frozen_v1/val"
    feature_config["_resolved_model_path"] = str(weights)
    feature_config["output"] = dict(feature_config["output"])
    feature_config["output"]["roi_padding_px"] = int(feature_config["output"].get("roi_padding_px", 12))

    from ultralytics import YOLO

    model = YOLO(str(weights))
    fixed_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(image_path)
        height, width = image.shape[:2]
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.inference_conf,
            iou=args.nms_iou,
            agnostic_nms=False,
            max_det=300,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )[0]
        candidates: list[tuple[int, float, np.ndarray, np.ndarray]] = []
        if result.boxes is not None and result.masks is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            boxes = result.boxes.xyxy.detach().cpu().numpy().astype(float)
            polygons = result.masks.xy
            if not (len(classes) == len(confidences) == len(boxes) == len(polygons)):
                raise ValueError(f"Prediction length mismatch: {image_path}")
            for class_id, confidence, box, polygon in zip(
                classes, confidences, boxes, polygons, strict=True
            ):
                if float(confidence) < args.analysis_conf:
                    continue
                mask = polygon_to_mask(np.asarray(polygon), height, width)
                if np.any(mask):
                    candidates.append((int(class_id), float(confidence), box, mask))

        source = manifest_index.get(image_path.stem)
        if source is None:
            raise KeyError(f"Image missing from dataset manifest: {image_path.stem}")
        image_record = {
            "new_stem": image_path.stem,
            "split": "val",
            "original_number": int(source["original_number"]),
            "original_stem": source["original_stem"],
            "source_group": source["source_group"],
            "resolved_image_path": str(image_path.resolve()),
        }

        for pred_index, (class_id, confidence, box, mask) in enumerate(candidates):
            if not 0 <= class_id < len(CLASS_NAMES):
                raise ValueError(f"Unexpected class {class_id}: {image_path}")
            prediction_id = f"{image_path.stem}__p{pred_index + 1:03d}"
            mask_path = fixed_mask_dir / f"{prediction_id}.png"
            mask_x1, mask_y1, mask_x2, mask_y2 = save_mask_crop(mask, mask_path)
            fixed_row = {
                "prediction_id": prediction_id,
                "image_id": image_path.stem,
                "image_path": str(image_path.resolve()),
                "pred_index": pred_index,
                "image_width": width,
                "image_height": height,
                "box_x1": float(box[0]),
                "box_y1": float(box[1]),
                "box_x2": float(box[2]),
                "box_y2": float(box[3]),
                "confidence": confidence,
                "baseline_class_id": class_id,
                "baseline_class_name": CLASS_NAMES[class_id],
                "fixed_mask_path": mask_path.relative_to(output).as_posix(),
                "fixed_mask_sha256": sha256(mask_path),
                "mask_x1": mask_x1,
                "mask_y1": mask_y1,
                "mask_x2": mask_x2,
                "mask_y2": mask_y2,
            }
            fixed_rows.append(fixed_row)
            feature = process_instance(
                image_bgr=image,
                raw_mask=mask.astype(np.uint8) * 255,
                confidence=confidence,
                image_record=image_record,
                instance_id=prediction_id,
                config=feature_config,
                output_root=feature_root,
                model_hash=weight_hash,
            )
            feature.update(
                {
                    "prediction_id": prediction_id,
                    "pred_index": pred_index,
                    "baseline_class_id": class_id,
                    "baseline_class_name": CLASS_NAMES[class_id],
                }
            )
            feature_rows.append(feature)

    fixed = pd.DataFrame(fixed_rows).sort_values(["image_id", "pred_index"]).reset_index(drop=True)
    features = pd.DataFrame(feature_rows).sort_values(["image_id", "pred_index"]).reset_index(drop=True)
    if fixed.empty or features.empty:
        raise RuntimeError("R0 generated no fixed predictions")
    if fixed["prediction_id"].duplicated().any() or features["prediction_id"].duplicated().any():
        raise ValueError("R0 prediction_id must be unique")
    if set(fixed["prediction_id"]) != set(features["prediction_id"]):
        raise ValueError("Fixed prediction and feature IDs differ")

    fixed_path = output / "fixed_predictions.csv"
    write_csv(fixed_path, fixed)
    write_csv(output / "prediction_features.csv", features)

    scored = apply_frozen_score(features, references, grading_config)

    stem_results: list[dict[str, Any]] = []
    cached_image_path = ""
    cached_image: np.ndarray | None = None
    for _, row in scored.iterrows():
        image_path = str(row["image_path"])
        if image_path != cached_image_path:
            cached_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if cached_image is None:
                raise FileNotFoundError(image_path)
            cached_image_path = image_path
        try:
            stem, _ = detect_stem(
                row,
                feature_workspace,
                cached_image,
                stem_config["stem_detection"],
            )
        except Exception as exc:
            stem = {
                "stem_auto_status": "ERROR",
                "stem_detection_warning": f"{type(exc).__name__}:{exc}",
            }
        stem_results.append(stem)
    scored = pd.concat([scored.reset_index(drop=True), pd.DataFrame(stem_results)], axis=1)

    replay = fixed.merge(
        scored[
            [
                "prediction_id",
                "rule_score",
                "rule_grade",
                "score_available",
                "stem_auto_status",
                "stem_detection_warning",
            ]
        ],
        on="prediction_id",
        how="left",
        validate="one_to_one",
    )
    replay["baseline_class_id"] = replay["baseline_class_id"].astype(int)
    replay["r0_score_class_id"] = replay["baseline_class_id"]
    score_applicable = replay["baseline_class_id"].ne(0) & replay["score_available"].fillna(False)
    replay.loc[score_applicable, "r0_score_class_id"] = replay.loc[
        score_applicable, "rule_grade"
    ].astype(int)
    replay["r0_score_stem_class_id"] = replay["r0_score_class_id"].astype(int)
    missing_stem = replay["baseline_class_id"].ne(0) & replay["stem_auto_status"].eq(STATUS_MISSING)
    replay.loc[missing_stem, "r0_score_stem_class_id"] = 3
    for variant in VARIANTS:
        class_column = f"{variant}_class_id"
        replay[class_column] = replay[class_column].astype(int)
        replay[f"{variant}_class_name"] = replay[class_column].map(dict(enumerate(CLASS_NAMES)))
    if not (
        replay["baseline_class_id"].eq(0)
        == replay["r0_score_class_id"].eq(0)
    ).all():
        raise AssertionError("R0-score changed the baseline UQ/qualified decision")
    if not (
        replay["baseline_class_id"].eq(0)
        == replay["r0_score_stem_class_id"].eq(0)
    ).all():
        raise AssertionError("R0-score-stem changed the baseline UQ/qualified decision")
    write_csv(output / "replay_predictions.csv", replay)

    variant_metrics: list[dict[str, Any]] = []
    all_class_metrics: list[dict[str, Any]] = []
    all_outcomes: list[dict[str, Any]] = []
    all_confusions: list[dict[str, Any]] = []
    for variant in VARIANTS:
        metrics, class_rows, outcomes, confusion_rows = evaluate_variant(
            variant,
            replay,
            images_dir,
            labels_dir,
            output,
            args.match_iou,
        )
        variant_metrics.append(metrics)
        all_class_metrics.extend(class_rows)
        all_outcomes.extend(outcomes)
        all_confusions.extend(confusion_rows)

    if args.max_images is None:
        baseline = next(row for row in variant_metrics if row["variant"] == "baseline")
        expected_overall = expected["overall"]
        exact_fields = {
            "images": int(expected_overall["images"]),
            "gt_instances": int(expected_overall["gt_instances"]),
            "predictions": int(expected_overall["predictions_at_analysis_conf"]),
            "localized_instances": int(expected_overall["localized_instances"]),
            "correct_instances": int(expected_overall["correct_instances"]),
        }
        differences = {
            key: {"actual": int(baseline[key]), "expected": wanted}
            for key, wanted in exact_fields.items()
            if int(baseline[key]) != wanted
        }
        if differences:
            raise ValueError(
                "Fixed prediction cache does not reproduce the frozen baseline: "
                f"{json.dumps(differences)}"
            )

    outcomes_frame = pd.DataFrame(all_outcomes)
    paired = {
        variant: paired_summary(outcomes_frame, variant)
        for variant in ("r0_score", "r0_score_stem")
    }
    write_csv(output / "variant_metrics.csv", pd.DataFrame(variant_metrics))
    write_csv(output / "class_metrics.csv", pd.DataFrame(all_class_metrics))
    write_csv(output / "variant_instance_outcomes.csv", outcomes_frame)
    write_csv(output / "confusion_counts.csv", pd.DataFrame(all_confusions))

    metrics_by_variant = {row["variant"]: row for row in variant_metrics}
    best_variant = max(VARIANTS, key=lambda name: metrics_by_variant[name]["macro_f1"])
    baseline_macro = metrics_by_variant["baseline"]["macro_f1"]
    best_macro = metrics_by_variant[best_variant]["macro_f1"]
    if best_variant == "baseline":
        conclusion = (
            "在完全固定预测实例的条件下，两种 partial rule replay 均未超过 baseline 的 Macro F1；"
            "现有证据不支持用本次自动规则重放替代四分类 Head。"
        )
    else:
        conclusion = (
            f"在完全固定预测实例的条件下，{best_variant} 的 Macro F1 从 "
            f"{baseline_macro:.4f} 变为 {best_macro:.4f}。该变化只来自最终类别重放，"
            "不来自定位、confidence 或 NMS；但 UQ 判断仍沿用 baseline。"
        )

    summary = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "experiment": "R0_fixed_prediction_partial_rule_replay",
        "status": "completed",
        "conclusion": conclusion,
        "inputs": {
            "dataset": str(dataset),
            "weights": str(weights),
            "weights_sha256": weight_hash,
            "feature_config": str(feature_config_path),
            "feature_config_sha256": sha256(feature_config_path),
            "stem_config": str(stem_config_path),
            "stem_config_sha256": sha256(stem_config_path),
            "reference_mode": args.reference_mode,
            "reference_size": args.reference_size,
            "reference_features": (
                str(reference_features_path)
                if args.reference_mode == "reference_features"
                else None
            ),
            "reference_features_sha256": (
                sha256(reference_features_path)
                if args.reference_mode == "reference_features"
                else None
            ),
            "old_candidates": str(old_candidates_path),
            "old_candidates_sha256": sha256(old_candidates_path),
            "recovered_reference": (
                str(recovered_reference_path) if recovered_reference_path else None
            ),
            "recovered_reference_sha256": (
                sha256(recovered_reference_path) if recovered_reference_path else None
            ),
            "grading_config": str(grading_config_path),
            "grading_config_sha256": sha256(grading_config_path),
            "split": "val",
            "test_accessed": False,
        },
        "protocol": {
            "imgsz": args.imgsz,
            "inference_confidence": args.inference_conf,
            "analysis_confidence": args.analysis_conf,
            "nms_iou": args.nms_iou,
            "agnostic_nms": False,
            "mask_dedup_iou": None,
            "match_iou": args.match_iou,
            "prediction_set": "cached_baseline_final_predictions",
            "allowed_mutation": "final_class_only",
            "unqualified_policy": "retain_baseline_UQ_vs_qualified_decision",
            "uncertain_stem_policy": "retain_score_grade_without_human_review",
        },
        "scoring_reconstruction": scoring_reconstruction,
        "stem_status_counts": {
            str(key): int(value)
            for key, value in replay["stem_auto_status"].fillna("MISSING_VALUE").value_counts().items()
        },
        "score_unavailable_predictions": int((~replay["score_available"].fillna(False)).sum()),
        "score_changed_predictions": int(
            replay["baseline_class_id"].ne(replay["r0_score_class_id"]).sum()
        ),
        "stem_overrides": int(
            replay["r0_score_class_id"].ne(replay["r0_score_stem_class_id"]).sum()
        ),
        "variant_metrics": variant_metrics,
        "paired_vs_baseline": paired,
        "artifacts": {
            "fixed_predictions": str(fixed_path),
            "fixed_predictions_sha256": sha256(fixed_path),
            "replay_predictions": str(output / "replay_predictions.csv"),
            "variant_metrics": str(output / "variant_metrics.csv"),
            "class_metrics": str(output / "class_metrics.csv"),
            "variant_instance_outcomes": str(output / "variant_instance_outcomes.csv"),
        },
        "interpretation_limits": {
            "partial_rule_replay_not_three_head_model": True,
            "unqualified_decision_is_baseline": True,
            "automatic_stem_not_human_confirmed": True,
            "single_val_split": True,
            "test_accessed": False,
            "label_changes_made": False,
            "oracle_99_92_not_model_performance": True,
        },
    }
    summary_path = output / "r0_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "R0_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
