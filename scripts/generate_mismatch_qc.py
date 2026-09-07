from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate spatial QC for images with instance-count mismatch.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--features", required=True)
    parser.add_argument("--count-audit", required=True)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--gt-duplicate-iou-threshold", type=float, default=0.8)
    return parser.parse_args()


def read_gt_masks(label_path: Path, height: int, width: int) -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in dict.fromkeys(lines):
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coordinates = np.asarray([float(value) for value in parts[1:]], dtype=np.float32).reshape(-1, 2)
        coordinates[:, 0] *= width
        coordinates[:, 1] *= height
        polygon = np.rint(coordinates).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        masks.append(mask)
    return masks


def read_prediction_mask(row: pd.Series, output_dir: Path, height: int, width: int) -> np.ndarray:
    roi_mask = cv2.imread(str(output_dir / str(row["raw_mask_path"])), cv2.IMREAD_GRAYSCALE)
    if roi_mask is None:
        raise IOError(f"Missing predicted mask: {row['raw_mask_path']}")
    x1, y1, x2, y2 = (int(row[name]) for name in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"))
    roi_mask = cv2.resize(roi_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((height, width), dtype=np.uint8)
    full[y1:y2, x1:x2] = roi_mask
    return full


def iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.count_nonzero((left > 0) & (right > 0)))
    union = int(np.count_nonzero((left > 0) | (right > 0)))
    return intersection / union if union else 0.0


def greedy_match(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], threshold: float) -> list[tuple[int, int, float]]:
    candidates = sorted(
        ((g, p, iou(gt, pred)) for g, gt in enumerate(gt_masks) for p, pred in enumerate(pred_masks)),
        key=lambda value: value[2],
        reverse=True,
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for gt_index, pred_index, overlap in candidates:
        if overlap < threshold:
            break
        if gt_index not in used_gt and pred_index not in used_pred:
            matches.append((gt_index, pred_index, overlap))
            used_gt.add(gt_index)
            used_pred.add(pred_index)
    return matches


def contours(mask: np.ndarray) -> list[np.ndarray]:
    result, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return result


def draw_masks(image: np.ndarray, masks: list[np.ndarray], color: tuple[int, int, int], prefix: str) -> np.ndarray:
    output = image.copy()
    for index, mask in enumerate(masks, start=1):
        mask_contours = contours(mask)
        cv2.drawContours(output, mask_contours, -1, color, 3)
        if mask_contours:
            x, y, _, _ = cv2.boundingRect(max(mask_contours, key=cv2.contourArea))
            cv2.putText(output, f"{prefix}{index}", (x, max(20, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return output


def titled_panel(image: np.ndarray, title: str, width: int = 640, height: int = 420) -> np.ndarray:
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    available = height - 42
    scale = min(width / image.shape[1], available / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    x = (width - resized.shape[1]) // 2
    y = 40 + (available - resized.shape[0]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (width - 1, 39), (8, 8, 8), -1)
    cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    features = pd.read_csv(output_dir / args.features)
    count_audit = pd.read_csv(output_dir / args.count_audit)
    mismatches = count_audit.loc[count_audit["count_match_status"] != "exact_count"].copy()
    qc_dir = output_dir / "count_mismatch_qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for _, audit_row in mismatches.iterrows():
        image_id = str(audit_row["image_id"])
        feature_rows = features.loc[features["image_id"] == image_id].sort_values("instance_id")
        image_path = Path(str(audit_row["image_path"]))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(image_path)
        height, width = image.shape[:2]
        gt_masks = read_gt_masks(args.labels_dir.resolve() / f"{image_id}.txt", height, width)
        pred_masks = [read_prediction_mask(row, output_dir, height, width) for _, row in feature_rows.iterrows()]
        matches = greedy_match(gt_masks, pred_masks, float(args.iou_threshold))
        matched_gt = {item[0] for item in matches}
        matched_pred = {item[1] for item in matches}
        unmatched_gt_indices = [index for index in range(len(gt_masks)) if index not in matched_gt]
        duplicate_like_gt_indices = [
            index
            for index in unmatched_gt_indices
            if max(
                (iou(gt_masks[index], other) for other_index, other in enumerate(gt_masks) if other_index != index),
                default=0.0,
            )
            >= float(args.gt_duplicate_iou_threshold)
        ]
        genuine_miss_gt_indices = [index for index in unmatched_gt_indices if index not in duplicate_like_gt_indices]
        duplicate_like_gt = [gt_masks[index] for index in duplicate_like_gt_indices]
        genuine_miss_gt = [gt_masks[index] for index in genuine_miss_gt_indices]
        unmatched_pred_indices = [index for index in range(len(pred_masks)) if index not in matched_pred]
        unmatched_pred = [pred_masks[index] for index in unmatched_pred_indices]
        unmatched_prediction_ids = [str(feature_rows.iloc[index]["instance_id"]) for index in unmatched_pred_indices]

        gt_panel = draw_masks(image, gt_masks, (0, 255, 0), "G")
        pred_panel = draw_masks(image, pred_masks, (255, 0, 255), "P")
        combined = draw_masks(image, duplicate_like_gt, (0, 255, 255), "DUP")
        combined = draw_masks(combined, genuine_miss_gt, (0, 0, 255), "MISS")
        combined = draw_masks(combined, unmatched_pred, (0, 165, 255), "EXTRA")
        panel = np.hstack(
            [
                titled_panel(gt_panel, f"Ground truth: {len(gt_masks)} (green)"),
                titled_panel(pred_panel, f"Predictions: {len(pred_masks)} (magenta)"),
                titled_panel(
                    combined,
                    (
                        f"matched={len(matches)}, duplicate-like={len(duplicate_like_gt)}, "
                        f"missed={len(genuine_miss_gt)}, extra={len(unmatched_pred)}"
                    ),
                ),
            ]
        )
        qc_path = qc_dir / f"{image_id}_count_mismatch.jpg"
        if not cv2.imwrite(str(qc_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise IOError(qc_path)
        rows.append(
            {
                "image_id": image_id,
                "gt_instances": len(gt_masks),
                "predicted_instances": len(pred_masks),
                "matched_at_iou_threshold": len(matches),
                "unmatched_gt": len(unmatched_gt_indices),
                "duplicate_like_gt": len(duplicate_like_gt),
                "genuine_unmatched_gt": len(genuine_miss_gt),
                "unmatched_predictions": len(unmatched_pred),
                "unmatched_prediction_ids": ";".join(unmatched_prediction_ids),
                "mean_matched_iou": float(np.mean([item[2] for item in matches])) if matches else np.nan,
                "minimum_matched_iou": float(np.min([item[2] for item in matches])) if matches else np.nan,
                "qc_image": qc_path.relative_to(output_dir).as_posix(),
            }
        )

    result = pd.DataFrame(rows)
    audit_stem = Path(args.count_audit).stem
    split_prefix = audit_stem.removesuffix("_instance_count_audit")
    result.to_csv(output_dir / f"{split_prefix}_spatial_mismatch_audit.csv", index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
