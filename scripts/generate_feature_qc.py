from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml


PANEL_WIDTH = 384
PANEL_HEIGHT = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate human-review QC panels for extracted fruit instances.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    return config


def resolve_from_config(config: dict[str, Any], value: str) -> Path:
    return (Path(config["_config_path"]).parent / value).resolve()


def read_artifact(output_root: Path, relative_path: Any, flags: int) -> np.ndarray | None:
    if relative_path is None or (isinstance(relative_path, float) and np.isnan(relative_path)):
        return None
    path = output_root / str(relative_path)
    return cv2.imread(str(path), flags)


def fit_panel(image: np.ndarray | None, title: str) -> np.ndarray:
    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 28, dtype=np.uint8)
    if image is not None and image.size:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            alpha = image[:, :, 3:4].astype(np.float32) / 255.0
            image = (image[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)
        available_height = PANEL_HEIGHT - 42
        scale = min(PANEL_WIDTH / image.shape[1], available_height / image.shape[0])
        new_size = (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale)))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
        x = (PANEL_WIDTH - resized.shape[1]) // 2
        y = 40 + (available_height - resized.shape[0]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (0, 0), (PANEL_WIDTH - 1, 39), (10, 10, 10), -1)
    cv2.putText(canvas, title, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def place_roi(full_shape: tuple[int, int], roi: tuple[int, int, int, int], roi_image: np.ndarray | None) -> np.ndarray:
    height, width = full_shape
    full = np.zeros((height, width), dtype=np.uint8)
    if roi_image is None:
        return full
    x1, y1, x2, y2 = roi
    target = cv2.resize(roi_image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    full[y1:y2, x1:x2] = target
    return full


def color_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    colored = np.zeros_like(image)
    colored[:, :] = color
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image, 1.0 - alpha, colored, alpha, 0)[selected]
    return overlay


def feature_summary_panel(row: pd.Series) -> np.ndarray:
    canvas = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 28, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (PANEL_WIDTH - 1, 39), (10, 10, 10), -1)
    cv2.putText(canvas, "18 extracted features", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    names = [
        "major_axis", "minor_axis", "area", "aspect_ratio", "L_mean", "L_std", "a_mean", "a_std", "b_mean",
        "b_std", "S_mean", "V_mean", "circularity", "solidity", "eccentricity", "dark_ratio",
        "abnormal_color_ratio", "max_damage_ratio",
    ]
    for index, name in enumerate(names):
        column = 0 if index < 9 else 1
        row_index = index if index < 9 else index - 9
        x = 7 + column * (PANEL_WIDTH // 2)
        y = 60 + row_index * 25
        value = row.get(name)
        text = f"{name}: {float(value):.4f}" if pd.notna(value) else f"{name}: NA"
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


def qc_reasons(features: pd.DataFrame, config: dict[str, Any]) -> dict[str, set[str]]:
    reasons: dict[str, set[str]] = {}

    def add(ids: list[str], reason: str) -> None:
        for instance_id in ids:
            reasons.setdefault(instance_id, set()).add(reason)

    qc = config["qc"]
    sample_size = min(int(qc["random_sample_size"]), len(features))
    random_ids = features.sample(n=sample_size, random_state=int(qc["random_seed"]))["instance_id"].tolist()
    add(random_ids, "random")
    failed = features.loc[~features["processing_success"].fillna(False), "instance_id"].tolist()
    add(failed, "processing_failure")
    edge = features.loc[features["touches_image_edge"].fillna(False), "instance_id"].tolist()
    add(edge, "touches_image_edge")
    low_conf = features.nsmallest(min(int(qc["extra_low_confidence"]), len(features)), "confidence")["instance_id"].tolist()
    add(low_conf, "low_confidence")
    high_damage = features.nlargest(min(int(qc["extra_high_damage"]), len(features)), "max_damage_ratio")["instance_id"].tolist()
    add(high_damage, "high_damage_candidate")
    high_highlight = features.nlargest(min(int(qc["extra_high_highlight"]), len(features)), "highlight_ratio")["instance_id"].tolist()
    add(high_highlight, "high_highlight")
    if "edge_abnormal_ratio" in features.columns and int(qc.get("extra_high_edge_abnormal", 0)) > 0:
        high_edge = features.nlargest(
            min(int(qc["extra_high_edge_abnormal"]), len(features)), "edge_abnormal_ratio"
        )["instance_id"].tolist()
        add(high_edge, "high_edge_abnormal")
    if "calyx_candidate_ratio" in features.columns and int(qc.get("extra_high_calyx", 0)) > 0:
        high_calyx = features.nlargest(
            min(int(qc["extra_high_calyx"]), len(features)), "calyx_candidate_ratio"
        )["instance_id"].tolist()
        add(high_calyx, "high_calyx_candidate")
    return reasons


def draw_qc(row: pd.Series, output_root: Path) -> np.ndarray:
    image = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Could not read image: {row['image_path']}")
    height, width = image.shape[:2]
    roi = tuple(int(row[name]) for name in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"))
    x1, y1, x2, y2 = roi
    mask_roi = read_artifact(output_root, row.get("mask_path"), cv2.IMREAD_GRAYSCALE)
    raw_mask_roi = read_artifact(output_root, row.get("raw_mask_path"), cv2.IMREAD_GRAYSCALE)
    border_removed_roi = read_artifact(output_root, row.get("border_removed_mask_path"), cv2.IMREAD_GRAYSCALE)
    color_mask_roi = read_artifact(output_root, row.get("color_mask_path"), cv2.IMREAD_GRAYSCALE)
    highlight_roi = read_artifact(output_root, row.get("highlight_mask_path"), cv2.IMREAD_GRAYSCALE)
    calyx_roi = read_artifact(output_root, row.get("calyx_mask_path"), cv2.IMREAD_GRAYSCALE)
    edge_abnormal_roi = read_artifact(output_root, row.get("edge_abnormal_mask_path"), cv2.IMREAD_GRAYSCALE)
    damage_eligible_roi = read_artifact(output_root, row.get("damage_eligible_mask_path"), cv2.IMREAD_GRAYSCALE)
    damage_roi = read_artifact(output_root, row.get("damage_mask_path"), cv2.IMREAD_GRAYSCALE)
    crop = read_artifact(output_root, row.get("crop_path"), cv2.IMREAD_UNCHANGED)
    mask = place_roi((height, width), roi, mask_roi)
    raw_mask = place_roi((height, width), roi, raw_mask_roi)
    border_removed = place_roi((height, width), roi, border_removed_roi)
    color_mask = place_roi((height, width), roi, color_mask_roi)
    highlight = place_roi((height, width), roi, highlight_roi)
    calyx = place_roi((height, width), roi, calyx_roi)
    edge_abnormal = place_roi((height, width), roi, edge_abnormal_roi)
    damage_eligible = place_roi((height, width), roi, damage_eligible_roi)
    damage = place_roi((height, width), roi, damage_roi)

    original_panel = image.copy()
    cv2.rectangle(original_panel, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 3)
    cv2.putText(
        original_panel,
        f"{row['instance_id']} conf={float(row['confidence']):.3f}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    contour_overlay = image.copy()
    contour_path = output_root / str(row["contour_path"])
    if contour_path.is_file():
        payload = json.loads(contour_path.read_text(encoding="utf-8"))
        points = np.asarray(payload.get("contour_points", []), dtype=np.int32)
        if len(points) >= 3:
            cv2.polylines(contour_overlay, [points], True, (0, 255, 0), 3)
    if pd.notna(row.get("major_x1")):
        cv2.line(
            contour_overlay,
            (int(round(row["major_x1"])), int(round(row["major_y1"]))),
            (int(round(row["major_x2"])), int(round(row["major_y2"]))),
            (0, 0, 255),
            4,
        )
        cv2.line(
            contour_overlay,
            (int(round(row["minor_x1"])), int(round(row["minor_y1"]))),
            (int(round(row["minor_x2"])), int(round(row["minor_y2"]))),
            (255, 255, 0),
            4,
        )

    refinement_overlay = color_overlay(image, border_removed, (0, 0, 255), 0.70)
    raw_contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    geometry_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(refinement_overlay, raw_contours, -1, (0, 255, 255), 2)
    cv2.drawContours(refinement_overlay, geometry_contours, -1, (0, 255, 0), 2)
    color_region_overlay = color_overlay(image, color_mask, (0, 255, 0), 0.45)
    highlight_overlay = color_overlay(image, highlight, (0, 255, 255))
    calyx_overlay = color_overlay(image, calyx, (255, 255, 0), 0.60)
    edge_overlay = color_overlay(image, edge_abnormal, (0, 165, 255), 0.60)
    damage_overlay = image.copy()
    ineligible_boundary = ((mask > 0) & (damage_eligible == 0)).astype(np.uint8) * 255
    damage_overlay = color_overlay(damage_overlay, ineligible_boundary, (96, 96, 96), 0.35)
    dark = ((damage == 85) | (damage == 255)).astype(np.uint8) * 255
    abnormal = ((damage == 170) | (damage == 255)).astype(np.uint8) * 255
    damage_overlay = color_overlay(damage_overlay, dark, (255, 0, 0), 0.55)
    damage_overlay = color_overlay(damage_overlay, abnormal, (255, 0, 255), 0.55)

    panels = [
        fit_panel(original_panel, "Original + ROI"),
        fit_panel(refinement_overlay, f"Mask refine: raw(yellow), kept(green), removed(red) {float(row['border_removed_ratio']):.3f}"),
        fit_panel(contour_overlay, "Contour + major(red) / minor(cyan)"),
        fit_panel(crop, "Background-excluded crop"),
        feature_summary_panel(row),
        fit_panel(color_region_overlay, "Actual color-statistics region (green)"),
        fit_panel(highlight_overlay, f"Highlight candidate ratio={float(row['highlight_ratio']):.4f}"),
        fit_panel(calyx_overlay, f"Calyx-like tip candidate (cyan) ratio={float(row.get('calyx_candidate_ratio', 0)):.4f}"),
        fit_panel(edge_overlay, f"Excluded edge abnormal (orange) ratio={float(row.get('edge_abnormal_ratio', 0)):.4f}"),
        fit_panel(damage_overlay, f"Interior: dark blue / abnormal magenta ratio={float(row['max_damage_ratio']):.4f}"),
    ]
    return np.vstack([np.hstack(panels[:5]), np.hstack(panels[5:])])


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_root = resolve_from_config(config, config["output"]["root"])
    feature_filename = str(config["output"].get("features_filename", "features_pilot.csv"))
    feature_path = output_root / feature_filename
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    features = pd.read_csv(feature_path)
    reasons = qc_reasons(features, config)
    qc_root = output_root / "qc_visualization"
    qc_root.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for instance_id in sorted(reasons):
        row = features.loc[features["instance_id"] == instance_id].iloc[0]
        panel = draw_qc(row, output_root)
        output_path = qc_root / f"{instance_id}_qc.jpg"
        if not cv2.imwrite(str(output_path), panel, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise IOError(output_path)
        index_rows.append(
            {
                "instance_id": instance_id,
                "image_id": row["image_id"],
                "selection_reason": ";".join(sorted(reasons[instance_id])),
                "qc_image": output_path.relative_to(output_root).as_posix(),
                "mask_ok": "",
                "geometry_mask_ok": "",
                "axes_ok": "",
                "color_region_ok": "",
                "damage_candidate_ok": "",
                "rot_candidate": "",
                "calyx_false_positive": "",
                "highlight_false_positive": "",
                "contact_false_positive": "",
                "other_issue": "",
                "reviewer_notes": "",
            }
        )
    index = pd.DataFrame(index_rows)
    index.to_csv(output_root / "qc_review.csv", index=False, encoding="utf-8-sig")
    print(f"QC panels written: {len(index)}")
    print(f"QC index: {output_root / 'qc_review.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
