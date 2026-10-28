from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


FEATURE_COLUMNS = [
    "major_axis",
    "minor_axis",
    "area",
    "aspect_ratio",
    "L_mean",
    "L_std",
    "a_mean",
    "a_std",
    "b_mean",
    "b_std",
    "S_mean",
    "V_mean",
    "circularity",
    "solidity",
    "eccentricity",
    "dark_ratio",
    "abnormal_color_ratio",
    "max_damage_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract interpretable features from YOLO instance masks.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    config["_config_path"] = str(path)
    return config


def resolve_from_config(config: dict[str, Any], value: str) -> Path:
    return (Path(config["_config_path"]).parent / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_kernel(value: int, name: str) -> int:
    value = int(value)
    if value < 0 or (value > 1 and value % 2 == 0):
        raise ValueError(f"{name} must be 0, 1, or a positive odd integer.")
    return value


def select_manifest_rows(manifest: pd.DataFrame, split: str, seed: int) -> list[dict[str, Any]]:
    subset = manifest.loc[manifest["split"] == split].copy()
    if subset.empty:
        raise ValueError(f"No manifest rows found for split={split!r}.")
    groups: list[list[dict[str, Any]]] = []
    for index, (_, group) in enumerate(subset.groupby("source_group", sort=True)):
        shuffled = group.sample(frac=1.0, random_state=seed + index).to_dict("records")
        groups.append(shuffled)
    ordered: list[dict[str, Any]] = []
    rank = 0
    while True:
        added = False
        for group in groups:
            if rank < len(group):
                ordered.append(group[rank])
                added = True
        if not added:
            break
        rank += 1
    return ordered


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.count_nonzero((left > 0) & (right > 0))
    if intersection == 0:
        return 0.0
    union = np.count_nonzero((left > 0) | (right > 0))
    return float(intersection / union) if union else 0.0


def polygon_to_mask(points: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if points is None or len(points) < 3:
        return mask
    polygon = np.rint(points).astype(np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def deduplicate_candidates(candidates: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    removed: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        duplicate_of = None
        duplicate_iou = 0.0
        for existing in kept:
            overlap = mask_iou(candidate["raw_mask"], existing["raw_mask"])
            if overlap >= threshold:
                duplicate_of = existing
                duplicate_iou = overlap
                break
        if duplicate_of is None:
            kept.append(candidate)
        else:
            removed.append(
                {
                    "model_index": candidate["model_index"],
                    "confidence": candidate["confidence"],
                    "duplicate_of_model_index": duplicate_of["model_index"],
                    "mask_iou": duplicate_iou,
                }
            )
    return kept, removed


def clean_analysis_mask(mask: np.ndarray, min_component_area: int, keep_largest: bool) -> tuple[np.ndarray, int, int]:
    binary = (mask > 0).astype(np.uint8)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    for label in range(1, component_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_area:
            components.append((label, area))
    output = np.zeros_like(mask)
    if components:
        selected = [max(components, key=lambda item: item[1])] if keep_largest else components
        for label, _ in selected:
            output[labels == label] = 255
    kept_area = int(np.count_nonzero(output))
    removed_area = int(np.count_nonzero(mask) - kept_area)
    return output, len(components), removed_area


def refine_geometry_mask(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int, float, int, float, str]:
    """Remove pale, low-chroma pixels only inside a narrow predicted-mask boundary band."""
    binary = (mask > 0).astype(np.uint8)
    empty_removed = np.zeros_like(mask)
    original_area = int(np.count_nonzero(binary))
    if not bool(config.get("enabled", False)) or original_area == 0:
        return mask.copy(), empty_removed, 0, 0.0, 0, 0.0, "disabled" if original_area else "empty_mask"

    max_depth = float(config["max_boundary_depth_px"])
    if max_depth <= 0:
        raise ValueError("max_boundary_depth_px must be positive.")
    _, a_star, b_star, saturation, value = color_spaces(image_bgr)
    chroma = np.sqrt(a_star * a_star + b_star * b_star)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    boundary_band = (binary > 0) & (distance <= max_depth)
    pale_background = (
        (saturation <= float(config["saturation_max"]))
        & (value >= float(config["value_min"]))
        & (chroma <= float(config["lab_chroma_max"]))
    )
    removed = boundary_band & pale_background
    refined = binary.copy()
    refined[removed] = 0

    refined_mask, _, _ = clean_analysis_mask(
        refined * 255,
        int(config["min_component_area_px"]),
        True,
    )
    refined_area = int(np.count_nonzero(refined_mask))
    candidate_area = original_area - refined_area
    candidate_ratio = candidate_area / original_area if original_area else 0.0
    if refined_area < int(config["min_remaining_area_px"]):
        return mask.copy(), empty_removed, 0, 0.0, candidate_area, candidate_ratio, "fallback_min_remaining_area"
    if candidate_ratio > float(config["max_removed_fraction"]):
        return mask.copy(), empty_removed, 0, 0.0, candidate_area, candidate_ratio, "fallback_max_removed_fraction"

    removed_mask = ((mask > 0) & (refined_mask == 0)).astype(np.uint8) * 255
    return refined_mask, removed_mask, candidate_area, candidate_ratio, candidate_area, candidate_ratio, "applied"


def main_contour(mask: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def axis_endpoints(center: tuple[float, float], length: float, angle_deg: float) -> tuple[float, float, float, float]:
    radians = math.radians(angle_deg)
    dx = math.cos(radians) * length / 2.0
    dy = math.sin(radians) * length / 2.0
    return center[0] - dx, center[1] - dy, center[0] + dx, center[1] + dy


def fit_axes(contour: np.ndarray) -> dict[str, Any]:
    if contour is None or len(contour) < 3:
        raise ValueError("Contour has fewer than 3 points.")
    method = "min_area_rect"
    if len(contour) >= 5:
        try:
            center, axes, angle = cv2.fitEllipse(contour)
            width, height = float(axes[0]), float(axes[1])
            if width > 0 and height > 0 and np.isfinite([*center, width, height, angle]).all():
                method = "fit_ellipse"
            else:
                raise ValueError("Non-finite ellipse parameters.")
        except (cv2.error, ValueError):
            center, axes, angle = cv2.minAreaRect(contour)
            width, height = float(axes[0]), float(axes[1])
    else:
        center, axes, angle = cv2.minAreaRect(contour)
        width, height = float(axes[0]), float(axes[1])
    if width <= 0 or height <= 0:
        raise ValueError("Axis length is zero.")
    if width >= height:
        major_axis, minor_axis, major_angle = width, height, float(angle)
    else:
        major_axis, minor_axis, major_angle = height, width, float(angle) + 90.0
    major_angle %= 180.0
    minor_angle = (major_angle + 90.0) % 180.0
    return {
        "method": method,
        "center_x": float(center[0]),
        "center_y": float(center[1]),
        "major_axis": major_axis,
        "minor_axis": minor_axis,
        "major_angle_deg": major_angle,
        "minor_angle_deg": minor_angle,
        "major_endpoints": axis_endpoints((float(center[0]), float(center[1])), major_axis, major_angle),
        "minor_endpoints": axis_endpoints((float(center[0]), float(center[1])), minor_axis, minor_angle),
    }


def color_spaces(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    lightness = lab[:, :, 0] * (100.0 / 255.0)
    a_star = lab[:, :, 1] - 128.0
    b_star = lab[:, :, 2] - 128.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    return lightness, a_star, b_star, saturation, value


def clean_candidate_region(region: np.ndarray, mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    output = ((region > 0) & (mask > 0)).astype(np.uint8) * 255
    open_size = validate_kernel(config["morphology_open_kernel"], "morphology_open_kernel")
    close_size = validate_kernel(config["morphology_close_kernel"], "morphology_close_kernel")
    if open_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        output = cv2.morphologyEx(output, cv2.MORPH_OPEN, kernel)
    if close_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    output[(mask == 0)] = 0
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats((output > 0).astype(np.uint8), 8)
    filtered = np.zeros_like(output)
    minimum = int(config["min_component_area_px"])
    for label in range(1, component_count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum:
            filtered[labels == label] = 255
    return filtered


def largest_component_area(mask: np.ndarray) -> int:
    component_count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if component_count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def safe_write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Could not write image: {path}")


def relative_artifact(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def empty_feature_row() -> dict[str, float]:
    return {name: float("nan") for name in FEATURE_COLUMNS}


def process_instance(
    image_bgr: np.ndarray,
    raw_mask: np.ndarray,
    confidence: float,
    image_record: dict[str, Any],
    instance_id: str,
    config: dict[str, Any],
    output_root: Path,
    model_hash: str,
) -> dict[str, Any]:
    height, width = image_bgr.shape[:2]
    features = empty_feature_row()
    errors: list[str] = []
    warnings: list[str] = []
    shape_success = color_success = damage_success = False
    axes: dict[str, Any] | None = None
    contour: np.ndarray | None = None

    lightness, a_star, b_star, saturation, value = color_spaces(image_bgr)

    analysis_cfg = config["analysis_mask"]
    component_mask, component_count, removed_component_area = clean_analysis_mask(
        raw_mask,
        int(analysis_cfg["min_component_area_px"]),
        bool(analysis_cfg["keep_largest_component"]),
    )
    geometry_cfg = config["geometry_refinement"]
    (
        analysis_mask,
        border_removed_mask,
        border_removed_area,
        border_removed_ratio,
        border_candidate_area,
        border_candidate_ratio,
        geometry_status,
    ) = refine_geometry_mask(image_bgr, component_mask, geometry_cfg)
    area = int(np.count_nonzero(analysis_mask))
    if area < int(analysis_cfg["min_instance_area_px"]):
        errors.append(f"mask_too_small:{area}")

    ys, xs = np.nonzero(analysis_mask)
    if len(xs):
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
    else:
        raw_ys, raw_xs = np.nonzero(raw_mask)
        if len(raw_xs):
            x1, x2 = int(raw_xs.min()), int(raw_xs.max()) + 1
            y1, y2 = int(raw_ys.min()), int(raw_ys.max()) + 1
        else:
            x1 = y1 = 0
            x2 = y2 = 1
        errors.append("empty_analysis_mask")

    padding = int(config["output"]["roi_padding_px"])
    roi_x1, roi_y1 = max(0, x1 - padding), max(0, y1 - padding)
    roi_x2, roi_y2 = min(width, x2 + padding), min(height, y2 + padding)
    touches_edge = bool(
        np.any(analysis_mask[0, :])
        or np.any(analysis_mask[-1, :])
        or np.any(analysis_mask[:, 0])
        or np.any(analysis_mask[:, -1])
    )

    artifact_paths = {
        "raw_mask_path": output_root / "instances" / "raw_masks" / f"{instance_id}.png",
        "mask_path": output_root / "instances" / "masks" / f"{instance_id}.png",
        "border_removed_mask_path": output_root / "instances" / "border_removed" / f"{instance_id}.png",
        "crop_path": output_root / "instances" / "crops" / f"{instance_id}.png",
        "contour_path": output_root / "instances" / "contours" / f"{instance_id}.json",
        "color_mask_path": output_root / "instances" / "color_masks" / f"{instance_id}.png",
        "highlight_mask_path": output_root / "instances" / "highlights" / f"{instance_id}.png",
        "calyx_mask_path": output_root / "instances" / "calyx_candidates" / f"{instance_id}.png",
        "edge_abnormal_mask_path": output_root / "instances" / "edge_abnormal" / f"{instance_id}.png",
        "damage_eligible_mask_path": output_root / "instances" / "damage_eligible" / f"{instance_id}.png",
        "damage_mask_path": output_root / "instances" / "damage_candidates" / f"{instance_id}.png",
    }
    safe_write(artifact_paths["raw_mask_path"], raw_mask[roi_y1:roi_y2, roi_x1:roi_x2])
    safe_write(artifact_paths["mask_path"], analysis_mask[roi_y1:roi_y2, roi_x1:roi_x2])
    safe_write(
        artifact_paths["border_removed_mask_path"],
        border_removed_mask[roi_y1:roi_y2, roi_x1:roi_x2],
    )
    crop_bgra = cv2.cvtColor(image_bgr[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2BGRA)
    crop_bgra[:, :, 3] = analysis_mask[roi_y1:roi_y2, roi_x1:roi_x2]
    safe_write(artifact_paths["crop_path"], crop_bgra)

    try:
        contour = main_contour(analysis_mask)
        if contour is None:
            raise ValueError("No valid contour.")
        perimeter = float(cv2.arcLength(contour, True))
        contour_area = float(cv2.contourArea(contour))
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        axes = fit_axes(contour)
        major_axis = float(axes["major_axis"])
        minor_axis = float(axes["minor_axis"])
        features.update(
            {
                "major_axis": major_axis,
                "minor_axis": minor_axis,
                "area": float(area),
                "aspect_ratio": major_axis / minor_axis if minor_axis > 0 else float("nan"),
                "circularity": (4.0 * math.pi * contour_area / (perimeter * perimeter))
                if perimeter > 0
                else float("nan"),
                "solidity": contour_area / hull_area if hull_area > 0 else float("nan"),
                "eccentricity": math.sqrt(max(0.0, 1.0 - (minor_axis / major_axis) ** 2)) if major_axis > 0 else float("nan"),
            }
        )
        shape_success = True
    except Exception as exc:  # instance-level isolation is intentional
        errors.append(f"shape:{type(exc).__name__}:{exc}")

    contour_payload = {
        "image_id": image_record["new_stem"],
        "instance_id": instance_id,
        "coordinate_system": "original_image_pixels",
        "roi_xyxy": [roi_x1, roi_y1, roi_x2, roi_y2],
        "contour_points": contour.reshape(-1, 2).astype(int).tolist() if contour is not None else [],
        "axes": axes,
    }
    artifact_paths["contour_path"].parent.mkdir(parents=True, exist_ok=True)
    artifact_paths["contour_path"].write_text(
        json.dumps(contour_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    highlight_cfg = config["highlight_filter"]
    erosion_px = int(highlight_cfg["boundary_erosion_px"])
    if erosion_px < 0:
        raise ValueError("boundary_erosion_px must be non-negative.")
    if erosion_px > 0:
        kernel_size = erosion_px * 2 + 1
        color_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        color_mask = cv2.erode(analysis_mask, color_kernel, iterations=1)
    else:
        color_mask = analysis_mask.copy()
    fruit = color_mask > 0
    color_mask_pixels = int(np.count_nonzero(fruit))
    static_highlight = np.zeros_like(fruit)
    relative_highlight = np.zeros_like(fruit)
    if bool(highlight_cfg["enabled"]):
        static_highlight = fruit & (saturation <= float(highlight_cfg["saturation_max"])) & (
            value >= float(highlight_cfg["value_min"])
        )
        if bool(highlight_cfg.get("relative_enabled", False)) and np.count_nonzero(fruit) >= 20:
            median_L_for_highlight = float(np.median(lightness[fruit]))
            mad_L_for_highlight = float(np.median(np.abs(lightness[fruit] - median_L_for_highlight)))
            scale_L_for_highlight = 1.4826 * max(
                mad_L_for_highlight,
                float(highlight_cfg["relative_min_L_mad"]),
            )
            percentile_limit = float(
                np.percentile(lightness[fruit], float(highlight_cfg["relative_L_percentile"]))
            )
            relative_L_limit = max(
                percentile_limit,
                median_L_for_highlight
                + float(highlight_cfg["relative_L_robust_z_min"]) * scale_L_for_highlight,
            )
            median_saturation = float(np.median(saturation[fruit]))
            relative_highlight = (
                fruit
                & (lightness >= relative_L_limit)
                & (saturation <= median_saturation - float(highlight_cfg["relative_saturation_drop_min"]))
            )
    highlight_raw = static_highlight | relative_highlight
    if all(
        name in highlight_cfg
        for name in ("morphology_open_kernel", "morphology_close_kernel", "min_component_area_px")
    ):
        highlight = clean_candidate_region(
            highlight_raw.astype(np.uint8) * 255,
            color_mask,
            highlight_cfg,
        ) > 0
    else:
        highlight = highlight_raw

    calyx_cfg = config.get("calyx_detection", {"enabled": False})
    calyx_mask = np.zeros_like(analysis_mask)
    if bool(calyx_cfg.get("enabled", False)) and axes is not None and np.count_nonzero(fruit) >= 20:
        median_a_for_calyx = float(np.median(a_star[fruit]))
        mad_a_for_calyx = float(np.median(np.abs(a_star[fruit] - median_a_for_calyx)))
        scale_a_for_calyx = 1.4826 * max(mad_a_for_calyx, float(calyx_cfg["min_a_mad"]))
        relative_a_limit = median_a_for_calyx - float(calyx_cfg["relative_a_z_min"]) * scale_a_for_calyx
        calyx_a_limit = min(float(calyx_cfg["absolute_a_max"]), relative_a_limit)
        geometry_distance = cv2.distanceTransform((analysis_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        yy, xx = np.indices(analysis_mask.shape, dtype=np.float32)
        angle = math.radians(float(axes["major_angle_deg"]))
        major_projection = np.abs(
            (xx - float(axes["center_x"])) * math.cos(angle)
            + (yy - float(axes["center_y"])) * math.sin(angle)
        )
        tip_zone = major_projection >= (
            float(calyx_cfg["tip_fraction"]) * float(axes["major_axis"]) / 2.0
        )
        calyx_raw = (
            fruit
            & (geometry_distance <= float(calyx_cfg["max_boundary_depth_px"]))
            & tip_zone
            & (a_star <= calyx_a_limit)
            & (value <= float(calyx_cfg["value_max"]))
        )
        calyx_mask = clean_candidate_region(
            calyx_raw.astype(np.uint8) * 255,
            analysis_mask,
            calyx_cfg,
        )
    calyx = calyx_mask > 0
    calyx_pixels = int(np.count_nonzero(calyx))
    calyx_candidate_ratio = calyx_pixels / area if area else float("nan")

    valid_color = fruit & ~highlight
    if bool(calyx_cfg.get("exclude_from_color", False)):
        valid_color &= ~calyx
    highlight_pixels = int(np.count_nonzero(highlight))
    static_highlight_pixels = int(np.count_nonzero(static_highlight))
    relative_highlight_pixels = int(np.count_nonzero(relative_highlight))
    valid_color_pixels = int(np.count_nonzero(valid_color))
    highlight_ratio = highlight_pixels / color_mask_pixels if color_mask_pixels else float("nan")
    highlight_static_ratio = static_highlight_pixels / color_mask_pixels if color_mask_pixels else float("nan")
    highlight_relative_ratio = relative_highlight_pixels / color_mask_pixels if color_mask_pixels else float("nan")
    safe_write(
        artifact_paths["color_mask_path"],
        (valid_color[roi_y1:roi_y2, roi_x1:roi_x2] * 255).astype(np.uint8),
    )
    safe_write(artifact_paths["highlight_mask_path"], (highlight[roi_y1:roi_y2, roi_x1:roi_x2] * 255).astype(np.uint8))
    safe_write(artifact_paths["calyx_mask_path"], calyx_mask[roi_y1:roi_y2, roi_x1:roi_x2])

    try:
        if valid_color_pixels < int(highlight_cfg["min_valid_color_pixels"]):
            raise ValueError(f"valid_color_pixels={valid_color_pixels}")
        features.update(
            {
                "L_mean": float(np.mean(lightness[valid_color])),
                "L_std": float(np.std(lightness[valid_color], ddof=0)),
                "a_mean": float(np.mean(a_star[valid_color])),
                "a_std": float(np.std(a_star[valid_color], ddof=0)),
                "b_mean": float(np.mean(b_star[valid_color])),
                "b_std": float(np.std(b_star[valid_color], ddof=0)),
                "S_mean": float(np.mean(saturation[valid_color])),
                "V_mean": float(np.mean(value[valid_color])),
            }
        )
        color_success = True
    except Exception as exc:
        errors.append(f"color:{type(exc).__name__}:{exc}")

    dark_mask = np.zeros_like(analysis_mask)
    abnormal_mask = np.zeros_like(analysis_mask)
    edge_abnormal_mask = np.zeros_like(analysis_mask)
    damage_cfg = config["damage_detection"]
    damage_boundary_px = int(damage_cfg["boundary_exclusion_px"])
    if damage_boundary_px < 0:
        raise ValueError("boundary_exclusion_px must be non-negative.")
    if damage_boundary_px > 0:
        damage_kernel_size = damage_boundary_px * 2 + 1
        damage_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (damage_kernel_size, damage_kernel_size))
        damage_interior = cv2.erode(analysis_mask, damage_kernel, iterations=1) > 0
    else:
        damage_interior = analysis_mask > 0
    edge_eligible = valid_color & ~damage_interior
    damage_eligible = valid_color & damage_interior
    if bool(calyx_cfg.get("exclude_from_damage", False)):
        edge_eligible &= ~calyx
        damage_eligible &= ~calyx
    damage_eligible_pixels = int(np.count_nonzero(damage_eligible))
    safe_write(
        artifact_paths["damage_eligible_mask_path"],
        (damage_eligible[roi_y1:roi_y2, roi_x1:roi_x2] * 255).astype(np.uint8),
    )
    try:
        if damage_eligible_pixels < int(highlight_cfg["min_valid_color_pixels"]):
            raise ValueError("Not enough valid color pixels for damage candidates.")
        if damage_cfg["dark_method"] != "robust_L":
            raise ValueError(f"Unsupported dark_method={damage_cfg['dark_method']!r}")
        median_L = float(np.median(lightness[damage_eligible]))
        mad_L = float(np.median(np.abs(lightness[damage_eligible] - median_L)))
        scale_L = 1.4826 * max(mad_L, float(damage_cfg["min_L_mad"]))
        relative_dark_limit = median_L - float(damage_cfg["dark_robust_z_min"]) * scale_L
        dark_limit = min(float(damage_cfg["dark_L_absolute_max"]), relative_dark_limit)
        dark_raw = damage_eligible & (lightness <= dark_limit)
        if damage_cfg["abnormal_method"] != "robust_ab":
            raise ValueError(f"Unsupported abnormal_method={damage_cfg['abnormal_method']!r}")
        median_a = float(np.median(a_star[damage_eligible]))
        median_b = float(np.median(b_star[damage_eligible]))
        mad_a = float(np.median(np.abs(a_star[damage_eligible] - median_a)))
        mad_b = float(np.median(np.abs(b_star[damage_eligible] - median_b)))
        minimum_mad = float(damage_cfg["min_channel_mad"])
        scale_a = 1.4826 * max(mad_a, minimum_mad)
        scale_b = 1.4826 * max(mad_b, minimum_mad)
        robust_distance = np.sqrt(((a_star - median_a) / scale_a) ** 2 + ((b_star - median_b) / scale_b) ** 2)
        abnormal_raw = damage_eligible & (robust_distance >= float(damage_cfg["robust_ab_z_min"]))
        damage_eligible_u8 = damage_eligible.astype(np.uint8) * 255
        dark_mask = clean_candidate_region(dark_raw.astype(np.uint8) * 255, damage_eligible_u8, damage_cfg)
        abnormal_mask = clean_candidate_region(abnormal_raw.astype(np.uint8) * 255, damage_eligible_u8, damage_cfg)
        edge_raw = edge_eligible & (
            (lightness <= dark_limit)
            | (robust_distance >= float(damage_cfg["robust_ab_z_min"]))
        )
        edge_cfg = config.get("edge_abnormal_detection", damage_cfg)
        edge_abnormal_mask = clean_candidate_region(
            edge_raw.astype(np.uint8) * 255,
            edge_eligible.astype(np.uint8) * 255,
            edge_cfg,
        )
        union_mask = cv2.bitwise_or(dark_mask, abnormal_mask)
        denominator = damage_eligible_pixels
        features["dark_ratio"] = float(np.count_nonzero(dark_mask) / denominator)
        features["abnormal_color_ratio"] = float(np.count_nonzero(abnormal_mask) / denominator)
        features["max_damage_ratio"] = float(largest_component_area(union_mask) / area) if area else float("nan")
        damage_success = True
    except Exception as exc:
        errors.append(f"damage:{type(exc).__name__}:{exc}")

    edge_abnormal_pixels = int(np.count_nonzero(edge_abnormal_mask))
    edge_abnormal_ratio = edge_abnormal_pixels / area if area else float("nan")
    safe_write(
        artifact_paths["edge_abnormal_mask_path"],
        edge_abnormal_mask[roi_y1:roi_y2, roi_x1:roi_x2],
    )

    damage_encoded = np.zeros_like(analysis_mask)
    damage_encoded[dark_mask > 0] = 85
    damage_encoded[abnormal_mask > 0] = 170
    damage_encoded[(dark_mask > 0) & (abnormal_mask > 0)] = 255
    safe_write(artifact_paths["damage_mask_path"], damage_encoded[roi_y1:roi_y2, roi_x1:roi_x2])

    px_per_mm = config["calibration"]["px_per_mm"]
    major_mm = minor_mm = area_mm2 = float("nan")
    if px_per_mm is not None and float(px_per_mm) > 0 and shape_success:
        px_per_mm = float(px_per_mm)
        major_mm = features["major_axis"] / px_per_mm
        minor_mm = features["minor_axis"] / px_per_mm
        area_mm2 = features["area"] / (px_per_mm**2)

    if component_count > 1:
        warnings.append(f"multiple_components:{component_count}")
    if removed_component_area > 0:
        warnings.append(f"removed_component_area:{removed_component_area}")

    row: dict[str, Any] = {
        "image_id": image_record["new_stem"],
        "instance_id": instance_id,
        "split": image_record["split"],
        "original_number": int(image_record["original_number"]),
        "original_stem": image_record["original_stem"],
        "source_group": image_record["source_group"],
        "image_path": str(Path(image_record["resolved_image_path"]).resolve()),
        "mask_source": "prediction",
        "model_path": str(Path(config["_resolved_model_path"])),
        "model_sha256": model_hash,
        "confidence": float(confidence),
        "image_width": width,
        "image_height": height,
        "roi_x1": roi_x1,
        "roi_y1": roi_y1,
        "roi_x2": roi_x2,
        "roi_y2": roi_y2,
        "centroid_x": float(xs.mean()) if len(xs) else float("nan"),
        "centroid_y": float(ys.mean()) if len(ys) else float("nan"),
        "touches_image_edge": touches_edge,
        "component_count": component_count,
        "removed_component_area": removed_component_area,
        "geometry_refinement_status": geometry_status,
        "border_removed_area": border_removed_area,
        "border_removed_ratio": border_removed_ratio,
        "border_candidate_area": border_candidate_area,
        "border_candidate_ratio": border_candidate_ratio,
        "contour_area": contour_area if shape_success else float("nan"),
        "axis_method": axes["method"] if axes else "",
        "axis_center_x": axes["center_x"] if axes else float("nan"),
        "axis_center_y": axes["center_y"] if axes else float("nan"),
        "major_angle_deg": axes["major_angle_deg"] if axes else float("nan"),
        "minor_angle_deg": axes["minor_angle_deg"] if axes else float("nan"),
        "major_x1": axes["major_endpoints"][0] if axes else float("nan"),
        "major_y1": axes["major_endpoints"][1] if axes else float("nan"),
        "major_x2": axes["major_endpoints"][2] if axes else float("nan"),
        "major_y2": axes["major_endpoints"][3] if axes else float("nan"),
        "minor_x1": axes["minor_endpoints"][0] if axes else float("nan"),
        "minor_y1": axes["minor_endpoints"][1] if axes else float("nan"),
        "minor_x2": axes["minor_endpoints"][2] if axes else float("nan"),
        "minor_y2": axes["minor_endpoints"][3] if axes else float("nan"),
        "highlight_ratio": highlight_ratio,
        "highlight_static_ratio": highlight_static_ratio,
        "highlight_relative_ratio": highlight_relative_ratio,
        "calyx_candidate_ratio": calyx_candidate_ratio,
        "edge_abnormal_ratio": edge_abnormal_ratio,
        "color_mask_pixel_count": color_mask_pixels,
        "valid_color_pixel_count": valid_color_pixels,
        "damage_eligible_pixel_count": damage_eligible_pixels,
        "major_axis_mm": major_mm,
        "minor_axis_mm": minor_mm,
        "area_mm2": area_mm2,
        "calibration_id": config["calibration"]["calibration_id"],
        "shape_success": shape_success,
        "color_success": color_success,
        "damage_success": damage_success,
        "processing_success": shape_success and color_success and damage_success,
        "error_reason": ";".join(errors),
        "warning_reason": ";".join(warnings),
    }
    row.update(features)
    for key, path in artifact_paths.items():
        row[key] = relative_artifact(path, output_root)
    return row


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_from_config(config, config["model"]["path"])
    manifest_path = resolve_from_config(config, config["data"]["manifest"])
    dataset_root = resolve_from_config(config, config["data"]["dataset_root"])
    output_root = resolve_from_config(config, config["output"]["root"])
    config["_resolved_model_path"] = str(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Extraction output is not empty; refusing to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    config_snapshot_path = output_root / "config_snapshot.yaml"
    shutil.copy2(Path(config["_config_path"]), config_snapshot_path)

    damage_cfg = config["damage_detection"]
    validate_kernel(damage_cfg["morphology_open_kernel"], "morphology_open_kernel")
    validate_kernel(damage_cfg["morphology_close_kernel"], "morphology_close_kernel")

    manifest = pd.read_csv(manifest_path)
    split = str(config["data"]["split"])
    selected = select_manifest_rows(manifest, split, int(config["data"]["random_seed"]))
    max_images_value = config["data"].get("max_images")
    target_instances_value = config["data"].get("target_instances")
    max_images = int(max_images_value) if max_images_value is not None else None
    target_instances = int(target_instances_value) if target_instances_value is not None else None
    if max_images is not None:
        selected = selected[:max_images]
    for record in selected:
        record["resolved_image_path"] = str(dataset_root / "images" / split / f"{record['new_stem']}.png")

    model_hash = sha256(model_path)
    model = YOLO(str(model_path))
    rows: list[dict[str, Any]] = []
    image_log: list[dict[str, Any]] = []
    duplicate_log: list[dict[str, Any]] = []

    for image_record in selected:
        image_id = image_record["new_stem"]
        image_path = Path(image_record["resolved_image_path"])
        log = {
            "image_id": image_id,
            "image_path": str(image_path),
            "status": "failed",
            "predicted_instances": 0,
            "kept_instances": 0,
            "deduplicated_instances": 0,
            "error_reason": "",
        }
        try:
            image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise IOError(f"Could not read image: {image_path}")
            predictions = model.predict(
                source=str(image_path),
                imgsz=int(config["model"]["imgsz"]),
                conf=float(config["model"]["confidence_threshold"]),
                iou=float(config["model"]["iou_threshold"]),
                device=str(config["model"]["device"]),
                retina_masks=bool(config["model"]["retina_masks"]),
                verbose=False,
            )
            result = predictions[0]
            if result.masks is None or result.boxes is None:
                log["status"] = "no_instances"
                image_log.append(log)
                continue
            polygons = result.masks.xy
            confidences = result.boxes.conf.detach().cpu().numpy().astype(float)
            candidates = []
            for model_index, (polygon, confidence) in enumerate(zip(polygons, confidences, strict=False)):
                raw_mask = polygon_to_mask(np.asarray(polygon), image_bgr.shape[0], image_bgr.shape[1])
                ys, xs = np.nonzero(raw_mask)
                if len(xs) == 0:
                    continue
                candidates.append(
                    {
                        "model_index": model_index,
                        "confidence": float(confidence),
                        "raw_mask": raw_mask,
                        "centroid_x": float(xs.mean()),
                        "centroid_y": float(ys.mean()),
                    }
                )
            log["predicted_instances"] = len(candidates)
            kept, removed = deduplicate_candidates(
                candidates, float(config["deduplication"]["mask_iou_threshold"])
            )
            kept.sort(key=lambda item: (item["centroid_y"], item["centroid_x"]))
            for item in removed:
                item["image_id"] = image_id
                duplicate_log.append(item)
            log["kept_instances"] = len(kept)
            log["deduplicated_instances"] = len(removed)
            for index, candidate in enumerate(kept, start=1):
                instance_id = f"{image_id}__i{index:03d}"
                try:
                    row = process_instance(
                        image_bgr=image_bgr,
                        raw_mask=candidate["raw_mask"],
                        confidence=candidate["confidence"],
                        image_record=image_record,
                        instance_id=instance_id,
                        config=config,
                        output_root=output_root,
                        model_hash=model_hash,
                    )
                    rows.append(row)
                except Exception as exc:  # one instance must not stop its image
                    failed = empty_feature_row()
                    failed.update(
                        {
                            "image_id": image_id,
                            "instance_id": instance_id,
                            "split": split,
                            "mask_source": "prediction",
                            "model_sha256": model_hash,
                            "confidence": candidate["confidence"],
                            "shape_success": False,
                            "color_success": False,
                            "damage_success": False,
                            "processing_success": False,
                            "error_reason": f"instance_unhandled:{type(exc).__name__}:{exc}",
                            "warning_reason": "",
                        }
                    )
                    rows.append(failed)
            log["status"] = "success"
        except Exception as exc:  # one image must not stop the pilot
            log["error_reason"] = f"{type(exc).__name__}:{exc}"
        image_log.append(log)
        if target_instances is not None and len(rows) >= target_instances:
            break

    features = pd.DataFrame(rows)
    if features.empty:
        raise RuntimeError("Pilot produced no instances.")
    ordered_columns = [
        "image_id",
        "instance_id",
        *FEATURE_COLUMNS,
        *[column for column in features.columns if column not in {"image_id", "instance_id", *FEATURE_COLUMNS}],
    ]
    features = features.loc[:, ordered_columns]
    feature_filename = str(config["output"].get("features_filename", "features_pilot.csv"))
    features.to_csv(output_root / feature_filename, index=False, encoding="utf-8-sig")
    pd.DataFrame(image_log).to_csv(output_root / "image_processing_log.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(duplicate_log).to_csv(output_root / "deduplication_log.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "run_type": str(config.get("run_type", "pilot")),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(Path(config["_config_path"])),
        "config_snapshot_path": str(config_snapshot_path),
        "config_sha256": sha256(config_snapshot_path),
        "config_version": config["config_version"],
        "model_path": str(model_path),
        "model_sha256": model_hash,
        "split": split,
        "images_attempted": len(image_log),
        "instances_written": len(features),
        "successful_instances": int(features["processing_success"].fillna(False).sum()),
        "failed_instances": int((~features["processing_success"].fillna(False)).sum()),
        "deduplicated_predictions": len(duplicate_log),
        "python": sys.version,
        "torch": torch.__version__,
        "opencv": cv2.__version__,
    }
    (output_root / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    minimum_instances = int(config.get("minimum_required_instances", 30))
    if len(features) < minimum_instances:
        raise RuntimeError(
            f"Extraction produced only {len(features)} instances; at least {minimum_instances} are required."
        )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("YOLO_CONFIG_DIR", str((Path(__file__).resolve().parents[1] / "environment" / "ultralytics_config")))
    raise SystemExit(main())
