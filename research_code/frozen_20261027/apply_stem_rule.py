from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import openpyxl
import pandas as pd
import yaml
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo


STATUS_PRESENT = "PRESENT"
STATUS_MISSING = "MISSING"
STATUS_UNCERTAIN = "UNCERTAIN"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect stalk evidence and apply the missing-stalk Grade-3 rule.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/size_priority_v1/size_priority_candidate_labels.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/stem_detection_v1.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/size_priority_stem_rule_v2"))
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def odd_kernel(value: int) -> np.ndarray:
    size = max(1, int(value))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def load_roi_mask(row: pd.Series, project_root: Path, column: str) -> np.ndarray:
    artifact_root = project_root / "outputs" / "features_frozen_v1" / str(row["split"])
    mask = cv2.imread(str(artifact_root / str(row[column])), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(artifact_root / str(row[column]))
    return mask > 0


def place_roi_mask(full_shape: tuple[int, int], roi_mask: np.ndarray, row: pd.Series) -> np.ndarray:
    output = np.zeros(full_shape, dtype=bool)
    x1, y1, x2, y2 = [int(row[name]) for name in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")]
    expected = (y2 - y1, x2 - x1)
    if roi_mask.shape != expected:
        roi_mask = cv2.resize(roi_mask.astype(np.uint8), (expected[1], expected[0]), interpolation=cv2.INTER_NEAREST) > 0
    output[y1:y2, x1:x2] = roi_mask
    return output


def axis_unit(row: pd.Series) -> np.ndarray:
    vector = np.array(
        [float(row["major_x2"]) - float(row["major_x1"]), float(row["major_y2"]) - float(row["major_y1"])],
        dtype=float,
    )
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        angle = math.radians(float(row["major_angle_deg"]))
        return np.array([math.cos(angle), math.sin(angle)], dtype=float)
    return vector / norm


def pca_alignment_and_slenderness(x: np.ndarray, y: np.ndarray, outward: np.ndarray, length: float) -> tuple[float, float]:
    if len(x) < 3:
        return 0.0, 0.0
    points = np.column_stack([x, y]).astype(float)
    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, int(np.argmax(values))]
    alignment = float(abs(np.dot(principal, outward)))
    mean_width = len(points) / max(length, 1.0)
    slenderness = float(length / max(mean_width, 1.0))
    return alignment, slenderness


def evaluate_tip(
    fruit: np.ndarray,
    calyx: np.ndarray,
    color_candidate: np.ndarray,
    center: np.ndarray,
    base_axis: np.ndarray,
    sign: int,
    config: dict[str, Any],
) -> tuple[dict[str, float | int], dict[str, np.ndarray | tuple[float, float]]]:
    yy, xx = np.indices(fruit.shape, dtype=np.float32)
    outward = base_axis * float(sign)
    fruit_y, fruit_x = np.nonzero(fruit)
    fruit_points = np.column_stack([fruit_x, fruit_y]).astype(float)
    projections = (fruit_points - center) @ outward
    maximum = float(projections.max())
    tip_points = fruit_points[projections >= maximum - 2.0]
    tip = np.median(tip_points, axis=0)

    dx = xx - float(tip[0])
    dy = yy - float(tip[1])
    longitudinal = dx * float(outward[0]) + dy * float(outward[1])
    perpendicular = np.abs(-dx * float(outward[1]) + dy * float(outward[0]))
    half_width = float(
        np.clip(
            float(config["corridor_half_width_fraction_minor_axis"]) * float(2.0 * np.sqrt(np.count_nonzero(fruit) / math.pi)),
            float(config["corridor_half_width_min_px"]),
            float(config["corridor_half_width_max_px"]),
        )
    )
    corridor = (
        (longitudinal >= -float(config["attach_depth_px"]))
        & (longitudinal <= float(config["search_length_px"]))
        & (perpendicular <= half_width)
    )
    raw = color_candidate & corridor
    raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, odd_kernel(int(config["morphology_close_kernel"]))) > 0
    connectivity = cv2.dilate(raw.astype(np.uint8), odd_kernel(int(config["morphology_dilate_kernel"]))) > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(connectivity.astype(np.uint8), connectivity=8)
    seed = (
        (longitudinal >= -float(config["attach_depth_px"]))
        & (longitudinal <= float(config["tip_seed_depth_px"]))
        & (perpendicular <= float(config["tip_seed_radius_px"]))
    )

    best = {
        "component_area_px": 0,
        "outside_area_px": 0,
        "outward_length_px": 0.0,
        "axis_alignment_cosine": 0.0,
        "slenderness": 0.0,
        "calyx_pixels": int(np.count_nonzero(calyx & corridor)),
    }
    selected_mask = np.zeros_like(fruit)
    for label in range(1, count):
        component_connectivity = labels == label
        if not np.any(component_connectivity & seed):
            continue
        component_raw = raw & component_connectivity
        component_area = int(np.count_nonzero(component_raw))
        if component_area < int(config["minimum_component_area_px"]):
            continue
        outside = component_raw & ~fruit & (longitudinal > 1.0)
        outside_area = int(np.count_nonzero(outside))
        if outside_area:
            outward_length = float(np.max(longitudinal[outside]))
            out_y, out_x = np.nonzero(outside)
            alignment, slenderness = pca_alignment_and_slenderness(out_x, out_y, outward, outward_length)
        else:
            outward_length = 0.0
            alignment = 0.0
            slenderness = 0.0
        evidence = outward_length + 0.15 * outside_area + 3.0 * alignment + 0.5 * slenderness
        best_evidence = (
            float(best["outward_length_px"])
            + 0.15 * int(best["outside_area_px"])
            + 3.0 * float(best["axis_alignment_cosine"])
            + 0.5 * float(best["slenderness"])
        )
        if evidence > best_evidence:
            best = {
                "component_area_px": component_area,
                "outside_area_px": outside_area,
                "outward_length_px": outward_length,
                "axis_alignment_cosine": alignment,
                "slenderness": slenderness,
                "calyx_pixels": int(np.count_nonzero(calyx & corridor)),
            }
            selected_mask = component_raw
    debug = {"corridor": corridor, "candidate": selected_mask, "tip": (float(tip[0]), float(tip[1]))}
    return best, debug


def evaluate_full_boundary(
    fruit: np.ndarray,
    calyx: np.ndarray,
    color_candidate: np.ndarray,
    center: np.ndarray,
    base_axis: np.ndarray,
    major_axis: float,
    config: dict[str, Any],
) -> tuple[dict[str, float | int], dict[str, np.ndarray | tuple[float, float]]]:
    distance_outside = cv2.distanceTransform((~fruit).astype(np.uint8), cv2.DIST_L2, 5)
    search_region = fruit | ((~fruit) & (distance_outside <= float(config["search_length_px"])))
    raw = color_candidate & search_region
    raw = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_CLOSE, odd_kernel(int(config["morphology_close_kernel"]))) > 0
    connectivity = cv2.dilate(raw.astype(np.uint8), odd_kernel(int(config["morphology_dilate_kernel"]))) > 0
    count, labels, _, _ = cv2.connectedComponentsWithStats(connectivity.astype(np.uint8), connectivity=8)
    attachment_kernel = odd_kernel(2 * int(config["boundary_attachment_radius_px"]) + 1)
    attachment_zone = cv2.dilate(fruit.astype(np.uint8), attachment_kernel) > 0

    best = {
        "component_area_px": 0,
        "outside_area_px": 0,
        "outward_length_px": 0.0,
        "axis_alignment_cosine": 0.0,
        "slenderness": 0.0,
        "calyx_pixels": 0,
        "attachment_tipness": 0.0,
    }
    selected_mask = np.zeros_like(fruit)
    selected_attachment = (float(center[0]), float(center[1]))
    for label in range(1, count):
        connected = labels == label
        if not np.any(connected & attachment_zone):
            continue
        component_raw = raw & connected
        component_area = int(np.count_nonzero(component_raw))
        if component_area < int(config["minimum_component_area_px"]):
            continue
        outside = component_raw & ~fruit
        outside_area = int(np.count_nonzero(outside))
        if outside_area:
            outward_length = float(np.max(distance_outside[outside]))
            out_y, out_x = np.nonzero(outside)
            distances = distance_outside[outside]
            farthest_index = int(np.argmax(distances))
            farthest = np.array([float(out_x[farthest_index]), float(out_y[farthest_index])])
            outward = farthest - center
            norm = float(np.linalg.norm(outward))
            outward = outward / norm if norm > 1e-6 else base_axis
            alignment, slenderness = pca_alignment_and_slenderness(out_x, out_y, outward, outward_length)
        else:
            outward_length = 0.0
            alignment = 0.0
            slenderness = 0.0
        near_boundary = component_raw & attachment_zone
        if np.any(near_boundary):
            attach_y, attach_x = np.nonzero(near_boundary)
            attachment = np.array([float(np.median(attach_x)), float(np.median(attach_y))])
            tipness = float(np.clip(abs(np.dot(attachment - center, base_axis)) / max(major_axis / 2.0, 1.0), 0.0, 2.0))
        else:
            attachment = center
            tipness = 0.0
        evidence = outward_length + 0.15 * outside_area + 3.0 * alignment + 0.5 * slenderness
        best_evidence = (
            float(best["outward_length_px"])
            + 0.15 * int(best["outside_area_px"])
            + 3.0 * float(best["axis_alignment_cosine"])
            + 0.5 * float(best["slenderness"])
        )
        if evidence > best_evidence:
            best = {
                "component_area_px": component_area,
                "outside_area_px": outside_area,
                "outward_length_px": outward_length,
                "axis_alignment_cosine": alignment,
                "slenderness": slenderness,
                "calyx_pixels": int(np.count_nonzero(calyx & component_raw)),
                "attachment_tipness": tipness,
            }
            selected_mask = component_raw
            selected_attachment = (float(attachment[0]), float(attachment[1]))
    display_band = (~fruit) & (distance_outside <= float(config["boundary_display_band_px"]))
    debug = {"corridor": display_band, "candidate": selected_mask, "tip": selected_attachment}
    return best, debug


def detect_stem(
    row: pd.Series,
    project_root: Path,
    image: np.ndarray,
    config: dict[str, Any],
    return_debug: bool = False,
) -> tuple[dict[str, object], dict[str, object] | None]:
    fruit_roi = load_roi_mask(row, project_root, "mask_path")
    calyx_roi = load_roi_mask(row, project_root, "calyx_mask_path")
    roi_x1, roi_y1, roi_x2, roi_y2 = [int(row[name]) for name in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")]
    expected = (roi_y2 - roi_y1, roi_x2 - roi_x1)
    if fruit_roi.shape != expected:
        fruit_roi = cv2.resize(fruit_roi.astype(np.uint8), (expected[1], expected[0]), interpolation=cv2.INTER_NEAREST) > 0
    if calyx_roi.shape != expected:
        calyx_roi = cv2.resize(calyx_roi.astype(np.uint8), (expected[1], expected[0]), interpolation=cv2.INTER_NEAREST) > 0
    margin = int(config["search_length_px"]) + int(config["corridor_half_width_max_px"]) + 20
    patch_x1 = max(0, roi_x1 - margin)
    patch_y1 = max(0, roi_y1 - margin)
    patch_x2 = min(image.shape[1], roi_x2 + margin)
    patch_y2 = min(image.shape[0], roi_y2 + margin)
    patch = image[patch_y1:patch_y2, patch_x1:patch_x2]
    fruit = np.zeros(patch.shape[:2], dtype=bool)
    calyx = np.zeros(patch.shape[:2], dtype=bool)
    local_x1, local_y1 = roi_x1 - patch_x1, roi_y1 - patch_y1
    local_x2, local_y2 = local_x1 + expected[1], local_y1 + expected[0]
    fruit[local_y1:local_y2, local_x1:local_x2] = fruit_roi
    calyx[local_y1:local_y2, local_x1:local_x2] = calyx_roi
    if np.count_nonzero(fruit) < 20:
        result = {
            "stem_auto_status": STATUS_UNCERTAIN,
            "stem_selected_tip": "none",
            "stem_outward_length_px": 0.0,
            "stem_outside_area_px": 0,
            "stem_component_area_px": 0,
            "stem_axis_alignment_cosine": 0.0,
            "stem_slenderness": 0.0,
            "stem_calyx_pixels_selected_tip": 0,
            "stem_evidence_score": 0.0,
            "stem_detection_warning": "fruit_mask_too_small",
        }
        return result, None

    filtered_patch = cv2.medianBlur(patch, int(config["median_blur_kernel"]))
    lab = cv2.cvtColor(filtered_patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(filtered_patch, cv2.COLOR_BGR2HSV).astype(np.float32)
    lightness = lab[:, :, 0] * (100.0 / 255.0)
    a_star = lab[:, :, 1] - 128.0
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0

    distance_outside = cv2.distanceTransform((~fruit).astype(np.uint8), cv2.DIST_L2, 5)
    background_ring = (~fruit) & (distance_outside >= 10.0) & (distance_outside <= 45.0)
    if np.count_nonzero(background_ring) < 100:
        background_ring = ~fruit
    background_L = float(np.median(lightness[background_ring]))
    fruit_a = a_star[fruit]
    median_a = float(np.median(fruit_a))
    mad_a = float(np.median(np.abs(fruit_a - median_a)))
    robust_scale_a = 1.4826 * max(mad_a, float(config["min_a_mad"]))
    relative_a_drop = max(
        float(config["relative_a_delta_min"]),
        float(config["relative_a_z_min"]) * robust_scale_a,
    )
    a_limit = min(float(config["absolute_a_max"]), median_a - relative_a_drop)
    L_limit = min(float(config["absolute_L_max"]), background_L - float(config["local_background_L_delta_min"]))
    color_candidate = (
        (hue >= float(config["hsv_hue_min"]))
        & (hue <= float(config["hsv_hue_max"]))
        & (a_star <= a_limit)
        & (lightness <= L_limit)
        & (saturation >= float(config["saturation_min"]))
        & (value <= float(config["value_max"]))
    )

    center = np.array(
        [float(row["axis_center_x"]) - patch_x1, float(row["axis_center_y"]) - patch_y1],
        dtype=float,
    )
    base_axis = axis_unit(row)
    metrics_positive, debug_positive = evaluate_tip(fruit, calyx, color_candidate, center, base_axis, 1, config)
    metrics_negative, debug_negative = evaluate_tip(fruit, calyx, color_candidate, center, base_axis, -1, config)
    metrics_by_sign = {1: metrics_positive, -1: metrics_negative}
    debug_by_sign = {1: debug_positive, -1: debug_negative}

    calyx_positive = int(metrics_positive["calyx_pixels"])
    calyx_negative = int(metrics_negative["calyx_pixels"])
    def evidence(metrics: dict[str, float | int]) -> float:
        return (
            float(metrics["outward_length_px"])
            + 0.15 * int(metrics["outside_area_px"])
            + 3.0 * float(metrics["axis_alignment_cosine"])
            + 0.5 * float(metrics["slenderness"])
        )

    if max(calyx_positive, calyx_negative) >= 10 and max(calyx_positive, calyx_negative) >= 1.5 * max(1, min(calyx_positive, calyx_negative)):
        axial_sign = 1 if calyx_positive > calyx_negative else -1
        axial_reason = "calyx_oriented_tip"
    else:
        axial_sign = 1 if evidence(metrics_positive) >= evidence(metrics_negative) else -1
        axial_reason = "strongest_tip_evidence"

    selected_sign = axial_sign
    selected = metrics_by_sign[axial_sign]
    selection_reason = axial_reason
    selected_debug = debug_by_sign[axial_sign]
    if bool(config.get("full_boundary_search", True)):
        boundary_metrics, boundary_debug = evaluate_full_boundary(
            fruit,
            calyx,
            color_candidate,
            center,
            base_axis,
            float(row["major_axis"]),
            config,
        )
        if evidence(boundary_metrics) > evidence(selected):
            selected = boundary_metrics
            selection_reason = "full_boundary_connected_component"
            selected_debug = boundary_debug
    length = float(selected["outward_length_px"])
    outside_area = int(selected["outside_area_px"])
    alignment = float(selected["axis_alignment_cosine"])
    slenderness = float(selected["slenderness"])
    present = (
        length >= float(config["present_min_outward_length_px"])
        and outside_area >= int(config["present_min_outside_area_px"])
        and alignment >= float(config["present_min_axis_alignment_cosine"])
        and slenderness >= float(config["present_min_slenderness"])
    )
    missing = length <= float(config["missing_max_outward_length_px"])
    if present:
        status = STATUS_PRESENT
    elif missing:
        status = STATUS_MISSING
    else:
        status = STATUS_UNCERTAIN

    length_scale = max(float(config["present_min_outward_length_px"]), 1.0)
    area_scale = max(float(config["present_min_outside_area_px"]), 1.0)
    evidence_score = float(
        np.clip(
            0.45 * length / length_scale
            + 0.25 * outside_area / area_scale
            + 0.20 * alignment / max(float(config["present_min_axis_alignment_cosine"]), 0.01)
            + 0.10 * slenderness / max(float(config["present_min_slenderness"]), 0.01),
            0.0,
            2.0,
        )
    )
    result = {
        "stem_auto_status": status,
        "stem_selected_tip": "full_boundary" if selection_reason == "full_boundary_connected_component" else ("axis_positive" if selected_sign == 1 else "axis_negative"),
        "stem_tip_selection_reason": selection_reason,
        "stem_outward_length_px": length,
        "stem_outside_area_px": outside_area,
        "stem_component_area_px": int(selected["component_area_px"]),
        "stem_axis_alignment_cosine": alignment,
        "stem_slenderness": slenderness,
        "stem_calyx_pixels_selected_tip": int(selected["calyx_pixels"]),
        "stem_attachment_tipness": float(selected.get("attachment_tipness", 1.0)),
        "stem_evidence_score": evidence_score,
        "stem_background_L": background_L,
        "stem_candidate_a_limit": a_limit,
        "stem_candidate_L_limit": L_limit,
        "stem_other_tip_outward_length_px": float(metrics_by_sign[-axial_sign]["outward_length_px"]),
        "stem_other_tip_outside_area_px": int(metrics_by_sign[-axial_sign]["outside_area_px"]),
        "stem_detection_warning": "" if status != STATUS_UNCERTAIN else "requires_manual_review",
    }
    if not return_debug:
        return result, None
    debug = {
        "patch_bounds": (patch_x1, patch_y1, patch_x2, patch_y2),
        "fruit": fruit,
        "calyx": calyx,
        "color_candidate": color_candidate,
        "selected_candidate": selected_debug["candidate"],
        "selected_corridor": selected_debug["corridor"],
        "other_corridor": debug_by_sign[1]["corridor"] | debug_by_sign[-1]["corridor"],
        "selected_tip_xy": selected_debug["tip"],
        "selected_sign": selected_sign,
    }
    return result, debug


def select_spread(pool: pd.DataFrame, n: int, used_instances: set[str], used_images: set[str]) -> pd.DataFrame:
    available = pool.loc[~pool["instance_id"].astype(str).isin(used_instances)].sort_values(
        ["stem_evidence_score", "instance_id"]
    )
    if len(available) <= n:
        positions = list(range(len(available)))
    else:
        positions = np.linspace(0, len(available) - 1, n).round().astype(int).tolist()
    primary = available.iloc[positions]
    selected: list[pd.Series] = []
    selected_ids: set[str] = set()
    for _, row in primary.iterrows():
        instance_id = str(row["instance_id"])
        image_id = str(row["image_id"])
        if image_id in used_images or instance_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(instance_id)
        used_instances.add(instance_id)
        used_images.add(image_id)
    for _, row in available.iterrows():
        if len(selected) >= n:
            break
        instance_id = str(row["instance_id"])
        image_id = str(row["image_id"])
        if instance_id in selected_ids or image_id in used_images:
            continue
        selected.append(row)
        selected_ids.add(instance_id)
        used_instances.add(instance_id)
        used_images.add(image_id)
    for _, row in available.iterrows():
        if len(selected) >= n:
            break
        instance_id = str(row["instance_id"])
        if instance_id in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(instance_id)
        used_instances.add(instance_id)
    return pd.DataFrame(selected)


def select_review(scored: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    used_instances: set[str] = set()
    used_images: set[str] = set()
    groups: list[pd.DataFrame] = []
    specifications = [
        ("auto_missing", scored["stem_auto_status"].eq(STATUS_MISSING), int(config["missing_candidates"]), "spread"),
        ("auto_uncertain", scored["stem_auto_status"].eq(STATUS_UNCERTAIN), int(config["uncertain_candidates"]), "spread"),
        ("weakest_auto_present", scored["stem_auto_status"].eq(STATUS_PRESENT), int(config["weakest_present_candidates"]), "low"),
        ("strongest_auto_present", scored["stem_auto_status"].eq(STATUS_PRESENT), int(config["strongest_present_candidates"]), "high"),
    ]
    for group, condition, count, mode in specifications:
        pool = scored.loc[condition].copy()
        if mode == "low":
            pool = pool.sort_values(["stem_evidence_score", "instance_id"]).head(max(count * 5, count))
        elif mode == "high":
            pool = pool.sort_values(["stem_evidence_score", "instance_id"], ascending=[False, True]).head(max(count * 5, count))
        selected = select_spread(pool, min(count, len(pool)), used_instances, used_images)
        selected["review_group"] = group
        groups.append(selected)
    review = pd.concat(groups, ignore_index=True)
    review.insert(0, "review_order", np.arange(1, len(review) + 1))
    review.insert(0, "review_id", [f"ST{index:03d}" for index in range(1, len(review) + 1)])
    review["review_image_file"] = review["review_id"] + "__" + review["instance_id"].astype(str) + ".png"
    review["review_stem_present"] = ""
    review["review_accept_auto_status"] = ""
    review["review_comment"] = ""
    return review


def place_image(canvas: np.ndarray, source: np.ndarray, left: int, top: int, width: int, height: int) -> None:
    h, w = source.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(source, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    rh, rw = resized.shape[:2]
    x = left + (width - rw) // 2
    y = top + (height - rh) // 2
    canvas[y : y + rh, x : x + rw] = resized


def make_review_image(
    row: pd.Series,
    project_root: Path,
    image: np.ndarray,
    detection_config: dict[str, Any],
    output_path: Path,
) -> None:
    _, debug = detect_stem(row, project_root, image, detection_config, return_debug=True)
    if debug is None:
        raise RuntimeError(f"No debug data for {row['instance_id']}")
    fruit = debug["fruit"]
    x1, y1, x2, y2 = debug["patch_bounds"]
    original = image[y1:y2, x1:x2].copy()
    overlay = original.copy()
    fruit_local = fruit
    selected_candidate = debug["selected_candidate"]
    selected_corridor = debug["selected_corridor"]
    other_corridor = debug["other_corridor"]
    contours, _ = cv2.findContours(fruit_local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 220, 0), 2)
    overlay[selected_corridor] = (0.65 * overlay[selected_corridor] + 0.35 * np.array([255, 210, 0])).astype(np.uint8)
    overlay[other_corridor] = (0.78 * overlay[other_corridor] + 0.22 * np.array([220, 220, 0])).astype(np.uint8)
    overlay[selected_candidate] = (255, 0, 255)
    tip_x, tip_y = debug["selected_tip_xy"]
    cv2.circle(overlay, (int(round(tip_x)), int(round(tip_y))), 6, (0, 255, 255), 2)

    canvas = np.full((450, 1160, 3), 247, dtype=np.uint8)
    title = f"{row['review_id']}  {row['instance_id']}  auto={row['stem_auto_status']}  [{row['review_group']}]"
    cv2.putText(canvas, title, (18, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (15, 15, 15), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Expanded original", (20, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
    cv2.putText(canvas, "Detection overlay", (390, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
    place_image(canvas, original, 20, 68, 330, 330)
    place_image(canvas, overlay, 390, 68, 330, 330)
    lines = [
        f"Base grade: {int(row['base_candidate_grade'])}",
        f"Provisional grade after rule: {int(row['provisional_grade_after_stem_rule'])}",
        f"Rule applied: {bool(row['stem_rule_applied'])}",
        f"Evidence score: {row['stem_evidence_score']:.2f}",
        f"Outward length: {row['stem_outward_length_px']:.1f}px",
        f"Outside area: {int(row['stem_outside_area_px'])}px",
        f"Axis alignment: {row['stem_axis_alignment_cosine']:.2f}",
        f"Slenderness: {row['stem_slenderness']:.2f}",
        f"Calyx pixels at selected tip: {int(row['stem_calyx_pixels_selected_tip'])}",
        f"Selected tip: {row['stem_selected_tip']}",
        "Green=fruit; magenta=connected stalk candidate",
        "Cyan=boundary attachment band; yellow=axial aid",
        "Confirm stalk presence from the original panel.",
    ]
    y = 78
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (752, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (20, 20, 20) if index < 4 else (48, 48, 48),
            2 if index == 1 else 1,
            cv2.LINE_AA,
        )
        y += 27
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(output_path)


def make_contact_sheets(review: pd.DataFrame, review_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, frame in review.groupby("review_group", sort=False):
        cols = 4
        rows = math.ceil(len(frame) / cols)
        tile_w, tile_h = 435, 169
        title_h = 56
        sheet = np.full((title_h + rows * tile_h, cols * tile_w, 3), 250, dtype=np.uint8)
        cv2.putText(sheet, f"Stem review: {group}  n={len(frame)}", (18, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (20, 20, 20), 2, cv2.LINE_AA)
        for index, (_, row) in enumerate(frame.iterrows()):
            panel = cv2.imread(str(review_dir / str(row["review_image_file"])), cv2.IMREAD_COLOR)
            if panel is None:
                raise FileNotFoundError(row["review_image_file"])
            panel = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            top = title_h + (index // cols) * tile_h
            left = (index % cols) * tile_w
            sheet[top : top + tile_h, left : left + tile_w] = panel
        if not cv2.imwrite(str(output_dir / f"{group}_contact_sheet.png"), sheet):
            raise IOError(group)


def write_review_workbook(review: pd.DataFrame, output_path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "StemReview"
    instructions = workbook.create_sheet("Instructions")
    rows = [
        ["果梗状态复核说明"],
        ["目标", "仅复核合格果是否有果梗；上游已经人工排除了霉变腐烂果。"],
        ["review_stem_present", "看 Expanded original：果梗存在填 YES，缺失填 NO，看不清填 UNCERTAIN。"],
        ["review_accept_auto_status", "自动判断正确填 YES，错误填 NO。"],
        ["review_comment", "记录被碗边干扰、果梗太细、果梗与其他果实重叠等原因。"],
        ["分级覆盖", "人工确认果梗缺失后，无论原 size_priority 为 1/2/3，都归入 3 级。"],
        ["自动不确定", "UNCERTAIN 不会静默降级；保留原等级并标记 needs_stem_review=TRUE。"],
        ["图例", "绿色=果体 Mask，紫色=与果体边缘相连的绿—褐色果梗候选，蓝色=全边界连接带，黄色=长轴辅助搜索范围。"],
    ]
    for row in rows:
        instructions.append(row)
    instructions.merge_cells("A1:B1")
    instructions["A1"].font = Font(bold=True, size=16, color="FFFFFF")
    instructions["A1"].fill = PatternFill("solid", fgColor="1F4E78")
    instructions.column_dimensions["A"].width = 28
    instructions.column_dimensions["B"].width = 92
    for row in instructions.iter_rows(min_row=2, max_col=2):
        row[0].font = Font(bold=True, color="1F4E78")
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    instructions.sheet_view.showGridLines = False

    for column_index, name in enumerate(review.columns, start=1):
        sheet.cell(1, column_index, name)
    for row_index, values in enumerate(review.itertuples(index=False, name=None), start=2):
        for column_index, value in enumerate(values, start=1):
            sheet.cell(row_index, column_index, None if pd.isna(value) else value)
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sheet.freeze_panes = "G2"
    sheet.sheet_view.showGridLines = False

    columns = {name: index + 1 for index, name in enumerate(review.columns)}
    yellow = PatternFill("solid", fgColor="FFF2CC")
    for row_index in range(2, len(review) + 2):
        image_cell = sheet.cell(row_index, columns["review_image_file"])
        image_cell.hyperlink = f"review_images/{image_cell.value}"
        image_cell.style = "Hyperlink"
        for name in ("review_stem_present", "review_accept_auto_status", "review_comment"):
            sheet.cell(row_index, columns[name]).fill = yellow

    status_validation = DataValidation(type="list", formula1='"YES,NO,UNCERTAIN"', allow_blank=True)
    accept_validation = DataValidation(type="list", formula1='"YES,NO"', allow_blank=True)
    sheet.add_data_validation(status_validation)
    sheet.add_data_validation(accept_validation)
    status_letter = openpyxl.utils.get_column_letter(columns["review_stem_present"])
    accept_letter = openpyxl.utils.get_column_letter(columns["review_accept_auto_status"])
    status_validation.add(f"{status_letter}2:{status_letter}{len(review)+1}")
    accept_validation.add(f"{accept_letter}2:{accept_letter}{len(review)+1}")

    auto_letter = openpyxl.utils.get_column_letter(columns["stem_auto_status"])
    status_colors = {STATUS_PRESENT: "C6E0B4", STATUS_MISSING: "F4B084", STATUS_UNCERTAIN: "FFE699"}
    for status, color in status_colors.items():
        sheet.conditional_formatting.add(
            f"{auto_letter}2:{auto_letter}{len(review)+1}",
            FormulaRule(formula=[f'{auto_letter}2="{status}"'], fill=PatternFill("solid", fgColor=color)),
        )

    preferred_widths = {
        "review_id": 11,
        "review_group": 24,
        "image_id": 14,
        "instance_id": 22,
        "stem_auto_status": 18,
        "review_image_file": 34,
        "review_stem_present": 21,
        "review_accept_auto_status": 24,
        "review_comment": 44,
    }
    for index, name in enumerate(review.columns, start=1):
        sheet.column_dimensions[openpyxl.utils.get_column_letter(index)].width = preferred_widths.get(name, 16)
        if pd.api.types.is_float_dtype(review[name]):
            for row_index in range(2, len(review) + 2):
                sheet.cell(row_index, index).number_format = "0.000"
    table = Table(displayName="StemStatusReview", ref=sheet.dimensions)
    table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True, showFirstColumn=False, showLastColumn=False)
    sheet.add_table(table)
    workbook.save(output_path)


def write_readme(output_dir: Path, counts: dict[str, int], overrides: int, uncertain: int) -> None:
    text = f"""# size_priority + 果梗缺失三级覆盖规则（v2）

本目录在 `size_priority_v1` 基础上加入以下业务规则：

> 对已经排除霉变、腐烂的不合格果，若确认果梗缺失，则候选等级统一覆盖为 3 级；果梗存在时保留原 size_priority 等级。

## 为什么需要单独检测

现有 YOLO 实例 Mask 主要贴合果体，细长果梗经常不在 Mask 内，所以不能用 `calyx_candidate_ratio` 直接代表果梗存在。本模块读取原始 RGB 图像，在果体长轴两端向外搜索 HSV 色相处于深绿—黄褐—褐色范围、比背景更暗、比红色果肉的 Lab `a*` 更低，并且与果端相连的细长结构。

- `PRESENT`：存在满足长度、面积、方向一致性和细长度条件的连接结构。
- `MISSING`：果端外侧几乎没有连接结构。
- `UNCERTAIN`：证据处于中间区间，可能受低分辨率、碗边、重叠或光照影响。

全部阈值保存在 `configs/stem_detection_v1.yaml`，没有硬编码到最终分级标准中。

## 当前自动初筛结果

- PRESENT：{counts.get(STATUS_PRESENT, 0):,}
- MISSING：{counts.get(STATUS_MISSING, 0):,}
- UNCERTAIN：{counts.get(STATUS_UNCERTAIN, 0):,}
- 因缺梗而从原 1/2 级覆盖到 3 级：{overrides:,}
- 仍需人工确认：{uncertain:,}

这些仍是候选结果，不是训练真值。`stem_review.xlsx` 会收录全部自动 `MISSING`，它们必须逐颗确认；另抽取部分 `UNCERTAIN` 和 `PRESENT` 做误差检查。

## 审核填写

- `review_stem_present`：有果梗填 `YES`，缺失填 `NO`，看不清填 `UNCERTAIN`。
- `review_accept_auto_status`：自动状态正确填 `YES`，错误填 `NO`。
- `review_comment`：记录碗边、重叠、过细或光照问题。

## 等级字段

- `base_candidate_grade`：原 size_priority 候选等级。
- `provisional_grade_after_stem_rule`：应用自动缺梗覆盖后的临时候选等级。
- `stem_rule_applied`：是否因为自动判为缺梗而覆盖成 3 级。
- `needs_stem_review`：自动状态为 UNCERTAIN，暂时保留原等级但不能视为已经确认。
- `candidate_label_status`：始终为 `candidate_only_not_training_truth`。

## 重要限制

果梗宽度只有少数像素，传统视觉检测难以保证完全准确。本模块的定位是减少人工工作量和执行已确定的业务规则；最终批量自动标注前，应使用本次复核结果校准阈值，并对所有自动缺梗候选做人工确认。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    input_path = args.input_csv if args.input_csv.is_absolute() else project_root / args.input_csv
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    detection_config = config["stem_detection"]
    review_config = config["review_sampling"]
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    required = {
        "image_id", "instance_id", "split", "image_path", "mask_path", "calyx_mask_path",
        "roi_x1", "roi_y1", "roi_x2", "roi_y2", "axis_center_x", "axis_center_y",
        "major_x1", "major_y1", "major_x2", "major_y2", "major_angle_deg", "minor_axis", "candidate_grade",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if data["instance_id"].duplicated().any():
        raise ValueError("instance_id must be unique")

    results: list[dict[str, object]] = []
    cached_path = ""
    cached_image: np.ndarray | None = None
    for _, row in data.iterrows():
        image_path = str(row["image_path"])
        if image_path != cached_path:
            cached_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if cached_image is None:
                raise FileNotFoundError(image_path)
            cached_path = image_path
        result, _ = detect_stem(row, project_root, cached_image, detection_config)
        results.append(result)
    stem = pd.DataFrame(results)
    scored = pd.concat([data.reset_index(drop=True), stem], axis=1)
    scored["base_candidate_grade"] = scored["candidate_grade"].astype(int)
    scored["stem_rule_applied"] = scored["stem_auto_status"].eq(STATUS_MISSING) & scored["base_candidate_grade"].ne(3)
    scored["provisional_grade_after_stem_rule"] = np.where(
        scored["stem_auto_status"].eq(STATUS_MISSING),
        int(config["grading_rule"]["missing_stem_grade"]),
        scored["base_candidate_grade"],
    ).astype(int)
    scored["needs_stem_review"] = scored["stem_auto_status"].eq(STATUS_UNCERTAIN)
    scored["stem_rule_status"] = np.where(
        scored["stem_auto_status"].eq(STATUS_MISSING),
        "AUTO_MISSING_OVERRIDE_TO_GRADE3_PENDING_REVIEW",
        np.where(
            scored["stem_auto_status"].eq(STATUS_UNCERTAIN),
            "BASE_GRADE_RETAINED_PENDING_STEM_REVIEW",
            "BASE_GRADE_RETAINED_AUTO_STEM_PRESENT",
        ),
    )
    scored["candidate_label_status"] = str(config["grading_rule"]["label_status"])
    scored["grading_scheme_version"] = "size_priority_stem_rule_v2"

    review = select_review(scored, review_config)
    review_dir = output_dir / "review_images"
    image_cache: dict[str, np.ndarray] = {}
    for _, row in review.iterrows():
        image_path = str(row["image_path"])
        if image_path not in image_cache:
            image_cache[image_path] = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_cache[image_path] is None:
            raise FileNotFoundError(image_path)
        make_review_image(row, project_root, image_cache[image_path], detection_config, review_dir / str(row["review_image_file"]))
    make_contact_sheets(review, review_dir, output_dir / "contact_sheets")

    priority = [
        "image_id", "instance_id", "split", "source_group", "grading_scheme_version", "candidate_label_status",
        "base_candidate_grade", "provisional_grade_after_stem_rule", "stem_rule_applied", "needs_stem_review",
        "stem_rule_status", "stem_auto_status", "stem_evidence_score", "stem_outward_length_px",
        "stem_outside_area_px", "stem_component_area_px", "stem_axis_alignment_cosine", "stem_slenderness",
        "stem_selected_tip", "stem_tip_selection_reason", "stem_calyx_pixels_selected_tip", "size_priority_score",
    ]
    remaining = [column for column in scored.columns if column not in priority]
    scored[priority + remaining].to_csv(output_dir / "size_priority_stem_rule_candidates.csv", index=False, encoding="utf-8-sig")

    review_columns = [
        "review_id", "review_order", "review_group", "image_id", "instance_id", "split", "source_group",
        "stem_auto_status", "base_candidate_grade", "provisional_grade_after_stem_rule", "stem_rule_applied",
        "needs_stem_review", "stem_evidence_score", "stem_outward_length_px", "stem_outside_area_px",
        "stem_axis_alignment_cosine", "stem_slenderness", "stem_calyx_pixels_selected_tip", "size_priority_score",
        "review_image_file", "review_stem_present", "review_accept_auto_status", "review_comment",
    ]
    review_export = review[review_columns].copy()
    review_export.to_csv(output_dir / "stem_review.csv", index=False, encoding="utf-8-sig")
    write_review_workbook(review_export, output_dir / "stem_review.xlsx")

    status_counts = {str(key): int(value) for key, value in scored["stem_auto_status"].value_counts().to_dict().items()}
    grade_distribution = (
        scored.groupby(["base_candidate_grade", "provisional_grade_after_stem_rule", "stem_auto_status"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    grade_distribution.to_csv(output_dir / "grade_transition_summary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "scheme_version": "size_priority_stem_rule_v2",
        "status": "candidate_only_not_training_truth",
        "input_csv": str(input_path.relative_to(project_root)).replace("\\", "/"),
        "input_sha256": sha256_file(input_path),
        "config": str(config_path.relative_to(project_root)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
        "instances": int(len(scored)),
        "stem_status_counts": status_counts,
        "grade3_overrides_from_grade1_or_2": int(scored["stem_rule_applied"].sum()),
        "needs_stem_review": int(scored["needs_stem_review"].sum()),
        "review_pack_instances": int(len(review)),
        "review_unique_images": int(review["image_id"].nunique()),
        "rule": "confirmed missing stalk => grade 3; rot/mold already excluded upstream",
        "software": {
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "opencv": cv2.__version__, "openpyxl": openpyxl.__version__, "pyyaml": yaml.__version__,
        },
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_readme(
        output_dir,
        status_counts,
        int(scored["stem_rule_applied"].sum()),
        int(scored["needs_stem_review"].sum()),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
