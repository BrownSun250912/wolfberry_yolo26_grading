from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml


REVIEWERS = ("lead", "reviewer_a", "reviewer_b")
FOLDER_NAMES = {
    "lead": "01_总负责人",
    "reviewer_a": "02_审核员A",
    "reviewer_b": "03_审核员B",
}
DISPLAY_NAMES = {
    "lead": "总负责人",
    "reviewer_a": "审核员A",
    "reviewer_b": "审核员B",
}
PAIR_MEMBERS = {
    "lead_a": ("lead", "reviewer_a"),
    "lead_b": ("lead", "reviewer_b"),
    "a_b": ("reviewer_a", "reviewer_b"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build three self-contained human review packages.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("review/three_person_audit_v1"),
    )
    return parser.parse_args()


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, quality: int = 90) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".jpg"
    params: list[int] = []
    if extension in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, encoded = cv2.imencode(extension, image, params)
    if not ok:
        raise IOError(f"Could not encode image: {path}")
    path.write_bytes(encoded.tobytes())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bool_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def fit_panel(image: np.ndarray, width: int, height: int, background: int = 24) -> np.ndarray:
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    ih, iw = image.shape[:2]
    scale = min(width / max(iw, 1), height / max(ih, 1))
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    x = (width - nw) // 2
    y = (height - nh) // 2
    canvas[y : y + nh, x : x + nw] = resized
    return canvas


def title_bar(lines: list[str], width: int, height: int = 82) -> np.ndarray:
    bar = np.full((height, width, 3), 12, dtype=np.uint8)
    for idx, line in enumerate(lines[:2]):
        y = 30 + idx * 32
        cv2.putText(
            bar,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72 if idx == 0 else 0.58,
            (255, 255, 255) if idx == 0 else (210, 210, 210),
            2 if idx == 0 else 1,
            cv2.LINE_AA,
        )
    return bar


def resolve_artifact(root: Path, split: str, relative: str) -> Path:
    return root / "outputs" / "features_frozen_v1" / split / Path(relative)


def paste_roi_mask(full_shape: tuple[int, int], local_mask: np.ndarray, row: pd.Series) -> np.ndarray:
    height, width = full_shape
    x1, y1, x2, y2 = (int(float(row[k])) for k in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    target_h, target_w = max(0, y2 - y1), max(0, x2 - x1)
    if local_mask.shape[:2] != (target_h, target_w):
        local_mask = cv2.resize(local_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    full = np.zeros((height, width), dtype=np.uint8)
    full[y1:y2, x1:x2] = local_mask
    return full


def mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def expanded_crop(image: np.ndarray, mask: np.ndarray, min_pad: int = 55) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return image.copy(), mask.copy()
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    span = max(x2 - x1, y2 - y1)
    pad = max(min_pad, int(round(span * 0.65)))
    h, w = image.shape[:2]
    xa, xb = max(0, x1 - pad), min(w, x2 + pad)
    ya, yb = max(0, y1 - pad), min(h, y2 + pad)
    return image[ya:yb, xa:xb].copy(), mask[ya:yb, xa:xb].copy()


def make_stem_preview(root: Path, row: pd.Series) -> np.ndarray:
    image = read_image(Path(str(row["image_path"])))
    split = str(row["split"])
    local_mask = read_image(resolve_artifact(root, split, str(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
    full_mask = paste_roi_mask(image.shape[:2], local_mask, row)

    full = image.copy()
    overlay = full.copy()
    overlay[full_mask > 0] = (0, 180, 255)
    full = cv2.addWeighted(full, 0.72, overlay, 0.28, 0)
    cv2.drawContours(full, mask_contours(full_mask), -1, (0, 255, 0), 3)
    ys, xs = np.nonzero(full_mask)
    if len(xs):
        cv2.rectangle(full, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 2)

    crop, crop_mask = expanded_crop(image, full_mask)
    crop_overlay = crop.copy()
    crop_overlay[crop_mask > 0] = (0, 180, 255)
    crop = cv2.addWeighted(crop, 0.68, crop_overlay, 0.32, 0)
    cv2.drawContours(crop, mask_contours(crop_mask), -1, (0, 255, 0), 3)

    body = np.hstack([fit_panel(full, 900, 560), fit_panel(crop, 500, 560)])
    evidence = str(row.get("stem_evidence_score", ""))
    bar = title_bar(
        [
            f"STEM REVIEW | {row['instance_id']} | AUTO=UNCERTAIN",
            f"Green contour = target fruit | evidence={evidence} | judge attached stalk only",
        ],
        body.shape[1],
    )
    return np.vstack([bar, body])


def normalized_polygons(label_path: Path, width: int, height: int) -> list[np.ndarray]:
    polygons: list[np.ndarray] = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        tokens = line.strip().split()
        if len(tokens) < 7:
            continue
        values = np.asarray([float(v) for v in tokens[1:]], dtype=np.float64).reshape(-1, 2)
        values[:, 0] *= width
        values[:, 1] *= height
        values[:, 0] = np.clip(values[:, 0], 0, width - 1)
        values[:, 1] = np.clip(values[:, 1], 0, height - 1)
        polygons.append(np.rint(values).astype(np.int32))
    return polygons


def clean_analysis_mask(mask: np.ndarray, min_component_area: int = 20) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    components = [(idx, int(stats[idx, cv2.CC_STAT_AREA])) for idx in range(1, count)]
    components = [item for item in components if item[1] >= min_component_area]
    output = np.zeros_like(mask)
    if components:
        label = max(components, key=lambda item: item[1])[0]
        output[labels == label] = 255
    return output


def refine_geometry_mask(image: np.ndarray, mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    original_area = int(np.count_nonzero(binary))
    if not config.get("enabled", False) or not original_area:
        return mask.copy()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    a_star = lab[:, :, 1] - 128.0
    b_star = lab[:, :, 2] - 128.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    chroma = np.sqrt(a_star * a_star + b_star * b_star)
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    boundary = (binary > 0) & (distance <= float(config["max_boundary_depth_px"]))
    pale = (
        (saturation <= float(config["saturation_max"]))
        & (value >= float(config["value_min"]))
        & (chroma <= float(config["lab_chroma_max"]))
    )
    refined = binary.copy()
    refined[boundary & pale] = 0
    refined_mask = clean_analysis_mask(refined * 255, int(config["min_component_area_px"]))
    refined_area = int(np.count_nonzero(refined_mask))
    removed_ratio = (original_area - refined_area) / original_area
    if refined_area < int(config["min_remaining_area_px"]):
        return mask.copy()
    if removed_ratio > float(config["max_removed_fraction"]):
        return mask.copy()
    return refined_mask


def main_contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("empty contour")
    return max(contours, key=cv2.contourArea)


def fit_axes(contour: np.ndarray) -> tuple[float, float]:
    if len(contour) >= 5:
        try:
            _, axes, _ = cv2.fitEllipse(contour)
            width, height = float(axes[0]), float(axes[1])
        except cv2.error:
            _, axes, _ = cv2.minAreaRect(contour)
            width, height = float(axes[0]), float(axes[1])
    else:
        _, axes, _ = cv2.minAreaRect(contour)
        width, height = float(axes[0]), float(axes[1])
    if width <= 0 or height <= 0:
        raise ValueError("invalid axes")
    return max(width, height), min(width, height)


def empirical_percentile(reference: np.ndarray, value: float) -> float:
    ref = np.sort(np.asarray(reference, dtype=float))
    left = np.searchsorted(ref, value, side="left")
    right = np.searchsorted(ref, value, side="right")
    return float(np.clip((left + right) / (2.0 * len(ref)), 0.0, 1.0))


def test_geometry_features(image: np.ndarray, polygon: np.ndarray, geometry_config: dict[str, Any]) -> tuple[dict[str, float], np.ndarray]:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    mask = clean_analysis_mask(mask)
    mask = refine_geometry_mask(image, mask, geometry_config)
    contour = main_contour(mask)
    perimeter = float(cv2.arcLength(contour, True))
    contour_area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    major, minor = fit_axes(contour)
    features = {
        "area": float(np.count_nonzero(mask)),
        "minor_axis": minor,
        "solidity": contour_area / hull_area if hull_area else 0.0,
        "aspect_ratio": major / minor if minor else 0.0,
        "circularity": 4.0 * math.pi * contour_area / (perimeter * perimeter) if perimeter else 0.0,
    }
    return features, mask


def score_test_instance(
    features: dict[str, float],
    references: dict[str, np.ndarray],
    grade_config: dict[str, Any],
) -> tuple[float, int, dict[str, float]]:
    pct = {name: empirical_percentile(references[name], value) for name, value in features.items()}
    aspect_typicality = float(np.clip(1.0 - 2.0 * abs(pct["aspect_ratio"] - 0.5), 0.0, 1.0))
    shape_score = 100.0 * (
        0.50 * pct["solidity"] + 0.30 * aspect_typicality + 0.20 * pct["circularity"]
    )
    weights = grade_config["score_formula"]["weights"]
    score = 100.0 * (
        float(weights["pct_area"]) * pct["area"]
        + float(weights["pct_minor_axis"]) * pct["minor_axis"]
    ) + float(weights["shape_score"]) * shape_score
    low = float(grade_config["thresholds"]["grade3_to_2"])
    high = float(grade_config["thresholds"]["grade2_to_1"])
    grade = 1 if score >= high else 3 if score < low else 2
    return score, grade, {**pct, "shape_score": shape_score, "aspect_typicality": aspect_typicality}


def make_test_preview(
    image: np.ndarray,
    polygon: np.ndarray,
    instance_id: str,
    base_grade: int,
    score: float,
    mask: np.ndarray,
) -> np.ndarray:
    full = image.copy()
    overlay = full.copy()
    overlay[mask > 0] = (0, 200, 255)
    full = cv2.addWeighted(full, 0.72, overlay, 0.28, 0)
    cv2.polylines(full, [polygon], True, (0, 255, 0), 3)
    ys, xs = np.nonzero(mask)
    if len(xs):
        cv2.rectangle(full, (int(xs.min()), int(ys.min())), (int(xs.max()), int(ys.max())), (0, 255, 255), 2)
    crop, crop_mask = expanded_crop(image, mask)
    crop_overlay = crop.copy()
    crop_overlay[crop_mask > 0] = (0, 200, 255)
    crop = cv2.addWeighted(crop, 0.68, crop_overlay, 0.32, 0)
    cv2.drawContours(crop, mask_contours(crop_mask), -1, (0, 255, 0), 3)
    body = np.hstack([fit_panel(full, 900, 560), fit_panel(crop, 500, 560)])
    bar = title_bar(
        [
            f"TEST DOUBLE REVIEW | {instance_id} | BASE=GRADE_{base_grade} | score={score:.2f}",
            "ROT/MOLD -> UNQUALIFIED; else MISSING STEM -> GRADE_3; otherwise confirm BASE grade",
        ],
        body.shape[1],
    )
    return np.vstack([bar, body])


def make_geometry_preview(
    root: Path,
    audit_row: pd.Series,
    manifest_row: pd.Series,
    candidates: pd.DataFrame,
    feature_index: pd.DataFrame,
) -> np.ndarray:
    split = str(manifest_row["split"])
    image_path = root / "data" / "split_v1" / "images" / split / f"{audit_row['image_id']}.png"
    label_path = root / "data" / "split_v1" / "labels" / split / f"{audit_row['image_id']}.txt"
    image = read_image(image_path)
    height, width = image.shape[:2]
    polygons = normalized_polygons(label_path, width, height)

    source_view = image.copy()
    for idx, polygon in enumerate(polygons, start=1):
        cv2.polylines(source_view, [polygon], True, (255, 255, 0), 3)
        center = tuple(np.rint(polygon.mean(axis=0)).astype(int))
        cv2.putText(source_view, f"P{idx:02d}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(source_view, f"P{idx:02d}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)

    candidate_view = image.copy()
    colors = [(0, 255, 0), (255, 0, 255), (0, 165, 255), (255, 128, 0), (0, 0, 255)]
    for idx, (_, candidate) in enumerate(candidates.sort_values("instance_id").iterrows(), start=1):
        instance_id = str(candidate["instance_id"])
        if instance_id not in feature_index.index:
            continue
        feature = feature_index.loc[instance_id]
        local_path = resolve_artifact(root, split, str(feature["mask_path"]))
        if not local_path.exists():
            continue
        local_mask = read_image(local_path, cv2.IMREAD_GRAYSCALE)
        full_mask = paste_roi_mask(image.shape[:2], local_mask, feature)
        color = colors[(idx - 1) % len(colors)]
        cv2.drawContours(candidate_view, mask_contours(full_mask), -1, color, 3)
        ys, xs = np.nonzero(full_mask)
        if len(xs):
            center = (int(np.median(xs)), int(np.median(ys)))
            label = f"I{idx:02d}:{candidate['four_class_label']}"
            cv2.putText(candidate_view, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(candidate_view, label, center, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    body = np.hstack([fit_panel(source_view, 700, 440), fit_panel(candidate_view, 700, 440)])
    bar = title_bar(
        [
            f"GEOMETRY CHECK | {audit_row['image_id']} | source polygons={len(polygons)} | graded instances={len(candidates)}",
            f"LEFT=Pxx original polygons | RIGHT=Ixx graded masks | reason={audit_row['reason']}",
        ],
        body.shape[1],
    )
    return np.vstack([bar, body])


def greedy_assign(groups: list[dict[str, Any]], targets: dict[str, float]) -> dict[str, str]:
    loads = {key: 0 for key in targets}
    assignments: dict[str, str] = {}
    for item in sorted(groups, key=lambda value: (-int(value["weight"]), str(value["id"]))):
        key = min(targets, key=lambda name: (loads[name] / max(targets[name], 1), loads[name], name))
        assignments[str(item["id"])] = key
        loads[key] += int(item["weight"])
    return assignments


def setup_reviewer_tree(folder: Path) -> None:
    categories = {
        "01_果梗复核": ("待审核", "有果梗_PRESENT", "无果梗_MISSING", "无法判断_UNCERTAIN"),
        "02_结构异常复核": ("待审核", "可修复保留_KEEP_FIX", "排除整图_EXCLUDE", "无法判断_UNCERTAIN"),
        "03_测试集双人复核": (
            "待审核",
            "不合格_UNQUALIFIED",
            "一级_GRADE1",
            "二级_GRADE2",
            "三级_GRADE3",
            "无法判断_UNCERTAIN",
        ),
    }
    for task, states in categories.items():
        for state in states:
            (folder / task / state).mkdir(parents=True, exist_ok=True)
    (folder / "争议备注.txt").write_text(
        "如有争议，每行填写：文件名<TAB>备注。不要修改原文件名。\n",
        encoding="utf-8-sig",
    )


def reviewer_readme(name: str, counts: dict[str, int], pair_counts: dict[str, int]) -> str:
    role_note = (
        "你是总负责人：不参与测试集初次双人审核；收回两位审核员的结果后，负责处理不一致和无法判断项。"
        if name == "lead"
        else "你是独立审核员：请按规则独立判断，不要先参考其他人的结果。"
    )
    return f"""# {DISPLAY_NAMES[name]}审核任务说明

{role_note}

## 本包工作量

- 果梗不确定实例：{counts['stem']} 颗（单人审核，全体三包之间不重复）。
- 结构异常图片：{counts['geometry']} 张（单人审核，全体三包之间不重复）。
- 独立测试集：{counts['test_images']} 张、{counts['test_instances']} 颗（每张由两个人独立审核）。
- 测试集配对来源：{json.dumps(pair_counts, ensure_ascii=False)}。

## 怎么填写

本审核包不需要打开Excel。每项任务都以图片文件表示：

1. 打开“待审核”中的图片。
2. 判断后，把图片移动到同任务下对应的结果文件夹。
3. 不要修改文件名，不要删除图片，不要把同一图片复制到多个结果文件夹。
4. 确实无法判断时放入“无法判断_UNCERTAIN”。有特殊情况写入根目录的“争议备注.txt”。

## 任务一：果梗复核

只判断目标枸杞是否保留了与果体相连的果梗：

- 有果梗_PRESENT：能看到与果体相连的细长果梗，颜色通常为深绿、橄榄绿至褐色。
- 无果梗_MISSING：只有果体或蒂痕，没有向外延伸的果梗。
- 无法判断_UNCERTAIN：遮挡、模糊、粘连或图像信息不足。

注意：红色果肉、暗斑、高光、碗沿接触点均不是果梗。只看绿色轮廓标出的目标果，不要判断旁边果实。

## 任务二：结构异常复核

检查左侧原始多边形 Pxx 与右侧已分级实例 Ixx 是否能一一对应：

- 可修复保留_KEEP_FIX：目标果实清楚，可通过补回、删除或重新对应一个轮廓修复。
- 排除整图_EXCLUDE：严重漏标、粘连、轮廓错位，无法可靠恢复。
- 无法判断_UNCERTAIN：仅凭检查图不能确定。

如能指出具体问题，可在“争议备注.txt”中写 Pxx 或 Ixx 编号。

## 任务三：测试集双人复核

图片标题中的 `BASE=GRADE_n` 是既定“大小优先＋形态”标准给出的基础等级：

1. 明显腐烂、霉变或黑褐色坏死，放入“不合格_UNQUALIFIED”。
2. 合格但果梗缺失，必须放入“三级_GRADE3”。
3. 不腐烂且果梗存在时，原则上按 BASE 等级放入一级、二级或三级。
4. 只是果体较小，不要直接判不合格；小果由 BASE 分数进入相应等级。
5. 如果轮廓或目标对应明显错误，放入“无法判断_UNCERTAIN”并备注。

测试集是最终模型评价依据，请独立判断。不要与另一位审核人讨论后再填写。

## 提交

完成后直接返回整个“{FOLDER_NAMES[name]}”文件夹。提交前确认三个“待审核”文件夹均为空；无法判断项应放到对应的 UNCERTAIN 文件夹，不能留在待审核中。
"""


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if root not in output.parents:
        raise ValueError("Output directory must be inside project root")

    held_path = root / "outputs/grading_standard_manual_v4_final/held_or_excluded_instances.csv"
    image_audit_path = root / "data/grading_four_class_v2/audit/image_audit.csv"
    manifest_path = root / "data/manifests/dataset_manifest.csv"
    class_manifest_path = root / "outputs/grading_standard_manual_v4_final/four_class_instance_manifest.csv"
    feature_path = root / "outputs/strict_rot_exclusion_v2/features_with_manual_rot.csv"
    reference_path = root / "outputs/size_priority_v1/scoring_reference_values.csv"
    grade_config_path = root / "outputs/grading_standard_manual_v4_final/grading_standard_config.json"
    feature_config_path = root / "configs/feature_extraction_frozen_v1_train.yaml"

    held = pd.read_csv(held_path, encoding="utf-8-sig")
    stem = held.loc[
        held["label_hold_reason"].astype(str).str.contains("UNCERTAIN", na=False)
    ].copy()
    if len(stem) != 930:
        raise ValueError(f"Expected 930 uncertain-stem rows, got {len(stem)}")
    if stem["instance_id"].duplicated().any():
        raise ValueError("Duplicate stem instance IDs")

    image_audit = pd.read_csv(image_audit_path, encoding="utf-8-sig")
    geometry = image_audit.loc[image_audit["status"].eq("HOLD")].copy()
    if len(geometry) != 36:
        raise ValueError(f"Expected 36 geometry HOLD images, got {len(geometry)}")

    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    test_manifest = manifest.loc[manifest["split"].eq("test")].copy()
    if len(test_manifest) != 129 or int(test_manifest["instance_count"].sum()) != 582:
        raise ValueError("Unexpected frozen test-set size")

    class_manifest = pd.read_csv(class_manifest_path, encoding="utf-8-sig")
    features = pd.read_csv(feature_path, encoding="utf-8-sig").set_index("instance_id", drop=False)
    reference_frame = pd.read_csv(reference_path, encoding="utf-8-sig")
    references = {
        feature: reference_frame.loc[
            reference_frame["feature"].eq(feature)
            & reference_frame["source_group"].astype(str).eq("ALL")
        ].sort_values("rank")["value"].to_numpy(dtype=float)
        for feature in ("area", "minor_axis", "solidity", "aspect_ratio", "circularity")
    }
    if any(not len(value) for value in references.values()):
        raise ValueError("Missing frozen ECDF references")
    grade_config = json.loads(grade_config_path.read_text(encoding="utf-8"))
    feature_config = yaml.safe_load(feature_config_path.read_text(encoding="utf-8"))
    geometry_config = feature_config["geometry_refinement"]

    stem_groups = [
        {"id": image_id, "weight": len(group)}
        for image_id, group in stem.groupby("image_id", sort=True)
    ]
    stem_assignment = greedy_assign(
        stem_groups,
        {"lead": 280.0, "reviewer_a": 325.0, "reviewer_b": 325.0},
    )
    geometry_assignment = greedy_assign(
        [{"id": row.image_id, "weight": 1} for row in geometry.itertuples()],
        {"lead": 10.0, "reviewer_a": 13.0, "reviewer_b": 13.0},
    )
    # The two independent test-set reviewers are Reviewer A and Reviewer B.
    # The lead does not perform the initial test review and only adjudicates disagreements.
    test_pair_assignment = {
        str(row.new_stem): "a_b" for row in test_manifest.itertuples()
    }

    reviewer_dirs = {key: output / FOLDER_NAMES[key] for key in REVIEWERS}
    for folder in reviewer_dirs.values():
        setup_reviewer_tree(folder)

    task_records: list[dict[str, Any]] = []
    counts = {name: defaultdict(int) for name in REVIEWERS}
    pair_counts_by_reviewer = {name: defaultdict(int) for name in REVIEWERS}

    for image_id, group in stem.groupby("image_id", sort=True):
        reviewer = stem_assignment[str(image_id)]
        for _, row in group.sort_values("instance_id").iterrows():
            preview = make_stem_preview(root, row)
            filename = f"{row['instance_id']}.jpg"
            relative = Path("01_果梗复核") / "待审核" / filename
            write_image(reviewer_dirs[reviewer] / relative, preview)
            counts[reviewer]["stem"] += 1
            task_records.append(
                {
                    "task_type": "stem",
                    "reviewer": reviewer,
                    "image_id": str(row["image_id"]),
                    "instance_id": str(row["instance_id"]),
                    "assignment_pair": None,
                    "preview": f"{FOLDER_NAMES[reviewer]}/{relative.as_posix()}",
                }
            )

    manifest_index = manifest.set_index("new_stem", drop=False)
    for _, audit_row in geometry.sort_values("image_id").iterrows():
        image_id = str(audit_row["image_id"])
        reviewer = geometry_assignment[image_id]
        manifest_row = manifest_index.loc[image_id]
        candidates = class_manifest.loc[class_manifest["image_id"].eq(image_id)].copy()
        preview = make_geometry_preview(root, audit_row, manifest_row, candidates, features)
        filename = f"{image_id}.jpg"
        relative = Path("02_结构异常复核") / "待审核" / filename
        write_image(reviewer_dirs[reviewer] / relative, preview)
        counts[reviewer]["geometry"] += 1
        task_records.append(
            {
                "task_type": "geometry",
                "reviewer": reviewer,
                "image_id": image_id,
                "instance_id": None,
                "assignment_pair": None,
                "reason": str(audit_row["reason"]),
                "preview": f"{FOLDER_NAMES[reviewer]}/{relative.as_posix()}",
            }
        )

    test_score_records: list[dict[str, Any]] = []
    for _, image_row in test_manifest.sort_values("new_stem").iterrows():
        image_id = str(image_row["new_stem"])
        pair = test_pair_assignment[image_id]
        members = PAIR_MEMBERS[pair]
        image_path = root / "data/split_v1/images/test" / f"{image_id}.png"
        label_path = root / "data/split_v1/labels/test" / f"{image_id}.txt"
        image = read_image(image_path)
        polygons = normalized_polygons(label_path, image.shape[1], image.shape[0])
        if len(polygons) != int(image_row["instance_count"]):
            raise ValueError(
                f"Test polygon count mismatch: {image_id}, labels={len(polygons)}, manifest={image_row['instance_count']}"
            )
        for index, polygon in enumerate(polygons, start=1):
            instance_id = f"{image_id}__i{index:03d}"
            feature_values, mask = test_geometry_features(image, polygon, geometry_config)
            score, base_grade, pct = score_test_instance(feature_values, references, grade_config)
            preview = make_test_preview(image, polygon, instance_id, base_grade, score, mask)
            filename = f"{instance_id}__BASE_G{base_grade}.jpg"
            for reviewer in members:
                relative = Path("03_测试集双人复核") / "待审核" / filename
                write_image(reviewer_dirs[reviewer] / relative, preview)
                counts[reviewer]["test_instances"] += 1
                pair_counts_by_reviewer[reviewer][pair] += 1
                task_records.append(
                    {
                        "task_type": "test",
                        "reviewer": reviewer,
                        "image_id": image_id,
                        "instance_id": instance_id,
                        "assignment_pair": pair,
                        "base_grade": base_grade,
                        "score": score,
                        "preview": f"{FOLDER_NAMES[reviewer]}/{relative.as_posix()}",
                    }
                )
            test_score_records.append(
                {
                    "image_id": image_id,
                    "instance_id": instance_id,
                    "assignment_pair": pair,
                    "base_grade": base_grade,
                    "manual_calibrated_score": score,
                    **feature_values,
                    **pct,
                }
            )
        for reviewer in members:
            counts[reviewer]["test_images"] += 1

    for reviewer in REVIEWERS:
        folder = reviewer_dirs[reviewer]
        readme = reviewer_readme(
            reviewer,
            {
                "stem": int(counts[reviewer]["stem"]),
                "geometry": int(counts[reviewer]["geometry"]),
                "test_images": int(counts[reviewer]["test_images"]),
                "test_instances": int(counts[reviewer]["test_instances"]),
            },
            {key: int(value) for key, value in sorted(pair_counts_by_reviewer[reviewer].items())},
        )
        (folder / "README.md").write_text(readme, encoding="utf-8")
        reviewer_manifest = [row for row in task_records if row["reviewer"] == reviewer]
        (folder / "assignment_manifest.json").write_text(
            json.dumps(reviewer_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    stem_total = sum(int(counts[name]["stem"]) for name in REVIEWERS)
    geometry_total = sum(int(counts[name]["geometry"]) for name in REVIEWERS)
    test_assignment_total = sum(int(counts[name]["test_instances"]) for name in REVIEWERS)
    if stem_total != 930 or geometry_total != 36 or test_assignment_total != 1164:
        raise ValueError(
            f"Assignment coverage failed: stem={stem_total}, geometry={geometry_total}, test assignments={test_assignment_total}"
        )

    test_scores_path = output / "test_base_scores.json"
    test_scores_path.write_text(json.dumps(test_score_records, ensure_ascii=False, indent=2), encoding="utf-8")
    master = {
        "package_version": "three_person_audit_v1",
        "review_policy": {
            "stem": "single review; grouped by image",
            "geometry": "single review",
            "test": "Reviewer A and Reviewer B independently review all test instances; lead adjudicates disagreements",
            "lead_workload": "lead has no initial test-set review assignment",
        },
        "source_totals": {
            "uncertain_stem_instances": 930,
            "uncertain_stem_images": int(stem["image_id"].nunique()),
            "geometry_hold_images": 36,
            "test_images": 129,
            "test_instances": 582,
            "test_review_assignments": 1164,
        },
        "reviewer_counts": {
            name: {
                "folder": FOLDER_NAMES[name],
                "stem_instances": int(counts[name]["stem"]),
                "geometry_images": int(counts[name]["geometry"]),
                "test_images": int(counts[name]["test_images"]),
                "test_instances": int(counts[name]["test_instances"]),
            }
            for name in REVIEWERS
        },
        "input_hashes": {
            str(path.relative_to(root)).replace("\\", "/"): sha256(path)
            for path in (
                held_path,
                image_audit_path,
                manifest_path,
                class_manifest_path,
                feature_path,
                reference_path,
                grade_config_path,
            )
        },
    }
    (output / "MASTER_ASSIGNMENT.json").write_text(
        json.dumps(master, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    overview_lines = [
        "# 三人审核分工总览",
        "",
        "本目录包含三份可直接派发的独立审核包。总负责人不参与测试集初审，只负责较少的果梗/结构任务和最终争议裁决。",
        "",
        "## 分工",
        "",
        "| 人员 | 果梗实例 | 结构异常图 | 测试图片 | 测试实例 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in REVIEWERS:
        overview_lines.append(
            f"| {DISPLAY_NAMES[name]} | {counts[name]['stem']} | {counts[name]['geometry']} | "
            f"{counts[name]['test_images']} | {counts[name]['test_instances']} |"
        )
    overview_lines.extend(
        [
            "",
            "## 重要规则",
            "",
            "- 果梗任务与结构异常任务在三人之间不重复。",
            "- 审核员A和审核员B分别独立审核全部129张测试图片、582颗测试实例，共1164次判断。",
            "- 总负责人不参与测试集初审；A、B结果不一致或无法判断的项目由总负责人裁决。",
            "- 所有审核均通过移动图片到分类文件夹完成，不要修改文件名。",
            "- 三个人完成后返回各自完整文件夹，后续可自动汇总结果并重建完整数据集。",
            "",
            "详细标准见每个人文件夹内的 README.md。",
        ]
    )
    (output / "README_分工总览.md").write_text("\n".join(overview_lines) + "\n", encoding="utf-8")
    print(json.dumps(master, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
