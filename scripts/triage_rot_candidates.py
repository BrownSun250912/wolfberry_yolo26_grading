from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml


DECISION_LABELS = {
    "AUTO_ROT_INTERNAL": "Clear internal rot candidate",
    "MANUAL_REVIEW_CALYX": "Manual review: abnormal region near calyx or fruit tip",
    "MANUAL_REVIEW_EDGE": "Manual review: non-calyx edge abnormality",
    "MANUAL_REVIEW_SEGMENTATION": "Manual review: segmentation not confirmed",
    "NO_CLEAR_INTERNAL_ROT": "No clear internal rot candidate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Conservative rot-candidate triage using saved OpenCV/YOLO artifacts."
    )
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def normalized_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).strip().upper()
    if text in {"", "34", "34.0", "NAN", "NONE"}:
        return ""
    if text == "NO0":
        return "NO"
    return text


def resolve_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def load_mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    return image


def largest_component_area(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def largest_component_overlap(mask: np.ndarray, overlap: np.ndarray) -> tuple[int, float]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0, 0.0
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == label
    area = int(np.count_nonzero(component))
    fraction = float(np.count_nonzero(component & overlap) / area) if area else 0.0
    return area, fraction


def make_contact_sheet(paths: list[Path], output_path: Path, columns: int) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    thumb_width, thumb_height = 560, 315
    thumbs = [cv2.resize(image, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA) for image in images]
    rows = (len(thumbs) + columns - 1) // columns
    canvas = np.full((rows * thumb_height, columns * thumb_width, 3), 28, dtype=np.uint8)
    for index, image in enumerate(thumbs):
        row, column = divmod(index, columns)
        y1, x1 = row * thumb_height, column * thumb_width
        canvas[y1 : y1 + thumb_height, x1 : x1 + thumb_width] = image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"Could not write contact sheet: {output_path}")


def overlay_color(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    active = mask > 0
    if not np.any(active):
        return
    layer = np.empty_like(image)
    layer[:] = color
    image[active] = cv2.addWeighted(image, 1.0 - alpha, layer, alpha, 0)[active]


def draw_visualization(
    row: pd.Series,
    masks: dict[str, np.ndarray],
    decision: str,
    output_path: Path,
) -> None:
    source = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f"Could not read source image: {row['image_path']}")

    x1, y1, x2, y2 = (int(row[name]) for name in ("roi_x1", "roi_y1", "roi_x2", "roi_y2"))
    roi = source[y1:y2, x1:x2].copy()
    if roi.shape[:2] != masks["fruit"].shape:
        raise ValueError(f"ROI/mask mismatch for {row['instance_id']}: {roi.shape[:2]} vs {masks['fruit'].shape}")

    annotated = roi.copy()
    overlay_color(annotated, masks["internal"], (190, 0, 190), 0.60)
    overlay_color(annotated, masks["edge"], (255, 190, 0), 0.45)
    overlay_color(annotated, masks["calyx"], (0, 220, 255), 0.60)
    overlay_color(annotated, masks["near_calyx"], (0, 80, 255), 0.75)
    contours, _ = cv2.findContours(masks["fruit"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(annotated, contours, -1, (0, 255, 0), 1)

    crop_h, crop_w = annotated.shape[:2]
    scale = min(420 / max(crop_w, 1), 330 / max(crop_h, 1))
    target = (max(1, int(crop_w * scale)), max(1, int(crop_h * scale)))
    left = cv2.resize(roi, target, interpolation=cv2.INTER_NEAREST)
    right = cv2.resize(annotated, target, interpolation=cv2.INTER_NEAREST)
    panel_h = max(left.shape[0], right.shape[0]) + 105
    panel_w = left.shape[1] + right.shape[1] + 30
    panel = np.full((panel_h, panel_w, 3), 25, dtype=np.uint8)
    panel[75 : 75 + left.shape[0], 5 : 5 + left.shape[1]] = left
    x_right = left.shape[1] + 25
    panel[75 : 75 + right.shape[0], x_right : x_right + right.shape[1]] = right

    title = f"{row.get('review_id', '')}  {row['instance_id']}  {decision}"
    metrics = (
        f"Lstd={float(row['L_std']):.2f}  bstd={float(row['b_std']):.2f}  "
        f"dark={float(row['dark_ratio']):.3f}  internal_max={float(row['max_damage_ratio']):.3f}  "
        f"calyx_adj={float(row['calyx_adjacent_abnormal_ratio']):.3f}"
    )
    cv2.putText(panel, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(panel, metrics, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "Original ROI", (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "green=fruit  magenta=internal  yellow=calyx  orange=near-calyx/tip  cyan=edge",
        (8, panel_h - 9),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), panel):
        raise IOError(f"Could not write visualization: {output_path}")


def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    evaluated = frame.loc[frame["ground_truth_available"]].copy()
    if evaluated.empty:
        return {
            "evaluation_available": False,
            "evaluated_instances": 0,
            "decision_counts": dict(Counter(frame["decision"])),
            "note": "No manual rot labels were supplied for this full-data run.",
        }
    y = evaluated["ground_truth_rot"].astype(bool)
    auto = evaluated["decision"].eq("AUTO_ROT_INTERNAL")
    routed = evaluated["decision"].isin(
        ["AUTO_ROT_INTERNAL", "MANUAL_REVIEW_CALYX", "MANUAL_REVIEW_EDGE"]
    )

    def metrics(predicted: pd.Series) -> dict[str, float | int | None]:
        tp = int((predicted & y).sum())
        fp = int((predicted & ~y).sum())
        fn = int((~predicted & y).sum())
        tn = int((~predicted & ~y).sum())
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None,
        }

    return {
        "evaluation_available": True,
        "evaluated_instances": int(len(evaluated)),
        "ground_truth_rot_yes": int(y.sum()),
        "ground_truth_rot_no": int((~y).sum()),
        "decision_counts": dict(Counter(evaluated["decision"])),
        "automatic_internal_rot": metrics(auto),
        "automatic_plus_location_review_queue": metrics(routed),
        "important_interpretation": [
            "Automatic precision is prioritized over recall; missed rot is expected.",
            "The manual queue is not a positive prediction and its precision is not a rot accuracy claim.",
            "These 240 samples were stratified for audit and are not prevalence-representative.",
            "Thresholds were selected with this reviewed set and require independent confirmation before final labeling.",
        ],
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    input_cfg = config["input"]
    output_root = resolve_path(base, config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)

    tables = []
    artifact_roots: dict[str, Path] = {}
    for split, table_path in input_cfg["feature_tables"].items():
        table = pd.read_csv(resolve_path(base, table_path), encoding="utf-8-sig")
        table["split"] = split
        tables.append(table)
        artifact_roots[split] = resolve_path(base, input_cfg["artifact_roots"][split])
    features = pd.concat(tables, ignore_index=True)

    scope = input_cfg["scope"]
    if scope == "reviewed_240":
        candidates = pd.read_csv(resolve_path(base, input_cfg["review_candidates"]), encoding="utf-8-sig")
        label_payload = json.loads(resolve_path(base, input_cfg["review_labels_json"]).read_text(encoding="utf-8"))
        labels = pd.DataFrame(label_payload["rows"])
        for name in ["segmentation_usable", "quality_status", "rot"]:
            labels[name] = labels[name].map(normalized_text)
        review = candidates.merge(
            labels[["review_id", "instance_id", "segmentation_usable", "quality_status", "rot"]],
            on=["review_id", "instance_id"],
            how="left",
            validate="one_to_one",
        )
        frame = review[["review_id", "instance_id", "segmentation_usable", "quality_status", "rot"]].merge(
            features,
            on="instance_id",
            how="left",
            validate="one_to_one",
        )
    elif scope == "all_instances":
        frame = features.copy()
        frame.insert(0, "review_id", frame["instance_id"])
        frame["segmentation_usable"] = ""
        frame["quality_status"] = ""
        frame["rot"] = ""
    else:
        raise ValueError(f"Unsupported input.scope={scope!r}")
    if frame["image_path"].isna().any():
        missing = frame.loc[frame["image_path"].isna(), "instance_id"].tolist()
        raise ValueError(f"Missing feature rows: {missing[:10]}")

    auto_cfg = config["auto_internal_rot"]
    calyx_cfg = config["manual_calyx_review"]
    edge_cfg = config["manual_edge_review"]
    results: list[dict[str, Any]] = []
    visual_masks: dict[str, dict[str, np.ndarray]] = {}

    for _, source_row in frame.iterrows():
        row = source_row.copy()
        split = str(row["split"])
        artifact_root = artifact_roots[split]
        fruit = load_mask(artifact_root / str(row["mask_path"])) > 0
        damage_encoded = load_mask(artifact_root / str(row["damage_mask_path"]))
        internal = damage_encoded > 0
        internal_dark = (damage_encoded == 85) | (damage_encoded == 255)
        calyx = load_mask(artifact_root / str(row["calyx_mask_path"])) > 0
        edge = load_mask(artifact_root / str(row["edge_abnormal_mask_path"])) > 0

        radius = int(calyx_cfg["neighborhood_radius_px"])
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        calyx_neighborhood = cv2.dilate(calyx.astype(np.uint8), kernel, iterations=1) > 0 if radius else calyx
        yy, xx = np.indices(fruit.shape, dtype=np.float32)
        xx += float(row["roi_x1"])
        yy += float(row["roi_y1"])
        angle = np.deg2rad(float(row["major_angle_deg"]))
        projection = np.abs(
            (xx - float(row["axis_center_x"])) * np.cos(angle)
            + (yy - float(row["axis_center_y"])) * np.sin(angle)
        )
        tip_zone = fruit & (
            projection >= float(calyx_cfg["tip_fraction"]) * float(row["major_axis"]) / 2.0
        )
        protected_calyx_or_tip = calyx_neighborhood | tip_zone
        near_calyx = (edge | internal) & protected_calyx_or_tip
        internal_far_calyx = internal & ~protected_calyx_or_tip
        edge_noncalyx = edge & ~protected_calyx_or_tip
        area = max(int(np.count_nonzero(fruit)), 1)
        internal_largest_px = largest_component_area(internal)
        internal_far_calyx_largest_px, internal_largest_dark_overlap_fraction = largest_component_overlap(
            internal_far_calyx, internal_dark
        )
        internal_far_calyx_largest_ratio = internal_far_calyx_largest_px / area
        internal_pixels = int(np.count_nonzero(internal))
        internal_calyx_overlap_fraction = (
            int(np.count_nonzero(internal & protected_calyx_or_tip)) / internal_pixels
            if internal_pixels else 0.0
        )
        calyx_adjacent_px = int(np.count_nonzero(near_calyx))
        calyx_adjacent_ratio = calyx_adjacent_px / area
        noncalyx_edge_ratio = int(np.count_nonzero(edge_noncalyx)) / area

        segmentation_ok = normalized_text(row["segmentation_usable"]) == "YES"
        strong_internal_signal = (
            float(row["b_std"]) >= float(auto_cfg["b_std_min"])
            and float(row["dark_ratio"]) >= float(auto_cfg["dark_ratio_min"])
            and float(row["max_damage_ratio"]) >= float(auto_cfg["max_damage_ratio_min"])
        )
        auto_internal = (
            strong_internal_signal
            and internal_far_calyx_largest_px >= int(auto_cfg["min_internal_candidate_area_px"])
            and internal_far_calyx_largest_ratio >= float(auto_cfg["min_far_from_calyx_component_ratio"])
            and internal_largest_dark_overlap_fraction
            >= float(auto_cfg["min_largest_component_dark_overlap_fraction"])
            and internal_calyx_overlap_fraction <= float(auto_cfg["max_calyx_overlap_fraction"])
        )
        calyx_review = (
            bool(calyx_cfg["enabled"])
            and (
                strong_internal_signal
                or not bool(calyx_cfg.get("require_auto_signal_thresholds", True))
            )
            and calyx_adjacent_px >= int(calyx_cfg["min_adjacent_abnormal_area_px"])
            and calyx_adjacent_ratio >= float(calyx_cfg["min_adjacent_abnormal_ratio"])
        )
        heterogeneous = (
            float(row["L_std"]) >= float(edge_cfg["heterogeneity_any"]["L_std_min"])
            or float(row["b_std"]) >= float(edge_cfg["heterogeneity_any"]["b_std_min"])
        )
        edge_review = (
            bool(edge_cfg["enabled"])
            and noncalyx_edge_ratio >= float(edge_cfg["edge_abnormal_ratio_min"])
            and heterogeneous
        )

        if bool(config["evaluation"]["require_segmentation_usable"]) and not segmentation_ok:
            decision = "MANUAL_REVIEW_SEGMENTATION"
        elif auto_internal:
            decision = "AUTO_ROT_INTERNAL"
        elif calyx_review:
            decision = "MANUAL_REVIEW_CALYX"
        elif edge_review:
            decision = "MANUAL_REVIEW_EDGE"
        else:
            decision = "NO_CLEAR_INTERNAL_ROT"

        quality = normalized_text(row["quality_status"])
        rot = normalized_text(row["rot"])
        gt_available = quality in {"QUALIFIED", "UNQUALIFIED"}
        gt_rot = rot == normalized_text(config["evaluation"]["positive_rot_value"])
        result = row.to_dict()
        result.update(
            {
                "decision": decision,
                "decision_description": DECISION_LABELS[decision],
                "internal_candidate_largest_px": internal_largest_px,
                "internal_far_calyx_largest_px": internal_far_calyx_largest_px,
                "internal_far_calyx_largest_ratio": internal_far_calyx_largest_ratio,
                "internal_largest_dark_overlap_fraction": internal_largest_dark_overlap_fraction,
                "internal_calyx_overlap_fraction": internal_calyx_overlap_fraction,
                "calyx_adjacent_abnormal_px": calyx_adjacent_px,
                "calyx_adjacent_abnormal_ratio": calyx_adjacent_ratio,
                "noncalyx_edge_abnormal_ratio": noncalyx_edge_ratio,
                "ground_truth_available": gt_available,
                "ground_truth_rot": gt_rot,
                "auto_rule_pass": auto_internal,
                "calyx_review_rule_pass": calyx_review,
                "edge_review_rule_pass": edge_review,
            }
        )
        results.append(result)
        visual_masks[str(row["instance_id"])] = {
            "fruit": fruit,
            "internal": internal_far_calyx,
            "calyx": calyx,
            "edge": edge_noncalyx,
            "near_calyx": near_calyx,
        }

    result_frame = pd.DataFrame(results)
    priority_columns = [
        "review_id", "image_id", "instance_id", "split", "decision", "decision_description",
        "segmentation_usable", "quality_status", "rot", "ground_truth_rot",
        "L_std", "b_std", "dark_ratio", "max_damage_ratio",
        "internal_candidate_largest_px", "calyx_adjacent_abnormal_px",
        "internal_far_calyx_largest_px", "internal_far_calyx_largest_ratio",
        "internal_largest_dark_overlap_fraction", "internal_calyx_overlap_fraction",
        "calyx_adjacent_abnormal_ratio", "noncalyx_edge_abnormal_ratio",
    ]
    remaining = [column for column in result_frame.columns if column not in priority_columns]
    ordered_frame = result_frame[priority_columns + remaining]
    ordered_frame.to_csv(
        output_root / "rot_triage_results.csv", index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL
    )
    queue_files = {
        "auto_internal_rot_candidates.csv": ["AUTO_ROT_INTERNAL"],
        "manual_review_calyx.csv": ["MANUAL_REVIEW_CALYX"],
        "manual_review_edge.csv": ["MANUAL_REVIEW_EDGE"],
        "manual_review_segmentation.csv": ["MANUAL_REVIEW_SEGMENTATION"],
    }
    for filename, decisions in queue_files.items():
        ordered_frame.loc[ordered_frame["decision"].isin(decisions)].to_csv(
            output_root / filename,
            index=False,
            encoding="utf-8-sig",
            quoting=csv.QUOTE_MINIMAL,
        )

    summary = evaluate(result_frame)
    summary["config_version"] = config["config_version"]
    summary["scope"] = scope
    summary["output_rows"] = int(len(result_frame))
    (output_root / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    visual_root = output_root / "visualizations"
    limit = int(config["output"]["visualization_limit_per_decision"])
    sort_columns = {
        "AUTO_ROT_INTERNAL": "max_damage_ratio",
        "MANUAL_REVIEW_CALYX": "calyx_adjacent_abnormal_ratio",
        "MANUAL_REVIEW_EDGE": "noncalyx_edge_abnormal_ratio",
        "MANUAL_REVIEW_SEGMENTATION": "max_damage_ratio",
    }
    for decision, sort_column in sort_columns.items():
        subset = result_frame.loc[result_frame["decision"].eq(decision)].sort_values(sort_column, ascending=False).head(limit)
        paths: list[Path] = []
        for _, row in subset.iterrows():
            output_path = visual_root / decision.lower() / f"{row['review_id']}_{row['instance_id']}.png"
            draw_visualization(row, visual_masks[str(row["instance_id"])], decision, output_path)
            paths.append(output_path)
        make_contact_sheet(
            paths,
            visual_root / f"{decision.lower()}_contact_sheet.png",
            int(config["output"]["contact_sheet_columns"]),
        )

    readme = f"""# Rot candidate triage pilot v1

This output is for assisted labeling, not a final grading standard.

- `AUTO_ROT_INTERNAL`: conservative internal-rot candidate. It passed all configured within-fruit thresholds.
- `MANUAL_REVIEW_CALYX`: abnormal pixels were adjacent to a calyx candidate or either longitudinal fruit tip; inspect manually.
- `MANUAL_REVIEW_EDGE`: a non-calyx edge abnormality may be mask/bowl contact; inspect manually.
- `MANUAL_REVIEW_SEGMENTATION`: the earlier audit did not confirm the mask as usable.
- `NO_CLEAR_INTERNAL_ROT`: no strong internal candidate was found. This does **not** prove absence of rot.

The automatic rule requires `b_std >= {auto_cfg['b_std_min']}`, `dark_ratio >= {auto_cfg['dark_ratio_min']}`,
`max_damage_ratio >= {auto_cfg['max_damage_ratio_min']}`, and an internal connected component of at least
`{auto_cfg['min_internal_candidate_area_px']}` pixels away from the dilated calyx neighborhood. The far-from-calyx
component must cover at least `{auto_cfg['min_far_from_calyx_component_ratio']}` of the fruit, while no more than
`{auto_cfg['max_calyx_overlap_fraction']}` of internal abnormal pixels may overlap the calyx neighborhood.
At least `{auto_cfg['min_largest_component_dark_overlap_fraction']}` of that largest component must itself be dark.
All measurements are calculated inside the refined fruit mask;
the existing high-light, calyx and boundary exclusions are inherited from feature extraction.

See `evaluation_summary.json` for the reviewed-set comparison and its limitations.

Queue files are written separately as `auto_internal_rot_candidates.csv`, `manual_review_calyx.csv`,
`manual_review_edge.csv`, and `manual_review_segmentation.csv`. The complete table remains in
`rot_triage_results.csv`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
