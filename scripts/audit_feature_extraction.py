from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CORE_FEATURES = [
    "major_axis", "minor_axis", "area", "aspect_ratio",
    "L_mean", "L_std", "a_mean", "a_std", "b_mean", "b_std", "S_mean", "V_mean",
    "circularity", "solidity", "eccentricity",
    "dark_ratio", "abnormal_color_ratio", "max_damage_ratio",
]

RATIO_COLUMNS = [
    "circularity", "solidity", "eccentricity", "highlight_ratio",
    "calyx_candidate_ratio", "edge_abnormal_ratio",
    "dark_ratio", "abnormal_color_ratio", "max_damage_ratio",
]

ARTIFACT_COLUMNS = [
    "raw_mask_path", "mask_path", "border_removed_mask_path", "crop_path", "contour_path",
    "color_mask_path", "highlight_mask_path", "calyx_mask_path", "edge_abnormal_mask_path",
    "damage_eligible_mask_path", "damage_mask_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a full feature-extraction split.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels-dir", required=True, type=Path)
    parser.add_argument("--split", required=True)
    return parser.parse_args()


def label_counts(path: Path) -> tuple[int, int, int]:
    if not path.is_file():
        return 0, 0, 0
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unique_lines = list(dict.fromkeys(lines))
    return len(lines), len(unique_lines), len(lines) - len(unique_lines)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    feature_path = output_dir / args.features
    image_log_path = output_dir / "image_processing_log.csv"
    labels_dir = args.labels_dir.resolve()
    if not feature_path.is_file() or not image_log_path.is_file() or not labels_dir.is_dir():
        raise FileNotFoundError("Feature CSV, image log, or label directory is missing.")

    features = pd.read_csv(feature_path)
    image_log = pd.read_csv(image_log_path)
    feature_counts = features.groupby("image_id").size().rename("feature_instance_count")
    audit = image_log.merge(feature_counts, how="left", left_on="image_id", right_index=True)
    audit["feature_instance_count"] = audit["feature_instance_count"].fillna(0).astype(int)

    counts = audit["image_id"].map(lambda image_id: label_counts(labels_dir / f"{image_id}.txt"))
    audit["gt_annotation_lines"] = counts.map(lambda value: value[0])
    audit["gt_unique_instances"] = counts.map(lambda value: value[1])
    audit["gt_duplicate_lines"] = counts.map(lambda value: value[2])
    audit["count_delta_vs_unique_gt"] = audit["feature_instance_count"] - audit["gt_unique_instances"]
    audit["count_match_status"] = np.select(
        [audit["count_delta_vs_unique_gt"] < 0, audit["count_delta_vs_unique_gt"] > 0],
        ["under_count", "over_count"],
        default="exact_count",
    )
    audit.to_csv(output_dir / f"{args.split}_instance_count_audit.csv", index=False, encoding="utf-8-sig")

    numeric = features.loc[:, CORE_FEATURES].apply(pd.to_numeric, errors="coerce")
    missing_core_values = int(numeric.isna().sum().sum())
    duplicate_instance_ids = int(features["instance_id"].duplicated().sum())
    invalid_axes = int(
        ((numeric["major_axis"] < numeric["minor_axis"]) | (numeric["minor_axis"] <= 0)).sum()
    )
    invalid_aspect = int((numeric["aspect_ratio"] < 1).sum())
    invalid_ratios = 0
    for column in RATIO_COLUMNS:
        values = pd.to_numeric(features[column], errors="coerce")
        invalid_ratios += int(((values < 0) | (values > 1) | values.isna()).sum())

    missing_artifacts: list[dict[str, str]] = []
    for _, row in features.iterrows():
        for column in ARTIFACT_COLUMNS:
            path = output_dir / str(row[column])
            if not path.is_file():
                missing_artifacts.append(
                    {"instance_id": str(row["instance_id"]), "column": column, "path": str(path)}
                )

    millimetre_columns = ["major_axis_mm", "minor_axis_mm", "area_mm2"]
    nonempty_mm_values = int(
        features.loc[:, millimetre_columns].apply(pd.to_numeric, errors="coerce").notna().sum().sum()
    )
    match_counts = audit["count_match_status"].value_counts()
    status_counts = image_log["status"].value_counts(dropna=False)
    refinement_counts = features["geometry_refinement_status"].value_counts(dropna=False)
    confidence = pd.to_numeric(features["confidence"], errors="coerce")

    summary = {
        "split": args.split,
        "feature_path": str(feature_path),
        "labels_dir": str(labels_dir),
        "images_in_log": int(len(image_log)),
        "image_status_counts": {str(key): int(value) for key, value in status_counts.items()},
        "feature_rows": int(len(features)),
        "unique_instance_ids": int(features["instance_id"].nunique()),
        "duplicate_instance_ids": duplicate_instance_ids,
        "successful_instances": int(features["processing_success"].fillna(False).sum()),
        "failed_instances": int((~features["processing_success"].fillna(False)).sum()),
        "missing_core_values": missing_core_values,
        "invalid_axes": invalid_axes,
        "invalid_aspect_ratio": invalid_aspect,
        "invalid_ratio_values": invalid_ratios,
        "artifact_references_checked": int(len(features) * len(ARTIFACT_COLUMNS)),
        "missing_artifacts": int(len(missing_artifacts)),
        "ground_truth_annotation_lines": int(audit["gt_annotation_lines"].sum()),
        "ground_truth_unique_instances": int(audit["gt_unique_instances"].sum()),
        "ground_truth_duplicate_lines": int(audit["gt_duplicate_lines"].sum()),
        "predicted_feature_instances": int(audit["feature_instance_count"].sum()),
        "image_count_match": {str(key): int(value) for key, value in match_counts.items()},
        "total_under_count": int(-audit.loc[audit["count_delta_vs_unique_gt"] < 0, "count_delta_vs_unique_gt"].sum()),
        "total_over_count": int(audit.loc[audit["count_delta_vs_unique_gt"] > 0, "count_delta_vs_unique_gt"].sum()),
        "confidence": {
            "min": float(confidence.min()),
            "median": float(confidence.median()),
            "mean": float(confidence.mean()),
            "max": float(confidence.max()),
        },
        "geometry_refinement_status": {str(key): int(value) for key, value in refinement_counts.items()},
        "calibration_ids": sorted(features["calibration_id"].dropna().astype(str).unique().tolist()),
        "nonempty_millimetre_values": nonempty_mm_values,
        "note": "Ground-truth comparison is count-only; it is not spatial instance matching or segmentation accuracy.",
    }
    (output_dir / f"{args.split}_extraction_quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(missing_artifacts).to_csv(
        output_dir / f"{args.split}_missing_artifacts.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
