from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 20260902
BOUNDARY_HIGH = "boundary_grade1_2"
BOUNDARY_LOW = "boundary_grade2_3"
STEM_MISSING_PATTERN = r"缺失果柄|没果柄"
MEASUREMENT_ERROR_PATTERN = r"碗边|框的有点问题"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the interpretable grading rule from completed boundary and stem reviews."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=Path("outputs/size_priority_stem_rule_v2/size_priority_stem_rule_candidates.csv"),
    )
    parser.add_argument(
        "--boundary-review",
        type=Path,
        default=Path("outputs/size_priority_v1/boundary_review.xlsx"),
    )
    parser.add_argument(
        "--stem-review",
        type=Path,
        default=Path("outputs/size_priority_stem_rule_v2/stem_review.xlsx"),
    )
    parser.add_argument(
        "--all-features-csv",
        type=Path,
        default=Path("outputs/strict_rot_exclusion_v1/features_with_manual_rot.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grading_standard_manual_v3"),
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--scheme-version",
        default="grading_standard_manual_v3",
        help="Version identifier recorded in the machine-readable grading configuration.",
    )
    parser.add_argument(
        "--accept-unreviewed-auto-missing",
        action="store_true",
        help="Treat unreviewed automatic MISSING results as grade-3 overrides after prior manual validation.",
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_unique(data: pd.DataFrame, name: str) -> None:
    duplicated = data.duplicated(["image_id", "instance_id"], keep=False)
    if duplicated.any():
        examples = data.loc[duplicated, ["image_id", "instance_id"]].head(5).to_dict("records")
        raise ValueError(f"{name} contains duplicate instance keys: {examples}")


def normalize_stem(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def score_from_weights(data: pd.DataFrame, weights: dict[str, float]) -> np.ndarray:
    return (
        100.0 * weights["pct_area"] * data["pct_area"].to_numpy(dtype=float)
        + 100.0 * weights["pct_minor_axis"] * data["pct_minor_axis"].to_numpy(dtype=float)
        + weights["shape_score"] * data["shape_score"].to_numpy(dtype=float)
    )


def fit_threshold(scores: np.ndarray, target_high: np.ndarray) -> dict[str, float | int]:
    scores = np.asarray(scores, dtype=float)
    target_high = np.asarray(target_high, dtype=bool)
    if len(scores) == 0 or target_high.all() or (~target_high).all():
        raise ValueError("Threshold calibration requires both classes.")
    unique = np.unique(scores)
    candidates = np.concatenate(
        ([unique[0] - 1e-9], (unique[:-1] + unique[1:]) / 2.0, [unique[-1] + 1e-9])
    )
    class_midpoint = float((np.median(scores[target_high]) + np.median(scores[~target_high])) / 2.0)
    best: tuple[tuple[float, float, float], dict[str, float | int]] | None = None
    for threshold in candidates:
        predicted = scores >= threshold
        tp = int(np.count_nonzero(predicted & target_high))
        tn = int(np.count_nonzero(~predicted & ~target_high))
        fp = int(np.count_nonzero(predicted & ~target_high))
        fn = int(np.count_nonzero(~predicted & target_high))
        sensitivity = tp / max(int(np.count_nonzero(target_high)), 1)
        specificity = tn / max(int(np.count_nonzero(~target_high)), 1)
        balanced_accuracy = (sensitivity + specificity) / 2.0
        accuracy = (tp + tn) / len(scores)
        record: dict[str, float | int] = {
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy),
            "accuracy": float(accuracy),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "n": int(len(scores)),
        }
        key = (balanced_accuracy, accuracy, -abs(float(threshold) - class_midpoint))
        if best is None or key > best[0]:
            best = (key, record)
    assert best is not None
    return best[1]


def prepare_boundary_review(boundary: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "review_id",
        "candidate_group",
        "image_id",
        "instance_id",
        "review_expected_grade",
        "review_comment",
    }
    missing = sorted(required - set(boundary.columns))
    if missing:
        raise ValueError(f"Boundary review missing columns: {missing}")
    validate_unique(boundary, "boundary review")
    boundary = boundary.copy()
    boundary["manual_expected_grade"] = pd.to_numeric(
        boundary["review_expected_grade"], errors="coerce"
    ).astype("Int64")
    invalid = boundary["manual_expected_grade"].isna() | ~boundary["manual_expected_grade"].isin([1, 2, 3])
    if invalid.any():
        raise ValueError(
            "Boundary review has blank/invalid expected grades: "
            + ", ".join(boundary.loc[invalid, "review_id"].astype(str).tolist())
        )
    comments = boundary["review_comment"].fillna("").astype(str)
    boundary["manual_stem_missing_comment"] = comments.str.contains(STEM_MISSING_PATTERN, regex=True)
    boundary["manual_measurement_error"] = comments.str.contains(MEASUREMENT_ERROR_PATTERN, regex=True)
    boundary["calibration_eligible"] = ~(
        boundary["manual_stem_missing_comment"] | boundary["manual_measurement_error"]
    )
    feature_columns = [
        "image_id",
        "instance_id",
        "pct_area",
        "pct_minor_axis",
        "shape_score",
        "size_priority_score",
        "stem_auto_status",
    ]
    boundary = boundary.merge(
        candidates[feature_columns],
        on=["image_id", "instance_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_candidate"),
        indicator=True,
    )
    if not boundary["_merge"].eq("both").all():
        missing_ids = boundary.loc[~boundary["_merge"].eq("both"), "instance_id"].astype(str).tolist()
        raise ValueError(f"Boundary review instances missing from candidate table: {missing_ids}")
    return boundary.drop(columns="_merge")


def choose_weights(boundary: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, float | int]], pd.DataFrame]:
    calibration = boundary.loc[boundary["calibration_eligible"]].copy()
    rows: list[dict[str, float]] = []
    best_key: tuple[float, float, float, float] | None = None
    best_weights: dict[str, float] | None = None
    best_metrics: dict[str, dict[str, float | int]] | None = None
    for area_percent in range(40, 91, 5):
        for minor_percent in range(5, 41, 5):
            shape_percent = 100 - area_percent - minor_percent
            if not 5 <= shape_percent <= 30:
                continue
            weights = {
                "pct_area": area_percent / 100.0,
                "pct_minor_axis": minor_percent / 100.0,
                "shape_score": shape_percent / 100.0,
                "color_uniformity_score": 0.0,
            }
            trial_score = score_from_weights(calibration, weights)
            metrics: dict[str, dict[str, float | int]] = {}
            for group in (BOUNDARY_HIGH, BOUNDARY_LOW):
                mask = calibration["candidate_group"].eq(group).to_numpy()
                grades = calibration.loc[mask, "manual_expected_grade"]
                target = grades.eq(1).to_numpy() if group == BOUNDARY_HIGH else grades.ne(3).to_numpy()
                metrics[group] = fit_threshold(trial_score[mask], target)
            mean_balanced_accuracy = float(
                np.mean([metrics[BOUNDARY_HIGH]["balanced_accuracy"], metrics[BOUNDARY_LOW]["balanced_accuracy"]])
            )
            mean_accuracy = float(np.mean([metrics[BOUNDARY_HIGH]["accuracy"], metrics[BOUNDARY_LOW]["accuracy"]]))
            size_weight = weights["pct_area"] + weights["pct_minor_axis"]
            key = (mean_balanced_accuracy, mean_accuracy, size_weight, weights["pct_area"])
            rows.append(
                {
                    **weights,
                    "mean_balanced_accuracy": mean_balanced_accuracy,
                    "mean_accuracy": mean_accuracy,
                    "grade1_2_balanced_accuracy": float(metrics[BOUNDARY_HIGH]["balanced_accuracy"]),
                    "grade2_3_balanced_accuracy": float(metrics[BOUNDARY_LOW]["balanced_accuracy"]),
                    "grade1_2_threshold": float(metrics[BOUNDARY_HIGH]["threshold"]),
                    "grade2_3_threshold": float(metrics[BOUNDARY_LOW]["threshold"]),
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best_weights = weights
                best_metrics = metrics
    assert best_weights is not None and best_metrics is not None
    grid = pd.DataFrame(rows).sort_values(
        ["mean_balanced_accuracy", "mean_accuracy", "pct_area"], ascending=[False, False, False]
    )
    return best_weights, best_metrics, grid


def bootstrap_thresholds(
    boundary: pd.DataFrame,
    weights: dict[str, float],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    calibration = boundary.loc[boundary["calibration_eligible"]].copy()
    calibration["manual_calibrated_score"] = score_from_weights(calibration, weights)
    rows: list[dict[str, object]] = []
    for group in (BOUNDARY_HIGH, BOUNDARY_LOW):
        subset = calibration.loc[calibration["candidate_group"].eq(group)].copy()
        subset["target_high"] = (
            subset["manual_expected_grade"].eq(1)
            if group == BOUNDARY_HIGH
            else subset["manual_expected_grade"].ne(3)
        )
        high = subset.loc[subset["target_high"]]
        low = subset.loc[~subset["target_high"]]
        values: list[float] = []
        for _ in range(iterations):
            sample = pd.concat(
                [
                    high.iloc[rng.integers(0, len(high), len(high))],
                    low.iloc[rng.integers(0, len(low), len(low))],
                ],
                ignore_index=True,
            )
            fitted = fit_threshold(
                sample["manual_calibrated_score"].to_numpy(dtype=float),
                sample["target_high"].to_numpy(dtype=bool),
            )
            values.append(float(fitted["threshold"]))
        rows.append(
            {
                "boundary": "grade1_2" if group == BOUNDARY_HIGH else "grade2_3",
                "iterations": iterations,
                "threshold_mean": float(np.mean(values)),
                "threshold_std": float(np.std(values, ddof=1)),
                "threshold_p2_5": float(np.quantile(values, 0.025)),
                "threshold_median": float(np.median(values)),
                "threshold_p97_5": float(np.quantile(values, 0.975)),
            }
        )
    return pd.DataFrame(rows)


def prepare_stem_review(stem: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    required = {"review_id", "image_id", "instance_id", "stem_auto_status", "review_stem_present"}
    missing = sorted(required - set(stem.columns))
    if missing:
        raise ValueError(f"Stem review missing columns: {missing}")
    validate_unique(stem, "stem review")
    stem = stem.copy()
    stem["manual_stem_present"] = stem["review_stem_present"].map(normalize_stem)
    invalid = ~stem["manual_stem_present"].isin(["YES", "NO", "UNCERTAIN"])
    if invalid.any():
        raise ValueError(
            "Stem review has blank/invalid values: "
            + ", ".join(stem.loc[invalid, "review_id"].astype(str).tolist())
        )
    keys = candidates[["image_id", "instance_id"]]
    checked = stem.merge(keys, on=["image_id", "instance_id"], how="left", indicator=True)
    if not checked["_merge"].eq("both").all():
        missing_ids = checked.loc[~checked["_merge"].eq("both"), "instance_id"].astype(str).tolist()
        raise ValueError(f"Stem review instances missing from candidate table: {missing_ids}")
    return stem


def add_stem_decisions(
    candidates: pd.DataFrame,
    boundary: pd.DataFrame,
    stem: pd.DataFrame,
    accept_unreviewed_auto_missing: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = candidates.copy()
    stem_manual = stem[
        ["review_id", "image_id", "instance_id", "manual_stem_present", "review_group", "review_comment"]
    ].rename(
        columns={
            "review_id": "stem_review_id",
            "review_group": "stem_review_group",
            "review_comment": "stem_review_comment",
        }
    )
    data = data.merge(stem_manual, on=["image_id", "instance_id"], how="left", validate="one_to_one")
    boundary_missing = boundary.loc[
        boundary["manual_stem_missing_comment"], ["review_id", "image_id", "instance_id"]
    ].rename(columns={"review_id": "boundary_stem_review_id"})
    data = data.merge(boundary_missing, on=["image_id", "instance_id"], how="left", validate="one_to_one")

    conflict = data["boundary_stem_review_id"].notna() & data["manual_stem_present"].eq("YES")
    if conflict.any():
        raise ValueError(
            "Boundary and stem reviews conflict for: "
            + ", ".join(data.loc[conflict, "instance_id"].astype(str).tolist())
        )

    boundary_only_missing = data["boundary_stem_review_id"].notna() & data["manual_stem_present"].isna()
    data.loc[boundary_only_missing, "manual_stem_present"] = "NO"
    data["stem_manual_source"] = ""
    data.loc[data["stem_review_id"].notna(), "stem_manual_source"] = "stem_review.xlsx"
    data.loc[boundary_only_missing, "stem_manual_source"] = "boundary_review_comment"

    data["stem_decision"] = "UNRESOLVED"
    data.loc[data["manual_stem_present"].eq("NO"), "stem_decision"] = "MISSING_CONFIRMED"
    data.loc[data["manual_stem_present"].eq("YES"), "stem_decision"] = "PRESENT_CONFIRMED"
    auto_present = data["manual_stem_present"].isna() & data["stem_auto_status"].eq("PRESENT")
    data.loc[auto_present, "stem_decision"] = "PRESENT_AUTO_QC"
    if accept_unreviewed_auto_missing:
        auto_missing = data["manual_stem_present"].isna() & data["stem_auto_status"].eq("MISSING")
        data.loc[auto_missing, "stem_decision"] = "MISSING_AUTO_PRIOR_VALIDATED"
    data["stem_confirmed_missing"] = data["stem_decision"].isin(
        ["MISSING_CONFIRMED", "MISSING_AUTO_PRIOR_VALIDATED"]
    )
    data["stem_unresolved"] = data["stem_decision"].eq("UNRESOLVED")

    audit_columns = [
        "image_id",
        "instance_id",
        "stem_auto_status",
        "stem_review_id",
        "boundary_stem_review_id",
        "manual_stem_present",
        "stem_manual_source",
        "stem_decision",
        "stem_review_group",
        "stem_review_comment",
    ]
    audit = data.loc[data["manual_stem_present"].notna(), audit_columns].copy()
    return data, audit


def build_distributions(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_name, subset in [("all", data), *[(str(k), v) for k, v in data.groupby("split")]]:
        for readiness, ready_subset in subset.groupby("label_readiness"):
            counts = ready_subset["recommended_candidate_grade"].value_counts()
            for grade in (1, 2, 3):
                rows.append(
                    {
                        "split": split_name,
                        "label_readiness": readiness,
                        "grade": grade,
                        "count": int(counts.get(grade, 0)),
                        "total_in_readiness_group": int(len(ready_subset)),
                    }
                )
    return pd.DataFrame(rows)


def build_image_level_manifest(
    all_features: pd.DataFrame,
    graded: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_unique(all_features, "all-features table")
    source_columns = [
        column
        for column in [
            "image_id",
            "instance_id",
            "split",
            "source_group",
            "image_path",
            "mask_path",
            "crop_path",
            "manual_rot_label",
            "manual_rot_exclude",
        ]
        if column in all_features.columns
    ]
    source = all_features[source_columns].copy()
    source["is_unqualified"] = (
        source["manual_rot_exclude"].astype(str).str.strip().str.upper().isin(["TRUE", "1", "YES", "Y"])
    )
    grade_columns = [
        "image_id",
        "instance_id",
        "recommended_candidate_grade",
        "label_readiness",
        "label_source_v3",
        "stem_decision",
    ]
    merged = source.merge(
        graded[grade_columns], on=["image_id", "instance_id"], how="left", validate="one_to_one"
    )
    overlap = merged["is_unqualified"] & merged["recommended_candidate_grade"].notna()
    if overlap.any():
        raise ValueError("An instance cannot be both strict-rot excluded and qualified graded.")
    merged["is_ready_qualified"] = merged["label_readiness"].eq("READY_CANDIDATE")
    merged["instance_accounted"] = merged["is_unqualified"] | merged["is_ready_qualified"]

    rows: list[dict[str, object]] = []
    for image_id, group in merged.groupby("image_id", sort=True):
        grades = group.loc[group["is_ready_qualified"], "recommended_candidate_grade"]
        rows.append(
            {
                "image_id": image_id,
                "split": str(group["split"].iloc[0]),
                "total_instances": int(len(group)),
                "unqualified_instances": int(group["is_unqualified"].sum()),
                "grade1_instances": int(grades.eq(1).sum()),
                "grade2_instances": int(grades.eq(2).sum()),
                "grade3_instances": int(grades.eq(3).sum()),
                "pending_or_excluded_instances": int((~group["instance_accounted"]).sum()),
                "all_instances_accounted": bool(group["instance_accounted"].all()),
            }
        )
    image_audit = pd.DataFrame(rows)
    complete_images = set(image_audit.loc[image_audit["all_instances_accounted"], "image_id"].astype(str))
    manifest = merged.loc[merged["image_id"].astype(str).isin(complete_images)].copy()
    manifest["four_class_label"] = ""
    manifest.loc[manifest["is_unqualified"], "four_class_label"] = "UNQUALIFIED"
    for grade in (1, 2, 3):
        manifest.loc[
            manifest["is_ready_qualified"] & manifest["recommended_candidate_grade"].eq(grade),
            "four_class_label",
        ] = f"GRADE_{grade}"
    if manifest["four_class_label"].eq("").any():
        raise ValueError("Complete-image manifest contains an unlabeled instance.")
    manifest_columns = [
        "image_id",
        "instance_id",
        "split",
        "source_group",
        "four_class_label",
        "recommended_candidate_grade",
        "is_unqualified",
        "label_source_v3",
        "stem_decision",
        "manual_rot_label",
        "image_path",
        "mask_path",
        "crop_path",
    ]
    manifest_columns = [column for column in manifest_columns if column in manifest.columns]
    return image_audit, manifest[manifest_columns]


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    candidate_path = resolve(root, args.candidate_csv)
    boundary_path = resolve(root, args.boundary_review)
    stem_path = resolve(root, args.stem_review)
    all_features_path = resolve(root, args.all_features_csv)
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(candidate_path, encoding="utf-8-sig").copy()
    all_features = pd.read_csv(all_features_path, encoding="utf-8-sig").copy()
    validate_unique(candidates, "candidate table")
    boundary = prepare_boundary_review(pd.read_excel(boundary_path), candidates)
    stem = prepare_stem_review(pd.read_excel(stem_path), candidates)

    weights, metrics, grid = choose_weights(boundary)
    low_threshold = float(metrics[BOUNDARY_LOW]["threshold"])
    high_threshold = float(metrics[BOUNDARY_HIGH]["threshold"])
    if not low_threshold < high_threshold:
        raise ValueError(f"Invalid calibrated thresholds: low={low_threshold}, high={high_threshold}")

    candidates["manual_calibrated_score"] = score_from_weights(candidates, weights)
    candidates["automatic_base_grade_v3"] = 2
    candidates.loc[candidates["manual_calibrated_score"] < low_threshold, "automatic_base_grade_v3"] = 3
    candidates.loc[candidates["manual_calibrated_score"] >= high_threshold, "automatic_base_grade_v3"] = 1

    boundary["manual_calibrated_score"] = score_from_weights(boundary, weights)
    boundary["automatic_base_grade_v3"] = 2
    boundary.loc[boundary["manual_calibrated_score"] < low_threshold, "automatic_base_grade_v3"] = 3
    boundary.loc[boundary["manual_calibrated_score"] >= high_threshold, "automatic_base_grade_v3"] = 1
    boundary["rule_matches_manual_grade"] = (
        boundary["automatic_base_grade_v3"].astype(int) == boundary["manual_expected_grade"].astype(int)
    )

    data, stem_audit = add_stem_decisions(
        candidates,
        boundary,
        stem,
        accept_unreviewed_auto_missing=args.accept_unreviewed_auto_missing,
    )
    boundary_manual = boundary[
        [
            "review_id",
            "image_id",
            "instance_id",
            "manual_expected_grade",
            "manual_measurement_error",
            "review_comment",
        ]
    ].rename(columns={"review_id": "boundary_review_id", "review_comment": "boundary_review_comment"})
    data = data.merge(boundary_manual, on=["image_id", "instance_id"], how="left", validate="one_to_one")

    data["automatic_rule_grade_v3"] = data["automatic_base_grade_v3"].astype(int)
    data.loc[data["stem_confirmed_missing"], "automatic_rule_grade_v3"] = 3
    data["recommended_candidate_grade"] = data["automatic_rule_grade_v3"].astype(int)
    valid_boundary_manual = data["manual_expected_grade"].notna() & ~data["manual_measurement_error"].fillna(False)
    data.loc[valid_boundary_manual, "recommended_candidate_grade"] = data.loc[
        valid_boundary_manual, "manual_expected_grade"
    ].astype(int)
    data.loc[data["stem_confirmed_missing"], "recommended_candidate_grade"] = 3
    data["stem_rule_applied_v3"] = data["stem_confirmed_missing"] & data["automatic_base_grade_v3"].ne(3)

    data["label_source_v3"] = "automatic_interpretable_rule"
    data.loc[valid_boundary_manual, "label_source_v3"] = "manual_boundary_review"
    data.loc[data["stem_confirmed_missing"], "label_source_v3"] = (
        data.loc[data["stem_confirmed_missing"], "label_source_v3"] + "+confirmed_missing_stem_override"
    )

    data["label_readiness"] = "READY_CANDIDATE"
    data["label_hold_reason"] = ""
    data.loc[data["stem_unresolved"], ["label_readiness", "label_hold_reason"]] = [
        "HOLD_FOR_STEM_REVIEW",
        "automatic stem status is UNCERTAIN and has no manual decision",
    ]
    measurement_error = data["manual_measurement_error"].fillna(False)
    data.loc[measurement_error, ["label_readiness", "label_hold_reason"]] = [
        "EXCLUDE_MEASUREMENT_ERROR",
        "manual review found bowl edge or mask boundary included in size measurement",
    ]
    data["candidate_label_status_v3"] = "candidate_only_not_final_commercial_truth"

    key_first = [
        "image_id",
        "instance_id",
        "split",
        "source_group",
        "recommended_candidate_grade",
        "label_readiness",
        "label_hold_reason",
        "label_source_v3",
        "manual_calibrated_score",
        "automatic_base_grade_v3",
        "automatic_rule_grade_v3",
        "stem_decision",
        "stem_confirmed_missing",
        "stem_unresolved",
        "stem_rule_applied_v3",
        "manual_expected_grade",
        "boundary_review_id",
        "stem_review_id",
        "candidate_label_status_v3",
    ]
    remaining = [column for column in data.columns if column not in key_first]
    data = data[key_first + remaining]

    bootstrap = bootstrap_thresholds(boundary, weights, args.bootstrap, args.seed)
    distributions = build_distributions(data)
    unresolved = data.loc[data["label_readiness"].ne("READY_CANDIDATE")].copy()
    image_audit, four_class_manifest = build_image_level_manifest(all_features, data)

    stem_crosstab = pd.crosstab(stem["stem_auto_status"], stem["manual_stem_present"])
    stem_metrics_rows: list[dict[str, object]] = []
    for auto_status in ("MISSING", "UNCERTAIN", "PRESENT"):
        row = stem_crosstab.loc[auto_status] if auto_status in stem_crosstab.index else pd.Series(dtype=int)
        total = int(row.sum())
        stem_metrics_rows.append(
            {
                "auto_status": auto_status,
                "manual_missing_NO": int(row.get("NO", 0)),
                "manual_present_YES": int(row.get("YES", 0)),
                "manual_uncertain": int(row.get("UNCERTAIN", 0)),
                "reviewed_total": total,
                "agreement_with_auto": (
                    float(row.get("NO", 0) / total)
                    if auto_status == "MISSING" and total
                    else float(row.get("YES", 0) / total)
                    if auto_status == "PRESENT" and total
                    else np.nan
                ),
            }
        )
    stem_metrics = pd.DataFrame(stem_metrics_rows)

    boundary_export_columns = [
        "review_id",
        "candidate_group",
        "image_id",
        "instance_id",
        "manual_expected_grade",
        "review_comment",
        "manual_stem_missing_comment",
        "manual_measurement_error",
        "calibration_eligible",
        "manual_calibrated_score",
        "automatic_base_grade_v3",
        "rule_matches_manual_grade",
        "pct_area",
        "pct_minor_axis",
        "shape_score",
        "size_priority_score",
    ]
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "grading_labels_candidate_v3.csv", index=False, encoding="utf-8-sig")
    data.loc[data["label_readiness"].eq("READY_CANDIDATE")].to_csv(
        output_dir / "training_ready_candidate_labels.csv", index=False, encoding="utf-8-sig"
    )
    four_class_manifest.to_csv(
        output_dir / "four_class_instance_manifest.csv", index=False, encoding="utf-8-sig"
    )
    unresolved.to_csv(output_dir / "held_or_excluded_instances.csv", index=False, encoding="utf-8-sig")
    boundary[boundary_export_columns].to_csv(
        audit_dir / "boundary_review_audit.csv", index=False, encoding="utf-8-sig"
    )
    stem_audit.to_csv(audit_dir / "stem_manual_audit.csv", index=False, encoding="utf-8-sig")
    stem_metrics.to_csv(audit_dir / "stem_review_evaluation.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(audit_dir / "grade_distribution.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(audit_dir / "threshold_bootstrap.csv", index=False, encoding="utf-8-sig")
    grid.head(25).to_csv(audit_dir / "weight_search_top25.csv", index=False, encoding="utf-8-sig")
    image_audit.to_csv(audit_dir / "image_level_dataset_readiness.csv", index=False, encoding="utf-8-sig")

    eligible_boundary = boundary.loc[boundary["calibration_eligible"]]
    summary = {
        "schema_version": 1,
        "scheme_version": args.scheme_version,
        "status": "candidate_only_not_final_commercial_truth",
        "inputs": {
            "candidate_csv": str(args.candidate_csv).replace("\\", "/"),
            "candidate_sha256": sha256_file(candidate_path),
            "boundary_review": str(args.boundary_review).replace("\\", "/"),
            "boundary_review_sha256": sha256_file(boundary_path),
            "stem_review": str(args.stem_review).replace("\\", "/"),
            "stem_review_sha256": sha256_file(stem_path),
            "all_features_csv": str(args.all_features_csv).replace("\\", "/"),
            "all_features_sha256": sha256_file(all_features_path),
        },
        "instances": int(len(data)),
        "manual_review_audit": {
            "boundary_rows": int(len(boundary)),
            "boundary_complete": int(boundary["manual_expected_grade"].notna().sum()),
            "boundary_calibration_eligible": int(boundary["calibration_eligible"].sum()),
            "boundary_stem_rule_rows": int(boundary["manual_stem_missing_comment"].sum()),
            "boundary_measurement_error_rows": int(boundary["manual_measurement_error"].sum()),
            "stem_rows": int(len(stem)),
            "stem_complete": int(stem["manual_stem_present"].isin(["YES", "NO", "UNCERTAIN"]).sum()),
        },
        "score_formula": {
            "formula": "100*(w_area*pct_area + w_minor*pct_minor_axis) + w_shape*shape_score",
            "weights": weights,
            "color_policy": "Color-uniformity features are retained but receive zero grade weight because the manual boundary audit contained no color-based decisions and illumination varied across days.",
            "weight_selection": "Transparent 5-percentage-point grid search; maximize mean balanced accuracy over the two manually reviewed boundaries; ties favor greater total size weight and then area weight.",
        },
        "thresholds": {
            "grade3_to_2": low_threshold,
            "grade2_to_1": high_threshold,
            "rule_grade_1": f"score >= {high_threshold:.6f}",
            "rule_grade_2": f"{low_threshold:.6f} <= score < {high_threshold:.6f}",
            "rule_grade_3": f"score < {low_threshold:.6f}",
        },
        "boundary_fit": {
            "grade1_2": metrics[BOUNDARY_HIGH],
            "grade2_3": metrics[BOUNDARY_LOW],
            "eligible_rows": int(len(eligible_boundary)),
            "eligible_manual_grade_counts": {
                str(int(k)): int(v)
                for k, v in eligible_boundary["manual_expected_grade"].value_counts().sort_index().items()
            },
        },
        "stem_review": {
            "confirmed_missing_distinct": int(data["stem_confirmed_missing"].sum()),
            "unreviewed_auto_missing_accepted": int(
                data["stem_decision"].eq("MISSING_AUTO_PRIOR_VALIDATED").sum()
            ),
            "confirmed_present_distinct": int(data["stem_decision"].eq("PRESENT_CONFIRMED").sum()),
            "auto_present_accepted_with_qc_flag": int(data["stem_decision"].eq("PRESENT_AUTO_QC").sum()),
            "unresolved": int(data["stem_unresolved"].sum()),
            "auto_missing_review_agreement": float(
                stem.loc[stem["stem_auto_status"].eq("MISSING"), "manual_stem_present"].eq("NO").mean()
            ),
            "sampled_auto_present_review_agreement": float(
                stem.loc[stem["stem_auto_status"].eq("PRESENT"), "manual_stem_present"].eq("YES").mean()
            ),
            "uncertain_reviewed_manual_counts": {
                str(k): int(v)
                for k, v in stem.loc[
                    stem["stem_auto_status"].eq("UNCERTAIN"), "manual_stem_present"
                ].value_counts().items()
            },
        },
        "output_counts": {
            "ready_candidate": int(data["label_readiness"].eq("READY_CANDIDATE").sum()),
            "held_for_stem_review": int(data["label_readiness"].eq("HOLD_FOR_STEM_REVIEW").sum()),
            "excluded_measurement_error": int(data["label_readiness"].eq("EXCLUDE_MEASUREMENT_ERROR").sum()),
            "confirmed_missing_stem_grade3_overrides": int(data["stem_rule_applied_v3"].sum()),
            "recommended_grade_counts_all": {
                str(int(k)): int(v)
                for k, v in data["recommended_candidate_grade"].value_counts().sort_index().items()
            },
            "recommended_grade_counts_ready": {
                str(int(k)): int(v)
                for k, v in data.loc[
                    data["label_readiness"].eq("READY_CANDIDATE"), "recommended_candidate_grade"
                ].value_counts().sort_index().items()
            },
            "complete_images_for_four_class_dataset": int(image_audit["all_instances_accounted"].sum()),
            "images_with_pending_or_excluded_instances": int((~image_audit["all_instances_accounted"]).sum()),
            "four_class_manifest_instances": int(len(four_class_manifest)),
            "four_class_manifest_counts": {
                str(k): int(v)
                for k, v in four_class_manifest["four_class_label"].value_counts().sort_index().items()
            },
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (output_dir / "grading_standard_config.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# 人工审核校准后的枸杞候选分级标准 {args.scheme_version}

本目录把已完成的 1/2、2/3 边界审核与果梗审核合并到同一个可追溯结果中。结果仍是后续数据集建设用的候选标签，不代表最终商业标准。

## 自动规则

- 校准分数：`{weights['pct_area']:.2f} × 面积百分位 + {weights['pct_minor_axis']:.2f} × 短径百分位 + {weights['shape_score']:.2f} × 形态分数`。各项最终统一到 0–100。
- 一级：分数 `>= {high_threshold:.6f}`。
- 二级：`{low_threshold:.6f} <= 分数 < {high_threshold:.6f}`。
- 三级：分数 `< {low_threshold:.6f}`。
- 已确认缺失果梗：无论基础分数是多少，覆盖为三级。
- 已人工审核的边界样本：在测量有效时，人工等级优先于自动规则；缺果梗规则仍具有最高优先级。

颜色均匀性特征没有删除，但本版不计入等级总分。原因是人工边界审核没有使用颜色问题作为分级理由，且数据分多天、不同光照拍摄；在没有颜色标定或明确人工颜色标准前，将其计入分数容易把光照差异误当成果实等级差异。

## 数据清理规则

- 边界表中注明“缺失果柄/没果柄”的记录不用于拟合尺寸分界，转交果梗覆盖规则。
- 注明“把碗边算入面积/框边缘有问题”的记录不用于拟合，也不进入训练候选集。
- 自动果梗状态为 `UNCERTAIN` 且没有人工结果的实例暂缓，不被静默判成有梗或无梗。

## 文件

- `grading_labels_candidate_v3.csv`：全部实例及规则、人工审核、果梗决策和可用状态。
- `training_ready_candidate_labels.csv`：当前可进入下一步数据集构建的候选子集。
- `four_class_instance_manifest.csv`：只保留“整张图内所有实例都有标签”的四分类清单，类别为不合格、一级、二级、三级；下一步据此转换 YOLO 标签。
- `held_or_excluded_instances.csv`：仍需果梗复核或因测量错误排除的实例。
- `audit/boundary_review_audit.csv`：边界人工审核清洗、校准与规则吻合情况。
- `audit/stem_manual_audit.csv`：合并后的人工果梗依据。
- `audit/stem_review_evaluation.csv`：自动果梗状态与人工结果对照。
- `audit/grade_distribution.csv`：按 train/val 和可用状态统计的等级数量。
- `audit/threshold_bootstrap.csv`：固定公式后，对人工边界样本做分层 bootstrap 得到的阈值波动范围。
- `audit/weight_search_top25.csv`：可解释权重网格搜索的前 25 组结果。
- `audit/image_level_dataset_readiness.csv`：逐图核对是否存在未定级实例，避免生成不完整 YOLO 标注。
- `grading_standard_config.json`：公式、阈值、输入哈希、审核数量及软件版本。

## 当前限制

边界样本只有 60 颗，而且来自旧规则边界附近，不是全体实例的随机样本。阈值适合作为自动标注候选标准，但在训练等级模型前仍应保留人工审核来源和暂缓状态。像素尺寸没有毫米标定，只适用于本批拍摄条件下的相对比较。
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary["output_counts"], ensure_ascii=False, indent=2))
    print(json.dumps({"weights": weights, "thresholds": summary["thresholds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
