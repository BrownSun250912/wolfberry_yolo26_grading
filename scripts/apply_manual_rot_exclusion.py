from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
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

ROT_MAP = {
    "y": "ROT",
    "yes": "ROT",
    "n": "NOT_ROT",
    "no": "NOT_ROT",
    "uncertain": "UNCERTAIN",
    "u": "UNCERTAIN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge strict manual rot labels into the frozen feature table and create a grading-analysis subset."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--review-xlsx", type=Path, required=True)
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("outputs/feature_analysis_frozen_v1/features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/strict_rot_exclusion_v1"),
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def normalize_manual_labels(review: pd.DataFrame) -> pd.DataFrame:
    required = {"image_id", "instance_id", "split", "rot"}
    missing = sorted(required - set(review.columns))
    if missing:
        raise KeyError(f"Review workbook is missing columns: {missing}")

    result = review.copy()
    result["instance_id"] = result["instance_id"].astype("string").str.strip()
    if result["instance_id"].isna().any() or result["instance_id"].eq("").any():
        raise ValueError("Review workbook contains a missing instance_id.")
    if result["instance_id"].duplicated().any():
        duplicates = result.loc[result["instance_id"].duplicated(keep=False), "instance_id"].tolist()
        raise ValueError(f"Review workbook contains duplicate instance_id values: {duplicates[:10]}")

    result["manual_rot_raw"] = result["rot"].astype("string").str.strip().str.lower()
    result["manual_rot_label"] = result["manual_rot_raw"].map(ROT_MAP)
    invalid = sorted(result.loc[result["manual_rot_label"].isna(), "manual_rot_raw"].dropna().unique())
    if invalid:
        raise ValueError(f"Unsupported values in rot column: {invalid}")
    if result["manual_rot_label"].isna().any():
        raise ValueError("Review workbook contains blank rot labels.")

    result["manual_rot_exclude"] = result["manual_rot_label"].isin(["ROT", "UNCERTAIN"])
    result["manual_label_policy"] = "strict_reviewer_y_means_rot; uncertain_excluded"
    return result


def compare_shared_data(review: pd.DataFrame, features: pd.DataFrame) -> dict[str, object]:
    shared_ids = sorted(set(review["instance_id"]) & set(features["instance_id"]))
    left = review.set_index("instance_id").loc[shared_ids]
    right = features.set_index("instance_id").loc[shared_ids]

    categorical_mismatches: dict[str, int] = {}
    for column in ("image_id", "split"):
        if column in left.columns and column in right.columns:
            mismatch = left[column].astype("string").fillna("").ne(right[column].astype("string").fillna(""))
            categorical_mismatches[column] = int(mismatch.sum())

    numeric_mismatches: dict[str, int] = {}
    for column in FEATURES:
        if column not in left.columns or column not in right.columns:
            continue
        left_values = pd.to_numeric(left[column], errors="coerce").to_numpy(dtype=float)
        right_values = pd.to_numeric(right[column], errors="coerce").to_numpy(dtype=float)
        mismatch = ~np.isclose(left_values, right_values, rtol=1e-9, atol=1e-9, equal_nan=True)
        numeric_mismatches[column] = int(mismatch.sum())

    if any(categorical_mismatches.values()) or any(numeric_mismatches.values()):
        raise ValueError(
            "Review workbook does not match the frozen feature source: "
            f"categorical={categorical_mismatches}, numeric={numeric_mismatches}"
        )

    return {
        "shared_instance_count": len(shared_ids),
        "categorical_mismatches": categorical_mismatches,
        "numeric_feature_mismatches": numeric_mismatches,
    }


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    review_path = resolve(project_root, args.review_xlsx).resolve()
    features_path = resolve(project_root, args.features_csv).resolve()
    output_dir = resolve(project_root, args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    review = pd.read_excel(review_path, sheet_name=0, engine="openpyxl")
    review = normalize_manual_labels(review)
    features = pd.read_csv(features_path, encoding="utf-8-sig")

    if "instance_id" not in features.columns:
        raise KeyError("Frozen feature table is missing instance_id.")
    if features["instance_id"].duplicated().any():
        raise ValueError("Frozen feature table contains duplicate instance_id values.")
    missing_features = [feature for feature in FEATURES if feature not in features.columns]
    if missing_features:
        raise KeyError(f"Frozen feature table is missing features: {missing_features}")

    features["instance_id"] = features["instance_id"].astype("string").str.strip()
    comparison = compare_shared_data(review, features)

    label_columns = [
        "instance_id",
        "manual_rot_raw",
        "manual_rot_label",
        "manual_rot_exclude",
        "manual_label_policy",
    ]
    label_table = review[label_columns].copy()
    merged = features.merge(label_table, on="instance_id", how="left", validate="one_to_one")
    merged["rot_reviewed"] = merged["manual_rot_label"].notna()
    merged["manual_rot_label"] = merged["manual_rot_label"].fillna("NOT_IN_REVIEW_QUEUE")
    merged["manual_rot_raw"] = merged["manual_rot_raw"].fillna("")
    merged["manual_rot_exclude"] = merged["manual_rot_exclude"].fillna(False).astype(bool)
    merged["manual_label_policy"] = merged["manual_label_policy"].fillna(
        "not_in_high_recall_rot_review_queue"
    )
    merged["grade_analysis_eligible"] = ~merged["manual_rot_exclude"]
    merged["grade_analysis_exclusion_reason"] = np.where(
        merged["manual_rot_label"].eq("ROT"),
        "manual_strict_rot",
        np.where(merged["manual_rot_label"].eq("UNCERTAIN"), "manual_rot_uncertain", ""),
    )

    merged = merged.sort_values(["split", "image_id", "instance_id"]).reset_index(drop=True)
    excluded = merged.loc[merged["manual_rot_exclude"]].copy()
    qualified = merged.loc[merged["grade_analysis_eligible"]].copy()

    feature_ids = set(features["instance_id"])
    missing_from_features = review.loc[~review["instance_id"].isin(feature_ids)].copy()
    missing_from_features["not_merged_reason"] = "not_present_after_frozen_qc_inclusion"

    manual_labels_out = review[
        [
            "image_id",
            "instance_id",
            "split",
            "manual_rot_raw",
            "manual_rot_label",
            "manual_rot_exclude",
            "manual_label_policy",
        ]
    ].sort_values(["split", "image_id", "instance_id"])

    manual_labels_out.to_csv(output_dir / "manual_rot_labels.csv", index=False, encoding="utf-8-sig")
    merged.to_csv(output_dir / "features_with_manual_rot.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(output_dir / "rot_excluded_features.csv", index=False, encoding="utf-8-sig")
    qualified.to_csv(output_dir / "qualified_features.csv", index=False, encoding="utf-8-sig")
    missing_from_features.to_csv(
        output_dir / "review_ids_not_in_feature_analysis.csv", index=False, encoding="utf-8-sig"
    )

    manual_counts = review["manual_rot_label"].value_counts(dropna=False).to_dict()
    merged_counts = merged["manual_rot_label"].value_counts(dropna=False).to_dict()
    summary = {
        "review_workbook": str(review_path),
        "frozen_features_source": str(features_path),
        "policy": {
            "y": "ROT and excluded",
            "n": "NOT_ROT and retained",
            "uncertain": "excluded under strict policy",
            "not_in_review_queue": "retained; not treated as manual NOT_ROT truth",
        },
        "review_rows": int(len(review)),
        "review_label_counts": {str(key): int(value) for key, value in manual_counts.items()},
        "frozen_feature_rows": int(len(features)),
        "merged_label_counts": {str(key): int(value) for key, value in merged_counts.items()},
        "excluded_rot_rows": int(len(excluded)),
        "qualified_rows": int(len(qualified)),
        "review_ids_not_in_frozen_features": int(len(missing_from_features)),
        "comparison": comparison,
    }
    (output_dir / "exclusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Strict manual rot exclusion v1",
                "",
                "This dataset uses the second reviewer workbook as the only manual rot source.",
                "The first submitted workbook is retained only as an obsolete archive and is not used.",
                "",
                "## Policy",
                "",
                "- `y` is normalized to `ROT` and excluded from grading analysis.",
                "- `n` is normalized to `NOT_ROT` and retained.",
                "- `uncertain` would be excluded under the strict policy.",
                "- Instances outside the review queue are retained as `NOT_IN_REVIEW_QUEUE`; this is not a manual healthy label.",
                "- Existing frozen QC exclusions remain upstream and are not reversed.",
                "- Pixel measurements remain in px / px^2 because no physical calibration is available.",
                "",
                "## Counts",
                "",
                f"- Manual review rows: {len(review):,}",
                f"- Frozen feature rows: {len(features):,}",
                f"- Strict rot exclusions matched to frozen features: {len(excluded):,}",
                f"- Remaining grading-analysis rows: {len(qualified):,}",
                f"- Review IDs already absent after frozen QC: {len(missing_from_features):,}",
                "",
                "## Outputs",
                "",
                "- `manual_rot_labels.csv`: normalized manual labels.",
                "- `features_with_manual_rot.csv`: full frozen table plus label lineage.",
                "- `rot_excluded_features.csv`: rows excluded as rot/uncertain.",
                "- `qualified_features.csv`: non-excluded rows used for later three-grade analysis.",
                "- `review_ids_not_in_feature_analysis.csv`: reviewed IDs absent after upstream frozen QC.",
                "- `exclusion_summary.json`: machine-readable validation summary.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
