from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


COLOR_WEIGHTS = {
    "L_std": 0.20,
    "a_std": 0.30,
    "b_std": 0.20,
    "max_damage_ratio": 0.30,
}
GLOBAL_FEATURES = ("area", "minor_axis", "solidity", "aspect_ratio", "circularity")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    if ref.size == 0:
        raise ValueError("Empty reference distribution")
    values = np.asarray(values, dtype=float)
    left = np.searchsorted(ref, values, side="left")
    right = np.searchsorted(ref, values, side="right")
    return np.clip((left + right) / (2.0 * ref.size), 0.0, 1.0)


def reference_values(
    references: pd.DataFrame, feature: str, source_group: str = "ALL"
) -> np.ndarray:
    rows = references.loc[
        references["feature"].eq(feature)
        & references["source_group"].astype(str).eq(str(source_group))
    ].sort_values("rank")
    if rows.empty:
        raise ValueError(f"Missing frozen reference: feature={feature}, group={source_group}")
    return rows["value"].to_numpy(dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score newly qualified instances with the previously frozen grading references."
    )
    parser.add_argument("--qualified-features", type=Path, required=True)
    parser.add_argument("--reference-values", type=Path, required=True)
    parser.add_argument("--size-config", type=Path, required=True)
    parser.add_argument("--old-candidates", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    args = parser.parse_args()

    data = pd.read_csv(args.qualified_features, encoding="utf-8-sig")
    references = pd.read_csv(args.reference_values, encoding="utf-8-sig")
    size_config = json.loads(args.size_config.read_text(encoding="utf-8"))
    old = pd.read_csv(args.old_candidates, encoding="utf-8-sig")

    if data["instance_id"].duplicated().any():
        raise ValueError("Qualified feature table contains duplicate instance_id values")
    required = {
        "image_id",
        "instance_id",
        "split",
        "source_group",
        *GLOBAL_FEATURES,
        *COLOR_WEIGHTS,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    scored = data.copy()
    for feature in GLOBAL_FEATURES:
        scored[f"pct_{feature}"] = empirical_percentile(
            reference_values(references, feature), scored[feature].to_numpy(dtype=float)
        )

    scored["aspect_typicality"] = np.clip(
        1.0 - 2.0 * np.abs(scored["pct_aspect_ratio"] - 0.5), 0.0, 1.0
    )
    scored["size_score"] = 100.0 * (
        0.65 * scored["pct_area"] + 0.35 * scored["pct_minor_axis"]
    )
    scored["shape_score"] = 100.0 * (
        0.50 * scored["pct_solidity"]
        + 0.30 * scored["aspect_typicality"]
        + 0.20 * scored["pct_circularity"]
    )

    for feature, weight in COLOR_WEIGHTS.items():
        output = np.full(len(scored), np.nan, dtype=float)
        for group, positions in scored.groupby("source_group", sort=True).indices.items():
            reference = reference_values(references, feature, str(group))
            values = scored.iloc[positions][feature].to_numpy(dtype=float)
            output[positions] = 1.0 - empirical_percentile(reference, values)
        scored[f"uniformity_{feature}"] = np.clip(output, 0.0, 1.0)

    scored["color_uniformity_score"] = 100.0 * sum(
        weight * scored[f"uniformity_{feature}"]
        for feature, weight in COLOR_WEIGHTS.items()
    )
    scored["size_priority_score"] = (
        0.55 * scored["size_score"]
        + 0.30 * scored["shape_score"]
        + 0.15 * scored["color_uniformity_score"]
    )
    low = float(size_config["thresholds"]["grade3_to_2"])
    high = float(size_config["thresholds"]["grade2_to_1"])
    scored["candidate_grade"] = np.where(
        scored["size_priority_score"].ge(high),
        1,
        np.where(scored["size_priority_score"].lt(low), 3, 2),
    ).astype(int)
    scored["candidate_label_status"] = "candidate_only_not_training_truth"
    scored["grading_scheme_version"] = "size_priority_v1_frozen_reference_reapplied"

    overlap = scored.merge(
        old,
        on="instance_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_new", "_old"),
    )
    if len(overlap) != len(old):
        raise ValueError(
            f"Old candidate preservation failed: overlap={len(overlap)}, old={len(old)}"
        )
    compare_columns = [
        "pct_area",
        "pct_minor_axis",
        "pct_solidity",
        "pct_aspect_ratio",
        "pct_circularity",
        "aspect_typicality",
        "size_score",
        "shape_score",
        "color_uniformity_score",
        "size_priority_score",
    ]
    max_differences: dict[str, float] = {}
    for column in compare_columns:
        difference = np.abs(
            overlap[f"{column}_new"].to_numpy(dtype=float)
            - overlap[f"{column}_old"].to_numpy(dtype=float)
        )
        max_differences[column] = float(np.nanmax(difference))
    old_grade_mismatches = int(
        np.count_nonzero(
            overlap["candidate_grade_new"].to_numpy(dtype=int)
            != overlap["candidate_grade_old"].to_numpy(dtype=int)
        )
    )
    if old_grade_mismatches or max(max_differences.values()) > 1e-10:
        raise ValueError(
            f"Frozen scoring does not reproduce old candidates: "
            f"grade_mismatches={old_grade_mismatches}, max_diff={max(max_differences.values())}"
        )

    preferred = [
        "image_id",
        "instance_id",
        "split",
        "source_group",
        "grading_scheme_version",
        "candidate_label_status",
        "candidate_grade",
        "size_priority_score",
        "size_score",
        "shape_score",
        "color_uniformity_score",
        "pct_area",
        "pct_minor_axis",
        "pct_solidity",
        "pct_aspect_ratio",
        "aspect_typicality",
        "pct_circularity",
        *[f"uniformity_{feature}" for feature in COLOR_WEIGHTS],
    ]
    remaining = [column for column in scored.columns if column not in preferred]
    scored = scored.sort_values(["split", "image_id", "instance_id"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    scored[preferred + remaining].to_csv(args.output_csv, index=False, encoding="utf-8-sig")

    audit = {
        "method": "reapply_previous_frozen_ECDF_references_and_thresholds",
        "qualified_instances": int(len(scored)),
        "previous_candidate_instances": int(len(old)),
        "newly_restored_nonrot_instances": int(len(scored) - len(old)),
        "old_candidate_grade_mismatches": old_grade_mismatches,
        "old_candidate_max_numeric_differences": max_differences,
        "candidate_grade_counts": {
            str(int(key)): int(value)
            for key, value in scored["candidate_grade"].value_counts().sort_index().items()
        },
        "inputs": {
            "qualified_features": str(args.qualified_features),
            "qualified_features_sha256": sha256(args.qualified_features),
            "reference_values": str(args.reference_values),
            "reference_values_sha256": sha256(args.reference_values),
            "size_config": str(args.size_config),
            "size_config_sha256": sha256(args.size_config),
            "old_candidates": str(args.old_candidates),
            "old_candidates_sha256": sha256(args.old_candidates),
        },
        "output_csv": str(args.output_csv),
        "output_sha256": sha256(args.output_csv),
    }
    args.audit_json.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
