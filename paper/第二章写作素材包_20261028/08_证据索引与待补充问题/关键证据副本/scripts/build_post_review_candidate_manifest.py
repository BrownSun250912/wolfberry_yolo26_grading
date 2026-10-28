from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply completed stem review to the train/val grade manifest.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grading_standard_post_review_v5"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output = args.output_dir.resolve() if args.output_dir.is_absolute() else (root / args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    candidate_path = root / "outputs/grading_standard_manual_v4_final/grading_labels_candidate_v3.csv"
    features_path = root / "outputs/strict_rot_exclusion_v2/features_with_manual_rot.csv"
    stem_path = root / "review/three_person_audit_v1/04_审核结果汇总/stem_results.json"
    candidate = pd.read_csv(candidate_path, encoding="utf-8-sig")
    features = pd.read_csv(features_path, encoding="utf-8-sig")
    stem = json.loads(stem_path.read_text(encoding="utf-8"))

    stem_labels = {}
    for instance_id, record in stem.items():
        labels = record.get("labels", [])
        if len(labels) != 1 or labels[0] not in {"PRESENT", "MISSING"}:
            raise ValueError(f"Invalid stem result: {instance_id} -> {labels}")
        stem_labels[instance_id] = labels[0]
    if len(stem_labels) != 930:
        raise ValueError(f"Expected 930 stem labels, got {len(stem_labels)}")

    rows = []
    excluded = []
    resolved_stem_counts = Counter()
    for _, row in candidate.iterrows():
        readiness = str(row["label_readiness"])
        instance_id = str(row["instance_id"])
        if readiness == "EXCLUDE_MEASUREMENT_ERROR":
            excluded.append(
                {
                    "image_id": str(row["image_id"]),
                    "instance_id": instance_id,
                    "reason": str(row["label_hold_reason"]),
                }
            )
            continue
        if readiness == "HOLD_FOR_STEM_REVIEW":
            if instance_id not in stem_labels:
                raise ValueError(f"Missing completed stem review: {instance_id}")
            stem_label = stem_labels[instance_id]
            resolved_stem_counts[stem_label] += 1
            if stem_label == "MISSING":
                grade = 3
                source = "human_stem_review_missing_override_grade3"
            else:
                grade = int(float(row["automatic_base_grade_v3"]))
                source = "human_stem_review_present_plus_frozen_grade_rule"
            stem_decision = f"{stem_label}_HUMAN_THREE_PERSON_BATCH"
        elif readiness == "READY_CANDIDATE":
            grade = int(float(row["recommended_candidate_grade"]))
            source = str(row["label_source_v3"])
            stem_decision = str(row["stem_decision"])
        else:
            raise ValueError(f"Unexpected label readiness: {readiness}")
        rows.append(
            {
                "image_id": str(row["image_id"]),
                "instance_id": instance_id,
                "split": str(row["split"]),
                "source_group": str(row["source_group"]),
                "four_class_label": f"GRADE_{grade}",
                "recommended_candidate_grade": grade,
                "is_unqualified": False,
                "label_source_v3": source,
                "stem_decision": stem_decision,
                "manual_rot_label": str(row["manual_rot_label"]),
                "image_path": str(row["image_path"]),
                "mask_path": str(row["mask_path"]),
                "crop_path": str(row["crop_path"]),
            }
        )

    rot = features.loc[
        features["manual_rot_exclude"].astype(str).str.upper().isin({"TRUE", "1", "YES"})
    ].copy()
    for _, row in rot.iterrows():
        rows.append(
            {
                "image_id": str(row["image_id"]),
                "instance_id": str(row["instance_id"]),
                "split": str(row["split"]),
                "source_group": str(row["source_group"]),
                "four_class_label": "UNQUALIFIED",
                "recommended_candidate_grade": None,
                "is_unqualified": True,
                "label_source_v3": "strict_completed_manual_rot_review",
                "stem_decision": "NOT_APPLICABLE_UNQUALIFIED",
                "manual_rot_label": str(row["manual_rot_label"]),
                "image_path": str(row["image_path"]),
                "mask_path": str(row["mask_path"]),
                "crop_path": str(row["crop_path"]),
            }
        )

    result = pd.DataFrame(rows).sort_values(["split", "image_id", "instance_id"]).reset_index(drop=True)
    if result.duplicated(["image_id", "instance_id"]).any():
        raise ValueError("Duplicate train/val candidate labels after review merge")
    if len(result) != 5084:
        raise ValueError(f"Expected 5084 usable feature instances, got {len(result)}")
    if not result["four_class_label"].isin({"UNQUALIFIED", "GRADE_1", "GRADE_2", "GRADE_3"}).all():
        raise ValueError("Unknown class in merged candidate manifest")
    result.to_csv(output / "four_class_instance_manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(excluded).to_csv(output / "excluded_measurement_errors.csv", index=False, encoding="utf-8-sig")
    summary = {
        "input_candidate_instances": len(candidate),
        "resolved_human_stem_instances": len(stem_labels),
        "resolved_stem_distribution": dict(sorted(resolved_stem_counts.items())),
        "manual_rot_unqualified_instances": len(rot),
        "excluded_measurement_error_instances": len(excluded),
        "output_instances": len(result),
        "output_images": int(result["image_id"].nunique()),
        "class_distribution": {
            key: int(value) for key, value in result["four_class_label"].value_counts().sort_index().items()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "README.md").write_text(
        "# 完成人工审核后的训练/验证候选标签\n\n"
        "本目录将三人果梗复核写回既定分级规则。人工确认无果梗的合格果统一覆盖为三级；"
        "人工确认有果梗的实例沿用冻结的大小优先＋形态基础等级。腐烂果继续使用严格人工腐烂标签。\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
