from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path


CATEGORY_BY_GRADE = {
    1: "一级_GRADE1",
    2: "二级_GRADE2",
    3: "三级_GRADE3",
}
LABEL_BY_CATEGORY_TOKEN = {
    "UNQUALIFIED": "UNQUALIFIED",
    "GRADE1": "GRADE1",
    "GRADE2": "GRADE2",
    "GRADE3": "GRADE3",
    "UNCERTAIN": "UNCERTAIN",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep completed manual adjudications and assign remaining pending test items by BASE grade."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("review/three_person_audit_v1/04_审核结果汇总"),
    )
    return parser.parse_args()


def instance_id(name: str) -> str:
    match = re.match(r"(wb_\d{6}__i\d{3})", Path(name).stem)
    if not match:
        raise ValueError(f"Cannot parse instance ID: {name}")
    return match.group(1)


def result_label(category: str) -> str:
    for token, label in LABEL_BY_CATEGORY_TOKEN.items():
        if token in category:
            return label
    raise ValueError(f"Unknown adjudication category: {category}")


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    summary = args.summary_dir.resolve() if args.summary_dir.is_absolute() else (root / args.summary_dir).resolve()
    adjudication = summary / "总负责人最终裁决"
    pending = adjudication / "待审核"
    if not pending.exists():
        raise FileNotFoundError(pending)

    manual_before: dict[str, str] = {}
    for category in sorted(path for path in adjudication.iterdir() if path.is_dir() and path.name != "待审核"):
        label = result_label(category.name)
        for path in category.glob("*.jpg"):
            item = instance_id(path.name)
            if item in manual_before:
                raise ValueError(f"Manual adjudication duplicated: {item}")
            manual_before[item] = label

    auto_log = []
    for source in sorted(pending.glob("*.jpg")):
        match = re.search(r"__BASE_G([123])", source.stem)
        if not match:
            raise ValueError(f"Pending file has no BASE grade: {source.name}")
        grade = int(match.group(1))
        destination = adjudication / CATEGORY_BY_GRADE[grade] / source.name
        if destination.exists():
            raise FileExistsError(destination)
        shutil.move(str(source), str(destination))
        auto_log.append(
            {
                "instance_id": instance_id(source.name),
                "base_grade": grade,
                "source": str(source),
                "destination": str(destination),
                "label_source": "frozen_base_rule_per_user_instruction",
            }
        )

    agreements = json.loads((summary / "test_agreements.json").read_text(encoding="utf-8"))
    disagreements = json.loads((summary / "test_disagreements.json").read_text(encoding="utf-8"))
    expected = {row["instance_id"] for row in agreements} | {row["instance_id"] for row in disagreements}
    final_adjudication: dict[str, list[str]] = defaultdict(list)
    for category in sorted(path for path in adjudication.iterdir() if path.is_dir() and path.name != "待审核"):
        label = result_label(category.name)
        for path in category.glob("*.jpg"):
            final_adjudication[instance_id(path.name)].append(label)
    duplicates = sorted(key for key, values in final_adjudication.items() if len(values) != 1)
    if duplicates:
        raise ValueError(f"Final adjudication contains duplicates: {duplicates[:10]}")
    if set(final_adjudication) != {row["instance_id"] for row in disagreements}:
        missing = sorted({row["instance_id"] for row in disagreements} - set(final_adjudication))
        extra = sorted(set(final_adjudication) - {row["instance_id"] for row in disagreements})
        raise ValueError(f"Adjudication coverage mismatch: missing={missing[:10]}, extra={extra[:10]}")

    final_rows = []
    for row in agreements:
        final_rows.append(
            {
                "instance_id": row["instance_id"],
                "final_label": row["final_label"],
                "label_source": "double_review_agreement",
            }
        )
    auto_ids = {row["instance_id"] for row in auto_log}
    for item, labels in sorted(final_adjudication.items()):
        final_rows.append(
            {
                "instance_id": item,
                "final_label": labels[0],
                "label_source": (
                    "frozen_base_rule_per_user_instruction"
                    if item in auto_ids
                    else "lead_manual_adjudication_missing_stem_or_other_manual_decision"
                ),
            }
        )
    if len(final_rows) != len(expected) or len({row["instance_id"] for row in final_rows}) != len(expected):
        raise ValueError("Final test label coverage is not unique and complete")
    distribution = Counter(row["final_label"] for row in final_rows)
    completion = {
        "expected_test_instances": len(expected),
        "double_review_agreements": len(agreements),
        "manual_adjudications_preserved": len(manual_before),
        "pending_assigned_by_frozen_base_rule": len(auto_log),
        "remaining_pending": len(list(pending.glob("*.jpg"))),
        "final_unique_labels": len(final_rows),
        "final_distribution": dict(sorted(distribution.items())),
        "ready_for_test_dataset_export": "UNCERTAIN" not in distribution,
    }
    (summary / "auto_base_assignment_log.json").write_text(
        json.dumps(auto_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (summary / "test_final_labels.json").write_text(
        json.dumps(sorted(final_rows, key=lambda row: row["instance_id"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (summary / "adjudication_completion.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
