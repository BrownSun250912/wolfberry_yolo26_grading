from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


TEST_LABELS = ("UNQUALIFIED", "GRADE1", "GRADE2", "GRADE3", "UNCERTAIN")
STEM_LABELS = ("PRESENT", "MISSING", "UNCERTAIN")
GEOMETRY_LABELS = ("KEEP_FIX", "EXCLUDE", "UNCERTAIN")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile and audit three-person folder-based review results.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--review-root",
        type=Path,
        default=Path("review/three_person_audit_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("review/three_person_audit_v1/04_审核结果汇总"),
    )
    return parser.parse_args()


def resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise IOError(path)
    path.write_bytes(data.tobytes())


def label_from_category(category: str, allowed: tuple[str, ...]) -> str:
    for label in allowed:
        if label in category:
            return label
    raise ValueError(f"Unknown result category: {category}")


def item_id_from_name(name: str, task_type: str) -> str:
    stem = Path(name).stem
    if task_type == "geometry":
        match = re.match(r"(wb_\d{6})", stem)
    else:
        match = re.match(r"(wb_\d{6}__i\d{3})", stem)
    if not match:
        raise ValueError(f"Could not parse task ID from {name}")
    return match.group(1)


def locate_returned_root(review_root: Path, candidates: list[Path]) -> Path:
    for candidate in candidates:
        path = review_root / candidate
        if (path / "assignment_manifest.json").exists():
            return path
    raise FileNotFoundError(f"Could not locate returned reviewer folder from {candidates}")


def collect_task(root: Path, task_dir: str, task_type: str) -> tuple[dict[str, list[dict[str, str]]], int]:
    task_root = root / task_dir
    if not task_root.exists():
        raise FileNotFoundError(task_root)
    allowed = TEST_LABELS if task_type == "test" else STEM_LABELS if task_type == "stem" else GEOMETRY_LABELS
    results: dict[str, list[dict[str, str]]] = defaultdict(list)
    pending = 0
    for category in sorted(path for path in task_root.iterdir() if path.is_dir()):
        files = sorted(category.glob("*.jpg"))
        if category.name == "待审核":
            pending += len(files)
            continue
        label = label_from_category(category.name, allowed)
        for path in files:
            item_id = item_id_from_name(path.name, task_type)
            results[item_id].append(
                {
                    "label": label,
                    "category": category.name,
                    "path": str(path),
                    "filename": path.name,
                }
            )
    return dict(results), pending


def expected_ids(folder: Path, task_type: str) -> set[str]:
    rows = json.loads((folder / "assignment_manifest.json").read_text(encoding="utf-8"))
    key = "image_id" if task_type == "geometry" else "instance_id"
    return {str(row[key]) for row in rows if row["task_type"] == task_type}


def task_audit(
    name: str,
    expected: set[str],
    results: dict[str, list[dict[str, str]]],
    pending: int,
) -> dict[str, Any]:
    actual = set(results)
    duplicate_ids = sorted(item_id for item_id, values in results.items() if len(values) != 1)
    return {
        "reviewer": name,
        "expected": len(expected),
        "unique_result_ids": len(actual),
        "result_files": sum(len(values) for values in results.values()),
        "pending_files": pending,
        "missing_ids": sorted(expected - actual),
        "unexpected_ids": sorted(actual - expected),
        "duplicate_or_conflicting_ids": duplicate_ids,
        "complete_unique_coverage": not pending and expected == actual,
    }


def labeled_records(results: dict[str, list[dict[str, str]]], task_type: str, reviewer: str) -> list[dict[str, Any]]:
    records = []
    for item_id, values in sorted(results.items()):
        records.append(
            {
                "task_type": task_type,
                "reviewer": reviewer,
                "item_id": item_id,
                "labels": [value["label"] for value in values],
                "files": [value["path"] for value in values],
            }
        )
    return records


def adjudication_panel(image: np.ndarray, instance_id: str, a_labels: list[str], b_labels: list[str]) -> np.ndarray:
    width = image.shape[1]
    bar = np.full((92, width, 3), 10, dtype=np.uint8)
    line1 = f"FINAL ADJUDICATION | {instance_id}"
    line2 = f"Reviewer A={' + '.join(a_labels) or 'MISSING'} | Reviewer B={' + '.join(b_labels) or 'MISSING'}"
    cv2.putText(bar, line1, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(bar, line2, (18, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, image])


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    review_root = resolve(project_root, args.review_root)
    output = resolve(project_root, args.output_dir)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if review_root not in output.parents:
        raise ValueError("Output must be inside the review root")

    roots = {
        "lead": locate_returned_root(review_root, [Path("01_总审核人"), Path("01_总负责人")]),
        "reviewer_a": locate_returned_root(
            review_root,
            [Path("02_审核员Aok/02_审核员A"), Path("02_审核员Aok"), Path("02_审核员A")],
        ),
        "reviewer_b": locate_returned_root(
            review_root,
            [Path("03_审核员Bok/03_审核员B"), Path("03_审核员Bok"), Path("03_审核员B")],
        ),
    }
    task_dirs = {
        "stem": "01_果梗复核",
        "geometry": "02_结构异常复核",
        "test": "03_测试集双人复核",
    }
    all_results: dict[str, dict[str, dict[str, list[dict[str, str]]]]] = {}
    audits: list[dict[str, Any]] = []
    for reviewer, folder in roots.items():
        all_results[reviewer] = {}
        for task_type, task_dir in task_dirs.items():
            results, pending = collect_task(folder, task_dir, task_type)
            expected = expected_ids(folder, task_type)
            all_results[reviewer][task_type] = results
            audits.append(task_audit(reviewer, expected, results, pending))

    output.mkdir(parents=True)
    combined_stem: dict[str, dict[str, Any]] = {}
    combined_geometry: dict[str, dict[str, Any]] = {}
    for reviewer in roots:
        for item_id, values in all_results[reviewer]["stem"].items():
            if item_id in combined_stem:
                raise ValueError(f"Stem assignment duplicated across reviewers: {item_id}")
            combined_stem[item_id] = {"reviewer": reviewer, "labels": [v["label"] for v in values]}
        for item_id, values in all_results[reviewer]["geometry"].items():
            if item_id in combined_geometry:
                raise ValueError(f"Geometry assignment duplicated across reviewers: {item_id}")
            combined_geometry[item_id] = {"reviewer": reviewer, "labels": [v["label"] for v in values]}

    expected_test = expected_ids(roots["reviewer_a"], "test")
    if expected_test != expected_ids(roots["reviewer_b"], "test"):
        raise ValueError("Reviewer A and B expected test sets differ")
    a_results = all_results["reviewer_a"]["test"]
    b_results = all_results["reviewer_b"]["test"]
    agreements: list[dict[str, Any]] = []
    disagreements: list[dict[str, Any]] = []
    for instance_id in sorted(expected_test | set(a_results) | set(b_results)):
        a_values = a_results.get(instance_id, [])
        b_values = b_results.get(instance_id, [])
        a_labels = [value["label"] for value in a_values]
        b_labels = [value["label"] for value in b_values]
        record = {
            "instance_id": instance_id,
            "reviewer_a_labels": a_labels,
            "reviewer_b_labels": b_labels,
            "reviewer_a_files": [value["path"] for value in a_values],
            "reviewer_b_files": [value["path"] for value in b_values],
        }
        if (
            len(a_labels) == 1
            and len(b_labels) == 1
            and a_labels[0] == b_labels[0]
            and a_labels[0] != "UNCERTAIN"
        ):
            record["final_label"] = a_labels[0]
            agreements.append(record)
        else:
            reasons = []
            if len(a_labels) != 1:
                reasons.append("reviewer_a_internal_duplicate_or_missing")
            if len(b_labels) != 1:
                reasons.append("reviewer_b_internal_duplicate_or_missing")
            if "UNCERTAIN" in a_labels + b_labels:
                reasons.append("reviewer_uncertain")
            if len(a_labels) == 1 and len(b_labels) == 1 and a_labels[0] != b_labels[0]:
                reasons.append("reviewers_disagree")
            record["reasons"] = reasons
            disagreements.append(record)

    adjudication_root = output / "总负责人最终裁决"
    for category in (
        "待审核",
        "不合格_UNQUALIFIED",
        "一级_GRADE1",
        "二级_GRADE2",
        "三级_GRADE3",
        "无法判断_UNCERTAIN",
    ):
        (adjudication_root / category).mkdir(parents=True, exist_ok=True)
    for record in disagreements:
        source_candidates = record["reviewer_a_files"] + record["reviewer_b_files"]
        if not source_candidates:
            raise FileNotFoundError(f"No preview available for {record['instance_id']}")
        source = Path(source_candidates[0])
        panel = adjudication_panel(
            read_image(source),
            record["instance_id"],
            record["reviewer_a_labels"],
            record["reviewer_b_labels"],
        )
        base_suffix = ""
        match = re.search(r"(__BASE_G\d+)", source.stem)
        if match:
            base_suffix = match.group(1)
        filename = f"{record['instance_id']}{base_suffix}.jpg"
        output_path = adjudication_root / "待审核" / filename
        write_image(output_path, panel)
        record["adjudication_file"] = str(output_path)

    notes = {}
    for reviewer, folder in roots.items():
        note_path = folder / "争议备注.txt"
        notes[reviewer] = note_path.read_text(encoding="utf-8-sig") if note_path.exists() else ""

    stem_distribution = Counter(
        labels[0]
        for row in combined_stem.values()
        if len(labels := row["labels"]) == 1
    )
    geometry_distribution = Counter(
        labels[0]
        for row in combined_geometry.values()
        if len(labels := row["labels"]) == 1
    )
    agreement_distribution = Counter(row["final_label"] for row in agreements)
    summary = {
        "reviewer_roots": {key: str(value) for key, value in roots.items()},
        "task_audits": audits,
        "stem": {
            "instances": len(combined_stem),
            "distribution": dict(sorted(stem_distribution.items())),
        },
        "geometry": {
            "images": len(combined_geometry),
            "distribution": dict(sorted(geometry_distribution.items())),
        },
        "test": {
            "expected_instances": len(expected_test),
            "agreements": len(agreements),
            "agreement_rate": len(agreements) / len(expected_test),
            "agreement_distribution": dict(sorted(agreement_distribution.items())),
            "needs_adjudication": len(disagreements),
            "reviewer_a_result_files": sum(len(v) for v in a_results.values()),
            "reviewer_b_result_files": sum(len(v) for v in b_results.values()),
            "reviewer_b_duplicate_or_conflicting_ids": sum(len(v) != 1 for v in b_results.values()),
        },
        "ready_to_rebuild_dataset": len(disagreements) == 0,
    }
    (output / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "stem_results.json").write_text(json.dumps(combined_stem, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "geometry_results.json").write_text(json.dumps(combined_geometry, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "test_agreements.json").write_text(json.dumps(agreements, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "test_disagreements.json").write_text(json.dumps(disagreements, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "reviewer_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    readme = f"""# 审核结果汇总与最终裁决

三份返回结果已做完整性与一致性审计。

## 已确认结果

- 果梗复核：{len(combined_stem)} 个实例；有果梗 {stem_distribution.get('PRESENT', 0)}，无果梗 {stem_distribution.get('MISSING', 0)}，无法判断 {stem_distribution.get('UNCERTAIN', 0)}。
- 结构异常复核：{len(combined_geometry)} 张；可修复保留 {geometry_distribution.get('KEEP_FIX', 0)}，排除 {geometry_distribution.get('EXCLUDE', 0)}，无法判断 {geometry_distribution.get('UNCERTAIN', 0)}。
- 测试集双人一致：{len(agreements)}/{len(expected_test)}（{len(agreements) / len(expected_test):.1%}）。
- 需要总负责人裁决：{len(disagreements)} 个测试实例。

## 总负责人怎么做

进入“总负责人最终裁决/待审核”，每张图片顶部显示审核员A和B的结果。

1. 独立判断最终类别。
2. 将图片移动到“不合格、一级、二级、三级”中的一个文件夹。
3. 确实无法决定时放入“无法判断_UNCERTAIN”，该实例不会进入最终测试集。
4. 不要修改文件名，也不要保留多份副本。

完成裁决后再进行最终标签合并和数据集重建。当前不要直接训练，因为测试集仍有争议标签。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
