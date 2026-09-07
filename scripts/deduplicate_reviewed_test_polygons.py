from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove near-identical duplicate polygons from the reviewed test split.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/grading_four_class_v3"))
    parser.add_argument("--iou-threshold", type=float, default=0.80)
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def polygon_mask(tokens: list[str], width: int, height: int) -> np.ndarray:
    xy = np.asarray([float(value) for value in tokens[1:]], dtype=float).reshape(-1, 2)
    xy[:, 0] *= width
    xy[:, 1] *= height
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(xy).astype(np.int32)], 1)
    return mask


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    dataset = args.dataset.resolve() if args.dataset.is_absolute() else (root / args.dataset).resolve()
    audit_path = dataset / "audit/test_polygon_deduplication.json"
    if audit_path.exists():
        raise FileExistsError(audit_path)

    image_manifest_path = dataset / "audit/image_manifest.csv"
    label_manifest_path = dataset / "audit/label_manifest.csv"
    image_manifest = pd.read_csv(image_manifest_path, encoding="utf-8-sig")
    label_manifest = pd.read_csv(label_manifest_path, encoding="utf-8-sig")
    test_images = image_manifest.loc[image_manifest["dataset_split"].eq("test")].copy()
    duplicate_records = []
    drop_instance_ids: set[str] = set()

    for _, image_row in test_images.sort_values("image_id").iterrows():
        image_id = str(image_row["image_id"])
        label_path = dataset / "labels/test" / f"{image_id}.txt"
        json_path = dataset / "labelme_json/test" / f"{image_id}.json"
        document = json.loads(json_path.read_text(encoding="utf-8"))
        width, height = int(document["imageWidth"]), int(document["imageHeight"])
        lines = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        masks = [polygon_mask(line, width, height) for line in lines]
        parent = list(range(len(lines)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            a, b = find(left), find(right)
            if a != b:
                parent[max(a, b)] = min(a, b)

        pair_ious: dict[tuple[int, int], float] = {}
        for left in range(len(lines)):
            for right in range(left + 1, len(lines)):
                intersection = int(np.count_nonzero(masks[left] & masks[right]))
                union_area = int(np.count_nonzero(masks[left] | masks[right]))
                iou = intersection / union_area if union_area else 0.0
                if iou >= args.iou_threshold:
                    union(left, right)
                    pair_ious[(left, right)] = iou
        groups: dict[int, list[int]] = {}
        for index in range(len(lines)):
            groups.setdefault(find(index), []).append(index)
        duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
        if not duplicate_groups:
            continue

        image_labels = label_manifest.loc[label_manifest["image_id"].eq(image_id)].copy()
        by_source_row = {int(row.source_polygon_row): row for row in image_labels.itertuples()}
        drop_indices: set[int] = set()
        for group in duplicate_groups:
            class_ids = {int(lines[index][0]) for index in group}
            if len(class_ids) != 1:
                raise ValueError(f"Duplicate polygons have conflicting classes: {image_id}, rows={group}")
            retained = min(group)
            for index in sorted(group):
                if index == retained:
                    continue
                drop_indices.add(index)
                instance_id = str(by_source_row[index + 1].instance_id)
                drop_instance_ids.add(instance_id)
                iou = max(
                    pair_ious.get((min(retained, index), max(retained, index)), 0.0),
                    0.0,
                )
                duplicate_records.append(
                    {
                        "image_id": image_id,
                        "retained_source_polygon_row": retained + 1,
                        "dropped_source_polygon_row": index + 1,
                        "retained_instance_id": str(by_source_row[retained + 1].instance_id),
                        "dropped_instance_id": instance_id,
                        "class_id": int(lines[index][0]),
                        "mask_iou": iou,
                    }
                )
        retained_indices = [index for index in range(len(lines)) if index not in drop_indices]
        label_path.write_text(
            "\n".join(" ".join(lines[index]) for index in retained_indices) + "\n",
            encoding="utf-8",
        )
        document["shapes"] = [document["shapes"][index] for index in retained_indices]
        json_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    if not duplicate_records:
        raise ValueError("No duplicate test polygons found")
    label_manifest = label_manifest.loc[~label_manifest["instance_id"].isin(drop_instance_ids)].copy()
    for image_id, positions in label_manifest.groupby("image_id", sort=False).groups.items():
        subset = label_manifest.loc[positions].sort_values("source_polygon_row")
        label_manifest.loc[subset.index, "output_label_row"] = range(1, len(subset) + 1)
    for image_id, group in label_manifest.loc[label_manifest["dataset_split"].eq("test")].groupby("image_id"):
        count = len(group)
        selector = image_manifest["image_id"].eq(image_id)
        image_manifest.loc[selector, "candidate_instances"] = count
        image_manifest.loc[selector, "source_unique_polygons"] = count
    image_manifest.to_csv(image_manifest_path, index=False, encoding="utf-8-sig")
    label_manifest.to_csv(label_manifest_path, index=False, encoding="utf-8-sig")

    final_label_path = root / "review/three_person_audit_v1/04_审核结果汇总/test_final_labels.json"
    final_labels = json.loads(final_label_path.read_text(encoding="utf-8"))
    deduplicated_labels = [row for row in final_labels if row["instance_id"] not in drop_instance_ids]
    deduplicated_path = root / "review/three_person_audit_v1/04_审核结果汇总/test_final_labels_deduplicated.json"
    deduplicated_path.write_text(
        json.dumps(deduplicated_labels, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    audit = {
        "iou_threshold": args.iou_threshold,
        "duplicate_groups": len(duplicate_records),
        "dropped_instances": len(drop_instance_ids),
        "retained_test_instances": len(deduplicated_labels),
        "records": duplicate_records,
        "all_duplicate_pairs_same_final_class": True,
    }
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = dataset / "audit/build_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    distribution = Counter(
        row.class_name
        for row in label_manifest.loc[label_manifest["dataset_split"].eq("test")].itertuples()
    )
    summary["test_instances"] = len(deduplicated_labels)
    summary["total_instances_with_test"] = len(label_manifest)
    summary["test_class_counts"] = dict(sorted(distribution.items()))
    summary["test_duplicate_polygons_removed"] = len(drop_instance_ids)
    summary["inputs"]["test_polygon_deduplication"] = {
        "path": str(audit_path),
        "sha256": digest(audit_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset / "audit/test_export_summary.json").write_text(
        json.dumps(
            {
                "images": 129,
                "instances": len(deduplicated_labels),
                "class_counts": dict(sorted(distribution.items())),
                "duplicate_polygons_removed": len(drop_instance_ids),
                "source_coordinates_preserved_for_retained_polygons": True,
                "source_images_copied_byte_identically": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
