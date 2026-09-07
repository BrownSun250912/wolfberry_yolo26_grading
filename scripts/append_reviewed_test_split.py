from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml


CLASS_ID = {"UNQUALIFIED": 0, "GRADE1": 1, "GRADE2": 2, "GRADE3": 3}
CLASS_NAME = {0: "unqualified", 1: "grade_1", 2: "grade_2", 3: "grade_3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append the completed double-reviewed test split to a built dataset.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/grading_four_class_v3"))
    return parser.parse_args()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    dataset = args.dataset.resolve() if args.dataset.is_absolute() else (root / args.dataset).resolve()
    if (dataset / "images/test").exists():
        raise FileExistsError("Test split already exists")
    label_source = root / "review/three_person_audit_v1/04_审核结果汇总/test_final_labels.json"
    labels = json.loads(label_source.read_text(encoding="utf-8"))
    by_instance = {row["instance_id"]: row for row in labels}
    if len(by_instance) != 582 or any(row["final_label"] not in CLASS_ID for row in labels):
        raise ValueError("Expected 582 unique finalized test labels")

    manifest_path = root / "data/manifests/dataset_manifest.csv"
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    test_manifest = manifest.loc[manifest["split"].eq("test")].sort_values("new_stem")
    if len(test_manifest) != 129 or int(test_manifest["instance_count"].sum()) != 582:
        raise ValueError("Unexpected original test manifest")

    for folder in ("images/test", "labels/test", "labelme_json/test"):
        (dataset / folder).mkdir(parents=True, exist_ok=False)

    image_rows = []
    label_rows = []
    source_dataset = root / "data/split_v1"
    for _, record in test_manifest.iterrows():
        image_id = str(record["new_stem"])
        source_image = source_dataset / "images/test" / f"{image_id}.png"
        source_txt = source_dataset / "labels/test" / f"{image_id}.txt"
        source_json = source_dataset / "labelme_json/test" / f"{image_id}.json"
        old_lines = [line.split() for line in source_txt.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if len(old_lines) != int(record["instance_count"]):
            raise ValueError(f"Source test line count mismatch: {image_id}")
        output_lines = []
        shapes = []
        for index, old in enumerate(old_lines, start=1):
            instance_id = f"{image_id}__i{index:03d}"
            if instance_id not in by_instance:
                raise ValueError(f"Missing final test label: {instance_id}")
            final = by_instance[instance_id]
            class_id = CLASS_ID[final["final_label"]]
            output_lines.append(" ".join([str(class_id), *old[1:]]))
            xy = [float(value) for value in old[1:]]
            points = [
                [xy[pos] * int(record["image_width"]), xy[pos + 1] * int(record["image_height"])]
                for pos in range(0, len(xy), 2)
            ]
            shapes.append(
                {
                    "label": CLASS_NAME[class_id],
                    "points": points,
                    "group_id": None,
                    "description": "",
                    "shape_type": "polygon",
                    "flags": {},
                    "mask": None,
                }
            )
            label_rows.append(
                {
                    "image_id": image_id,
                    "instance_id": instance_id,
                    "original_split": "test",
                    "dataset_split": "test",
                    "source_polygon_row": index,
                    "output_label_row": index,
                    "class_id": class_id,
                    "class_name": CLASS_NAME[class_id],
                    "match_iou": 1.0,
                    "match_margin": 1.0,
                    "grade_source": final["label_source"],
                    "stem_decision": "TEST_FINAL_REVIEW_OR_FROZEN_BASE_RULE",
                    "mask_label_source": "original_human_polygon",
                }
            )
        output_image = dataset / "images/test" / source_image.name
        output_txt = dataset / "labels/test" / source_txt.name
        output_json = dataset / "labelme_json/test" / source_json.name
        shutil.copy2(source_image, output_image)
        output_txt.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
        document = {
            "version": "5.5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": f"../../images/test/{image_id}.png",
            "imageData": None,
            "imageHeight": int(record["image_height"]),
            "imageWidth": int(record["image_width"]),
        }
        output_json.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        image_rows.append(
            {
                "image_id": image_id,
                "original_split": "test",
                "candidate_instances": len(old_lines),
                "original_manifest_instances": len(old_lines),
                "source_unique_polygons": len(old_lines),
                "status": "EXPORT",
                "reason": "",
                "match_min_iou": 1.0,
                "match_min_margin": 1.0,
                "duplicate_group": image_id,
                "boundary_review_used": False,
                "dataset_split": "test",
                "source_image": str(source_image),
                "image_sha256": digest(source_image),
                "source_txt_sha256": digest(source_txt),
                "source_json_sha256": digest(source_json),
                "output_image": f"images/test/{image_id}.png",
                "output_label": f"labels/test/{image_id}.txt",
            }
        )

    image_manifest_path = dataset / "audit/image_manifest.csv"
    label_manifest_path = dataset / "audit/label_manifest.csv"
    image_manifest = pd.read_csv(image_manifest_path, encoding="utf-8-sig")
    label_manifest = pd.read_csv(label_manifest_path, encoding="utf-8-sig")
    image_manifest = pd.concat([image_manifest, pd.DataFrame(image_rows)], ignore_index=True)
    label_manifest = pd.concat([label_manifest, pd.DataFrame(label_rows)], ignore_index=True)
    if image_manifest["image_id"].duplicated().any() or label_manifest["instance_id"].duplicated().any():
        raise ValueError("Appending test introduced duplicate IDs")
    image_manifest.to_csv(image_manifest_path, index=False, encoding="utf-8-sig")
    label_manifest.to_csv(label_manifest_path, index=False, encoding="utf-8-sig")

    config_path = dataset / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["test"] = "images/test"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    summary_path = dataset / "audit/build_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    test_distribution = Counter(CLASS_NAME[CLASS_ID[row["final_label"]]] for row in labels)
    summary.update(
        {
            "test_available": True,
            "test_graded": True,
            "test_images": 129,
            "test_instances": 582,
            "total_images_with_test": int(len(image_manifest)),
            "total_instances_with_test": int(len(label_manifest)),
            "test_class_counts": dict(sorted(test_distribution.items())),
            "label_policy": "human polygons + manual rot/stem review + frozen interpretable grade rule + double-reviewed test",
        }
    )
    summary["inputs"]["test_final_labels"] = {
        "path": str(label_source),
        "sha256": digest(label_source),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (dataset / "audit/test_export_summary.json").write_text(
        json.dumps(
            {
                "images": 129,
                "instances": 582,
                "class_counts": dict(sorted(test_distribution.items())),
                "source_coordinates_preserved": True,
                "source_images_copied_byte_identically": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
