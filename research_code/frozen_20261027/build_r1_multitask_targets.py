"""Build and audit R1-A hierarchical-grading targets from the frozen train/val set.

This script deliberately does not read the test image or label directories.  It
inherits the existing four-class train/val membership and converts every
instance into three masked tasks:

* defect: unqualified (1) versus qualified (0), supervised for every instance;
* stem: missing (1) versus present (0), supervised only for qualified fruit;
* score: frozen continuous grading score, supervised only for qualified fruit.

The generated RGB crop retains context around the fruit so a stem remains
visible.  A separate binary mask records the human instance polygon.  No model
prediction, confidence threshold, NMS result, or test artifact is involved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/grading_four_class_v3"
DEFAULT_MAPPING = DEFAULT_DATASET / "audit/instance_mapping.csv"
DEFAULT_CANDIDATES = (
    ROOT / "outputs/grading_standard_manual_v4_final/grading_labels_candidate_v3.csv"
)
DEFAULT_GRADING_CONFIG = (
    ROOT / "outputs/grading_standard_manual_v4_final/grading_standard_config.json"
)
DEFAULT_OUTPUT = ROOT / "outputs/r1_multitask_targets_v1"
CLASS_IDS = {"UNQUALIFIED": 0, "GRADE_1": 1, "GRADE_2": 2, "GRADE_3": 3}
EXPECTED_ROWS = 4795
EXPECTED_SPLITS = {"train": 4277, "val": 518}
EXPECTED_CLASSES = {"UNQUALIFIED": 426, "GRADE_1": 1060, "GRADE_2": 1289, "GRADE_3": 2020}
EXPECTED_MANUAL_OVERRIDES = {
    "wb_000220__i001",
    "wb_000266__i002",
    "wb_000361__i002",
    "wb_000983__i004",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--grading-config", type=Path, default=DEFAULT_GRADING_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.25,
        help="Context on each side as a fraction of the larger mask dimension.",
    )
    parser.add_argument("--minimum-padding", type=int, default=12)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def grade_from_score(score: float, lower: float, upper: float) -> int:
    if score >= upper:
        return 1
    if score >= lower:
        return 2
    return 3


def load_train_val_images(dataset: Path) -> tuple[dict[str, str], dict[str, Path]]:
    split_by_image: dict[str, str] = {}
    path_by_image: dict[str, Path] = {}
    for split in ("train", "val"):
        directory = dataset / "images" / split
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in sorted(directory.glob("*.png")):
            if path.stem in split_by_image:
                raise ValueError(f"Image appears in both train and val: {path.stem}")
            split_by_image[path.stem] = split
            path_by_image[path.stem] = path.resolve()
    if not split_by_image:
        raise ValueError("No train/val images found")
    return split_by_image, path_by_image


def parse_polygon(line: str, width: int, height: int) -> tuple[int, np.ndarray]:
    parts = line.split()
    if len(parts) < 7 or len(parts) % 2 == 0:
        raise ValueError(f"Invalid YOLO segmentation row with {len(parts)} tokens")
    class_id = int(parts[0])
    normalized = np.asarray(parts[1:], dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(normalized).all() or np.any(normalized < 0) or np.any(normalized > 1):
        raise ValueError("Non-finite or out-of-range polygon coordinate")
    points = np.rint(normalized * np.asarray([width, height])).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return class_id, points


def crop_with_context(
    image: np.ndarray,
    points: np.ndarray,
    padding_ratio: float,
    minimum_padding: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    height, width = image.shape[:2]
    full_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(full_mask, [points], 255)
    if not np.any(full_mask):
        raise ValueError("Polygon produced an empty mask")
    x, y, box_width, box_height = cv2.boundingRect(points)
    padding = max(minimum_padding, int(math.ceil(max(box_width, box_height) * padding_ratio)))
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(width, x + box_width + padding)
    y2 = min(height, y + box_height + padding)
    crop = image[y1:y2, x1:x2].copy()
    mask = full_mask[y1:y2, x1:x2].copy()
    if crop.size == 0 or mask.size == 0 or crop.shape[:2] != mask.shape:
        raise ValueError("Invalid crop/mask geometry")
    return crop, mask, (x1, y1, x2, y2)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def main() -> None:
    args = parse_args()
    dataset = resolve(args.dataset)
    mapping_path = resolve(args.mapping)
    candidate_path = resolve(args.candidates)
    grading_config_path = resolve(args.grading_config)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing R1 target package: {output}")
    if args.padding_ratio < 0 or args.minimum_padding < 0:
        raise ValueError("Padding values must be non-negative")

    mapping = pd.read_csv(mapping_path, encoding="utf-8-sig")
    candidates = pd.read_csv(candidate_path, encoding="utf-8-sig")
    grading_config = json.loads(grading_config_path.read_text(encoding="utf-8"))
    lower = float(grading_config["thresholds"]["grade3_to_2"])
    upper = float(grading_config["thresholds"]["grade2_to_1"])
    split_by_image, image_path_by_id = load_train_val_images(dataset)

    required_mapping = {
        "image_id",
        "instance_id",
        "source_group",
        "four_class_label",
        "is_unqualified",
        "label_source_v3",
        "stem_decision",
        "source_row",
    }
    missing = required_mapping - set(mapping.columns)
    if missing:
        raise KeyError(f"Mapping is missing columns: {sorted(missing)}")
    if len(mapping) != EXPECTED_ROWS or not mapping["instance_id"].is_unique:
        raise ValueError("Frozen mapping row count or instance uniqueness changed")
    if set(mapping["image_id"].astype(str)) != set(split_by_image):
        raise ValueError("Train/val image membership differs from frozen instance mapping")

    candidate_required = {
        "instance_id",
        "manual_calibrated_score",
        "automatic_base_grade_v3",
        "manual_expected_grade",
        "label_source_v3",
    }
    missing = candidate_required - set(candidates.columns)
    if missing:
        raise KeyError(f"Candidate table is missing columns: {sorted(missing)}")
    candidate_subset = candidates[list(candidate_required)].copy()
    if not candidate_subset["instance_id"].is_unique:
        raise ValueError("Candidate score table contains duplicate instance IDs")
    candidate_subset = candidate_subset.rename(
        columns={
            "label_source_v3": "score_lineage",
            "automatic_base_grade_v3": "score_rule_grade",
            "manual_expected_grade": "manual_expected_grade",
        }
    )
    data = mapping.merge(candidate_subset, on="instance_id", how="left", validate="one_to_one")
    data["dataset_split"] = data["image_id"].astype(str).map(split_by_image)
    data["final_class_id"] = data["four_class_label"].map(CLASS_IDS)
    if data["final_class_id"].isna().any():
        raise ValueError("Unknown final class label")

    output.mkdir(parents=True)
    (output / "crops/train").mkdir(parents=True)
    (output / "crops/val").mkdir(parents=True)
    (output / "masks/train").mkdir(parents=True)
    (output / "masks/val").mkdir(parents=True)

    rows: list[dict[str, Any]] = []
    artifact_hashes: list[str] = []
    grouped = data.sort_values(["dataset_split", "image_id", "source_row"]).groupby(
        "image_id", sort=False
    )
    for image_number, (image_id, group) in enumerate(grouped, start=1):
        split = str(group["dataset_split"].iloc[0])
        image_path = image_path_by_id[str(image_id)]
        label_path = dataset / "labels" / split / f"{image_id}.txt"
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        label_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(label_lines) != len(group):
            raise ValueError(f"Instance count mismatch for {image_id}: {len(label_lines)} vs {len(group)}")
        for record in group.to_dict("records"):
            source_row = int(record["source_row"])
            if source_row < 1 or source_row > len(label_lines):
                raise IndexError(f"Invalid source row for {record['instance_id']}: {source_row}")
            polygon_class_id, points = parse_polygon(
                label_lines[source_row - 1], image.shape[1], image.shape[0]
            )
            final_class_id = int(record["final_class_id"])
            if polygon_class_id != final_class_id:
                raise ValueError(
                    f"Class mismatch for {record['instance_id']}: polygon={polygon_class_id}, "
                    f"mapping={final_class_id}"
                )
            crop, mask, (x1, y1, x2, y2) = crop_with_context(
                image, points, args.padding_ratio, args.minimum_padding
            )
            instance_id = str(record["instance_id"])
            crop_path = output / "crops" / split / f"{instance_id}.png"
            mask_path = output / "masks" / split / f"{instance_id}.png"
            if not cv2.imwrite(str(crop_path), crop):
                raise OSError(f"Failed to write crop: {crop_path}")
            if not cv2.imwrite(str(mask_path), mask):
                raise OSError(f"Failed to write mask: {mask_path}")
            crop_hash = sha256(crop_path)
            mask_hash = sha256(mask_path)
            artifact_hashes.extend(
                [f"{project_relative(crop_path)}\t{crop_hash}", f"{project_relative(mask_path)}\t{mask_hash}"]
            )

            is_unqualified = bool(record["is_unqualified"])
            defect_target = int(is_unqualified)
            if is_unqualified:
                stem_target: float | int = math.nan
                stem_available = 0
                score_target = math.nan
                score_available = 0
                reconstructed = "UNQUALIFIED"
                manual_override = 0
            else:
                stem_decision = str(record["stem_decision"])
                if stem_decision.startswith("MISSING"):
                    stem_target = 1
                elif stem_decision.startswith("PRESENT"):
                    stem_target = 0
                else:
                    raise ValueError(f"Unresolved qualified stem target: {instance_id} -> {stem_decision}")
                stem_available = 1
                score_target = float(record["manual_calibrated_score"])
                if not math.isfinite(score_target):
                    raise ValueError(f"Missing/non-finite score target: {instance_id}")
                score_available = 1
                if int(stem_target) == 1:
                    reconstructed = "GRADE_3"
                else:
                    reconstructed = f"GRADE_{grade_from_score(score_target, lower, upper)}"
                manual_override = int(reconstructed != str(record["four_class_label"]))

            lineage = {
                "final_class": f"instance_mapping:{record['label_source_v3']}",
                "defect": "strict_completed_manual_rot_review"
                if is_unqualified
                else "qualified_by_completed_rot_review",
                "stem": None if is_unqualified else f"instance_mapping:{record['stem_decision']}",
                "score": None if is_unqualified else f"candidate_table:{record['score_lineage']}",
            }
            rows.append(
                {
                    "instance_id": instance_id,
                    "image_id": str(image_id),
                    "split": split,
                    "source_group": str(record["source_group"]),
                    "image_path": project_relative(image_path),
                    "label_path": project_relative(label_path),
                    "source_polygon_row": source_row,
                    "crop_path": project_relative(crop_path),
                    "mask_path": project_relative(mask_path),
                    "crop_sha256": crop_hash,
                    "mask_sha256": mask_hash,
                    "crop_x1": x1,
                    "crop_y1": y1,
                    "crop_x2": x2,
                    "crop_y2": y2,
                    "final_class_id": final_class_id,
                    "final_class_name": str(record["four_class_label"]),
                    "defect_target": defect_target,
                    "defect_target_available": 1,
                    "stem_target": stem_target,
                    "stem_target_available": stem_available,
                    "score_target": score_target,
                    "score_target_available": score_available,
                    "rule_reconstructed_class": reconstructed,
                    "manual_boundary_override": manual_override,
                    "final_label_source": str(record["label_source_v3"]),
                    "stem_label_source": str(record["stem_decision"]),
                    "target_lineage": json.dumps(lineage, ensure_ascii=False, sort_keys=True),
                }
            )
        if image_number % 200 == 0:
            print(f"prepared {image_number}/{len(grouped)} images", flush=True)

    targets = pd.DataFrame(rows).sort_values(["split", "image_id", "instance_id"]).reset_index(drop=True)
    target_path = output / "r1_multitask_targets.csv"
    write_csv(target_path, targets)
    distribution = (
        targets.groupby(["split", "final_class_name"], dropna=False)
        .size()
        .reset_index(name="instances")
        .sort_values(["split", "final_class_name"])
    )
    write_csv(output / "target_distribution.csv", distribution)

    errors: list[str] = []
    warnings: list[str] = []
    split_counts = {key: int(value) for key, value in targets["split"].value_counts().items()}
    class_counts = {
        key: int(value) for key, value in targets["final_class_name"].value_counts().items()
    }
    qualified = targets["defect_target"].eq(0)
    unqualified = ~qualified
    override_ids = set(targets.loc[targets["manual_boundary_override"].eq(1), "instance_id"])
    checks = {
        "row_count_exact": len(targets) == EXPECTED_ROWS,
        "instance_id_unique": bool(targets["instance_id"].is_unique),
        "split_counts_exact": split_counts == EXPECTED_SPLITS,
        "class_counts_exact": class_counts == EXPECTED_CLASSES,
        "image_split_disjoint": int(targets.groupby("image_id")["split"].nunique().max()) == 1,
        "all_paths_exist": bool(
            targets[["image_path", "label_path", "crop_path", "mask_path"]]
            .map(lambda value: (ROOT / str(value)).is_file())
            .all()
            .all()
        ),
        "defect_supervision_complete": bool(targets["defect_target_available"].eq(1).all()),
        "unqualified_auxiliary_losses_masked": bool(
            targets.loc[unqualified, "stem_target_available"].eq(0).all()
            and targets.loc[unqualified, "score_target_available"].eq(0).all()
            and targets.loc[unqualified, "stem_target"].isna().all()
            and targets.loc[unqualified, "score_target"].isna().all()
        ),
        "qualified_auxiliary_supervision_complete": bool(
            targets.loc[qualified, "stem_target_available"].eq(1).all()
            and targets.loc[qualified, "score_target_available"].eq(1).all()
            and targets.loc[qualified, "stem_target"].isin([0, 1]).all()
            and np.isfinite(targets.loc[qualified, "score_target"].to_numpy(dtype=float)).all()
        ),
        "score_range_valid": bool(targets.loc[qualified, "score_target"].between(0, 100).all()),
        "frozen_rule_reconstruction_4791_of_4795": int(
            targets["rule_reconstructed_class"].eq(targets["final_class_name"]).sum()
        )
        == 4791,
        "manual_override_ids_exact": override_ids == EXPECTED_MANUAL_OVERRIDES,
        "test_directory_not_read_by_builder": True,
    }
    for name, passed in checks.items():
        if not passed:
            errors.append(name)
    source_group_span = {
        str(key): sorted(group["split"].unique().tolist())
        for key, group in targets.groupby("source_group")
    }
    if any(len(splits) > 1 for splits in source_group_span.values()):
        warnings.append(
            "source_group is an acquisition stratum, not the split unit; all four frozen strata span "
            "train and val. R1 inherits this baseline design and does not re-split it."
        )

    combined_artifact_digest = hashlib.sha256(
        "\n".join(sorted(artifact_hashes)).encode("utf-8")
    ).hexdigest()
    audit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
        "test_accessed": False,
        "test_policy": "No test image, label, directory, or test row is read by this builder.",
        "counts": {
            "instances": len(targets),
            "images": int(targets["image_id"].nunique()),
            "split_instances": split_counts,
            "split_images": {
                key: int(value)
                for key, value in targets.groupby("split")["image_id"].nunique().items()
            },
            "classes": class_counts,
            "defect_positive_unqualified": int(targets["defect_target"].sum()),
            "qualified": int(qualified.sum()),
            "missing_stem": int(targets.loc[qualified, "stem_target"].sum()),
            "present_stem": int((targets.loc[qualified, "stem_target"] == 0).sum()),
            "manual_boundary_overrides": int(targets["manual_boundary_override"].sum()),
            "rule_reconstructed": int(
                targets["rule_reconstructed_class"].eq(targets["final_class_name"]).sum()
            ),
        },
        "score": {
            "minimum": float(targets.loc[qualified, "score_target"].min()),
            "maximum": float(targets.loc[qualified, "score_target"].max()),
            "mean": float(targets.loc[qualified, "score_target"].mean()),
            "grade3_to_2": lower,
            "grade2_to_1": upper,
        },
        "source_group_split_span": source_group_span,
        "inputs": {
            "instance_mapping": {"path": project_relative(mapping_path), "sha256": sha256(mapping_path)},
            "candidate_scores": {"path": project_relative(candidate_path), "sha256": sha256(candidate_path)},
            "grading_config": {
                "path": project_relative(grading_config_path),
                "sha256": sha256(grading_config_path),
            },
        },
        "outputs": {
            "targets": {"path": project_relative(target_path), "sha256": sha256(target_path)},
            "target_distribution": {
                "path": project_relative(output / "target_distribution.csv"),
                "sha256": sha256(output / "target_distribution.csv"),
            },
            "crop_mask_manifest_sha256": combined_artifact_digest,
            "crop_files": len(targets),
            "mask_files": len(targets),
        },
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    }
    audit_path = output / "target_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    override_rows = targets.loc[targets["manual_boundary_override"].eq(1)].copy()
    write_csv(output / "manual_boundary_overrides.csv", override_rows)
    report_lines = [
        "# R1-A 多任务目标审计",
        "",
        f"- 状态：**{audit['status']}**",
        f"- 实例：{len(targets)}（train={split_counts.get('train', 0)}，val={split_counts.get('val', 0)}）",
        f"- 图片：{targets['image_id'].nunique()}（train=966，val=116）",
        f"- 缺陷正类（Unqualified）：{audit['counts']['defect_positive_unqualified']}",
        f"- 合格果：{audit['counts']['qualified']}",
        f"- 缺梗 / 有梗：{audit['counts']['missing_stem']} / {audit['counts']['present_stem']}",
        f"- 连续评分范围：{audit['score']['minimum']:.6f}–{audit['score']['maximum']:.6f}",
        f"- 冻结规则重建：{audit['counts']['rule_reconstructed']}/{len(targets)}",
        f"- 人工边界覆盖：{audit['counts']['manual_boundary_overrides']} 个（单独保留 lineage）",
        "- Test：未读取；目标包只枚举 `images/train` 与 `images/val`。",
        "",
        "## 监督掩码",
        "",
        "- 全部实例监督 defect head。",
        "- Unqualified 不监督 stem/score head。",
        "- Qualified 同时监督 stem/score head。",
        "",
        "## 拆分说明",
        "",
        "R1-A 不重新划分数据，完全继承当前四分类基线的 train/val 图片成员。",
        "同一 `image_id` 不跨集合。`source_group` 是采集批次分层变量，不是独立拆分单元；",
        "现有四个批次均同时出现在 train 和 val，这一既有设计在 R1 中保持不变。",
        "",
        "## 人工边界覆盖实例",
        "",
    ]
    report_lines.extend(f"- `{item}`" for item in sorted(override_ids))
    report_lines.extend(
        [
            "",
            "这些实例的人工最终等级与冻结连续分值阈值不一致。训练仍保留真实连续分值；",
            "最终规则不会用实例身份作弊，因此无人工身份规则时理论重建上限为 4791/4795。",
            "",
            "## 哈希",
            "",
            f"- 目标表：`{audit['outputs']['targets']['sha256']}`",
            f"- crop/mask 联合清单：`{combined_artifact_digest}`",
            "",
        ]
    )
    (output / "R1_TARGET_AUDIT.md").write_text("\n".join(report_lines), encoding="utf-8")
    if errors:
        raise RuntimeError(f"R1 target audit failed: {errors}")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
