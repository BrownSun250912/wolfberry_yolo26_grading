from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build transparent multi-tier rot screening queues.")
    parser.add_argument("--config", required=True, type=Path)
    return parser.parse_args()


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    value = str(value).strip().upper()
    return "" if value in {"", "34", "34.0", "NAN", "NONE"} else value


def add_tiers(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    signals = list(config["signals"])
    for level in ["broad", "medium"]:
        columns = []
        for signal in signals:
            column = f"{level}_{signal}_pass"
            result[column] = pd.to_numeric(result[signal], errors="coerce") >= float(
                config["signals"][signal][level]
            )
            columns.append(column)
        result[f"{level}_signal_count"] = result[columns].sum(axis=1).astype(int)
        result[f"{level}_pass"] = result[f"{level}_signal_count"] >= int(
            config["minimum_signals"][level]
        )

    result["high_confidence_pass"] = result["decision"].eq("AUTO_ROT_INTERNAL")
    result["rot_exclusion_candidate"] = (
        result["broad_signal_count"] >= int(config["minimum_signals"]["exclusion"])
    ) | result["high_confidence_pass"]
    result["screening_tier"] = np.select(
        [
            result["high_confidence_pass"],
            result["medium_pass"],
            result["broad_pass"],
        ],
        ["HIGH_CONFIDENCE", "MEDIUM_CANDIDATE", "BROAD_CANDIDATE"],
        default="LOW_EVIDENCE",
    )
    location = config["location_flags"]
    calyx = (
        pd.to_numeric(result["calyx_adjacent_abnormal_px"], errors="coerce").fillna(0)
        >= int(location["calyx_adjacent_min_px"])
    ) & (
        pd.to_numeric(result["calyx_adjacent_abnormal_ratio"], errors="coerce").fillna(0)
        >= float(location["calyx_adjacent_min_ratio"])
    )
    edge = (
        pd.to_numeric(result["noncalyx_edge_abnormal_ratio"], errors="coerce").fillna(0)
        >= float(location["edge_abnormal_min_ratio"])
    )
    result["location_flag"] = np.select(
        [calyx, edge],
        ["CALYX_OR_TIP", "EDGE_OR_CONTACT"],
        default="INTERIOR_OR_UNRESOLVED",
    )
    result["needs_rot_review"] = result["screening_tier"].ne("LOW_EVIDENCE")
    processing_ok = result["processing_success"].astype(str).str.upper().eq("TRUE")
    reviewed_segmentation = result["segmentation_usable"].map(text)
    segmentation_ok = reviewed_segmentation.eq("YES") | reviewed_segmentation.eq("")
    result["grade_analysis_candidate"] = (
        ~result["rot_exclusion_candidate"] & processing_ok & segmentation_ok
    )
    return result


def reviewed_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    usable = frame.loc[frame["segmentation_usable"].map(text).eq("YES")].copy()
    y = usable["rot"].map(text).eq("YES")

    def calculate(predicted: pd.Series) -> dict[str, float | int | None]:
        tp = int((predicted & y).sum())
        fp = int((predicted & ~y).sum())
        fn = int((~predicted & y).sum())
        tn = int((~predicted & ~y).sum())
        return {
            "selected": int(predicted.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        }

    return {
        "usable_reviewed_instances": int(len(usable)),
        "rot_positive": int(y.sum()),
        "rot_negative": int((~y).sum()),
        "high_confidence": calculate(usable["high_confidence_pass"]),
        "medium_or_higher": calculate(usable["medium_pass"] | usable["high_confidence_pass"]),
        "broad_or_higher": calculate(usable["broad_pass"] | usable["high_confidence_pass"]),
        "high_recall_exclusion": calculate(usable["rot_exclusion_candidate"]),
        "warning": (
            "The reviewed set was feature-stratified and the same labels were used to calibrate thresholds; "
            "these are apparent screening metrics, not independent test performance."
        ),
    }


def write_outputs(frame: pd.DataFrame, root: Path, prefix: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / f"{prefix}_all_instances.csv", index=False, encoding="utf-8-sig")
    frame.loc[frame["rot_exclusion_candidate"]].to_csv(
        root / f"{prefix}_rot_exclusion_candidates.csv", index=False, encoding="utf-8-sig"
    )
    frame.loc[frame["grade_analysis_candidate"]].to_csv(
        root / f"{prefix}_grade_analysis_candidates.csv", index=False, encoding="utf-8-sig"
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    root = resolve(base, config["output"]["root"])

    reviewed = pd.read_csv(resolve(base, config["input"]["reviewed_results"]), encoding="utf-8-sig")
    full = pd.read_csv(resolve(base, config["input"]["full_results"]), encoding="utf-8-sig")
    reviewed = add_tiers(reviewed, config)
    full = add_tiers(full, config)
    write_outputs(reviewed, root, "reviewed_240")
    write_outputs(full, root, "full")

    metrics = reviewed_metrics(reviewed)
    metrics["config_version"] = config["config_version"]
    metrics["full_instance_count"] = int(len(full))
    metrics["full_tier_counts"] = {
        str(key): int(value) for key, value in full["screening_tier"].value_counts().items()
    }
    metrics["full_broad_or_higher_count"] = int(
        (full["broad_pass"] | full["high_confidence_pass"]).sum()
    )
    metrics["full_medium_or_higher_count"] = int(
        (full["medium_pass"] | full["high_confidence_pass"]).sum()
    )
    metrics["full_rot_exclusion_candidate_count"] = int(full["rot_exclusion_candidate"].sum())
    metrics["full_grade_analysis_candidate_count"] = int(full["grade_analysis_candidate"].sum())
    (root / "screening_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    readme = """# Rot screening tiers v2

This corrects the interpretation of the earlier conservative triage. The high-confidence list is not the estimated
number of rotten fruits. It is only the smallest, strongest-evidence queue.

- `screening_tier` retains HIGH/MEDIUM/BROAD/LOW evidence levels in the complete table for traceability.

For constructing the later grade-standard analysis pool, `rot_exclusion_candidate` uses a deliberately wider rule:
any one of the five BROAD signals triggers temporary exclusion. `grade_analysis_candidate` is the complementary,
successfully processed pool. Exclusion is reversible and must not be converted directly into a rot ground-truth label.

All tiers are candidates for assisted labeling, not final labels. `location_flag` separates calyx/tip and edge/contact
interference for manual review. Do not write these tiers directly into YOLO labels without checking the images.
"""
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
