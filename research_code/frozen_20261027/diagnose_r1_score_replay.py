"""Post-hoc validation component diagnosis; no training or label/threshold edits."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPPER = 65.62684031710077
LOWER = 37.40298225745565


def score_grade(scores):
    values = np.asarray(scores, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Scores must be finite")
    return np.where(values >= UPPER, 1, np.where(values >= LOWER, 2, 3))


def compose_variants(predictions):
    if predictions.prediction_id.isna().any() or not predictions.prediction_id.is_unique:
        raise ValueError("Prediction identities must be complete and unique")
    for column in ("defect_probability", "stem_missing_probability"):
        values = predictions[column].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Invalid {column}")
    if not predictions.baseline_class_id.isin([0, 1, 2, 3]).all():
        raise ValueError("Invalid baseline class")
    rule = predictions.rule_grade.to_numpy(float)
    if not np.isin(rule, [1, 2, 3]).all():
        raise ValueError("Invalid rule grade")
    learned = score_grade(predictions.score_prediction)
    missing = predictions.stem_missing_probability.to_numpy() >= 0.5
    baseline_uq = predictions.baseline_class_id.to_numpy() == 0
    learned_uq = predictions.defect_probability.to_numpy() >= 0.5
    rule_path = np.where(missing, 3, rule).astype(int)
    learned_path = np.where(missing, 3, learned)
    return {
        "h1_rule_score": np.where(baseline_uq, 0, rule_path),
        "fixed_gate_learned_score": np.where(baseline_uq, 0, learned_path),
        "learned_gate_rule_score": np.where(learned_uq, 0, rule_path),
        "full_r1": np.where(learned_uq, 0, learned_path),
    }


def score_stats(reference, prediction):
    error = np.asarray(prediction, float) - np.asarray(reference, float)
    if not len(error) or not np.isfinite(error).all():
        raise ValueError("Score metric population must be nonempty and finite")
    absolute = np.abs(error)
    return {"n": len(error), "mae": float(absolute.mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))),
            "bias_prediction_minus_reference": float(error.mean()),
            "median_absolute_error": float(np.median(absolute)),
            "p90_absolute_error": float(np.quantile(absolute, .9)),
            "max_absolute_error": float(absolute.max()),
            "absolute_error_gt_5": int((absolute > 5).sum()),
            "absolute_error_gt_10": int((absolute > 10).sum())}


def paired_counts(reference_correct, candidate_correct):
    left, right = np.asarray(reference_correct, bool), np.asarray(candidate_correct, bool)
    if left.shape != right.shape:
        raise ValueError("Paired populations differ")
    return {"both_correct": int((left & right).sum()),
            "lost_correct": int((left & ~right).sum()),
            "gained_correct": int((~left & right).sum()),
            "both_wrong": int((~left & ~right).sum()),
            "net_correct_change": int(right.sum() - left.sum())}


def main():
    from evaluate_r1a_frozen_predictions import replace_classes, variant_summary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/reports/r1_score_diagnostic_20261007")
    output = parser.parse_args().output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    paths = {
        "predictions": "outputs/reports/h1_frozen_replay_20261006/h1_predictions.csv",
        "legacy": "outputs/reports/r1a_hierarchical_grader_e200_frozen_replay_20260929/r1a_predictions.csv",
        "outcomes": "outputs/reports/h1_valfix_replay_20261006/variant_instance_outcomes.csv",
        "targets": "data/grading_four_class_v3_valfix_20261006/audit/r1_multitask_targets.csv",
        "gt_crops": "models/experiments/r1a_hierarchical_grader_e200_resume_epoch150_seed20260929/best_val_predictions.csv",
        "fixed": "outputs/reports/yolo26n_offlineaug_e200_r0_fixed_replay_maskfix_20260929/replay_predictions.csv",
    }
    paths = {key: ROOT / value for key, value in paths.items()}
    inputs = {k: {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for k, p in paths.items()}
    pred, fixed = pd.read_csv(paths["predictions"]), pd.read_csv(paths["fixed"])
    variants = compose_variants(pred)
    if len(pred) != 549 or not np.array_equal(variants["h1_rule_score"], pred.h1_class_id):
        raise ValueError("H1 reconstruction or population changed")
    if not np.array_equal(variants["full_r1"], pred.r1a_class_id):
        raise ValueError("Full R1 reconstruction differs")
    if not fixed.image_path.map(lambda p: Path(p).parent.name == "val").all():
        raise ValueError("Only validation inputs permitted")
    if set(fixed.prediction_id) != set(pred.prediction_id):
        raise ValueError("Fixed prediction identities differ")
    check = pred.merge(pd.read_csv(paths["legacy"]), on="prediction_id", validate="one_to_one", suffixes=("", "_legacy"))
    if len(check) != 549 or not check.r1a_class_id.eq(check.r1a_class_id_legacy).all():
        raise ValueError("Legacy R1 class decisions differ")
    old = pd.read_csv(paths["outcomes"])
    base = old.loc[old.variant.eq("baseline")].copy()
    summaries, outcomes, class_frames, major = [], [], [], []
    for name, classes in variants.items():
        replacement = pd.DataFrame({"prediction_id": pred.prediction_id, "r1a_class_id": classes})
        frame = replace_classes(base, replacement)
        frame["variant"] = name
        summary, class_frame = variant_summary(frame, name)
        if tuple(summary[k] for k in ("gt_instances", "localized_instances", "predictions", "false_positive_predictions")) != (518, 500, 549, 49):
            raise ValueError("Frozen population changed")
        summary["correct_gt_fraction_including_misses"] = summary["correct_instances"] / 518
        summaries.append(summary)
        outcomes.append(frame)
        class_frames.append(class_frame)
        local = frame.loc[frame.record_type.eq("ground_truth") & frame.localization_matched.astype(bool)]
        gt, pr = local.gt_class_id.to_numpy(int), local.pred_class_id.to_numpy(int)
        major.append({"variant": name, "localized_n": len(local),
                      "uq_gate_mismatches": int(((gt == 0) != (pr == 0)).sum()),
                      "grade_1_to_3": int(((gt == 1) & (pr == 3)).sum()),
                      "grade_3_to_1": int(((gt == 3) & (pr == 1)).sum()),
                      "adjacent_qualified_errors": int(((gt > 0) & (pr > 0) & (np.abs(gt - pr) == 1)).sum())})
    if summaries[0]["correct_instances"] != int(old.loc[old.variant.eq("h1"), "class_correct"].sum()) or summaries[-1]["correct_instances"] != int(old.loc[old.variant.eq("r1a"), "class_correct"].sum()):
        raise ValueError("Corrected metric reproduction failed")
    target = pd.read_csv(paths["targets"])
    target = target.loc[target.split.eq("val")].copy()
    target["gt_index"] = target.source_polygon_row.astype(int) - 1
    gt = base.loc[base.record_type.eq("ground_truth")].merge(target, on=["image_id", "gt_index"], validate="one_to_one", how="left")
    if len(gt) != 518 or gt.instance_id.isna().any() or not gt.gt_class_id.eq(gt.final_class_id).all():
        raise ValueError("GT/target identity or corrected class mismatch")
    matched = gt.loc[gt.localization_matched.astype(bool)].merge(pred, on="prediction_id", suffixes=("", "_cache"), validate="one_to_one")
    for name, values in variants.items():
        matched[name] = matched.prediction_id.map(dict(zip(pred.prediction_id, values)))
    scored = matched.loc[matched.defect_target.eq(0) & matched.score_target_available.eq(1)].copy()
    if len(scored) != 448:
        raise ValueError("Expected 448 matched qualified score targets")
    scored["reference_score_grade"] = score_grade(scored.score_target)
    scored["learned_score_grade"] = score_grade(scored.score_prediction)
    scored["reference_boundary_distance"] = np.minimum(abs(scored.score_target - UPPER), abs(scored.score_target - LOWER))
    scored["reference_within_1pt_boundary"] = scored.reference_boundary_distance <= 1
    scored["score_path_active"] = scored.baseline_class_id.ne(0) & scored.stem_missing_probability.lt(.5)
    for source, column in (("learned", "score_prediction"), ("rule", "rule_score")):
        scored[source + "_score_error"] = scored[column] - scored.score_target
        scored[source + "_absolute_error"] = scored[source + "_score_error"].abs()
    scored["absolute_error_change_learned_minus_rule"] = scored.learned_absolute_error - scored.rule_absolute_error
    scored["score_swap_lost_correct"] = scored.h1_rule_score.eq(scored.gt_class_id) & scored.fixed_gate_learned_score.ne(scored.gt_class_id)
    scored["score_swap_gained_correct"] = scored.h1_rule_score.ne(scored.gt_class_id) & scored.fixed_gate_learned_score.eq(scored.gt_class_id)
    scored["image_path"] = scored.prediction_id.map(fixed.set_index("prediction_id").image_path)
    score_summaries = []
    groups = {"all_matched_qualified": np.ones(len(scored), bool), "score_path_active": scored.score_path_active,
              "reference_within_1pt_boundary": scored.reference_within_1pt_boundary,
              "reference_outside_1pt_boundary": ~scored.reference_within_1pt_boundary}
    for group, mask in groups.items():
        subset = scored.loc[mask]
        if not len(subset):
            continue
        for source, column in (("rule", "rule_score"), ("learned", "score_prediction")):
            row = {"population": group, "score_source": source, **score_stats(subset.score_target, subset[column])}
            row["score_only_grade_agreement"] = float(np.mean(score_grade(subset[column]) == subset.reference_score_grade))
            score_summaries.append(row)
    gt_pair = scored.merge(pd.read_csv(paths["gt_crops"])[["instance_id", "score_prediction"]], on="instance_id", suffixes=("", "_gt_crop"), validate="one_to_one")
    gt_crop_stats = score_stats(gt_pair.score_target, gt_pair.score_prediction_gt_crop)
    transitions = []
    for left, right in [("h1_rule_score", "fixed_gate_learned_score"), ("fixed_gate_learned_score", "full_r1"),
                        ("h1_rule_score", "learned_gate_rule_score"), ("learned_gate_rule_score", "full_r1")]:
        transitions.append({"reference": left, "candidate": right, **paired_counts(matched[left].eq(matched.gt_class_id), matched[right].eq(matched.gt_class_id))})
    boundary = []
    for near in (True, False):
        sub = scored.loc[scored.reference_within_1pt_boundary.eq(near)]
        boundary.append({"reference_within_1pt": near, "n": len(sub), "lost_correct": int(sub.score_swap_lost_correct.sum()), "gained_correct": int(sub.score_swap_gained_correct.sum())})
    distribution, stratified = [], []
    for source, column in (("rule", "rule_absolute_error"), ("learned", "learned_absolute_error")):
        bins = pd.cut(scored[column], [-np.inf, 1, 3, 5, 10, np.inf], labels=["0_to_1", "over_1_to_3", "over_3_to_5", "over_5_to_10", "over_10"])
        distribution.extend({"source": source, "absolute_error_band": str(label), "n": int(count)} for label, count in bins.value_counts(sort=False).items())
    for key in ("source_group", "stem_target", "gt_class_id"):
        for value, sub in scored.groupby(key, dropna=False):
            stratified.append({"group_by": key, "value": str(value), **score_stats(sub.score_target, sub.score_prediction)})
    info = {"generated_utc": datetime.now(timezone.utc).isoformat(), "label_version": "v3_valfix_20261006_first_7_only",
            "training_performed": False, "labels_changed": False, "test_accessed_this_run": False,
            "thresholds_changed": False, "inputs": inputs, "thresholds": {"grade_1": UPPER, "grade_2": LOWER},
            "cache_max_score_delta_vs_legacy": float(abs(check.score_prediction - check.score_prediction_legacy).max()),
            "variants": summaries, "paired_transitions": transitions, "major_errors": major,
            "boundary_transitions": boundary, "score_metrics": score_summaries,
            "gt_crop_score_metrics_on_same_448": gt_crop_stats,
            "scope_notes": ["Same 549 proposals, 500 matches, 18 missed GT, 49 unmatched predictions.",
                            "One-point reference boundary band and five/ten-point error bins are descriptive, not new correctness tolerances.",
                            "No samples excluded from headline metrics. UQ is a separate gate, not an ordinal grade.",
                            "Hybrids are post-hoc diagnostics, not new trained models or independent test evidence.",
                            "Second review pending corrections not included. Scores remain rule-generated.",
                            "GT-crop comparison uses an existing lower-precision cache; small numerical differences are not crop-only effects."]}
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(summaries).to_csv(output / "variant_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(class_frames).to_csv(output / "class_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(outcomes).to_csv(output / "variant_instance_outcomes.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(score_summaries).to_csv(output / "score_metrics.csv", index=False, encoding="utf-8-sig")
    scored.sort_values(["learned_absolute_error", "instance_id"], ascending=[False, True]).to_csv(output / "score_cases_ranked.csv", index=False, encoding="utf-8-sig")
    scored.loc[scored.score_swap_lost_correct | scored.score_swap_gained_correct].to_csv(output / "score_swap_changed_cases.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(distribution).to_csv(output / "score_error_distribution.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(stratified).to_csv(output / "score_subgroups.csv", index=False, encoding="utf-8-sig")
    (output / "summary.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != inputs[key]["sha256"]:
            raise RuntimeError(f"Source changed: {key}")
    print(pd.DataFrame(summaries)[["variant", "correct_instances", "classification_accuracy_when_localized"]].to_string(index=False))
    print(pd.DataFrame(score_summaries)[["population", "score_source", "n", "mae", "absolute_error_gt_10"]].to_string(index=False))
    print(json.dumps({"transitions": transitions, "boundary": boundary, "major": major}, indent=2))
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
