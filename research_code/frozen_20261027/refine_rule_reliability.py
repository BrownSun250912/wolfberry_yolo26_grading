"""Fair train-only reliability calibration of the actual direct-rule comparator.

Compute scores on prediction masks, not GT masks/targets. Same frozen stem and
high-confidence scope. Wait for queued exclusive benchmark before any extraction.
This is a necessary comparator, not an additional proposed innovation.
"""
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO
from ultralytics.utils import LOGGER

import extract_features as extraction
import run_r0_fixed_prediction_replay as rules
import run_r1_ap_followup as harness
import refine_score_ranking as ranking
from run_refine_module_experiments import STAGE, OUT
from run_r1_ap_followup import ROOT, EX, DETECTOR, STEM, TARGETS, load_cache, policy, save, sha256
from analyze_val_instance_errors import polygon_to_mask
from analyze_r1_ap_bootstrap import pure_match
from train_r1_confidence_followup import confidences
from evaluate_refine_score_direction import score_diagnostics

BASE = STAGE / "direct_rule_reliability"
REPORT = OUT / "direct_rule_reliability"
HISTORICAL = ROOT / "outputs/reports/h1_frozen_replay_20261006/h1_predictions.csv"
CFG = rules.DEFAULT_FEATURE_CONFIG


def shape_features(image, mask, config):
    """Exact geometry subsection of historical process_instance, no file writes."""
    # The detector helper yields bool. Historical extraction used uint8 PNG
    # masks; the geometry fallback returns its input dtype, which OpenCV cannot
    # contour if bool. Normalize the representation, not the foreground pixels.
    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    analysis = config["analysis_mask"]
    cleaned, _, _ = extraction.clean_analysis_mask(
        mask, int(analysis["min_component_area_px"]), bool(analysis["keep_largest_component"]))
    refined = extraction.refine_geometry_mask(image, cleaned, config["geometry_refinement"])[0]
    contour = extraction.main_contour(refined)
    if contour is None:
        raise ValueError("Direct rule has no valid contour")
    perimeter = float(cv2.arcLength(contour, True))
    area = float(cv2.contourArea(contour))
    hull = float(cv2.contourArea(cv2.convexHull(contour)))
    axes = extraction.fit_axes(contour)
    if perimeter <= 0 or hull <= 0:
        raise ValueError("Direct rule geometry degenerate")
    return {"area": float(np.count_nonzero(refined)), "minor_axis": float(axes["minor_axis"]),
            "aspect_ratio": float(axes["major_axis"] / axes["minor_axis"]),
            "circularity": 4 * math.pi * area / perimeter**2, "solidity": area / hull}


def register():
    inputs = {"detector": DETECTOR, "stem": STEM, "targets": TARGETS, "historical_val_rules": HISTORICAL,
              "feature_config": CFG, "frozen_grading_config": rules.DEFAULT_GRADING_CONFIG,
              "frozen_ecdf_anchors": rules.DEFAULT_OLD_CANDIDATES,
              "calibration_roles": EX / "calibration_image_split.csv"}
    scripts = [Path(__file__), Path(rules.__file__), Path(extraction.__file__), Path(ranking.__file__)]
    plan = {"question": "Does learned scoring improve the full pipeline over direct predicted-mask rules under identical fresh ranking?",
            "role": "Strong comparator, not a novel proposed model; retain negative results",
            "score": "Historical frozen ECDF/formula on ORIGINAL PREDICTION masks with same boundary refinement; never GT scores",
            "scope": "All raw-conf>=.25 predictions with valid frozen scoring/stem, including unmatched predictions; class0 labels/conf unchanged",
            "low_conf": "Retain existing scores as original score-residual comparison does; no class changes below .25",
            "stem": "Original frozen learned R1 stem, same .5 override", "rank_seeds": ranking.RANK_SEEDS,
            "rank_fit": "Same MLP14-16-1, BCE, AdamW.003wd.0001, max300/patience30, original train-image roles and fit-only normalizer",
            "selection": "No development-selected seed or hyperparameter; fixed rank08 primary; report all three",
            "gate": "Val predicted-mask rule scores must reproduce all549 historical high scores within1e-6 and grades exactly BEFORE fitting",
            "inference_eligibility_not_GT_matching": True,
            "GPU_serialization": "Wait until final_runtime_status completed before reconstruction; no concurrent measurement",
            "inputs": {k: {"path": str(p), "sha256": sha256(p)} for k, p in inputs.items()},
            "sources": {p.name: sha256(p) for p in scripts}, "test_accessed": False, "primary_replaced": False}
    BASE.mkdir(exist_ok=True); REPORT.mkdir(exist_ok=True)
    path = BASE / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Direct-rule plan/source drift")
    save(path, plan)
    return plan


def reconstruct(entries, split, plan, detector, config, references, grading):
    if split not in ["train", "val"]:
        raise ValueError("Test access prohibited")
    directory = BASE / ("scores_" + split); directory.mkdir(exist_ok=True)
    revised = []; records = []; started = time.monotonic()
    for n, e in enumerate(entries):
        if (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists():
            raise RuntimeError("Explicit quota stop; partial cache retained")
        path = directory / (e["image_id"] + ".pt")
        if path.exists():
            item = torch.load(path, map_location="cpu", weights_only=False)
            if item["plan"] != plan or item["image_sha256"] != e["image_sha256"]:
                raise ValueError("Direct-rule partial cache lineage drift")
        else:
            if sha256(Path(e["image_path"])) != e["image_sha256"]:
                raise ValueError("Direct-rule image drift")
            r = detector.predict(e["image_path"], imgsz=640, conf=.001, iou=.7, max_det=300,
                                 retina_masks=True, half=False, device=0, verbose=False)[0]
            if (not np.array_equal(r.boxes.cls.cpu().numpy().astype(np.int64), e["baseline"])
                    or not np.array_equal(r.boxes.conf.cpu().numpy(), e["conf"])
                    or not np.array_equal(r.boxes.xyxy.cpu().numpy(), e["boxes"])):
                raise ValueError("Direct-rule detector prediction drift")
            values = e["score"].copy(); raw = []; ids = []; failures = []
            eligible = np.flatnonzero((e["conf"] >= .25) & np.isfinite(values) & np.isfinite(e["stem"]))
            for j in eligible:
                mask = polygon_to_mask(r.masks.xy[j], e["image_height"], e["image_width"])
                if int(np.count_nonzero(mask)) != int(e["mask_area"][j]):
                    raise ValueError("Direct-rule predicted-mask area drift")
                try:
                    raw.append(shape_features(r.orig_img, mask, config)); ids.append(int(j))
                except ValueError as exc:
                    failures.append({"index": int(j), "reason": str(exc)})
            if raw:
                scored = rules.apply_frozen_score(pd.DataFrame(raw), references, grading)
                if not scored.score_available.all():
                    raise ValueError("Nonfinite direct-rule scores")
                values[ids] = scored.rule_score.to_numpy(float)
            item = {"image_id": e["image_id"], "image_sha256": e["image_sha256"], "plan": plan,
                    "scores": values, "eligible_indices": eligible, "rule_updated_indices": ids,
                    "fallbacks": failures, "scope_determined_without_GT": True}
            torch.save(item, path)
        revised.append(dict(e, score=item["scores"]))
        records.append({"image_id": e["image_id"], "path": str(path), "sha256": sha256(path),
                        "updated": len(item["rule_updated_indices"]), "fallbacks": item["fallbacks"]})
        if (n+1) % 100 == 0:
            print(f"DIRECT RULE {split} {n+1}/{len(entries)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    save(directory / "manifest.json", {"plan": plan, "entries": records, "images": len(entries),
         "updated": sum(r["updated"] for r in records), "fallback_n": sum(len(r["fallbacks"]) for r in records),
         "test_accessed": False})
    return revised


def verify_historical(entries):
    historical = pd.read_csv(HISTORICAL).set_index("prediction_id"); n = 0; maximum = 0.
    for e in entries:
        for j, pid in enumerate(e["old_prediction_ids"]):
            if e["conf"][j] < .25:
                continue
            row = historical.loc[pid]
            delta = abs(float(e["score"][j])-float(row.rule_score))
            maximum = max(maximum, delta); n += 1
            grade = 1 if e["score"][j] >= harness.HIGH else 2 if e["score"][j] >= harness.LOW else 3
            if grade != int(row.rule_grade) or delta > 1e-6:
                raise ValueError(f"Historical direct-rule score reproduction failed {pid}: {delta}")
    if n != 549:
        raise ValueError("Historical direct-rule cohort differs")
    result = {"status": "PASS", "n": n, "max_score_delta": maximum, "GT_substitution": False}
    save(OUT / "direct_rule_reconstruction_parity.json", result)
    return result


def main():
    torch.set_num_threads(2); cv2.setNumThreads(1); LOGGER.setLevel(50)
    harness.match_flags = pure_match; harness.OUT = REPORT
    plan = register(); started = time.monotonic()
    save(OUT / "direct_rule_reliability_status.json", {"status": "waiting_final_exclusive_benchmark"})
    while True:
        path = OUT / "final_runtime_status.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if state.get("status") == "completed":
            break
        if (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists():
            return
        if state.get("status") in ["failed", "crashed", "budget_stop"]:
            raise RuntimeError("Inspect prerequisite benchmark status before direct-rule extraction")
        if time.monotonic()-started > 14400:
            raise TimeoutError("Direct-rule prerequisite wait exceeded")
        time.sleep(15)
    config = yaml.safe_load(CFG.read_text(encoding="utf-8"))
    grading = json.loads(rules.DEFAULT_GRADING_CONFIG.read_text(encoding="utf-8"))
    anchors = pd.read_csv(rules.DEFAULT_OLD_CANDIDATES)
    references, recovery, _ = rules.recover_references_from_candidate_ecdf(anchors, rules.DEFAULT_REFERENCE_SIZE)
    save(BASE / "reference_reconstruction.json", {"recovery": recovery,
         "validation": rules.validate_reconstructed_scoring(anchors, references, grading)})
    detector = YOLO(str(DETECTOR))
    save(OUT / "direct_rule_reliability_status.json", {"status": "validation_rule_reconstruction"})
    val = reconstruct(load_cache("val"), "val", plan, detector, config, references, grading)
    parity = verify_historical(val)
    save(OUT / "direct_rule_reliability_status.json", {"status": "train_rule_reconstruction", "parity": parity})
    train = reconstruct(load_cache("train"), "train", plan, detector, config, references, grading)
    del detector; torch.cuda.empty_cache()
    data = ranking.population(train); ranking.BASE = BASE / "reliability"; rows = []
    for seed in ranking.RANK_SEEDS:
        save(OUT / "direct_rule_reliability_status.json", {"status": "ranking_fit", "rank_seed": seed})
        head, cp, fitted = ranking.fit("direct_rule", seed, data, {"plan": plan, "parity": parity})
        ranks = confidences(head, cp, val, "product")
        result, _, _ = harness.evaluate(val, f"direct_rule_rank{seed}", [policy(e, True, True, True) for e in val], ranks)
        result.update(score_diagnostics(val)); result.update({"rank_seed": seed, "held_bce": fitted["held_bce"],
             "budget_partial": fitted["budget_partial"]})
        rows.append(result); pd.DataFrame(rows).to_csv(OUT / "direct_rule_reliability_development.csv", index=False)
        print(f"DIRECT RULE RANK seed{seed} AP50={result['metrics/mAP50(M)']:.8f} correct={result['correct_500']}", flush=True)
        if fitted["budget_partial"]:
            save(OUT / "direct_rule_reliability_status.json", {"status": "partial_fit_stop", "rank_seed": seed})
            return
    save(OUT / "direct_rule_reliability_status.json", {"status": "completed", "fits": len(rows),
         "test_accessed": False, "primary_replaced": False, "parity": parity})


if __name__ == "__main__":
    main()
