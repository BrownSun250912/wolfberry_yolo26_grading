"""Train-only reliability refits for internally selected CONTROL score residual.

All eligible predictions are scored from inference-time frozen fusion features;
GT matching never determines which prediction receives a new score.
"""
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

import refine_score_direction as score
import run_r1_ap_followup as harness
from run_r1_ap_followup import load_cache, save, sha256, policy
from run_refine_module_experiments import STAGE, OUT
from cache_r1_quality_features import load_quality, EXQ
from train_r1_confidence_followup import features, ReliabilityHead, confidences, FEATURES
from analyze_r1_ap_bootstrap import pure_match
from evaluate_refine_score_direction import score_diagnostics

BASE = STAGE / "score_reliability"
RANK_SEEDS = [20261008, 20261009, 20261010]


def register():
    selection = json.loads((OUT / "score_direction_selection.json").read_text(encoding="utf-8"))
    if selection["selected_by_internal_held_only"]["kind"] != "control":
        raise ValueError("This fast path is only mathematically valid for adapter-free CONTROL")
    plan = {"purpose": "Separate new score classification from reliability ranking, avoid old class-conditioned ranker transplant",
            "selected_family": "control, selected on original internal held means BEFORE development evaluation",
            "score_seeds": score.SEEDS, "ranking_seeds": RANK_SEEDS,
            "reference": "Also refit unchanged frozen scores at all three ranking seeds using identical protocol",
            "ranking_population": "All original training predictions except baseline class0; original calibration image roles seed20261008",
            "score_application": "All valid baseline!=0 raw-conf>=.25 predictions, including false positives, using cached inference fusion",
            "normalization": "Ranking features normalized on calibration-fit rows ONLY; score normalization saved from score-fit ONLY",
            "optimizer": "MLP14-16-1, AdamW .003 wd .0001, batch256, max300 epochs patience30, minimum held BCE",
            "confidence": "Fixed product p(reliable)*original_conf; baseline class0 confidence unchanged",
            "selection": "No best development score/rank seed; report3x3 nested seeds and fixed score22/rank08",
            "nested_seeds_not_independent_datasets": True, "test_accessed": False, "primary_replaced": False,
            "script_sha256": sha256(Path(__file__)), "source_context": score.binding(),
            "calibration_roles_sha256": sha256(harness.EX / "calibration_image_split.csv"),
            "train_features_manifest_sha256": sha256(EXQ / "features_train_manifest.json"),
            "val_features_manifest_sha256": sha256(EXQ / "features_val_manifest.json")}
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Reliability plan drift")
    save(path, plan)
    return plan


def revised(entries, quality, cp):
    delta = nn.Sequential(nn.Linear(256, 32), nn.SiLU(), nn.Linear(32, 1))
    delta.load_state_dict({k[6:]: value for k,value in cp["state"].items() if k.startswith("delta.")})
    delta.eval(); result = []
    mean, std = cp["state"]["mean"], cp["state"]["std"]
    with torch.inference_mode():
        for e, q in zip(entries, quality):
            if e["image_id"] != q["image_id"] or len(e["score"]) != len(q["features"]):
                raise ValueError("Inference feature alignment differs")
            values = e["score"].copy()
            valid = np.asarray(q["valid_indices"], dtype=int)
            active = valid[(e["baseline"][valid] != 0) & (e["conf"][valid] >= .25) & np.isfinite(values[valid])]
            if len(active):
                correction = .05 * torch.tanh(delta((q["features"][active]-mean)/std).squeeze(1))
                base = torch.tensor(values[active].astype(np.float32)/100)
                values[active] = (base+correction).clamp(0,1).numpy()*100
            result.append(dict(e, score=values))
    return result


def population(entries):
    roles = pd.read_csv(harness.EX / "calibration_image_split.csv").set_index("image_id")
    parts = {r: [[], []] for r in ["fit", "held"]}
    for e in entries:
        x, cls = features(e)
        y = pure_match(cls, e["gt_cls"], e["mask_iou"])[:, 0].astype(np.float32)
        use = e["baseline"] != 0
        role = "held" if roles.loc[e["image_id"], "role"] == "calibration_holdout" else "fit"
        parts[role][0].append(x[use]); parts[role][1].append(y[use])
    fx, fy = [np.concatenate(a) for a in parts["fit"]]
    hx, hy = [np.concatenate(a) for a in parts["held"]]
    mean, std = fx.mean(0), fx.std(0); std[std < 1e-6] = 1
    return torch.tensor((fx-mean)/std), torch.tensor(fy), torch.tensor((hx-mean)/std), torch.tensor(hy), mean, std


def fit(name, seed, data, context):
    directory = BASE / name / f"rank_seed{seed}"
    if directory.exists():
        raise FileExistsError("Inspect existing reliability fit; no silent retry: " + str(directory))
    directory.mkdir(parents=True)
    torch.manual_seed(seed); np.random.seed(seed)
    model = ReliabilityHead("mlp_confidence")
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.0001)
    generator = torch.Generator().manual_seed(seed)
    x,y,vx,vy,mean,std = data
    best, best_epoch = float("inf"), 0
    history = []; started = time.monotonic(); stopped = False
    for epoch in range(301):
        if epoch:
            model.train()
            for ids in torch.randperm(len(x), generator=generator).split(256):
                opt.zero_grad(); nn.functional.binary_cross_entropy_with_logits(model(x[ids]), y[ids]).backward(); opt.step()
        model.eval()
        with torch.inference_mode():
            held = float(nn.functional.binary_cross_entropy_with_logits(model(vx), vy))
        history.append({"epoch": epoch, "held_bce": held})
        if held < best-1e-6:
            best, best_epoch = held, epoch
            torch.save({"family": "mlp_confidence", "head_state": model.state_dict(), "mean": mean, "std": std,
                        "seed": seed, "best_epoch": best_epoch, "holdout_bce": best,
                        "feature_names": FEATURES, "context": context}, directory / "best.pt")
        torch.save({"head_state": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                    "generator_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(),
                    "context": context}, directory / "last.pt")
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        stopped = (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists() or time.time()+60 >= 1792694245 or time.monotonic()-started > 1200
        if epoch-best_epoch >= 30 or stopped:
            break
    cp = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(cp["head_state"]); model.eval()
    clone = ReliabilityHead("mlp_confidence"); clone.load_state_dict(cp["head_state"]); clone.eval()
    with torch.inference_mode():
        if not torch.equal(model(vx), clone(vx)):
            raise ValueError("Reliability roundtrip failed")
    record = {"name": name, "rank_seed": seed, "held_bce": best, "best_epoch": best_epoch, "epochs": epoch,
              "wall_seconds": time.monotonic()-started, "budget_partial": stopped,
              "checkpoint_sha256": sha256(directory / "best.pt"), "context": context}
    save(directory / "completion.json", record)
    return model, cp, record


def main():
    torch.set_num_threads(2); harness.match_flags = pure_match
    harness.OUT = OUT / "score_reliability_metrics"; harness.OUT.mkdir(parents=True, exist_ok=True)
    plan = register()
    train, val = load_cache("train"), load_cache("val")
    quality_train, quality_val = load_quality("train"), load_quality("val")
    records = []
    arms = [("original_score", None)] + [(f"control_score{seed}", seed) for seed in score.SEEDS]
    for name, score_seed in arms:
        if (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists() or time.time()+180 >= 1792694245:
            return
        cp_path = None if score_seed is None else score.STAGE / f"control_seed{score_seed}" / "best.pt"
        if cp_path:
            cp = torch.load(cp_path, map_location="cpu", weights_only=False)
            changed_train, changed_val = revised(train, quality_train, cp), revised(val, quality_val, cp)
            observed = score_diagnostics(changed_val)
            previous = pd.read_csv(OUT / "score_direction_development.csv").query("kind == 'control' and seed == @score_seed").iloc[0]
            raw, _, _ = harness.evaluate(changed_val, name + "_raw", [policy(e,True,True,True) for e in changed_val])
            if abs(observed["score_MAE"] - previous.score_MAE) > .0001 or raw["correct_500"] != previous.correct_500:
                raise ValueError("Fast fusion scoring differs from full prefix evaluation")
        else:
            changed_train, changed_val = train, val
        context = {"plan": plan, "score_checkpoint_sha256": None if cp_path is None else sha256(cp_path)}
        data = population(changed_train)
        for rank_seed in RANK_SEEDS:
            if (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists():
                return
            save(OUT / "score_reliability_status.json", {"status": "running", "name": name, "rank_seed": rank_seed})
            model, cp, fitted = fit(name, rank_seed, data, context)
            ranks = confidences(model, cp, changed_val, "product")
            result, _, _ = harness.evaluate(changed_val, f"{name}_rank{rank_seed}", [policy(e,True,True,True) for e in changed_val], ranks)
            result.update(score_diagnostics(changed_val))
            result.update({"score_seed": score_seed, "arm": name, "rank_seed": rank_seed,
                           "held_bce": fitted["held_bce"], "budget_partial": fitted["budget_partial"]})
            records.append(result)
            pd.DataFrame(records).to_csv(OUT / "score_reliability_development.csv", index=False)
            print(f"SCORE RELIABILITY {name} rank{rank_seed} AP50={result['metrics/mAP50(M)']:.8f} AP95={result['metrics/mAP50-95(M)']:.8f} correct={result['correct_500']}", flush=True)
            if fitted["budget_partial"]:
                return
    save(OUT / "score_reliability_status.json", {"status": "completed", "ranking_fits": len(records),
         "test_accessed": False, "primary_replaced": False, "nested_seeds_not_independent_datasets": True})


if __name__ == "__main__":
    main()
