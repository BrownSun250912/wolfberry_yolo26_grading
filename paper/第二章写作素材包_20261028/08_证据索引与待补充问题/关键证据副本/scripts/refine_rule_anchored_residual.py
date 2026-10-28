"""Exploratory correction of prediction-rule scores, with paired learned-base control.

After the direct-rule comparator is reproduced, test whether frozen crop fusion
can correct its prediction-mask measurement error against unchanged references.
Same MLP/loss/dose/normalizer for both bases. No development-selected checkpoints.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import refine_score_direction as direction
from refine_score_residual_ablation import Residual, metrics
import refine_score_ranking as ranking
import refine_rule_reliability as direct
import run_r1_ap_followup as harness
from run_refine_module_experiments import STAGE, OUT
from run_r1_ap_followup import save, sha256, load_cache, policy, LOW, HIGH
from cache_r1_quality_features import load_quality
from train_r1_confidence_followup import confidences
from analyze_r1_ap_bootstrap import pure_match
from evaluate_refine_score_direction import score_diagnostics

BASE = STAGE / "rule_anchored_residual"
CONFIG = {"head": "mlp32", "bound": .05, "ordinal": .1}
KINDS = ["learned_base_control", "rule_base_residual"]


def register():
    plan = {"question": "Can an equal-capacity learned correction improve direct prediction-rule scoring, not merely improve a weaker learned-score base?",
            "registered_after": "Direct-rule development counterexample and historical549 parity; exploratory follow-up, not original preregistration",
            "arms": KINDS, "zero_fit_reference": "Unmodified direct-rule scores, same inference scope and frozen stem",
            "feature": "Frozen R1 fusion256 already used for stem; includes visual and geometry, not pure visual",
            "config": CONFIG, "seeds": direction.SEEDS, "device": "CPU FP32 threads2",
            "population": "Identical original matched qualified score3053fit761held image roles, original prefix IDs; no new labels",
            "normalizer": "Fit-only frozen fusion mean/std, identical between arms",
            "optimization": "AdamW.001wd.001,batch128,SmoothL1beta.025+.1ordinalBCEwidth.05+.1delta_squared,clipnorm5",
            "checkpoint": "Minimum heldMAE improvement1e-4,epoch0included,max60,patience12",
            "pairing": "Same full initial MLP state and shuffled sample IDs for each seed; randomized arm order within seed",
            "internal_practical_gate": "Rule residual meanMAE<=direct rule zero meanMAE-.05 AND <=paired learned-base control meanMAE-.05; RMSE<=direct rule RMSE+.02",
            "development": "Evaluate both arms all3seeds and zero rule, report even if gate fails; no development-based seed selection",
            "ranking": "Rule-residual all3score seeds x3ranking seeds, same original calibration roles and fit-only scaling; no rank transplant",
            "deployment_seed": "Fixed score22/rank08; positive gate not automatic production replacement",
            "deadline": "Inherited frozen rank fitter retains original window boundary; partials reported, never overwrite or silently restart",
            "test_accessed": False, "primary_replaced": False,
            "script_sha256": sha256(Path(__file__)), "direction_binding": direction.binding(),
            "model_source_sha256": sha256(Path(__file__).with_name("refine_score_residual_ablation.py")),
            "direct_rule_plan_sha256": sha256(direct.BASE / "plan.json"),
            "ranking_source_sha256": sha256(Path(ranking.__file__))}
    BASE.mkdir(exist_ok=True)
    path = BASE / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Rule-anchored residual source/plan drift")
    save(path, plan)
    return plan


def load_rule_entries(split):
    entries = load_cache(split); directory = direct.BASE / ("scores_" + split)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if len(entries) != len(manifest["entries"]):
        raise ValueError("Rule cache cohort changed")
    result = []
    for e, item in zip(entries, manifest["entries"]):
        if e["image_id"] != item["image_id"] or sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Rule cache identity/hash mismatch")
        scores = torch.load(item["path"], map_location="cpu", weights_only=False)
        if scores["image_sha256"] != e["image_sha256"]:
            raise ValueError("Rule source image differs")
        result.append(dict(e, score=scores["scores"]))
    return entries, result


def prepare():
    cache = torch.load(direction.STAGE / "train_prefix.pt", map_location="cpu", weights_only=False)
    if cache["binding"] != direction.binding():
        raise ValueError("Rule-anchored original prefix binding differs")
    learned, rule_entries = load_rule_entries("train")
    rule = {e["image_id"]: e for e in rule_entries}
    quality = {e["image_id"]: e for e in load_quality("train")}; parts = {}
    for role, rows in cache["data"].items():
        features = torch.stack([quality[r["image_id"]]["features"][r["prediction_index"]] for r in rows["ids"]])
        bases = torch.tensor([rule[r["image_id"]]["score"][r["prediction_index"]]/100 for r in rows["ids"]], dtype=torch.float32)
        if not torch.isfinite(bases).all() or not torch.isfinite(features).all():
            raise ValueError("Rule-anchored score population nonfinite")
        parts[role] = {"features": features, "learned_base": rows["base"], "rule_base": bases,
                       "target": rows["target"], "ids": rows["ids"]}
    if [len(parts[r]["target"]) for r in ["fit", "held"]] != [3053, 761]:
        raise ValueError("Rule-anchored original population differs")
    mean = parts["fit"]["features"].mean(0)
    std = parts["fit"]["features"].std(0, unbiased=False).clamp_min(1e-6)
    zero = {}
    for name, key in [("learned_base", "learned_base"), ("direct_rule", "rule_base")]:
        error = (parts["held"][key]-parts["held"]["target"])*100
        zero[name] = {"MAE": float(error.abs().mean()), "RMSE": float(error.square().mean().sqrt()), "n": len(error)}
    save(OUT / "rule_anchored_zero_references.json", zero)
    return parts, mean, std, zero


def fit(kind, seed, allparts, mean, std, plan):
    directory = BASE / f"{kind}_seed{seed}"
    if directory.exists():
        raise FileExistsError("Inspect existing rule-anchored fit before any restart")
    directory.mkdir(); torch.manual_seed(seed); np.random.seed(seed)
    model = Residual(CONFIG, mean, std)
    torch.save(model.state_dict(), directory / "initial_state.pt")
    key = "rule_base" if kind == "rule_base_residual" else "learned_base"
    parts = {r: {"features": p["features"], "base": p[key], "target": p["target"]} for r, p in allparts.items()}
    generator = torch.Generator().manual_seed(seed)
    first = torch.randperm(len(parts["fit"]["base"]), generator=torch.Generator().manual_seed(seed))[:32].tolist()
    save(directory / "first32_ids.json", [allparts["fit"]["ids"][j] for j in first])
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    thresholds = torch.tensor([LOW/100, HIGH/100]); best = float("inf"); best_epoch = 0
    history = []; started = time.monotonic(); partial = False
    for epoch in range(61):
        if epoch:
            model.train(); f = parts["fit"]
            for ids in torch.randperm(len(f["base"]), generator=generator).split(128):
                pred, delta = model(f["features"][ids], f["base"][ids]); y = f["target"][ids]
                loss = nn.functional.smooth_l1_loss(pred, y, beta=.025)+.1*delta.square().mean()
                loss += .1*nn.functional.binary_cross_entropy_with_logits((pred[:, None]-thresholds)/.05, (y[:, None]>=thresholds).float())
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
        held = metrics(model, parts["held"]); history.append({"epoch": epoch, **held})
        if held["MAE"] < best-1e-4:
            best, best_epoch = held["MAE"], epoch
            torch.save({"state": model.state_dict(), "config": CONFIG, "kind": kind, "seed": seed,
                        "held": held, "epoch": epoch, "plan": plan}, directory / "best.pt")
        torch.save({"state": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                    "generator_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(), "plan": plan}, directory / "last.pt")
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        partial = (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists() or time.monotonic()-started > 900
        if epoch-best_epoch >= 12 or partial:
            break
    cp = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    clone = Residual(CONFIG, mean, std); clone.load_state_dict(cp["state"])
    if abs(metrics(clone, parts["held"])["MAE"]-cp["held"]["MAE"]) > 1e-7:
        raise ValueError("Rule-anchored roundtrip differs")
    record = {"kind": kind, "seed": seed, "held_MAE": best, "held_RMSE": cp["held"]["RMSE"],
              "best_epoch": best_epoch, "epochs": epoch, "budget_partial": partial,
              "wall_seconds": time.monotonic()-started, "checkpoint_sha256": sha256(directory / "best.pt"), "test_accessed": False}
    save(directory / "completion.json", record)
    print(f"RULE ANCHOR {kind} seed{seed} MAE={best:.6f} RMSE={record['held_RMSE']:.6f}", flush=True)
    return record


@torch.inference_mode()
def revise(entries, quality, cp):
    model = Residual(cp["config"], cp["state"]["mean"], cp["state"]["std"])
    model.load_state_dict(cp["state"]); model.eval(); result = []
    for e, q in zip(entries, quality):
        if e["image_id"] != q["image_id"]:
            raise ValueError("Rule residual inference features misaligned")
        values = e["score"].copy(); valid = np.asarray(q["valid_indices"], dtype=int)
        ids = valid[(e["baseline"][valid] != 0) & (e["conf"][valid] >= .25) & np.isfinite(values[valid])]
        if len(ids):
            values[ids] = model(q["features"][ids], torch.tensor(values[ids].astype(np.float32)/100))[0].numpy()*100
        result.append(dict(e, score=values))
    return result


def main():
    torch.set_num_threads(2); harness.match_flags = pure_match
    harness.OUT = OUT / "rule_anchored_metrics"; harness.OUT.mkdir(exist_ok=True)
    plan = register(); started = time.monotonic()
    save(OUT / "rule_anchored_status.json", {"status": "waiting_direct_rule"})
    while True:
        path = OUT / "direct_rule_reliability_status.json"
        state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if state.get("status") == "completed":
            break
        if state.get("status") in ["crashed", "failed", "partial_fit_stop"]:
            raise RuntimeError("Inspect incomplete direct-rule prerequisite")
        if (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists():
            return
        if time.monotonic()-started > 14400:
            raise TimeoutError("Rule-anchored prerequisite wait exceeded")
        time.sleep(15)
    parts, mean, std, zero = prepare(); records = []
    for seed in direction.SEEDS:
        for kind in np.random.default_rng(seed).permutation(KINDS).tolist():
            save(OUT / "rule_anchored_status.json", {"status": "score_fit", "kind": kind, "seed": seed})
            records.append(fit(kind, seed, parts, mean, std, plan))
            pd.DataFrame(records).to_csv(OUT / "rule_anchored_fits.csv", index=False)
            if records[-1]["budget_partial"]:
                return
        dirs = [BASE / f"{k}_seed{seed}" for k in KINDS]
        states = [torch.load(d / "initial_state.pt", map_location="cpu", weights_only=False) for d in dirs]
        if not all(torch.equal(states[0][k], states[1][k]) for k in states[0]):
            raise ValueError("Rule-anchored paired initialization differs")
        orders = [json.loads((d / "first32_ids.json").read_text(encoding="utf-8")) for d in dirs]
        if orders[0] != orders[1]:
            raise ValueError("Rule-anchored paired shuffle differs")
    means = pd.DataFrame(records).groupby("kind")[["held_MAE", "held_RMSE"]].mean()
    means.to_csv(OUT / "rule_anchored_means.csv")
    r = means.loc["rule_base_residual"]; c = means.loc["learned_base_control"]
    eligible = bool(r.held_MAE <= zero["direct_rule"]["MAE"]-.05 and r.held_MAE <= c.held_MAE-.05
                    and r.held_RMSE <= zero["direct_rule"]["RMSE"]+.02)
    save(OUT / "rule_anchored_selection.json", {"eligible": eligible, "internal_means": means.to_dict("index"),
         "zero": zero, "pairing_audit": "PASS", "development_not_used": True})
    learned_val, rule_val = load_rule_entries("val"); qv = load_quality("val")
    _, rule_train = load_rule_entries("train"); qt = load_quality("train")
    rawrows = []; rankrows = []; ranking.BASE = BASE / "reliability"
    for seed in direction.SEEDS:
        for kind in KINDS:
            path = BASE / f"{kind}_seed{seed}/best.pt"; cp = torch.load(path, map_location="cpu", weights_only=False)
            changed_val = revise(rule_val if kind == "rule_base_residual" else learned_val, qv, cp)
            result, _, _ = harness.evaluate(changed_val, f"{kind}_seed{seed}_raw", [policy(e, True, True, True) for e in changed_val])
            result.update(score_diagnostics(changed_val)); result.update({"kind": kind, "score_seed": seed})
            rawrows.append(result); pd.DataFrame(rawrows).to_csv(OUT / "rule_anchored_development.csv", index=False)
            if kind != "rule_base_residual":
                continue
            changed_train = revise(rule_train, qt, cp); data = ranking.population(changed_train)
            for rank_seed in ranking.RANK_SEEDS:
                save(OUT / "rule_anchored_status.json", {"status": "rank_fit", "score_seed": seed, "rank_seed": rank_seed})
                head, rank_cp, fitted = ranking.fit(f"rule_base_score{seed}", rank_seed, data,
                                                  {"plan": plan, "score_checkpoint_sha256": sha256(path)})
                ranks = confidences(head, rank_cp, changed_val, "product")
                result, _, _ = harness.evaluate(changed_val, f"rule_base_score{seed}_rank{rank_seed}",
                                               [policy(e, True, True, True) for e in changed_val], ranks)
                result.update(score_diagnostics(changed_val)); result.update({"score_seed": seed, "rank_seed": rank_seed,
                     "held_bce": fitted["held_bce"], "budget_partial": fitted["budget_partial"], "internal_eligible": eligible})
                rankrows.append(result); pd.DataFrame(rankrows).to_csv(OUT / "rule_anchored_reliability.csv", index=False)
                print(f"RULE ANCHOR RANK score{seed} rank{rank_seed} AP50={result['metrics/mAP50(M)']:.8f}", flush=True)
                if fitted["budget_partial"]:
                    save(OUT / "rule_anchored_status.json", {"status": "partial_rank_stop"})
                    return
    save(OUT / "rule_anchored_status.json", {"status": "completed", "score_fits": 6, "rank_fits": len(rankrows),
         "internal_eligible": eligible, "test_accessed": False, "primary_replaced": False})


if __name__ == "__main__":
    main()
