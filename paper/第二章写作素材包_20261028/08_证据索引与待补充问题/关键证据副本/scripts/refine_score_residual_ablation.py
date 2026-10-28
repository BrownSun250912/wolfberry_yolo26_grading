"""Train-only mechanism ablation for bounded visual score residuals (CPU).

Fresh same-device control: do not pool these fits with previous GPU fits.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

import refine_score_direction as direction
from cache_r1_quality_features import load_quality
from run_r1_ap_followup import save, sha256, LOW, HIGH
from run_refine_module_experiments import STAGE, OUT

BASE = STAGE / "score_residual_ablation"
CONFIGS = {
    "control": {"head": "mlp32", "bound": .05, "ordinal": .1},
    "bias_only": {"head": "bias", "bound": .05, "ordinal": .1},
    "linear": {"head": "linear", "bound": .05, "ordinal": .1},
    "tight2": {"head": "mlp32", "bound": .02, "ordinal": .1},
    "wide10": {"head": "mlp32", "bound": .10, "ordinal": .1},
    "no_ordinal": {"head": "mlp32", "bound": .05, "ordinal": 0.},
    "strong_ordinal": {"head": "mlp32", "bound": .05, "ordinal": .3},
}
SEEDS = [20261022, 20261023, 20261024]


class ConstantHead(nn.Module):
    def __init__(self):
        super().__init__(); self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, feature):
        return self.bias.expand(len(feature), 1)


class Residual(nn.Module):
    def __init__(self, config, mean, std):
        super().__init__(); self.config = dict(config)
        if config["head"] == "bias":
            self.head = ConstantHead()
        elif config["head"] == "linear":
            self.head = nn.Linear(256, 1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        else:
            self.head = nn.Sequential(nn.Linear(256,32),nn.SiLU(),nn.Linear(32,1))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("mean", mean.clone()); self.register_buffer("std", std.clone())

    def forward(self, features, base):
        delta = self.config["bound"] * torch.tanh(self.head((features-self.mean)/self.std).squeeze(1))
        return (base+delta).clamp(0,1), delta


def register():
    BASE.mkdir(parents=True, exist_ok=True)
    plan = {"question": "Does score gain need instance visual context, bounded correction, or ordinal auxiliary objective?",
            "registered_after": "Directional screening and selected CONTROL development/ranking/uncertainty, exploratory follow-up",
            "configs": CONFIGS, "seeds": SEEDS, "device": "CPU FP32 threads2; fresh same-device paired control",
            "population": "Original score-prefix fit3053/held761 matched prediction instances; unchanged targets and image roles",
            "features": "Frozen original R1 fusion256, all-prediction quality cache; no new encoder fitting",
            "normalization": "fit rows only", "base": "Original frozen geometry score, unchanged",
            "optimization": "AdamW .001 wd .001, batch128, SmoothL1 beta.025 + ordinal_weight*BCE width.05 + .1*delta^2, gradnorm5",
            "stopping": "held MAE min, epoch0 eligible, patience12, max60, common budget marker, source window boundary",
            "family_selection": "Mean held MAE across three seeds, RMSE tie; no development selection",
            "development": "After internal selection, evaluate selected family and same-stage control across3seeds; refit ranking only for selected family",
            "independence": "Inherited teacher overlap; original internal holdout not independent full-system test",
            "test_accessed": False, "primary_replaced": False, "script_sha256": sha256(Path(__file__)),
            "context": direction.binding()}
    path = BASE / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Residual ablation source/plan drift")
    save(path, plan)
    return plan


def data():
    cache = torch.load(direction.STAGE / "train_prefix.pt", map_location="cpu", weights_only=False)
    if cache["binding"] != direction.binding():
        raise ValueError("Original prefix drift")
    quality = {e["image_id"]: e for e in load_quality("train")}
    parts = {}
    for role, rows in cache["data"].items():
        features = torch.stack([quality[row["image_id"]]["features"][row["prediction_index"]] for row in rows["ids"]])
        if not torch.isfinite(features).all():
            raise ValueError("Nonfinite fused features")
        parts[role] = {"features": features, "base": rows["base"], "target": rows["target"]}
    mean = parts["fit"]["features"].mean(0)
    std = parts["fit"]["features"].std(0, unbiased=False).clamp_min(1e-6)
    torch.save({"parts": parts, "mean": mean, "std": std, "context": direction.binding()}, BASE / "features.pt")
    return parts, mean, std


@torch.inference_mode()
def metrics(model, part):
    model.eval(); pred = model(part["features"],part["base"])[0]
    error = (pred-part["target"])*100
    return {"MAE": float(error.abs().mean()), "RMSE": float(error.square().mean().sqrt()),
            "n": len(error), "abs_gt10": int((error.abs()>10).sum())}


def fit(kind, seed, parts, mean, std, plan):
    directory = BASE / f"{kind}_seed{seed}"
    if directory.exists():
        raise FileExistsError("No silent retry of residual fit: " + str(directory))
    directory.mkdir()
    torch.manual_seed(seed); np.random.seed(seed)
    model = Residual(CONFIGS[kind], mean, std)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    generator = torch.Generator().manual_seed(seed)
    limits = torch.tensor([LOW/100,HIGH/100])
    history = []; best = float("inf"); best_epoch = 0; started = time.monotonic(); partial = False
    for epoch in range(61):
        if epoch:
            model.train(); f = parts["fit"]
            for ids in torch.randperm(len(f["base"]), generator=generator).split(128):
                pred,delta = model(f["features"][ids],f["base"][ids]); target = f["target"][ids]
                loss = nn.functional.smooth_l1_loss(pred,target,beta=.025) + .1*delta.square().mean()
                loss = loss + CONFIGS[kind]["ordinal"]*nn.functional.binary_cross_entropy_with_logits((pred[:,None]-limits)/.05,(target[:,None]>=limits).float())
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),5); opt.step()
        held = metrics(model,parts["held"]); fit_metric = metrics(model,parts["fit"])
        history.append({"epoch": epoch, "held_MAE": held["MAE"], "held_RMSE": held["RMSE"], "fit_MAE": fit_metric["MAE"]})
        if held["MAE"] < best-1e-4:
            best,best_epoch = held["MAE"],epoch
            torch.save({"state": model.state_dict(), "config": CONFIGS[kind], "kind": kind, "seed": seed,
                        "epoch": epoch, "held": held, "plan": plan}, directory / "best.pt")
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        torch.save({"state": model.state_dict(), "optimizer": opt.state_dict(), "epoch": epoch,
                    "generator_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(), "plan": plan}, directory / "last.pt")
        partial = (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists() or time.time()+90>=1792694245 or time.monotonic()-started>900
        if epoch-best_epoch >= 12 or partial:
            break
    cp = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    clone = Residual(cp["config"],mean,std); clone.load_state_dict(cp["state"])
    if abs(metrics(clone,parts["held"])["MAE"]-cp["held"]["MAE"]) > 1e-7:
        raise ValueError("Residual CPU roundtrip differs")
    record = {"kind": kind, "seed": seed, "held_MAE": best, "held_RMSE": cp["held"]["RMSE"],
              "best_epoch": best_epoch, "epochs": epoch, "parameters": sum(p.numel() for p in model.parameters()),
              "budget_partial": partial, "wall_seconds": time.monotonic()-started,
              "checkpoint_sha256": sha256(directory / "best.pt"), "test_accessed": False}
    save(directory / "completion.json", record)
    print(f"RESIDUAL {kind} seed{seed} held_MAE={best:.6f} best_epoch={best_epoch}/{epoch}", flush=True)
    return record


def main():
    torch.set_num_threads(2); plan = register(); parts,mean,std = data()
    order = [(kind,seed) for kind in CONFIGS for seed in SEEDS]
    order = [order[i] for i in np.random.default_rng(20261028).permutation(len(order))]
    save(BASE / "schedule.json", order); records = []
    for kind,seed in order:
        if (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists() or time.time()+180>=1792694245:
            break
        save(OUT / "score_residual_ablation_status.json", {"status": "running", "kind": kind, "seed": seed})
        records.append(fit(kind,seed,parts,mean,std,plan))
        pd.DataFrame(records).to_csv(OUT / "score_residual_ablation_fits.csv", index=False)
    if len(records) == len(order) and not any(r["budget_partial"] for r in records):
        frame = pd.DataFrame(records)
        means = frame.groupby("kind").agg(n=("seed","size"),MAE=("held_MAE","mean"),MAE_sd=("held_MAE","std"),RMSE=("held_RMSE","mean")).reset_index()
        means.to_csv(OUT / "score_residual_ablation_means.csv",index=False)
        choice = means.sort_values(["MAE","RMSE"]).iloc[0].to_dict()
        save(OUT / "score_residual_ablation_selection.json", {"choice": choice,"chosen_on_internal_held_only": True,
             "fixed_score_seed": 20261022, "fixed_rank_seed": 20261008, "development_not_used_for_choice": True})
        save(OUT / "score_residual_ablation_status.json", {"status": "completed", "fits": len(records), "test_accessed": False})


if __name__ == "__main__":
    main()
