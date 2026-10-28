"""Prediction-crop score residuals with frozen R1 and directional adapters.

Separate registered stage from native head fitting. Uses original-train image
roles, existing cached RGB+mask inputs, and rule scores; no test access. Prefix
features are recomputed from the original R1, never borrowed from changed YOLO.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from cache_r1_encoder_inputs import load_inputs, decode
from cache_r1_quality_features import EXQ
from train_r1a_hierarchical_grader import R1HierarchicalGrader
from run_r1_ap_followup import load_cache, STEM, TARGETS, save, sha256, LOW, HIGH, policy, evaluate
import run_r1_ap_followup as harness
from analyze_r1_ap_bootstrap import pure_match
from run_refine_module_experiments import STAGE as NATIVE_STAGE, OUT
from yolo26_refine_gates import IdentityFeatureGate

STAGE = NATIVE_STAGE / "score_direction"
SEEDS = [20261022, 20261023, 20261024]
KINDS = ["control", "strip", "deg", "partial"]


def binding():
    return {"script_sha256": sha256(Path(__file__)), "adapter_sha256": sha256(Path(__file__).with_name("yolo26_refine_gates.py")),
            "R1_sha256": sha256(STEM), "targets_sha256": sha256(TARGETS),
            "roles_sha256": sha256(EXQ / "head_image_split.csv"),
            "input_manifest_sha256": sha256(EXQ / "encoder_inputs/train_manifest.json"),
            "test_accessed": False}


def plan():
    STAGE.mkdir(parents=True, exist_ok=True)
    p = {"date": "2026-10-22", "question": "Do directional crop features improve rule-score residual learning over equal-capacity head-only learning?",
         "kinds": KINDS, "seeds": SEEDS, "context": binding(),
         "prefix": "Original pretrained R1 visual blocks0:4 frozen, cached in FP32; adapter at192x10x10 before last down block",
         "tail": "Original last down block, global pooling, geometry and fusion frozen/eval; only adapter and delta head trained",
         "population": "Original train matched predictions, raw confidence>=.25, baselineclass!=0, available qualified rule score; same image roles",
         "residual": "base_rule_prediction01 + .05*tanh(MLP32(normalized_new_fusion)), clipped0..1; zero final layer",
         "normalization": "Mean/std of unchanged fusion on fit rows only",
         "loss": "SmoothL1 beta .025 + .1 ordinal BCE width .05 at unchanged LOW/HIGH + .1 correction squared",
         "optimizer": "AdamW lr.001 weight_decay.001; batch128; fixed paired order; FP32",
         "checkpoint_selection": "Internal held MAE, epoch-zero included, improvement1e-4; patience12, max60 epochs",
         "selection": "Family mean held MAE across3 seeds; RMSE tie. Developer evaluation only after training selection.",
         "augmentation": "None in feature-cache domain; same cached actual prediction crops for every arm",
         "ranking": "Original confidence for all score comparisons. Old class-dependent ranker not silently reused after label changes.",
         "caveat": "Original R1 and score teacher have seen original training population; internal held is not independent system test",
         "primary_model_replaced": False, "budget_stop": "Common native STOP_AFTER_CURRENT_EPOCH flag; original quota window only"}
    path = STAGE / "plan.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != p:
        raise ValueError("Score stage changed")
    save(path, p)
    return p


def load_original(device="cpu"):
    checkpoint = torch.load(STEM, map_location="cpu", weights_only=False)
    model = R1HierarchicalGrader()
    model.load_state_dict(checkpoint["model_state"])
    model.eval().requires_grad_(False)
    return model.to(device)


def prepare():
    torch.set_num_threads(4)
    plan()
    path = STAGE / "train_prefix.pt"
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data["binding"] != binding():
            raise ValueError("Prefix source drift")
        return
    original = load_original()
    roles = pd.read_csv(EXQ / "head_image_split.csv").set_index("image_id")
    target = pd.read_csv(TARGETS).set_index("instance_id")
    entries = load_cache("train")
    inputs = load_inputs("train")
    parts = {r: {"prefix": [], "geometry": [], "base": [], "target": [], "ids": []} for r in ["fit", "held"]}
    with torch.inference_mode():
        for n, (e, item) in enumerate(zip(entries, inputs)):
            if e["image_id"] != item["image_id"] or item["split"] != "train":
                raise ValueError("Training alignment mismatch")
            keep = []
            labels = []
            ids = []
            for k, matched in enumerate(item["matched"]):
                j = int(item["prediction_indices"][k])
                row = target.loc[matched["instance_id"]]
                if row.split != "train":
                    raise ValueError("Development target entered fit")
                if e["baseline"][j] == 0 or e["conf"][j] < .25 or not np.isfinite(e["score"][j]):
                    continue
                if int(row.score_target_available) != 1 or int(row.defect_target) != 0:
                    continue
                keep.append(k); labels.append(float(row.score_target) / 100)
                ids.append({"image_id": e["image_id"], "instance_id": row.name, "prediction_index": j})
            if not keep:
                continue
            role = "held" if roles.loc[e["image_id"], "role"] == "head_holdout" else "fit"
            d = parts[role]
            tensor = decode(item["image_uint8"][keep])
            prefix = torch.cat([original.visual[:4](batch) for batch in tensor.split(32)])
            d["prefix"].append(prefix.clone())
            d["geometry"].append(item["geometry"][keep].clone())
            d["base"].append(torch.tensor([e["score"][i["prediction_index"]] / 100 for i in ids], dtype=torch.float32))
            d["target"].append(torch.tensor(labels, dtype=torch.float32)); d["ids"].extend(ids)
            if (n+1) % 150 == 0:
                print(f"SCORE PREFIX {n+1}/{len(entries)} CPU", flush=True)
    data = {r: {k: (v if k == "ids" else torch.cat(v)) for k, v in part.items()} for r, part in parts.items()}
    with torch.inference_mode():
        f = data["fit"]
        image = original.visual[4:](f["prefix"])
        features = original.fusion(torch.cat((image, original.geometry(f["geometry"])), dim=1))
        mean = features.mean(0); std = features.std(0, unbiased=False).clamp_min(1e-6)
    for r in data:
        if not len(data[r]["target"]):
            raise ValueError("Empty score population")
    torch.save({"binding": binding(), "data": data, "mean": mean, "std": std}, path)
    save(STAGE / "population.json", {"fit": len(data["fit"]["target"]), "held": len(data["held"]["target"]),
         "prefix_shape": list(data["fit"]["prefix"].shape), "cache_sha256": sha256(path), "test_accessed": False})
    print("SCORE PREFIX COMPLETE " + json.dumps({r: len(d["target"]) for r, d in data.items()}), flush=True)


class DirectionScore(nn.Module):
    def __init__(self, kind, original, mean, std):
        super().__init__()
        self.tail = copy.deepcopy(original.visual[4:]).eval().requires_grad_(False)
        self.geometry = copy.deepcopy(original.geometry).eval().requires_grad_(False)
        self.fusion = copy.deepcopy(original.fusion).eval().requires_grad_(False)
        state = torch.get_rng_state()
        self.adapter = nn.Identity() if kind == "control" else IdentityFeatureGate(192, kind)
        torch.set_rng_state(state)
        self.delta = nn.Sequential(nn.Linear(256, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.delta[-1].weight); nn.init.zeros_(self.delta[-1].bias)
        self.register_buffer("mean", mean.clone()); self.register_buffer("std", std.clone())

    def train(self, mode=True):
        super().train(mode)
        self.tail.eval(); self.geometry.eval(); self.fusion.eval()
        return self

    def forward(self, prefix, geometry, base):
        visual = self.tail(self.adapter(prefix))
        feature = self.fusion(torch.cat((visual, self.geometry(geometry)), dim=1))
        delta = .05 * torch.tanh(self.delta((feature-self.mean)/self.std).squeeze(1))
        return (base+delta).clamp(0, 1), delta


@torch.inference_mode()
def metrics(model, data):
    model.eval()
    pred = torch.cat([model(data["prefix"][i], data["geometry"][i], data["base"][i])[0]
                      for i in torch.arange(len(data["target"]), device=data["target"].device).split(256)])
    diff = (pred-data["target"]) * 100
    return {"MAE": float(diff.abs().mean()), "RMSE": float(diff.square().mean().sqrt()),
            "abs_gt10": int((diff.abs()>10).sum()), "n": len(diff)}


def fit(kind, seed, data, mean, std, original):
    directory = STAGE / f"{kind}_seed{seed}"
    if directory.exists():
        record = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        if record["binding"] != binding():
            raise ValueError("Completed score fit drift")
        return record
    directory.mkdir()
    torch.manual_seed(seed); np.random.seed(seed)
    model = DirectionScore(kind, original, mean, std).to("cuda:0")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=.001)
    generator = torch.Generator().manual_seed(seed)
    thresholds = torch.tensor([LOW/100, HIGH/100], device="cuda:0")
    best = float("inf"); best_epoch = 0; history = []; started = time.monotonic()
    for epoch in range(61):
        if epoch:
            model.train()
            for ids in torch.randperm(len(data["fit"]["target"]), generator=generator).split(128):
                ids = ids.to("cuda:0"); f = data["fit"]
                pred, delta = model(f["prefix"][ids], f["geometry"][ids], f["base"][ids]); y = f["target"][ids]
                loss = nn.functional.smooth_l1_loss(pred, y, beta=.025) + .1*delta.square().mean()
                loss = loss + .1*nn.functional.binary_cross_entropy_with_logits((pred[:, None]-thresholds)/.05, (y[:, None]>=thresholds).float())
                optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5); optimizer.step()
        held = metrics(model, data["held"]); history.append({"epoch": epoch, **held})
        if held["MAE"] < best - 1e-4:
            best, best_epoch = held["MAE"], epoch
            torch.save({"kind": kind, "seed": seed, "state": {k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        "binding": binding(), "held": held, "epoch": epoch}, directory / "best.pt")
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        torch.save({"state": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                    "generator_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(), "binding": binding()}, directory / "last.pt")
        if epoch % 5 == 0:
            print(f"SCORE {kind} seed{seed} epoch{epoch} MAE={held['MAE']:.5f} best={best:.5f}", flush=True)
        if epoch-best_epoch >= 12 or (NATIVE_STAGE / "STOP_AFTER_CURRENT_EPOCH").exists():
            break
    best_cp = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
    record = {"kind": kind, "seed": seed, "best_epoch": best_epoch, "epochs_completed": epoch, "held": best_cp["held"],
              "parameters_trained": sum(p.numel() for p in model.parameters() if p.requires_grad), "binding": binding(),
              "checkpoint_sha256": sha256(directory / "best.pt"), "wall_seconds": time.monotonic()-started,
              "budget_partial": (NATIVE_STAGE / "STOP_AFTER_CURRENT_EPOCH").exists(), "test_accessed": False}
    save(directory / "completion.json", record)
    return record


def train():
    torch.set_num_threads(4); plan()
    cached = torch.load(STAGE / "train_prefix.pt", map_location="cpu", weights_only=False)
    if cached["binding"] != binding():
        raise ValueError("Score cache drift")
    data = {r: {k: v.to("cuda:0") for k,v in part.items() if k != "ids"} for r,part in cached["data"].items()}
    original = load_original()
    order = list(itertools.product(KINDS, SEEDS))
    order = [order[i] for i in np.random.default_rng(20261022).permutation(len(order))]
    save(STAGE / "run_order.json", order)
    fits = []
    for kind, seed in order:
        if (NATIVE_STAGE / "STOP_AFTER_CURRENT_EPOCH").exists():
            break
        record = fit(kind, seed, data, cached["mean"], cached["std"], original); fits.append(record)
        rows = [{"kind": f["kind"], "seed": f["seed"], "held_MAE": f["held"]["MAE"], "held_RMSE": f["held"]["RMSE"],
                 "best_epoch": f["best_epoch"], "epochs": f["epochs_completed"], "parameters": f["parameters_trained"]} for f in fits]
        pd.DataFrame(rows).to_csv(OUT / "score_direction_fits.csv", index=False)
    if len(fits) == len(order):
        frame = pd.DataFrame(rows)
        family = frame.groupby("kind").agg(n=("seed", "size"), MAE=("held_MAE", "mean"), RMSE=("held_RMSE", "mean")).reset_index()
        family.to_csv(OUT / "score_direction_families.csv", index=False)
        selected = family.sort_values(["MAE", "RMSE"]).iloc[0].to_dict()
        save(OUT / "score_direction_selection.json", {"selected_by_internal_held_only": selected, "fixed_deployment_seed": 20261022,
             "all12_complete": True, "test_accessed": False, "primary_model_replaced": False})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("action", choices=["prepare", "train"])
    action = parser.parse_args().action
    prepare() if action == "prepare" else train()
