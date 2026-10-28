"""Isolated paired native-head experiments with exact experiment provenance.

Uses the existing train-only internal image split. Does not evaluate development
or touch test during training. Each fit is a fresh equal-dose dose, not a resume
with a changed scheduler. Training epoch callbacks exclude final_eval callbacks.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import LOGGER

from cache_r1_quality_color import DETECTOR
from run_r1_ap_followup import ROOT, TARGETS, save, sha256
from seeded_native_trainer import SeededMask50Trainer
from yolo26_refine_gates import install

STAGE = ROOT / "models/experiments/refine_module_20261022"
OUT = ROOT / "outputs/reports/refine_module_20261022"
CONFIG = ROOT / "configs/native_yolo_eca_head_followup.yaml"
KINDS = ["control", "strip", "deg", "partial"]


class RefineTrainer(SeededMask50Trainer):
    gate_kind = "control"
    gate_kernel = 7

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=None, verbose=verbose)
        install(model, self.gate_kind, self.gate_kernel)
        if weights:
            model.load(weights)
        return model


class StripTrainer(RefineTrainer):
    gate_kind = "strip"


class DegTrainer(RefineTrainer):
    gate_kind = "deg"


class PartialTrainer(RefineTrainer):
    gate_kind = "partial"


TRAINERS = {"control": RefineTrainer, "strip": StripTrainer, "deg": DegTrainer, "partial": PartialTrainer}


def sources():
    files = [Path(__file__), Path(__file__).with_name("yolo26_refine_gates.py"),
             Path(__file__).with_name("seeded_native_trainer.py"), Path(__file__).with_name("yolo26_identity_eca.py")]
    return {p.name: sha256(p) for p in files}


def prepare_plan():
    STAGE.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "models/experiments/r1_quality_followup_20261009/native_yolo_eca_stage/dataset_manifest.json"
    if not CONFIG.exists() or not source.exists():
        raise FileNotFoundError("Existing isolated native dataset is required")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest["images_by_split"] != {"train": 2319, "val": 193}:
        raise ValueError("Dataset split drift")
    value = {
        "date": "2026-10-22", "kinds": KINDS, "seeds": [20261022, 20261023, 20261024],
        "hypotheses": {"strip": "Directional features help grading/mask contours", "deg": "Directional prior adds value beyond local strip mixing",
                       "partial": "Partial channel mixing yields a useful lower-cost adapter"},
        "detector": str(DETECTOR), "detector_sha256": sha256(DETECTOR),
        "targets_sha256": sha256(TARGETS), "config_sha256": sha256(CONFIG), "dataset_manifest": manifest,
        "installed_ultralytics": ultralytics.__version__, "source_hashes": sources(),
        "dose_policy": "Short 8-epoch screen; promising families enter fresh paired 40-epoch fits. Never pool different doses.",
        "selection": "Internal 193-image training holdout Mask AP50, epoch-zero eligible; no development labels select checkpoints",
        "freeze": "Everything before final Segment26; train Segment26 and adapter only; frozen BN eval inherited",
        "optimizer": {"type": "AdamW", "lr": 0.0001, "lrf": 1.0, "weight_decay": 0.0005, "batch": 8, "nbs": 64, "warmup_epochs": 0},
        "augmentation": "Existing original plus two offline variants for fit images; online augmentation off",
        "precision": "FP32", "gate_range": [0.8, 1.2], "identity_initialization": True,
        "kernel": 7, "deg_bins": 18, "partial_ratio": 0.25,
        "baseline_caveat": "Pretrained backbone has seen the original training population including internal holdout; not independent test",
        "protected": ["original labels", "rules", "thresholds", "original checkpoints", "sealed test"],
        "primary_model_replaced": False, "test_accessed": False,
        "budget": {"primary_start_used_percent": 4, "starting_balance": 556.4829495, "reset_forbidden": True,
                   "paid_credits_authorized": False, "window_resets_at": 1792694245,
                   "begin_wrapup_percent": 92, "stop_new_fits_percent": 96, "absolute_stop_percent": 99},
        "fit_hard_timeout_seconds": 7200,
    }
    path = STAGE / "plan.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            # Allow one explicitly logged correction before ANY fit exists.
            # Old bytes retained; no completed/partial fit may be reinterpreted.
            old_plan = json.loads(path.read_text(encoding="utf-8"))
            if list(STAGE.glob("*/context.json")) or (STAGE / "plan_pretraining_revision.json").exists():
                raise ValueError("Plan/source changed after commitment; explicit new stage required")
            save(STAGE / "plan_pretraining_revision.json", {"reason": "Remove unimplemented AP95 tie-break before any training",
                 "old_plan": old_plan, "old_sha256": sha256(path), "no_fit_started": True})
            save(path, value)
    else:
        save(path, value)
    return value


def run(kind, seed, epochs, tag):
    torch.set_num_threads(4)
    LOGGER.setLevel(30)
    plan = prepare_plan()
    name = f"{tag}_{kind}_seed{seed}_e{epochs}"
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("Unsafe tag")
    directory = STAGE / name
    if (directory / "completion.json").exists():
        record = json.loads((directory / "completion.json").read_text(encoding="utf-8"))
        if record["source_hashes"] != sources():
            raise ValueError("Source drift for completed fit")
        print("FIT ALREADY COMPLETE " + name, flush=True)
        return record
    if directory.exists():
        raise FileExistsError("Partial fit exists; explicit recovery required: " + str(directory))
    if (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists():
        raise RuntimeError("Quota stop marker is present")
    model = YOLO(str(DETECTOR))
    head_index = len(model.model.model) - 1
    started_at = time.monotonic()
    max_time = plan["fit_hard_timeout_seconds"]

    def started(trainer):
        if Path(trainer.save_dir).resolve() != directory.resolve():
            raise ValueError("Unexpected save path")
        bad = [n for n, p in trainer.model.named_parameters() if p.requires_grad and not n.startswith(f"model.{head_index}.")]
        if bad:
            raise ValueError("Backbone unexpectedly unfrozen: " + str(bad[:3]))
        context = {"name": name, "pid": os.getpid(), "kind": kind, "seed": seed, "epochs": epochs,
                   "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source_hashes": sources(),
                   "plan_sha256": sha256(STAGE / "plan.json"),
                   "trainable_parameters": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                   "total_parameters": sum(p.numel() for p in trainer.model.parameters()),
                   "adapter_parameters": sum(p.numel() for n, p in trainer.model.named_parameters() if "refine_gates" in n)}
        save(directory / "context.json", context)
        validator = trainer.get_validator()
        metrics = validator(model=copy.deepcopy(trainer.ema.ema))
        fitness = float(metrics["metrics/mAP50(M)"])
        trainer.best_fitness = fitness
        checkpoint = directory / "weights/initial_reference.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": -1, "best_fitness": fitness, "model": None, "ema": copy.deepcopy(trainer.ema.ema).half(),
                    "updates": 0, "optimizer": None, "train_args": vars(trainer.args), "train_metrics": metrics,
                    "version": ultralytics.__version__, "date": context["created_at"], "reference_only": True}, checkpoint)
        save(directory / "initial_reference.json", {"metrics": metrics, "fitness": fitness, "checkpoint_sha256": sha256(checkpoint)})
        print(f"INITIAL {name} mask50={fitness:.8f}", flush=True)

    def finished_epoch(trainer):
        if not trainer.validator.training:
            save(directory / "final_validation_only.json", {"metrics": trainer.metrics, "not_training_epoch": True})
            return
        trace = directory / "held_trace.csv"
        rows = pd.read_csv(trace).to_dict("records") if trace.exists() else []
        if len(rows) != int(trainer.epoch):
            raise ValueError("Epoch gap/duplicate")
        rows.append({"epoch": int(trainer.epoch), "mask_ap50": float(trainer.metrics["metrics/mAP50(M)"]),
                     "mask_ap95": float(trainer.metrics["metrics/mAP50-95(M)"]), "elapsed_seconds": time.monotonic() - started_at})
        pd.DataFrame(rows).to_csv(trace, index=False)
        print(f"EPOCH {name} {trainer.epoch + 1}/{epochs} AP50={rows[-1]['mask_ap50']:.6f} AP95={rows[-1]['mask_ap95']:.6f}", flush=True)
        if time.monotonic() - started_at > max_time:
            save(directory / "timeout.json", {"hard_timeout_seconds": max_time, "graceful_epoch_boundary": True})
            trainer.stop = True
        if (STAGE / "STOP_AFTER_CURRENT_EPOCH").exists():
            trainer.stop = True

    def snapshot(trainer):
        path = directory / "weights/resumable.pt"
        temporary = path.with_suffix(".copying")
        shutil.copy2(trainer.last, temporary)
        os.replace(temporary, path)

    model.add_callback("on_train_start", started)
    model.add_callback("on_fit_epoch_end", finished_epoch)
    model.add_callback("on_model_save", snapshot)
    model.train(data=str(CONFIG), epochs=epochs, batch=8, imgsz=640, device=0, workers=4,
                seed=seed, deterministic=True, amp=False, optimizer="AdamW", lr0=1e-4, lrf=1.0,
                weight_decay=0.0005, nbs=64, warmup_epochs=0.0, freeze=list(range(head_index)),
                patience=0, save=True, save_period=-1, plots=False, project=str(STAGE), name=name,
                exist_ok=False, cache=False, close_mosaic=0, mosaic=0., mixup=0., copy_paste=0., cutmix=0.,
                hsv_h=0., hsv_s=0., hsv_v=0., degrees=0., translate=0., scale=0., shear=0., perspective=0.,
                flipud=0., fliplr=0., fraction=1., mask_ratio=4, overlap_mask=True, trainer=TRAINERS[kind])
    trace = pd.read_csv(directory / "held_trace.csv")
    ref = json.loads((directory / "initial_reference.json").read_text(encoding="utf-8"))
    best = directory / "weights/best.pt"
    trained_best = float(trace.mask_ap50.max())
    selected = best if best.exists() and trained_best > ref["fitness"] else directory / "weights/initial_reference.pt"
    record = {"name": name, "kind": kind, "seed": seed, "requested_epochs": epochs, "epochs_completed": len(trace),
              "equal_dose_completed": len(trace) == epochs, "source_hashes": sources(),
              "selected_checkpoint": str(selected), "selected_checkpoint_sha256": sha256(selected),
              "initial_held_mask_ap50": ref["fitness"], "best_held_mask_ap50": max(ref["fitness"], trained_best),
              "best_trained_held_mask_ap50": trained_best, "wall_seconds": time.monotonic() - started_at,
              "test_accessed": False, "development_used_in_selection": False}
    save(directory / ("completion.json" if record["equal_dose_completed"] else "partial.json"), record)
    print("FIT RESULT " + json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--tag", default="screen")
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare_plan(), ensure_ascii=False), flush=True)
    elif args.kind:
        run(args.kind, args.seed, args.epochs, args.tag)
    else:
        parser.error("Specify --kind or --prepare")


if __name__ == "__main__":
    main()
