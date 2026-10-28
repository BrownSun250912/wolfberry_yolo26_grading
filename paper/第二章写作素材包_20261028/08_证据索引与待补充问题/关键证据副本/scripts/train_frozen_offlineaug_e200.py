"""Train or resume the frozen 200-epoch YOLO26n-seg Mask-mAP50 baseline.

Fresh runs begin with the local official checkpoint; --resume preserves the
optimizer, EMA, schedule, and best fitness from the interrupted last.pt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "configs/grading_four_class_v3_augmented_delivery_local.yaml"
WEIGHTS = ROOT / "models/pretrained/yolo26n-seg.pt"
WEIGHTS_SHA256 = "361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f"
PROJECT = ROOT / "models/experiments"
RUN_NAME = "yolo26n_seg_offlineaug_e200_mask50_seed20260915"
SEED = 20260915
EPOCHS = 200
TRACE_FIELDS = ("epoch", "mask_map50", "mask_map50_95", "default_fitness", "selection_fitness", "best_mask_map50", "is_new_best")


class Mask50SegmentationTrainer(SegmentationTrainer):
    """Replace only the checkpoint fitness; retain genuine validation metrics."""

    def validate(self):
        previous_best = self.best_fitness
        metrics, default_fitness = super().validate()
        if metrics is None:
            raise RuntimeError("Validation returned no metrics; cannot select best.pt")
        value = float(metrics["metrics/mAP50(M)"])
        if not math.isfinite(value):
            raise RuntimeError(f"Nonfinite validation Mask mAP50: {value}")
        self.default_fitness = float(default_fitness)
        self.best_fitness = value if previous_best is None else max(previous_best, value)
        self.mask50_new_best = previous_best is None or value >= previous_best
        # In this Ultralytics version save_model() writes best.pt when
        # self.best_fitness == self.fitness. Equal scores favor the later epoch.
        return metrics, value


def preflight(resume: bool = False) -> None:
    run_dir = PROJECT / RUN_NAME
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    if not DATA.is_file() or not WEIGHTS.is_file():
        raise FileNotFoundError(f"Missing data YAML or pretrained weights: {DATA}, {WEIGHTS}")
    actual_hash = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()
    if actual_hash != WEIGHTS_SHA256:
        raise ValueError(f"Pretrained checkpoint changed: {actual_hash}")
    base = ROOT / "data/grading_four_class_v3_augmented_delivery"
    expected = {"images/train": 2898, "labels/train": 2898, "images/val": 116, "labels/val": 116}
    for relative, count in expected.items():
        extension = "*.png" if relative.startswith("images") else "*.txt"
        actual = sum(1 for _ in (base / relative).glob(extension))
        if actual != count:
            raise ValueError(f"Unexpected {relative} count: {actual} != {count}")
    if resume:
        last = run_dir / "weights/last.pt"
        trace = run_dir / "mask50_selection_trace.csv"
        results = run_dir / "results.csv"
        if not last.is_file() or not trace.is_file() or not results.is_file():
            raise FileNotFoundError("Resuming requires last.pt, results.csv and mask50_selection_trace.csv")
        with trace.open(newline="", encoding="utf-8") as stream:
            trace_rows = list(csv.DictReader(stream))
        with results.open(newline="", encoding="utf-8") as stream:
            result_rows = list(csv.DictReader(stream))
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)  # trusted local training output
        completed = int(checkpoint["epoch"]) + 1
        if not (0 < completed < EPOCHS and completed == len(trace_rows) == len(result_rows)):
            raise ValueError(
                f"Checkpoint/CSV mismatch: checkpoint={completed}, trace={len(trace_rows)}, results={len(result_rows)}"
            )
        if [int(r["epoch"]) for r in result_rows] != list(range(1, completed + 1)):
            raise ValueError("Non-contiguous results.csv epochs")
        if [int(r["epoch"]) for r in trace_rows] != list(range(1, completed + 1)):
            raise ValueError("Non-contiguous selection trace epochs")
        if checkpoint.get("optimizer") is None or checkpoint.get("ema") is None:
            raise ValueError("Checkpoint lacks optimizer or EMA; cannot faithfully resume")
        if not math.isclose(float(checkpoint["best_fitness"]), float(trace_rows[-1]["best_mask_map50"])):
            raise ValueError("Checkpoint best fitness does not match the Mask-mAP50 trace")
        args = checkpoint["train_args"]
        if args["epochs"] != EPOCHS or args["seed"] != SEED or Path(args["data"]).resolve() != DATA.resolve():
            raise ValueError("Checkpoint does not belong to the frozen training protocol")
        print(f"RESUME VERIFIED: completed={completed}/{EPOCHS}, best_mask50={checkpoint['best_fitness']}", flush=True)
    print(f"PREFLIGHT OK: {RUN_NAME}; {expected}; pretrained_sha256={actual_hash}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Check frozen inputs without training")
    parser.add_argument("--resume", action="store_true", help="Continue the interrupted run from its own last.pt")
    options = parser.parse_args()
    preflight(resume=options.resume)
    if options.check_only:
        return

    trace = PROJECT / RUN_NAME / "mask50_selection_trace.csv"

    def record_epoch(trainer):
        epoch = trainer.epoch + 1
        if epoch > EPOCHS:  # final_eval triggers this callback a second time
            return
        row = {
            "epoch": epoch,
            "mask_map50": trainer.metrics["metrics/mAP50(M)"],
            "mask_map50_95": trainer.metrics["metrics/mAP50-95(M)"],
            "default_fitness": trainer.default_fitness,
            "selection_fitness": trainer.fitness,
            "best_mask_map50": trainer.best_fitness,
            "is_new_best": int(trainer.mask50_new_best),
        }
        with trace.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=TRACE_FIELDS)
            if stream.tell() == 0:
                writer.writeheader()
            writer.writerow(row)

    model = YOLO(str(PROJECT / RUN_NAME / "weights/last.pt" if options.resume else WEIGHTS))
    model.add_callback("on_fit_epoch_end", record_epoch)
    model.train(
        trainer=Mask50SegmentationTrainer,
        data=str(DATA),
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        epochs=EPOCHS,
        patience=0,
        batch=8,
        imgsz=640,
        seed=SEED,
        device="0",
        workers=4,
        deterministic=True,
        amp=False,
        cache=False,
        end2end=True,
        optimizer="AdamW",
        lr0=0.00125,
        lrf=0.01,
        cos_lr=False,
        momentum=0.9,
        weight_decay=0.0005,
        nbs=64,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.0,
        mask_ratio=4,
        overlap_mask=True,
        val=True,
        split="val",
        plots=True,
        save=True,
        save_period=-1,
        resume=options.resume,
        compile=False,
        multi_scale=0.0,
        rect=False,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        close_mosaic=0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        erasing=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        bgr=0.0,
    )
    with trace.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != EPOCHS or not (PROJECT / RUN_NAME / "weights/best.pt").is_file():
        raise RuntimeError(f"Incomplete training or best checkpoint: {trace}")
    best = [row for row in rows if row["is_new_best"] == "1"][-1]
    print(f"COMPLETE: {EPOCHS} epochs; best epoch={best['epoch']}; val Mask mAP50={best['mask_map50']}", flush=True)
    from report_frozen_offlineaug_e200 import generate, inputs

    complete = inputs()
    if complete is None:
        raise RuntimeError("Cannot create paper result: incomplete final evidence")
    generate(*complete)


if __name__ == "__main__":
    main()
