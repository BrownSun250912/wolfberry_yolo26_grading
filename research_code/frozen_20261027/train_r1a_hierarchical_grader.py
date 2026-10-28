"""Train the independent R1-A crop+mask hierarchical grader.

The detector/segmenter is not imported or modified here.  This script consumes
the audited train/val instance package and learns defect, stem, and continuous
score outputs from a four-channel RGB+mask crop plus explicit mask geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "outputs/r1_multitask_targets_v1/r1_multitask_targets.csv"
DEFAULT_OUTPUT = ROOT / "models/experiments/r1a_hierarchical_grader_seed20260929"
LOWER_THRESHOLD = 37.40298225745565
UPPER_THRESHOLD = 65.62684031710077
CLASS_NAMES = ("UNQUALIFIED", "GRADE_1", "GRADE_2", "GRADE_3")
GEOMETRY_NAMES = (
    "log_mask_area",
    "log_mask_width",
    "log_mask_height",
    "log_crop_width",
    "log_crop_height",
    "mask_area_ratio",
)


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    epochs: int
    patience: int
    batch_size: int
    image_size: int
    learning_rate: float
    weight_decay: float
    lambda_stem: float
    lambda_score: float
    num_workers: int
    amp: bool
    device: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-stem", type=float, default=1.0)
    parser.add_argument("--lambda-score", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Resume model and optimizer state from a prior last.pt into a new output directory.",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Run through the requested total epoch even when validation does not improve.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_determinism(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def compute_geometry(mask: np.ndarray, crop_shape: tuple[int, int]) -> np.ndarray:
    binary = mask > 0
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        raise ValueError("Empty mask")
    mask_area = float(len(xs))
    mask_width = float(xs.max() - xs.min() + 1)
    mask_height = float(ys.max() - ys.min() + 1)
    crop_height, crop_width = crop_shape
    return np.asarray(
        [
            math.log1p(mask_area),
            math.log1p(mask_width),
            math.log1p(mask_height),
            math.log1p(float(crop_width)),
            math.log1p(float(crop_height)),
            mask_area / float(crop_width * crop_height),
        ],
        dtype=np.float32,
    )


def scan_geometry(frame: pd.DataFrame) -> np.ndarray:
    values: list[np.ndarray] = []
    for row in frame.itertuples(index=False):
        crop = cv2.imread(str(ROOT / row.crop_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(ROOT / row.mask_path), cv2.IMREAD_GRAYSCALE)
        if crop is None or mask is None:
            raise FileNotFoundError(f"Missing crop/mask for {row.instance_id}")
        if crop.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Crop/mask shape mismatch for {row.instance_id}")
        values.append(compute_geometry(mask, crop.shape[:2]))
    return np.stack(values)


def letterbox(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    scale = min(size / width, size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    resized_mask = cv2.resize(mask, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    mask_canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    canvas[y : y + new_height, x : x + new_width] = resized_image
    mask_canvas[y : y + new_height, x : x + new_width] = resized_mask
    return canvas, mask_canvas


class R1Dataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        geometry: np.ndarray,
        geometry_mean: np.ndarray,
        geometry_std: np.ndarray,
        image_size: int,
        augment: bool,
        seed: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.geometry = geometry.astype(np.float32)
        self.geometry_mean = geometry_mean.astype(np.float32)
        self.geometry_std = geometry_std.astype(np.float32)
        self.image_size = image_size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image = cv2.imread(str(ROOT / str(row["crop_path"])), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(ROOT / str(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(str(row["instance_id"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment:
            if self.rng.random() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if self.rng.random() < 0.5:
                image = np.ascontiguousarray(image[::-1, :])
                mask = np.ascontiguousarray(mask[::-1, :])
            contrast = float(self.rng.uniform(0.85, 1.15))
            brightness = float(self.rng.uniform(-12.0, 12.0))
            image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
        image, mask = letterbox(image, mask, self.image_size)
        image_float = image.astype(np.float32) / 255.0
        # ImageNet normalization is used as a stable scale even though weights start from scratch.
        image_float = (image_float - np.asarray([0.485, 0.456, 0.406], np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], np.float32
        )
        mask_float = (mask.astype(np.float32) / 255.0)[..., None]
        tensor = torch.from_numpy(np.concatenate([image_float, mask_float], axis=2).transpose(2, 0, 1))
        geometry = (self.geometry[index] - self.geometry_mean) / self.geometry_std
        stem_target = 0.0 if pd.isna(row["stem_target"]) else float(row["stem_target"])
        score_target = 0.0 if pd.isna(row["score_target"]) else float(row["score_target"]) / 100.0
        return {
            "image": tensor,
            "geometry": torch.from_numpy(geometry),
            "defect": torch.tensor(float(row["defect_target"]), dtype=torch.float32),
            "stem": torch.tensor(stem_target, dtype=torch.float32),
            "stem_available": torch.tensor(float(row["stem_target_available"]), dtype=torch.float32),
            "score": torch.tensor(score_target, dtype=torch.float32),
            "score_available": torch.tensor(float(row["score_target_available"]), dtype=torch.float32),
            "final_class": torch.tensor(int(row["final_class_id"]), dtype=torch.long),
            "instance_id": str(row["instance_id"]),
        }


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


def down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.SiLU(inplace=True),
        ResidualBlock(out_channels),
    )


class R1HierarchicalGrader(nn.Module):
    def __init__(self, geometry_dim: int = len(GEOMETRY_NAMES)) -> None:
        super().__init__()
        self.visual = nn.Sequential(
            down_block(4, 32),
            down_block(32, 64),
            down_block(64, 128),
            down_block(128, 192),
            down_block(192, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.geometry = nn.Sequential(
            nn.Linear(geometry_dim, 32),
            nn.SiLU(inplace=True),
            nn.Linear(32, 32),
            nn.SiLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(288, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.20),
        )
        self.defect_head = nn.Linear(256, 1)
        self.stem_head = nn.Linear(256, 1)
        self.score_head = nn.Linear(256, 1)

    def forward(self, image: torch.Tensor, geometry: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.fusion(torch.cat([self.visual(image), self.geometry(geometry)], dim=1))
        return {
            "defect_logit": self.defect_head(features).squeeze(1),
            "stem_logit": self.stem_head(features).squeeze(1),
            "score": torch.sigmoid(self.score_head(features).squeeze(1)),
        }


def binary_counts(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    target = target.astype(bool)
    prediction = prediction.astype(bool)
    tp = int(np.sum(target & prediction))
    fp = int(np.sum(~target & prediction))
    fn = int(np.sum(target & ~prediction))
    tn = int(np.sum(~target & ~prediction))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def four_class_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    rows = []
    f1_values = []
    for class_id, name in enumerate(CLASS_NAMES):
        stats = binary_counts(target == class_id, prediction == class_id)
        rows.append({"class_id": class_id, "class_name": name, **stats})
        f1_values.append(float(stats["f1"]))
    return {
        "accuracy": float(np.mean(target == prediction)),
        "correct": int(np.sum(target == prediction)),
        "total": int(len(target)),
        "macro_f1": float(np.mean(f1_values)),
        "per_class": rows,
    }


def hierarchical_class(defect_probability: np.ndarray, stem_probability: np.ndarray, score: np.ndarray) -> np.ndarray:
    result = np.full(len(score), 3, dtype=np.int64)
    qualified = defect_probability < 0.5
    result[~qualified] = 0
    has_stem = stem_probability < 0.5
    scored = qualified & has_stem
    result[scored & (score >= UPPER_THRESHOLD)] = 1
    result[scored & (score >= LOWER_THRESHOLD) & (score < UPPER_THRESHOLD)] = 2
    result[scored & (score < LOWER_THRESHOLD)] = 3
    return result


def masked_mean(loss: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    return (loss * available).sum() / available.sum().clamp_min(1.0)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    defect_loss_fn: nn.Module,
    stem_loss_fn: nn.Module,
    score_loss_fn: nn.Module,
    lambda_stem: float,
    lambda_score: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp_enabled: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_defect = 0.0
    total_stem = 0.0
    total_score = 0.0
    total_items = 0
    predictions: list[dict[str, Any]] = []
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            geometry = batch["geometry"].to(device, non_blocking=True)
            defect = batch["defect"].to(device)
            stem = batch["stem"].to(device)
            stem_available = batch["stem_available"].to(device)
            score = batch["score"].to(device)
            score_available = batch["score_available"].to(device)
            final_class = batch["final_class"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(image, geometry)
                defect_loss = defect_loss_fn(output["defect_logit"], defect).mean()
                stem_loss = masked_mean(stem_loss_fn(output["stem_logit"], stem), stem_available)
                score_loss = masked_mean(score_loss_fn(output["score"], score), score_available)
                loss = defect_loss + lambda_stem * stem_loss + lambda_score * score_loss
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            batch_size = len(batch["instance_id"])
            total_items += batch_size
            total_loss += float(loss.detach()) * batch_size
            total_defect += float(defect_loss.detach()) * batch_size
            total_stem += float(stem_loss.detach()) * batch_size
            total_score += float(score_loss.detach()) * batch_size
            defect_probability = torch.sigmoid(output["defect_logit"]).detach().cpu().numpy()
            stem_probability = torch.sigmoid(output["stem_logit"]).detach().cpu().numpy()
            score_value = output["score"].detach().cpu().numpy() * 100.0
            for index, instance_id in enumerate(batch["instance_id"]):
                predictions.append(
                    {
                        "instance_id": instance_id,
                        "final_class_target": int(final_class[index].cpu()),
                        "defect_target": int(defect[index].cpu()),
                        "stem_target": int(stem[index].cpu()),
                        "stem_available": int(stem_available[index].cpu()),
                        "score_target": float(score[index].cpu()) * 100.0,
                        "score_available": int(score_available[index].cpu()),
                        "defect_probability": float(defect_probability[index]),
                        "stem_probability": float(stem_probability[index]),
                        "score_prediction": float(score_value[index]),
                    }
                )
    prediction_frame = pd.DataFrame(predictions)
    final_prediction = hierarchical_class(
        prediction_frame["defect_probability"].to_numpy(),
        prediction_frame["stem_probability"].to_numpy(),
        prediction_frame["score_prediction"].to_numpy(),
    )
    prediction_frame["final_class_prediction"] = final_prediction
    target = prediction_frame["final_class_target"].to_numpy(dtype=int)
    defect_stats = binary_counts(
        prediction_frame["defect_target"].to_numpy(dtype=int),
        prediction_frame["defect_probability"].to_numpy() >= 0.5,
    )
    stem_rows = prediction_frame["stem_available"].eq(1)
    stem_stats = binary_counts(
        prediction_frame.loc[stem_rows, "stem_target"].to_numpy(dtype=int),
        prediction_frame.loc[stem_rows, "stem_probability"].to_numpy() >= 0.5,
    )
    score_rows = prediction_frame["score_available"].eq(1)
    score_error = (
        prediction_frame.loc[score_rows, "score_prediction"]
        - prediction_frame.loc[score_rows, "score_target"]
    ).to_numpy()
    metrics = {
        "loss": total_loss / total_items,
        "defect_loss": total_defect / total_items,
        "stem_loss": total_stem / total_items,
        "score_loss": total_score / total_items,
        "defect": defect_stats,
        "stem": stem_stats,
        "score_mae": float(np.mean(np.abs(score_error))),
        "score_rmse": float(np.sqrt(np.mean(np.square(score_error)))),
        "four_class": four_class_metrics(target, final_prediction),
    }
    return metrics, prediction_frame


def better(metrics: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    current_key = (
        metrics["four_class"]["macro_f1"],
        metrics["four_class"]["per_class"][0]["f1"],
        metrics["stem"]["f1"],
        -metrics["score_mae"],
    )
    best_key = (
        best["four_class"]["macro_f1"],
        best["four_class"]["per_class"][0]["f1"],
        best["stem"]["f1"],
        -best["score_mae"],
    )
    return current_key > best_key


def checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: TrainConfig,
    geometry_mean: np.ndarray,
    geometry_std: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "geometry_names": GEOMETRY_NAMES,
            "geometry_mean": geometry_mean.tolist(),
            "geometry_std": geometry_std.tolist(),
            "thresholds": {"defect": 0.5, "stem": 0.5, "grade3_to_2": LOWER_THRESHOLD, "grade2_to_1": UPPER_THRESHOLD},
            "val_metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    targets_path = resolve(args.targets)
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {output}")
    output.mkdir(parents=True)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
    resume_path = resolve(args.resume_checkpoint) if args.resume_checkpoint is not None else None
    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lambda_stem=args.lambda_stem,
        lambda_score=args.lambda_score,
        num_workers=args.num_workers,
        amp=not args.no_amp,
        device=args.device,
    )
    set_determinism(config.seed)
    data = pd.read_csv(targets_path, encoding="utf-8-sig")
    if len(data) != 4795 or set(data["split"]) != {"train", "val"}:
        raise ValueError("Unexpected target package")
    train = data.loc[data["split"].eq("train")].reset_index(drop=True)
    val = data.loc[data["split"].eq("val")].reset_index(drop=True)
    if len(train) != 4277 or len(val) != 518:
        raise ValueError("Frozen split counts changed")
    train_geometry = scan_geometry(train)
    val_geometry = scan_geometry(val)
    geometry_mean = train_geometry.mean(axis=0)
    geometry_std = train_geometry.std(axis=0)
    geometry_std[geometry_std < 1e-6] = 1.0
    (output / "geometry_stats.json").write_text(
        json.dumps(
            {
                "names": GEOMETRY_NAMES,
                "train_mean": geometry_mean.tolist(),
                "train_std": geometry_std.tolist(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    resume_checkpoint: dict[str, Any] | None = None
    completed_before_resume = 0
    if resume_path is not None:
        if not resume_path.is_file():
            raise FileNotFoundError(resume_path)
        resume_checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        completed_before_resume = int(resume_checkpoint["epoch"])
        if completed_before_resume >= config.epochs:
            raise ValueError(
                f"Resume checkpoint is already at epoch {completed_before_resume}, "
                f"not below requested total {config.epochs}"
            )
        if not np.allclose(
            np.asarray(resume_checkpoint["geometry_mean"], dtype=np.float32),
            geometry_mean,
            rtol=0,
            atol=1e-6,
        ) or not np.allclose(
            np.asarray(resume_checkpoint["geometry_std"], dtype=np.float32),
            geometry_std,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("Geometry normalization changed since the resume checkpoint")
        prior_config = resume_checkpoint["config"]
        immutable = {
            "seed": config.seed,
            "batch_size": config.batch_size,
            "image_size": config.image_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "lambda_stem": config.lambda_stem,
            "lambda_score": config.lambda_score,
        }
        for name, value in immutable.items():
            if prior_config[name] != value:
                raise ValueError(
                    f"Resume-incompatible config change for {name}: "
                    f"checkpoint={prior_config[name]!r}, requested={value!r}"
                )
    augmentation_seed = config.seed + completed_before_resume
    train_dataset = R1Dataset(
        train,
        train_geometry,
        geometry_mean,
        geometry_std,
        config.image_size,
        True,
        augmentation_seed,
    )
    val_dataset = R1Dataset(
        val, val_geometry, geometry_mean, geometry_std, config.image_size, False, config.seed
    )
    generator = torch.Generator().manual_seed(augmentation_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    device = torch.device(config.device)
    model = R1HierarchicalGrader().to(device)
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"], strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    qualified_train = int(train["defect_target"].eq(0).sum())
    defect_train = int(train["defect_target"].eq(1).sum())
    present_train = int((train["stem_target"] == 0).sum())
    missing_train = int((train["stem_target"] == 1).sum())
    defect_pos_weight = qualified_train / defect_train
    stem_pos_weight = present_train / missing_train
    defect_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(defect_pos_weight, device=device), reduction="none"
    )
    stem_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(stem_pos_weight, device=device), reduction="none"
    )
    score_loss_fn = nn.SmoothL1Loss(beta=0.05, reduction="none")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        remaining_epochs = config.epochs - completed_before_resume
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=remaining_epochs, eta_min=1e-5
        )
        scheduler_policy = (
            "resume optimizer LR, then new cosine decay over remaining "
            f"{remaining_epochs} epochs to eta_min=1e-5"
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=1e-5
        )
        scheduler_policy = f"single cosine decay over {config.epochs} epochs to eta_min=1e-5"
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    run_config = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "targets": {"path": str(targets_path), "sha256": sha256(targets_path)},
        "model": {
            "name": "R1HierarchicalGrader",
            "initialization": "random_kaiming_pytorch_default_no_external_pretrained_weights",
            "parameters": parameter_count,
            "input": f"RGB+mask {config.image_size}x{config.image_size} plus {len(GEOMETRY_NAMES)} geometry features",
        },
        "loss": {
            "defect": "BCEWithLogitsLoss",
            "defect_pos_weight": defect_pos_weight,
            "stem": "BCEWithLogitsLoss",
            "stem_pos_weight": stem_pos_weight,
            "score": "SmoothL1Loss(beta=0.05) on score/100",
            "lambda_stem": config.lambda_stem,
            "lambda_score": config.lambda_score,
        },
        "selection": "val GT-crop hierarchical four-class macro F1; ties: UQ F1, stem F1, lower score MAE, earlier epoch",
        "early_stopping_enabled": not args.no_early_stop,
        "scheduler_policy": scheduler_policy,
        "resume": None
        if resume_path is None
        else {
            "checkpoint": str(resume_path),
            "checkpoint_sha256": sha256(resume_path),
            "completed_epoch": completed_before_resume,
            "optimizer_state_restored": True,
            "model_state_restored": True,
            "scheduler_state_restored": False,
            "scheduler_reset_reason": (
                "total epoch budget extended; decay current LR smoothly to "
                f"epoch {config.epochs}"
            ),
            "augmentation_rng_note": "new deterministic stream seeded with seed + completed_epoch; not bitwise-identical to an uninterrupted run",
        },
        "test_accessed": False,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
    }
    (output / "training_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    best_predictions: pd.DataFrame | None = None
    if resume_path is not None:
        source_dir = resume_path.parent
        source_history_path = source_dir / "history.csv"
        source_best_path = source_dir / "best.pt"
        source_predictions_path = source_dir / "best_val_predictions.csv"
        for required in (source_history_path, source_best_path, source_predictions_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        source_history = pd.read_csv(source_history_path, encoding="utf-8-sig")
        if int(source_history["epoch"].iloc[-1]) != completed_before_resume:
            raise ValueError("Source history does not end at the resume checkpoint epoch")
        history = source_history.to_dict("records")
        prior_best = torch.load(source_best_path, map_location="cpu", weights_only=False)
        best_metrics = prior_best["val_metrics"]
        best_epoch = int(prior_best["epoch"])
        best_predictions = pd.read_csv(source_predictions_path, encoding="utf-8-sig")
        epochs_without_improvement = completed_before_resume - best_epoch
        shutil.copy2(source_best_path, output / "best.pt")
        shutil.copy2(source_predictions_path, output / "best_val_predictions.csv")
        pd.DataFrame(history).to_csv(output / "history.csv", index=False, encoding="utf-8-sig")
    for epoch in range(completed_before_resume + 1, config.epochs + 1):
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            device,
            defect_loss_fn,
            stem_loss_fn,
            score_loss_fn,
            config.lambda_stem,
            config.lambda_score,
            optimizer,
            scaler,
            amp_enabled,
        )
        val_metrics, val_predictions = run_epoch(
            model,
            val_loader,
            device,
            defect_loss_fn,
            stem_loss_fn,
            score_loss_fn,
            config.lambda_stem,
            config.lambda_score,
            None,
            scaler,
            amp_enabled,
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_macro_f1": train_metrics["four_class"]["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_macro_f1": val_metrics["four_class"]["macro_f1"],
            "val_accuracy": val_metrics["four_class"]["accuracy"],
            "val_correct": val_metrics["four_class"]["correct"],
            "val_uq_f1": val_metrics["four_class"]["per_class"][0]["f1"],
            "val_defect_f1": val_metrics["defect"]["f1"],
            "val_stem_f1": val_metrics["stem"]["f1"],
            "val_score_mae": val_metrics["score_mae"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False, encoding="utf-8-sig")
        checkpoint(
            output / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            config,
            geometry_mean,
            geometry_std,
            val_metrics,
        )
        if better(val_metrics, best_metrics):
            best_metrics = val_metrics
            best_epoch = epoch
            best_predictions = val_predictions.copy()
            epochs_without_improvement = 0
            checkpoint(
                output / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                config,
                geometry_mean,
                geometry_std,
                val_metrics,
            )
            best_predictions.to_csv(
                output / "best_val_predictions.csv", index=False, encoding="utf-8-sig"
            )
        else:
            epochs_without_improvement += 1
        print(
            f"epoch {epoch:03d}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val={val_metrics['four_class']['correct']}/{val_metrics['four_class']['total']} "
            f"macro_f1={val_metrics['four_class']['macro_f1']:.4f} "
            f"uq_f1={val_metrics['four_class']['per_class'][0]['f1']:.4f} "
            f"stem_f1={val_metrics['stem']['f1']:.4f} "
            f"score_mae={val_metrics['score_mae']:.3f} "
            f"best_epoch={best_epoch}",
            flush=True,
        )
        if (
            not args.no_early_stop
            and epoch >= 10
            and epochs_without_improvement >= config.patience
        ):
            print(f"early stopping after {epoch} epochs", flush=True)
            break
    if best_metrics is None or best_predictions is None:
        raise RuntimeError("No best checkpoint was produced")
    result = {
        "schema_version": 1,
        "status": "COMPLETED",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "resumed_from_epoch": completed_before_resume,
        "best_metrics": best_metrics,
        "best_checkpoint_sha256": sha256(output / "best.pt"),
        "last_checkpoint_sha256": sha256(output / "last.pt"),
        "test_accessed": False,
        "scope": "GT-crop diagnostic and checkpoint selection only; frozen-Seg prediction replay is still required",
    }
    (output / "training_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    per_class = pd.DataFrame(best_metrics["four_class"]["per_class"])
    per_class.to_csv(output / "best_val_class_metrics.csv", index=False, encoding="utf-8-sig")
    report = [
        "# R1-A 独立分层分级器训练结果",
        "",
        f"- 状态：{result['status']}",
        f"- best epoch：{best_epoch}",
        f"- 完成 epoch：{len(history)}",
        f"- 续训起点：epoch {completed_before_resume if completed_before_resume else '无（从头训练）'}",
        f"- GT-crop val 正确：{best_metrics['four_class']['correct']}/{best_metrics['four_class']['total']}",
        f"- GT-crop val Accuracy：{best_metrics['four_class']['accuracy']:.4f}",
        f"- GT-crop val Macro F1：{best_metrics['four_class']['macro_f1']:.4f}",
        f"- UQ F1：{best_metrics['four_class']['per_class'][0]['f1']:.4f}",
        f"- 缺梗 F1：{best_metrics['stem']['f1']:.4f}",
        f"- Score MAE：{best_metrics['score_mae']:.4f}",
        "- Test：未访问。",
        "",
        "这只是 GT crop 上的分级器诊断与选模结果，不是最终系统结果。下一步必须在冻结的 Seg baseline val 预测实例上重放，且不得改变预测数量、confidence 或 NMS。",
        "",
    ]
    (output / "TRAINING_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
