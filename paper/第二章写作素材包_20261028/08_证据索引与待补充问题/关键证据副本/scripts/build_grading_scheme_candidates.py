from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import cv2
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import cohen_kappa_score


RANDOM_SEED = 20260902

SCHEMES = {
    "balanced": {
        "label": "Balanced (recommended)",
        "weights": {"size_score": 0.40, "shape_score": 0.35, "color_uniformity_score": 0.25},
        "method": "weighted_mean",
    },
    "size_priority": {
        "label": "Size priority",
        "weights": {"size_score": 0.55, "shape_score": 0.30, "color_uniformity_score": 0.15},
        "method": "weighted_mean",
    },
    "appearance_priority": {
        "label": "Shape/color priority",
        "weights": {"size_score": 0.25, "shape_score": 0.40, "color_uniformity_score": 0.35},
        "method": "weighted_mean",
    },
    "bottleneck_guard": {
        "label": "Bottleneck guard",
        "weights": {"size_score": 0.35, "shape_score": 0.35, "color_uniformity_score": 0.30},
        "method": "0.70 weighted mean + 0.30 minimum component, plus grade gates",
    },
}

COLOR_FEATURE_WEIGHTS = {
    "L_std": 0.20,
    "a_std": 0.30,
    "b_std": 0.20,
    "max_damage_ratio": 0.30,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build several interpretable three-grade candidate schemes.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("outputs/strict_rot_exclusion_v1/qualified_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/grading_scheme_candidates_v1"),
    )
    parser.add_argument("--lower-quantile", type=float, default=0.30)
    parser.add_argument("--upper-quantile", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float))
    if ref.size == 0:
        raise ValueError("Empty empirical reference distribution.")
    x = np.asarray(values, dtype=float)
    left = np.searchsorted(ref, x, side="left")
    right = np.searchsorted(ref, x, side="right")
    return np.clip((left + right) / (2.0 * ref.size), 0.0, 1.0)


def add_component_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame.copy()
    train = scored["split"].eq("train")
    if not train.any():
        raise ValueError("A train split is required to fit score references.")

    reference_rows: list[dict[str, object]] = []

    for feature in ("area", "minor_axis", "solidity", "aspect_ratio", "circularity"):
        reference = scored.loc[train, feature].to_numpy(dtype=float)
        scored[f"pct_{feature}"] = empirical_percentile(reference, scored[feature].to_numpy(dtype=float))
        reference_rows.append(
            {
                "reference_scope": "global_train",
                "source_group": "ALL",
                "feature": feature,
                "n": int(reference.size),
                "min": float(np.min(reference)),
                "P10": float(np.quantile(reference, 0.10)),
                "P25": float(np.quantile(reference, 0.25)),
                "median": float(np.quantile(reference, 0.50)),
                "P75": float(np.quantile(reference, 0.75)),
                "P90": float(np.quantile(reference, 0.90)),
                "max": float(np.max(reference)),
            }
        )

    scored["aspect_typicality"] = np.clip(1.0 - 2.0 * np.abs(scored["pct_aspect_ratio"] - 0.5), 0.0, 1.0)
    scored["size_score"] = 100.0 * (0.65 * scored["pct_area"] + 0.35 * scored["pct_minor_axis"])
    scored["shape_score"] = 100.0 * (
        0.50 * scored["pct_solidity"]
        + 0.30 * scored["aspect_typicality"]
        + 0.20 * scored["pct_circularity"]
    )

    for feature, weight in COLOR_FEATURE_WEIGHTS.items():
        output = np.full(len(scored), np.nan, dtype=float)
        for group, group_index in scored.groupby("source_group", sort=True).groups.items():
            group_positions = scored.index.get_indexer(group_index)
            reference = scored.loc[train & scored["source_group"].eq(group), feature].to_numpy(dtype=float)
            if reference.size < 30:
                reference = scored.loc[train, feature].to_numpy(dtype=float)
                scope = "global_train_fallback"
            else:
                scope = "source_group_train"
            values = scored.loc[group_index, feature].to_numpy(dtype=float)
            output[group_positions] = 1.0 - empirical_percentile(reference, values)
            reference_rows.append(
                {
                    "reference_scope": scope,
                    "source_group": str(group),
                    "feature": feature,
                    "n": int(reference.size),
                    "min": float(np.min(reference)),
                    "P10": float(np.quantile(reference, 0.10)),
                    "P25": float(np.quantile(reference, 0.25)),
                    "median": float(np.quantile(reference, 0.50)),
                    "P75": float(np.quantile(reference, 0.75)),
                    "P90": float(np.quantile(reference, 0.90)),
                    "max": float(np.max(reference)),
                    "component_weight": weight,
                }
            )
        scored[f"uniformity_{feature}"] = np.clip(output, 0.0, 1.0)

    scored["color_uniformity_score"] = 100.0 * sum(
        weight * scored[f"uniformity_{feature}"] for feature, weight in COLOR_FEATURE_WEIGHTS.items()
    )
    for column in ("size_score", "shape_score", "color_uniformity_score"):
        scored[column] = scored[column].clip(0.0, 100.0)

    return scored, pd.DataFrame(reference_rows)


def weighted_score(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    return sum(frame[column] * weight for column, weight in weights.items())


def assign_schemes(
    scored: pd.DataFrame, lower_quantile: float, upper_quantile: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = scored.copy()
    train = result["split"].eq("train")
    parameter_rows: list[dict[str, object]] = []

    gate_thresholds = {
        "size_grade1_min": float(result.loc[train, "size_score"].quantile(0.50)),
        "shape_grade1_min": float(result.loc[train, "shape_score"].quantile(0.35)),
        "color_grade1_min": float(result.loc[train, "color_uniformity_score"].quantile(0.35)),
        "size_grade3_max": float(result.loc[train, "size_score"].quantile(0.10)),
        "shape_grade3_max": float(result.loc[train, "shape_score"].quantile(0.10)),
        "color_grade3_max": float(result.loc[train, "color_uniformity_score"].quantile(0.10)),
    }

    for scheme, config in SCHEMES.items():
        base = weighted_score(result, config["weights"])
        if scheme == "bottleneck_guard":
            minimum = result[["size_score", "shape_score", "color_uniformity_score"]].min(axis=1)
            score = 0.70 * base + 0.30 * minimum
        else:
            score = base
        score_column = f"score_{scheme}"
        grade_column = f"grade_{scheme}"
        result[score_column] = score

        low = float(result.loc[train, score_column].quantile(lower_quantile))
        high = float(result.loc[train, score_column].quantile(upper_quantile))
        grade = np.where(result[score_column] >= high, 1, np.where(result[score_column] < low, 3, 2)).astype(int)

        if scheme == "bottleneck_guard":
            grade1_allowed = (
                result["size_score"].ge(gate_thresholds["size_grade1_min"])
                & result["shape_score"].ge(gate_thresholds["shape_grade1_min"])
                & result["color_uniformity_score"].ge(gate_thresholds["color_grade1_min"])
            )
            grade3_forced = (
                result["size_score"].lt(gate_thresholds["size_grade3_max"])
                | result["shape_score"].lt(gate_thresholds["shape_grade3_max"])
                | result["color_uniformity_score"].lt(gate_thresholds["color_grade3_max"])
            )
            grade = np.where((grade == 1) & ~grade1_allowed, 2, grade)
            grade = np.where(grade3_forced, 3, grade)

        result[grade_column] = grade
        row = {
            "scheme": scheme,
            "scheme_label": config["label"],
            "method": config["method"],
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "score_threshold_grade3_to_2": low,
            "score_threshold_grade2_to_1": high,
            "size_weight": config["weights"]["size_score"],
            "shape_weight": config["weights"]["shape_score"],
            "color_uniformity_weight": config["weights"]["color_uniformity_score"],
        }
        if scheme == "bottleneck_guard":
            row.update(gate_thresholds)
        parameter_rows.append(row)

    grade_columns = [f"grade_{scheme}" for scheme in SCHEMES]
    result["scheme_grade_min"] = result[grade_columns].min(axis=1)
    result["scheme_grade_max"] = result[grade_columns].max(axis=1)
    result["scheme_grade_span"] = result["scheme_grade_max"] - result["scheme_grade_min"]
    result["schemes_unanimous"] = result["scheme_grade_span"].eq(0)
    result["scheme_score_std"] = result[[f"score_{scheme}" for scheme in SCHEMES]].std(axis=1, ddof=0)
    result["candidate_label_status"] = "exploratory_candidate_not_training_truth"
    return result, pd.DataFrame(parameter_rows)


def bootstrap_thresholds(
    scored: pd.DataFrame, lower_quantile: float, upper_quantile: float, seed: int, iterations: int = 500
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    train = scored.loc[scored["split"].eq("train")]
    rows: list[dict[str, object]] = []
    for scheme in SCHEMES:
        values = train[f"score_{scheme}"].to_numpy(dtype=float)
        lows = np.empty(iterations, dtype=float)
        highs = np.empty(iterations, dtype=float)
        for iteration in range(iterations):
            sample = rng.choice(values, size=values.size, replace=True)
            lows[iteration] = np.quantile(sample, lower_quantile)
            highs[iteration] = np.quantile(sample, upper_quantile)
        for boundary, estimates in (("grade3_to_2", lows), ("grade2_to_1", highs)):
            rows.append(
                {
                    "scheme": scheme,
                    "boundary": boundary,
                    "bootstrap_iterations": iterations,
                    "mean": float(estimates.mean()),
                    "std": float(estimates.std(ddof=1)),
                    "P2_5": float(np.quantile(estimates, 0.025)),
                    "median": float(np.quantile(estimates, 0.50)),
                    "P97_5": float(np.quantile(estimates, 0.975)),
                }
            )
    return pd.DataFrame(rows)


def build_scheme_summaries(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    distribution_rows: list[dict[str, object]] = []
    for scheme, config in SCHEMES.items():
        column = f"grade_{scheme}"
        for split in ("all", "train", "val"):
            subset = scored if split == "all" else scored.loc[scored["split"].eq(split)]
            for grade in (1, 2, 3):
                count = int(subset[column].eq(grade).sum())
                distribution_rows.append(
                    {
                        "scheme": scheme,
                        "scheme_label": config["label"],
                        "split": split,
                        "grade": grade,
                        "count": count,
                        "share": float(count / len(subset)),
                    }
                )
    distributions = pd.DataFrame(distribution_rows)

    agreement = pd.DataFrame(index=SCHEMES, columns=SCHEMES, dtype=float)
    kappa = pd.DataFrame(index=SCHEMES, columns=SCHEMES, dtype=float)
    pair_rows: list[dict[str, object]] = []
    for left in SCHEMES:
        for right in SCHEMES:
            left_values = scored[f"grade_{left}"].to_numpy(dtype=int)
            right_values = scored[f"grade_{right}"].to_numpy(dtype=int)
            exact = float(np.mean(left_values == right_values))
            weighted = float(cohen_kappa_score(left_values, right_values, weights="quadratic"))
            agreement.loc[left, right] = exact
            kappa.loc[left, right] = weighted
            if list(SCHEMES).index(left) < list(SCHEMES).index(right):
                pair_rows.append(
                    {
                        "scheme_1": left,
                        "scheme_2": right,
                        "exact_agreement": exact,
                        "quadratic_weighted_kappa": weighted,
                    }
                )
    return distributions, agreement, kappa, pd.DataFrame(pair_rows)


def select_unique(
    pool: pd.DataFrame,
    n: int,
    used_instances: set[str],
    used_images: set[str],
    sort_columns: list[str],
    ascending: list[bool],
) -> pd.DataFrame:
    available = pool.loc[~pool["instance_id"].isin(used_instances)].sort_values(sort_columns, ascending=ascending)
    selected_rows: list[pd.Series] = []
    for _, row in available.iterrows():
        if row["image_id"] in used_images:
            continue
        selected_rows.append(row)
        used_instances.add(str(row["instance_id"]))
        used_images.add(str(row["image_id"]))
        if len(selected_rows) == n:
            break
    if len(selected_rows) < n:
        for _, row in available.iterrows():
            if row["instance_id"] in {item["instance_id"] for item in selected_rows}:
                continue
            selected_rows.append(row)
            used_instances.add(str(row["instance_id"]))
            if len(selected_rows) == n:
                break
    return pd.DataFrame(selected_rows)


def select_review_candidates(scored: pd.DataFrame, parameters: pd.DataFrame) -> pd.DataFrame:
    used_instances: set[str] = set()
    used_images: set[str] = set()
    groups: list[pd.DataFrame] = []

    balanced_params = parameters.set_index("scheme").loc["balanced"]
    low = float(balanced_params["score_threshold_grade3_to_2"])
    high = float(balanced_params["score_threshold_grade2_to_1"])
    working = scored.copy()
    working["distance_to_balanced_low"] = (working["score_balanced"] - low).abs()
    working["distance_to_balanced_high"] = (working["score_balanced"] - high).abs()
    mean_score = working[[f"score_{scheme}" for scheme in SCHEMES]].mean(axis=1)
    working["mean_scheme_score"] = mean_score

    for grade in (1, 2, 3):
        pool = working.loc[working["schemes_unanimous"] & working["grade_balanced"].eq(grade)].copy()
        target = float(pool["mean_scheme_score"].median())
        pool["representative_distance"] = (pool["mean_scheme_score"] - target).abs()
        chosen = select_unique(
            pool, 20, used_instances, used_images, ["representative_distance", "instance_id"], [True, True]
        )
        chosen["candidate_group"] = f"consensus_grade{grade}"
        groups.append(chosen)

    for name, distance_column in (
        ("balanced_boundary_grade1_2", "distance_to_balanced_high"),
        ("balanced_boundary_grade2_3", "distance_to_balanced_low"),
    ):
        chosen = select_unique(
            working, 25, used_instances, used_images, [distance_column, "instance_id"], [True, True]
        )
        chosen["candidate_group"] = name
        groups.append(chosen)

    disagreement = working.loc[~working["schemes_unanimous"]].copy()
    chosen = select_unique(
        disagreement,
        40,
        used_instances,
        used_images,
        ["scheme_grade_span", "scheme_score_std", "instance_id"],
        [False, False, True],
    )
    chosen["candidate_group"] = "scheme_disagreement"
    groups.append(chosen)

    candidates = pd.concat(groups, ignore_index=True)
    candidates.insert(0, "review_order", np.arange(1, len(candidates) + 1))
    candidates.insert(0, "review_id", [f"G{index:04d}" for index in range(1, len(candidates) + 1)])
    candidates["review_preferred_scheme"] = ""
    candidates["review_expected_grade"] = ""
    candidates["review_comment"] = ""
    candidates["review_image_file"] = candidates["review_id"] + "__" + candidates["instance_id"] + ".png"
    return candidates


def place_image(canvas: np.ndarray, source: np.ndarray, left: int, top: int, width: int, height: int) -> None:
    h, w = source.shape[:2]
    scale = min(width / max(w, 1), height / max(h, 1))
    resized = cv2.resize(
        source,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    rh, rw = resized.shape[:2]
    y = top + (height - rh) // 2
    x = left + (width - rw) // 2
    canvas[y : y + rh, x : x + rw] = resized


def make_review_image(row: pd.Series, project_root: Path, output_path: Path) -> None:
    image = cv2.imread(str(Path(row["image_path"])), cv2.IMREAD_COLOR)
    split_root = project_root / "outputs" / "features_frozen_v1" / str(row["split"])
    crop = cv2.imread(str(split_root / str(row["crop_path"])), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(split_root / str(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
    if image is None or crop is None or mask is None:
        raise FileNotFoundError(f"Missing image artifact for {row['instance_id']}")

    x1, y1, x2, y2 = [int(row[name]) for name in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")]
    original_roi = image[y1:y2, x1:x2].copy()
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = original_roi.copy()
    cv2.drawContours(outlined, contours, -1, (0, 255, 0), 2)

    if crop.ndim == 3 and crop.shape[2] == 4:
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
        isolated = (crop[:, :, :3].astype(np.float32) * alpha + 32.0 * (1.0 - alpha)).astype(np.uint8)
    else:
        isolated = crop[:, :, :3]

    canvas = np.full((420, 1120, 3), 247, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{row['review_id']}  {row['instance_id']}  [{row['candidate_group']}]",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (18, 18, 18),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "Original ROI + contour", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1)
    cv2.putText(canvas, "Background-excluded crop", (380, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (45, 45, 45), 1)
    place_image(canvas, outlined, 20, 65, 320, 320)
    place_image(canvas, isolated, 380, 65, 320, 320)

    grades = "  ".join(
        [
            f"BAL={int(row['grade_balanced'])}",
            f"SIZE={int(row['grade_size_priority'])}",
            f"APPEAR={int(row['grade_appearance_priority'])}",
            f"GUARD={int(row['grade_bottleneck_guard'])}",
        ]
    )
    lines = [
        f"Components: size={row['size_score']:.1f}  shape={row['shape_score']:.1f}",
        f"            color_uniformity={row['color_uniformity_score']:.1f}",
        f"Candidate grades: {grades}",
        f"Balanced score={row['score_balanced']:.1f}",
        f"Area={row['area']:.0f}px2  minor={row['minor_axis']:.1f}px",
        f"Aspect={row['aspect_ratio']:.2f}  solidity={row['solidity']:.3f}",
        f"Lstd={row['L_std']:.2f}  astd={row['a_std']:.2f}  bstd={row['b_std']:.2f}",
        f"max_damage_ratio={row['max_damage_ratio']:.3f}",
        f"Batch={row['source_group']}  split={row['split']}",
        "These are candidate grades, not training truth.",
    ]
    y = 78
    for line in lines:
        cv2.putText(canvas, line, (730, y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (30, 30, 30), 1, cv2.LINE_AA)
        y += 31
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(output_path)


def make_contact_sheets(candidates: pd.DataFrame, review_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, group_frame in candidates.groupby("candidate_group", sort=False):
        cols = 4
        rows = math.ceil(len(group_frame) / cols)
        tile_w, tile_h = 420, 158
        title_h = 55
        sheet = np.full((title_h + rows * tile_h, cols * tile_w, 3), 250, dtype=np.uint8)
        cv2.putText(
            sheet,
            f"{group}  n={len(group_frame)}",
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for index, (_, row) in enumerate(group_frame.iterrows()):
            panel = cv2.imread(str(review_dir / row["review_image_file"]), cv2.IMREAD_COLOR)
            if panel is None:
                raise FileNotFoundError(row["review_image_file"])
            panel = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            top = title_h + (index // cols) * tile_h
            left = (index % cols) * tile_w
            sheet[top : top + tile_h, left : left + tile_w] = panel
        if not cv2.imwrite(str(output_dir / f"{group}_contact_sheet.png"), sheet):
            raise IOError(group)


def configure_plots() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def make_plots(
    scored: pd.DataFrame,
    distributions: pd.DataFrame,
    agreement: pd.DataFrame,
    parameters: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_plots()

    all_dist = distributions.loc[distributions["split"].eq("all")]
    pivot = all_dist.pivot(index="scheme", columns="grade", values="count").reindex(SCHEMES)
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    bottom = np.zeros(len(pivot))
    colors = {1: "#009E73", 2: "#E69F00", 3: "#D55E00"}
    for grade in (1, 2, 3):
        values = pivot[grade].to_numpy(dtype=float)
        ax.bar([SCHEMES[name]["label"] for name in pivot.index], values, bottom=bottom, color=colors[grade], label=f"Grade {grade}")
        bottom += values
    ax.set(title="Candidate grade counts by scheme", ylabel="Instance count")
    ax.tick_params(axis="x", rotation=18)
    ax.legend(frameon=False, ncol=3)
    fig.savefig(output_dir / "scheme_grade_distribution.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    image = ax.imshow(agreement.to_numpy(dtype=float), cmap="Blues", vmin=0, vmax=1)
    labels = [SCHEMES[name]["label"] for name in agreement.index]
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for row in range(len(labels)):
        for col in range(len(labels)):
            value = agreement.iloc[row, col]
            ax.text(col, row, f"{value:.1%}", ha="center", va="center", color="white" if value > 0.75 else "black")
    ax.set_title("Exact agreement between candidate schemes")
    fig.colorbar(image, ax=ax, shrink=0.8, label="Exact agreement")
    fig.savefig(output_dir / "scheme_agreement_heatmap.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    color_map = {1: "#009E73", 2: "#E69F00", 3: "#D55E00"}
    for grade in (1, 2, 3):
        subset = scored["grade_balanced"].eq(grade)
        axes[0].scatter(scored.loc[subset, "size_score"], scored.loc[subset, "shape_score"], s=12, alpha=0.45, c=color_map[grade], label=f"Grade {grade}")
        axes[1].scatter(scored.loc[subset, "size_score"], scored.loc[subset, "color_uniformity_score"], s=12, alpha=0.45, c=color_map[grade], label=f"Grade {grade}")
    axes[0].set(title="Balanced scheme: size vs shape", xlabel="Size score", ylabel="Shape score", xlim=(0, 100), ylim=(0, 100))
    axes[1].set(title="Balanced scheme: size vs color uniformity", xlabel="Size score", ylabel="Color-uniformity score", xlim=(0, 100), ylim=(0, 100))
    axes[0].legend(frameon=False)
    fig.savefig(output_dir / "balanced_component_scatter.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.8), layout="constrained")
    for scheme, config in SCHEMES.items():
        ax.hist(scored[f"score_{scheme}"], bins=40, histtype="step", linewidth=1.5, label=config["label"])
        row = parameters.set_index("scheme").loc[scheme]
        ax.axvline(float(row["score_threshold_grade3_to_2"]), color="#666666", alpha=0.25, linewidth=0.8)
        ax.axvline(float(row["score_threshold_grade2_to_1"]), color="#666666", alpha=0.25, linewidth=0.8)
    ax.set(title="Candidate score distributions and train-derived boundaries", xlabel="Candidate score (0-100)", ylabel="Instance count")
    ax.legend(frameon=False)
    fig.savefig(output_dir / "scheme_score_distributions.png", dpi=300)
    plt.close(fig)


def write_readme(output_dir: Path, row_count: int, candidate_count: int, unanimous_share: float) -> None:
    text = f"""# 枸杞三级候选方案审核包（v1）

本目录基于严格排除腐烂后的 {row_count:,} 个实例生成。所有结果都是探索性候选，不是正式等级标签，不得直接用于训练。

## 共同的三个维度

- 尺寸分数：`65% area percentile + 35% minor_axis percentile`，越大分数越高。
- 形态分数：`50% solidity percentile + 30% aspect-ratio typicality + 20% circularity percentile`。
- 颜色均匀性分数：反向使用 `L_std`、`a_std`、`b_std`、`max_damage_ratio`；颜色参考分布按 `source_group` 的 train 数据拟合，以降低拍摄光线批次差异。
- 尺寸和形态参考只用全局 train 数据拟合，不按批次归一化，以免抹掉真实大小和形态差异。

## 四套方案

1. `balanced`：尺寸 40%、形态 35%、颜色均匀性 25%，推荐先审核。
2. `size_priority`：尺寸 55%、形态 30%、颜色均匀性 15%，更接近“大果优先”。
3. `appearance_priority`：尺寸 25%、形态 40%、颜色均匀性 35%，更强调规整和均匀。
4. `bottleneck_guard`：尺寸 35%、形态 35%、颜色均匀性 30%，同时加入短板惩罚与一级/三级门槛。

前三套按 train 候选分数的 P30、P70 划分三级；短板型在此基础上增加门槛。P30/P70 只是便于审核的工程候选线，不是已确定的商品标准。

四套方案完全一致的实例占 {unanimous_share:.1%}。审核包抽取 {candidate_count} 个不同实例，覆盖一致一级、二级、三级，均衡方案两条边界以及方案冲突样本。

## 审核方法

优先查看 `contact_sheets/` 总览，再按 `candidate_review.csv` 打开 `review_images/` 中对应图片。建议填写：

- `review_preferred_scheme`：你整体更认可哪套方案。
- `review_expected_grade`：你认为该实例更接近 1、2 或 3。
- `review_comment`：记录“太小、太细、形态不规整、颜色不均匀、算法评分与肉眼冲突”等原因。

重点审核 `balanced_boundary_grade1_2`、`balanced_boundary_grade2_3` 和 `scheme_disagreement`，不用重新标完全部 {row_count:,} 颗。

## 主要文件

- `all_candidate_scheme_scores.csv`：全部实例的分项分数、四套总分和候选等级。
- `scheme_parameters.csv`：权重、P30/P70 分界值和短板门槛。
- `scheme_grade_distribution.csv`：各方案在 train、val 和全部数据中的等级数量。
- `scheme_pairwise_metrics.csv`：方案之间的一致率和二次加权 Kappa。
- `threshold_bootstrap_stability.csv`：500 次重抽样的候选分界稳定性。
- `candidate_review.csv`：人工审核索引。
- `review_images/`、`contact_sheets/`：审核图片。
- `plots/`：方案数量、一致性和分项分布图。
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    features_path = args.features_csv if args.features_csv.is_absolute() else project_root / args.features_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(features_path, encoding="utf-8-sig")
    required = {
        "image_id",
        "instance_id",
        "split",
        "source_group",
        "area",
        "minor_axis",
        "solidity",
        "aspect_ratio",
        "circularity",
        *COLOR_FEATURE_WEIGHTS,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if data["instance_id"].duplicated().any():
        raise ValueError("instance_id must be unique.")
    if data[list(required - {"image_id", "instance_id", "split", "source_group"})].isna().any().any():
        raise ValueError("Required feature values contain missing data.")

    scored, references = add_component_scores(data)
    scored, parameters = assign_schemes(scored, args.lower_quantile, args.upper_quantile)
    bootstrap = bootstrap_thresholds(scored, args.lower_quantile, args.upper_quantile, args.seed)
    distributions, agreement, kappa, pair_metrics = build_scheme_summaries(scored)
    candidates = select_review_candidates(scored, parameters)

    review_dir = output_dir / "review_images"
    for _, row in candidates.iterrows():
        make_review_image(row, project_root, review_dir / row["review_image_file"])
    make_contact_sheets(candidates, review_dir, output_dir / "contact_sheets")
    make_plots(scored, distributions, agreement, parameters, output_dir / "plots")

    scored.to_csv(output_dir / "all_candidate_scheme_scores.csv", index=False, encoding="utf-8-sig")
    references.to_csv(output_dir / "score_reference_quantiles.csv", index=False, encoding="utf-8-sig")
    parameters.to_csv(output_dir / "scheme_parameters.csv", index=False, encoding="utf-8-sig")
    distributions.to_csv(output_dir / "scheme_grade_distribution.csv", index=False, encoding="utf-8-sig")
    agreement.to_csv(output_dir / "scheme_exact_agreement_matrix.csv", encoding="utf-8-sig")
    kappa.to_csv(output_dir / "scheme_quadratic_kappa_matrix.csv", encoding="utf-8-sig")
    pair_metrics.to_csv(output_dir / "scheme_pairwise_metrics.csv", index=False, encoding="utf-8-sig")
    bootstrap.to_csv(output_dir / "threshold_bootstrap_stability.csv", index=False, encoding="utf-8-sig")

    review_columns = [
        "review_id",
        "review_order",
        "candidate_group",
        "image_id",
        "instance_id",
        "split",
        "source_group",
        "size_score",
        "shape_score",
        "color_uniformity_score",
        *[item for scheme in SCHEMES for item in (f"score_{scheme}", f"grade_{scheme}")],
        "scheme_grade_span",
        "area",
        "minor_axis",
        "aspect_ratio",
        "solidity",
        "circularity",
        "L_std",
        "a_std",
        "b_std",
        "max_damage_ratio",
        "review_image_file",
        "review_preferred_scheme",
        "review_expected_grade",
        "review_comment",
    ]
    candidates[review_columns].to_csv(output_dir / "candidate_review.csv", index=False, encoding="utf-8-sig")

    unanimous_share = float(scored["schemes_unanimous"].mean())
    summary = {
        "analysis_type": "exploratory_interpretable_grading_candidates",
        "input_instances": int(len(scored)),
        "train_instances": int(scored["split"].eq("train").sum()),
        "val_instances": int(scored["split"].eq("val").sum()),
        "schemes": list(SCHEMES),
        "threshold_source": "train only",
        "lower_quantile": args.lower_quantile,
        "upper_quantile": args.upper_quantile,
        "unanimous_instance_count": int(scored["schemes_unanimous"].sum()),
        "unanimous_share": unanimous_share,
        "review_candidate_count": int(len(candidates)),
        "random_seed": args.seed,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "status": "candidate_only_not_training_truth",
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(output_dir, len(scored), len(candidates), unanimous_share)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
