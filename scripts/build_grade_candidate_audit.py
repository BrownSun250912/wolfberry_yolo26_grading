from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import RobustScaler


RANDOM_SEED = 20260829
FEATURES = [
    "major_axis",
    "minor_axis",
    "area",
    "aspect_ratio",
    "L_mean",
    "L_std",
    "a_mean",
    "a_std",
    "b_mean",
    "b_std",
    "S_mean",
    "V_mean",
    "circularity",
    "solidity",
    "eccentricity",
    "dark_ratio",
    "abnormal_color_ratio",
    "max_damage_ratio",
]

COLOR_AUDIT_FEATURES = ["L_mean", "a_mean", "S_mean", "L_std", "a_std", "b_std"]
SHAPE_AUDIT_FEATURES = ["aspect_ratio", "circularity", "solidity"]
DAMAGE_AUDIT_FEATURES = ["dark_ratio", "max_damage_ratio"]
SIZE_AUDIT_FEATURES = ["major_axis", "minor_axis", "area"]

GROUP_LABELS = {
    "typical_center": "典型中心样本",
    "small_size": "明显偏小候选",
    "large_size": "明显偏大候选",
    "color_atypical": "颜色异常候选",
    "shape_atypical": "形态异常候选",
    "damage_high": "疑似损伤较高候选",
    "signal_conflict": "多维信号冲突样本",
    "middle_ambiguous": "中间模糊样本",
}

GROUP_COLORS = {
    "typical_center": "#0072B2",
    "small_size": "#E69F00",
    "large_size": "#009E73",
    "color_atypical": "#CC79A7",
    "shape_atypical": "#56B4E9",
    "damage_high": "#D55E00",
    "signal_conflict": "#000000",
    "middle_ambiguous": "#7F7F7F",
}

GROUP_MARKERS = {
    "typical_center": "o",
    "small_size": "v",
    "large_size": "^",
    "color_atypical": "s",
    "shape_atypical": "D",
    "damage_high": "X",
    "signal_conflict": "P",
    "middle_ambiguous": "h",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an unlabeled grade-coverage audit and review pack.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, default=Path("outputs/feature_analysis_frozen_v1/features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grade_candidate_audit_v1"))
    parser.add_argument("--samples-per-group", type=int, default=30)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def robust_parameters(values: pd.Series) -> dict[str, float | str]:
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    median = float(numeric.median())
    q25 = float(numeric.quantile(0.25))
    q75 = float(numeric.quantile(0.75))
    iqr = q75 - q25
    scale = iqr / 1.349 if iqr > 0 else 0.0
    method = "IQR/1.349"
    if not np.isfinite(scale) or scale <= 1e-12:
        mad = float(np.median(np.abs(numeric.to_numpy() - median)))
        scale = 1.4826 * mad
        method = "1.4826*MAD"
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = float(numeric.std(ddof=0))
        method = "population_std_fallback"
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
        method = "unit_fallback"
    return {
        "median": median,
        "q25": q25,
        "q75": q75,
        "iqr": iqr,
        "scale": scale,
        "scale_method": method,
    }


def add_robust_scores(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scored = frame.copy()
    parameter_rows: list[dict[str, float | str]] = []
    for feature in FEATURES:
        params = robust_parameters(scored[feature])
        scored[f"rz_{feature}"] = (scored[feature].astype(float) - float(params["median"])) / float(params["scale"])
        scored[f"pct_{feature}"] = scored[feature].rank(method="average", pct=True)
        parameter_rows.append({"feature": feature, **params})

    size_percentiles = scored[[f"pct_{name}" for name in SIZE_AUDIT_FEATURES]].to_numpy(dtype=float)
    scored["size_percentile"] = np.median(size_percentiles, axis=1)
    scored["size_tail_score"] = 2.0 * np.abs(scored["size_percentile"] - 0.5)

    color_z = np.clip(
        np.abs(scored[[f"rz_{name}" for name in COLOR_AUDIT_FEATURES]].to_numpy(dtype=float)),
        0,
        8,
    )
    shape_z = np.clip(
        np.abs(scored[[f"rz_{name}" for name in SHAPE_AUDIT_FEATURES]].to_numpy(dtype=float)),
        0,
        8,
    )
    damage_z = np.clip(
        np.maximum(
            scored["rz_dark_ratio"].to_numpy(dtype=float),
            scored["rz_max_damage_ratio"].to_numpy(dtype=float),
        ),
        0,
        12,
    )

    scored["color_atypicality_raw"] = np.sqrt(np.mean(np.square(color_z), axis=1))
    scored["shape_atypicality_raw"] = np.sqrt(np.mean(np.square(shape_z), axis=1))
    scored["damage_evidence_raw"] = damage_z
    scored["color_atypicality_pct"] = scored["color_atypicality_raw"].rank(method="average", pct=True)
    scored["shape_atypicality_pct"] = scored["shape_atypicality_raw"].rank(method="average", pct=True)
    scored["damage_evidence_pct"] = scored["damage_evidence_raw"].rank(method="average", pct=True)
    scored["max_nonsize_anomaly_pct"] = scored[
        ["color_atypicality_pct", "shape_atypicality_pct", "damage_evidence_pct"]
    ].max(axis=1)
    overall_raw = np.column_stack(
        [
            np.abs(scored["rz_area"].to_numpy(dtype=float)),
            scored["color_atypicality_raw"].to_numpy(dtype=float),
            scored["shape_atypicality_raw"].to_numpy(dtype=float),
            scored["damage_evidence_raw"].to_numpy(dtype=float),
        ]
    ).max(axis=1)
    scored["overall_atypicality_raw"] = overall_raw
    scored["overall_atypicality_pct"] = scored["overall_atypicality_raw"].rank(method="average", pct=True)

    large_abnormal = (scored["size_percentile"] >= 0.75) & (scored["max_nonsize_anomaly_pct"] >= 0.85)
    small_typical = (scored["size_percentile"] <= 0.25) & (scored["max_nonsize_anomaly_pct"] <= 0.45)
    scored["conflict_type"] = np.select(
        [large_abnormal, small_typical],
        ["large_but_abnormal", "small_but_otherwise_typical"],
        default="",
    )
    scored["conflict_score"] = np.where(
        large_abnormal,
        scored["size_percentile"] + scored["max_nonsize_anomaly_pct"],
        np.where(
            small_typical,
            (1.0 - scored["size_percentile"]) + (1.0 - scored["max_nonsize_anomaly_pct"]),
            0.0,
        ),
    )
    return scored, pd.DataFrame(parameter_rows)


def candidate_masks(scored: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "signal_conflict": scored["conflict_type"].ne(""),
        "typical_center": (
            scored["size_percentile"].between(0.35, 0.65)
            & (scored["color_atypicality_pct"] <= 0.50)
            & (scored["shape_atypicality_pct"] <= 0.50)
            & (scored["damage_evidence_pct"] <= 0.50)
        ),
        "middle_ambiguous": (
            scored["size_percentile"].between(0.30, 0.70)
            & scored["overall_atypicality_pct"].between(0.60, 0.80)
            & (scored["max_nonsize_anomaly_pct"] < 0.90)
        ),
        "damage_high": scored["damage_evidence_pct"] >= 0.90,
        "color_atypical": scored["color_atypicality_pct"] >= 0.90,
        "shape_atypical": scored["shape_atypicality_pct"] >= 0.90,
        "small_size": scored["size_percentile"] <= 0.10,
        "large_size": scored["size_percentile"] >= 0.90,
    }


def diverse_random_sample(
    pool: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
    used_instances: set[str],
    used_images: set[str],
) -> pd.DataFrame:
    available = pool.loc[~pool["instance_id"].isin(used_instances)].copy()
    if len(available) < count:
        raise RuntimeError(f"Candidate pool too small after deduplication: requested={count}, available={len(available)}")
    available["_random"] = rng.random(len(available))
    available = available.sort_values(["_random", "source_group", "image_id"]).drop(columns="_random")

    selected_rows: list[pd.Series] = []
    selected_ids: set[str] = set()
    for prefer_new_image in (True, False):
        for _, row in available.iterrows():
            if len(selected_rows) >= count:
                break
            if row["instance_id"] in selected_ids:
                continue
            if prefer_new_image and row["image_id"] in used_images:
                continue
            selected_rows.append(row)
            selected_ids.add(str(row["instance_id"]))
            used_images.add(str(row["image_id"]))
        if len(selected_rows) >= count:
            break
    result = pd.DataFrame(selected_rows).reset_index(drop=True)
    if len(result) != count:
        raise RuntimeError(f"Could only select {len(result)} of {count} rows.")
    used_instances.update(result["instance_id"].astype(str))
    return result


def select_candidates(
    scored: pd.DataFrame,
    per_group: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    masks = candidate_masks(scored)
    rng = np.random.default_rng(seed)
    used_instances: set[str] = set()
    used_images: set[str] = set()
    selections: list[pd.DataFrame] = []
    pool_rows: list[dict[str, int | str]] = []

    order = [
        "signal_conflict",
        "typical_center",
        "middle_ambiguous",
        "damage_high",
        "color_atypical",
        "shape_atypical",
        "small_size",
        "large_size",
    ]
    for group in order:
        pool = scored.loc[masks[group]].copy()
        pool_rows.append({"candidate_group": group, "candidate_pool_size": int(len(pool)), "selected": per_group})
        if group == "signal_conflict":
            first_target = per_group // 2
            first_val_target = max(1, round(first_target * 0.10))
            second_target = per_group - first_target
            second_val_target = max(1, round(second_target * 0.10))
            first_val = diverse_random_sample(
                pool.loc[(pool["conflict_type"] == "large_but_abnormal") & (pool["split"] == "val")],
                first_val_target,
                rng,
                used_instances,
                used_images,
            )
            first_train = diverse_random_sample(
                pool.loc[(pool["conflict_type"] == "large_but_abnormal") & (pool["split"] == "train")],
                first_target - first_val_target,
                rng,
                used_instances,
                used_images,
            )
            second_val = diverse_random_sample(
                pool.loc[(pool["conflict_type"] == "small_but_otherwise_typical") & (pool["split"] == "val")],
                second_val_target,
                rng,
                used_instances,
                used_images,
            )
            second_train = diverse_random_sample(
                pool.loc[(pool["conflict_type"] == "small_but_otherwise_typical") & (pool["split"] == "train")],
                second_target - second_val_target,
                rng,
                used_instances,
                used_images,
            )
            selected = pd.concat([first_val, first_train, second_val, second_train], ignore_index=True)
        else:
            val_target = max(1, round(per_group * 0.10))
            val_selected = diverse_random_sample(
                pool.loc[pool["split"] == "val"],
                val_target,
                rng,
                used_instances,
                used_images,
            )
            train_selected = diverse_random_sample(
                pool.loc[pool["split"] == "train"],
                per_group - val_target,
                rng,
                used_instances,
                used_images,
            )
            selected = pd.concat([val_selected, train_selected], ignore_index=True)
        selected.insert(0, "candidate_group", group)
        selected.insert(1, "candidate_group_cn", GROUP_LABELS[group])
        selections.append(selected)

    candidates = pd.concat(selections, ignore_index=True)
    candidates = candidates.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    candidates.insert(0, "review_order", np.arange(1, len(candidates) + 1))
    candidates.insert(1, "review_id", [f"R{value:04d}" for value in candidates["review_order"]])
    return candidates, pd.DataFrame(pool_rows)


def cluster_structure_audit(scored: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    matrix = scored[FEATURES].to_numpy(dtype=float)
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    transformed = np.clip(scaler.fit_transform(matrix), -6.0, 6.0)
    pca = PCA(n_components=2, svd_solver="full")
    pca_scores = pca.fit_transform(transformed)
    pca_frame = scored[["image_id", "instance_id", "split"]].copy()
    pca_frame["PC1_robust"] = pca_scores[:, 0]
    pca_frame["PC2_robust"] = pca_scores[:, 1]

    rows: list[dict[str, float | int | str]] = []
    rng = np.random.default_rng(seed)
    for k in (2, 3, 4, 5):
        model = KMeans(n_clusters=k, n_init=30, random_state=seed)
        labels = model.fit_predict(transformed)
        silhouette = float(
            silhouette_score(
                transformed,
                labels,
                sample_size=min(3000, len(transformed)),
                random_state=seed,
            )
        )
        ch = float(calinski_harabasz_score(transformed, labels))
        db = float(davies_bouldin_score(transformed, labels))
        counts = np.bincount(labels, minlength=k)
        stability_values: list[float] = []
        subset_size = int(round(len(transformed) * 0.80))
        for bootstrap_index in range(20):
            subset = np.sort(rng.choice(len(transformed), size=subset_size, replace=False))
            trial = KMeans(n_clusters=k, n_init=15, random_state=seed + bootstrap_index + 1)
            trial_labels = trial.fit_predict(transformed[subset])
            stability_values.append(float(adjusted_rand_score(labels[subset], trial_labels)))
        rows.append(
            {
                "k": k,
                "silhouette": silhouette,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
                "bootstrap_ari_mean": float(np.mean(stability_values)),
                "bootstrap_ari_std": float(np.std(stability_values, ddof=1)),
                "smallest_cluster_n": int(counts.min()),
                "largest_cluster_n": int(counts.max()),
                "smallest_cluster_share": float(counts.min() / counts.sum()),
                "largest_cluster_share": float(counts.max() / counts.sum()),
                "cluster_sizes": ";".join(str(int(value)) for value in sorted(counts.tolist())),
            }
        )
    diagnostics = pd.DataFrame(rows)
    pca_meta = {
        "PC1_explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
        "PC2_explained_variance_ratio": float(pca.explained_variance_ratio_[1]),
        "PC1_PC2_cumulative": float(pca.explained_variance_ratio_[:2].sum()),
    }
    return diagnostics, pca_frame, pca_meta


def make_review_image(row: pd.Series, project_root: Path, output_path: Path) -> None:
    image = cv2.imread(str(Path(row["image_path"])), cv2.IMREAD_COLOR)
    split_root = project_root / "outputs" / "features_frozen_v1" / str(row["split"])
    crop = cv2.imread(str(split_root / str(row["crop_path"])), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(split_root / str(row["mask_path"])), cv2.IMREAD_GRAYSCALE)
    if image is None or crop is None or mask is None:
        raise FileNotFoundError(f"Missing image artifact for {row['instance_id']}")

    x1, y1, x2, y2 = [int(row[name]) for name in ("roi_x1", "roi_y1", "roi_x2", "roi_y2")]
    original_roi = image[y1:y2, x1:x2].copy()
    if original_roi.size == 0:
        raise ValueError(f"Empty ROI for {row['instance_id']}")
    contour_list, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outlined = original_roi.copy()
    cv2.drawContours(outlined, contour_list, -1, (0, 255, 0), 2)

    if crop.shape[2] == 4:
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
        rgb_crop = crop[:, :, :3].astype(np.float32)
        background = np.full_like(rgb_crop, 32, dtype=np.float32)
        isolated = (rgb_crop * alpha + background * (1.0 - alpha)).astype(np.uint8)
    else:
        isolated = crop[:, :, :3]

    canvas = np.full((330, 720, 3), 245, dtype=np.uint8)
    cv2.putText(
        canvas,
        f"{row['review_id']}  {row['instance_id']}",
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(canvas, "Original ROI + refined contour", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)
    cv2.putText(canvas, "Background-excluded crop", (380, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 40, 40), 1)

    def place(source: np.ndarray, left: int, top: int, width: int, height: int) -> None:
        h, w = source.shape[:2]
        scale = min(width / max(w, 1), height / max(h, 1))
        resized = cv2.resize(source, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape[:2]
        y = top + (height - rh) // 2
        x = left + (width - rw) // 2
        canvas[y : y + rh, x : x + rw] = resized

    place(outlined, 20, 65, 320, 240)
    place(isolated, 380, 65, 320, 240)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"Could not write review image: {output_path}")


def make_contact_sheets(candidates: pd.DataFrame, review_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, group_frame in candidates.sort_values("review_order").groupby("candidate_group", sort=False):
        rows, cols = 6, 5
        tile_w, tile_h = 350, 165
        title_h = 60
        sheet = np.full((title_h + rows * tile_h, cols * tile_w, 3), 250, dtype=np.uint8)
        cv2.putText(
            sheet,
            f"{group} - {GROUP_LABELS[group]} - n={len(group_frame)}",
            (18, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        for index, (_, row) in enumerate(group_frame.iterrows()):
            panel = cv2.imread(str(review_dir / str(row["review_image_file"])), cv2.IMREAD_COLOR)
            if panel is None:
                raise FileNotFoundError(row["review_image_file"])
            panel = cv2.resize(panel, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            y = title_h + (index // cols) * tile_h
            x = (index % cols) * tile_w
            sheet[y : y + tile_h, x : x + tile_w] = panel
        path = output_dir / f"{group}_contact_sheet.png"
        if not cv2.imwrite(str(path), sheet):
            raise IOError(path)


def configure_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#9CA3AF",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_score_distributions(scored: pd.DataFrame, output_path: Path) -> None:
    configure_plot_style()
    columns = [
        ("size_percentile", "Relative size percentile"),
        ("color_atypicality_pct", "Color atypicality percentile"),
        ("shape_atypicality_pct", "Shape atypicality percentile"),
        ("damage_evidence_pct", "Damage-candidate percentile"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained")
    for ax, (column, title) in zip(axes.flat, columns):
        ax.hist(scored[column], bins=np.linspace(0, 1, 31), color="#0072B2", edgecolor="white", linewidth=0.5)
        ax.axvline(0.10, color="#D55E00", linestyle="--", linewidth=1.5, label="10th / 90th percentile")
        ax.axvline(0.90, color="#D55E00", linestyle="--", linewidth=1.5)
        ax.set(title=title, xlabel="Empirical percentile", ylabel="Instance count", xlim=(0, 1))
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Full-dataset coverage scores (n=5,088)", fontsize=15)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_candidate_pca(
    pca_scores: pd.DataFrame,
    candidates: pd.DataFrame,
    pca_meta: dict[str, float],
    output_path: Path,
) -> None:
    configure_plot_style()
    merged = pca_scores.merge(candidates[["instance_id", "candidate_group"]], on="instance_id", how="left")
    fig, ax = plt.subplots(figsize=(10, 7), layout="constrained")
    ax.scatter(
        merged["PC1_robust"],
        merged["PC2_robust"],
        s=8,
        color="#D1D5DB",
        alpha=0.35,
        linewidths=0,
        label="All included instances",
    )
    for group in GROUP_LABELS:
        subset = merged.loc[merged["candidate_group"] == group]
        ax.scatter(
            subset["PC1_robust"],
            subset["PC2_robust"],
            s=42,
            color=GROUP_COLORS[group],
            marker=GROUP_MARKERS[group],
            edgecolor="white",
            linewidth=0.5,
            label=f"{group} (n={len(subset)})",
        )
    ax.axhline(0, color="#6B7280", linewidth=0.8)
    ax.axvline(0, color="#6B7280", linewidth=0.8)
    ax.set(
        title="Stratified review candidates over full robust-PCA space",
        xlabel=f"Robust PC1 ({pca_meta['PC1_explained_variance_ratio']:.1%} variance)",
        ylabel=f"Robust PC2 ({pca_meta['PC2_explained_variance_ratio']:.1%} variance)",
    )
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_group_profiles(candidates: pd.DataFrame, output_path: Path) -> None:
    configure_plot_style()
    profile_columns = [
        "size_percentile",
        "size_tail_score",
        "color_atypicality_pct",
        "shape_atypicality_pct",
        "damage_evidence_pct",
        "overall_atypicality_pct",
    ]
    order = list(GROUP_LABELS)
    matrix = candidates.groupby("candidate_group")[profile_columns].median().reindex(order)
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    image = ax.imshow(matrix.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(profile_columns)), [name.replace("_", "\n") for name in profile_columns])
    ax.set_yticks(range(len(order)), order)
    ax.set_title("Median score profile of each review group")
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix.iat[y, x]
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.35 or value > 0.72 else "black", fontsize=8)
    fig.colorbar(image, ax=ax, label="Median percentile / tail score")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_cluster_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> None:
    configure_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    axes[0].plot(diagnostics["k"], diagnostics["silhouette"], color="#0072B2", marker="o", label="Silhouette")
    axes[0].axhline(0.25, color="#D55E00", linestyle="--", linewidth=1.2, label="0.25 descriptive reference")
    axes[0].set(title="Separation by requested cluster count", xlabel="k", ylabel="Silhouette coefficient", xticks=diagnostics["k"])
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].errorbar(
        diagnostics["k"],
        diagnostics["bootstrap_ari_mean"],
        yerr=diagnostics["bootstrap_ari_std"],
        color="#009E73",
        marker="s",
        capsize=4,
        label="Bootstrap ARI mean ± SD",
    )
    axes[1].set(title="Cluster stability under 80% resampling", xlabel="k", ylabel="Adjusted Rand index", xticks=diagnostics["k"], ylim=(-0.05, 1.05))
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Exploratory cluster-structure audit; not grade labels", fontsize=14)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_outputs(
    project_root: Path,
    output_dir: Path,
    scored: pd.DataFrame,
    robust_params: pd.DataFrame,
    candidates: pd.DataFrame,
    pool_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    pca_scores: pd.DataFrame,
    pca_meta: dict[str, float],
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "review_images"
    contact_dir = output_dir / "contact_sheets"
    plot_dir = output_dir / "audit_plots"
    review_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidates.copy()
    candidates["review_image_file"] = [f"{row.review_id}_{row.instance_id}.png" for row in candidates.itertuples()]
    candidates["review_image_path"] = [str((review_dir / name).resolve()) for name in candidates["review_image_file"]]
    candidates["contact_sheet_path"] = [
        str((contact_dir / f"{group}_contact_sheet.png").resolve()) for group in candidates["candidate_group"]
    ]
    for _, row in candidates.iterrows():
        make_review_image(row, project_root, review_dir / str(row["review_image_file"]))
    make_contact_sheets(candidates, review_dir, contact_dir)

    scored.to_csv(output_dir / "all_instance_coverage_scores.csv", index=False, encoding="utf-8-sig")
    robust_params.to_csv(output_dir / "robust_scaling_parameters.csv", index=False, encoding="utf-8-sig")
    pool_summary.to_csv(output_dir / "candidate_group_summary.csv", index=False, encoding="utf-8-sig")
    diagnostics.to_csv(output_dir / "cluster_structure_diagnostics.csv", index=False, encoding="utf-8-sig")
    pca_scores.to_csv(output_dir / "robust_pca_scores.csv", index=False, encoding="utf-8-sig")

    audit_columns = [
        "review_order",
        "review_id",
        "candidate_group",
        "candidate_group_cn",
        "conflict_type",
        "image_id",
        "instance_id",
        "split",
        "source_group",
        "review_image_path",
        "contact_sheet_path",
        "size_percentile",
        "size_tail_score",
        "color_atypicality_pct",
        "shape_atypicality_pct",
        "damage_evidence_pct",
        "overall_atypicality_pct",
        *FEATURES,
    ]
    candidate_audit = candidates[audit_columns].copy()
    candidate_audit.to_csv(output_dir / "candidate_samples_with_scores.csv", index=False, encoding="utf-8-sig")

    review_columns = [
        "review_order",
        "review_id",
        "candidate_group",
        "candidate_group_cn",
        "image_id",
        "instance_id",
        "split",
        "review_image_path",
        "contact_sheet_path",
    ]
    review = candidates[review_columns].copy()
    for column in [
        "segmentation_usable",
        "quality_status",
        "rot",
        "too_small",
        "wrinkled",
        "damaged",
        "deformed",
        "color_problem",
        "other_defect",
        "manual_grade",
        "review_confidence",
        "review_comment",
    ]:
        review[column] = ""
    review.to_csv(output_dir / "candidate_review.csv", index=False, encoding="utf-8-sig")

    plot_score_distributions(scored, plot_dir / "score_distributions.png")
    plot_candidate_pca(pca_scores, candidates, pca_meta, plot_dir / "candidate_pca.png")
    plot_group_profiles(candidates, plot_dir / "candidate_group_profiles.png")
    plot_cluster_diagnostics(diagnostics, plot_dir / "cluster_structure_diagnostics.png")

    group_counts = candidates.groupby(["candidate_group", "split"]).size().unstack(fill_value=0)
    best_row = diagnostics.loc[diagnostics["silhouette"].idxmax()]
    summary = {
        "analysis_type": "exploratory_unlabeled_coverage_audit",
        "input_instances": int(len(scored)),
        "candidate_instances": int(len(candidates)),
        "candidate_groups": int(candidates["candidate_group"].nunique()),
        "samples_per_group": int(candidates.groupby("candidate_group").size().min()),
        "unique_candidate_images": int(candidates["image_id"].nunique()),
        "candidate_split_counts": candidates["split"].value_counts().to_dict(),
        "group_split_counts": group_counts.to_dict(orient="index"),
        "random_seed": int(seed),
        "best_silhouette_k": int(best_row["k"]),
        "best_silhouette": float(best_row["silhouette"]),
        "best_k_bootstrap_ari_mean": float(best_row["bootstrap_ari_mean"]),
        "robust_pca": pca_meta,
        "important_limitations": [
            "Candidate groups are feature-coverage strata, not grade labels.",
            "Atypicality indicates difference from the dataset center, not poor quality.",
            "Pixel size comparisons are valid only under the existing acquisition scale.",
            "Thresholds used for candidate pools are sampling rules, not final grading thresholds.",
        ],
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    features_path = args.features_csv if args.features_csv.is_absolute() else project_root / args.features_csv
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    frame = pd.read_csv(features_path)
    missing = [name for name in FEATURES if name not in frame.columns]
    if missing:
        raise KeyError(f"Missing features: {missing}")
    if frame[FEATURES].isna().any().any():
        raise ValueError("Core feature table contains missing values.")
    if not frame["instance_id"].is_unique:
        raise ValueError("instance_id is not unique.")

    scored, robust_params = add_robust_scores(frame)
    candidates, pool_summary = select_candidates(scored, args.samples_per_group, args.seed)
    diagnostics, pca_scores, pca_meta = cluster_structure_audit(scored, args.seed)
    summary = write_outputs(
        project_root,
        output_dir,
        scored,
        robust_params,
        candidates,
        pool_summary,
        diagnostics,
        pca_scores,
        pca_meta,
        args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
