from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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

UNITS = {
    "major_axis": "px",
    "minor_axis": "px",
    "area": "px^2",
    "aspect_ratio": "ratio",
    "L_mean": "Lab L*",
    "L_std": "Lab L*",
    "a_mean": "Lab a*",
    "a_std": "Lab a*",
    "b_mean": "Lab b*",
    "b_std": "Lab b*",
    "S_mean": "HSV S [0,1]",
    "V_mean": "HSV V [0,1]",
    "circularity": "ratio",
    "solidity": "ratio",
    "eccentricity": "ratio",
    "dark_ratio": "ratio",
    "abnormal_color_ratio": "ratio",
    "max_damage_ratio": "ratio",
}

PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "black": "#000000",
    "grid": "#D1D5DB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create descriptive statistics and figures for the frozen feature dataset.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--input-features-csv",
        type=Path,
        default=None,
        help="Optional pre-filtered feature CSV. When omitted, rebuild the frozen train+val QC dataset.",
    )
    parser.add_argument("--correlation-threshold", type=float, default=0.90)
    return parser.parse_args()


def read_prefiltered_dataset(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data = pd.read_csv(path, encoding="utf-8-sig")
    data = data.sort_values(["split", "image_id", "instance_id"]).reset_index(drop=True)
    audit_columns = [
        column
        for column in (
            "image_id",
            "instance_id",
            "split",
            "rot_reviewed",
            "manual_rot_raw",
            "manual_rot_label",
            "manual_rot_exclude",
            "grade_analysis_eligible",
            "grade_analysis_exclusion_reason",
        )
        if column in data.columns
    ]
    inclusion_log = data[audit_columns].copy()
    inclusion_log["include_in_analysis"] = True
    per_split = {
        str(split): {"included_rows": int(count)}
        for split, count in data["split"].value_counts().sort_index().items()
    }
    summary: dict[str, object] = {
        "input_rows": int(len(data)),
        "included_rows": int(len(data)),
        "excluded_rows": 0,
        "per_split": per_split,
        "filter_policy": [
            "input already passed the frozen QC inclusion policy",
            "strict manual rot and uncertain labels were excluded upstream",
            "instances outside the high-recall rot review queue were retained",
            "no feature-magnitude threshold was used to create grade labels",
        ],
    }
    return data, inclusion_log, summary


def read_extra_prediction_ids(path: Path) -> set[str]:
    audit = pd.read_csv(path)
    result: set[str] = set()
    for value in audit["unmatched_prediction_ids"].dropna().astype(str):
        result.update(item for item in value.split(";") if item)
    return result


def build_dataset(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    frames: list[pd.DataFrame] = []
    extra_ids: set[str] = set()
    per_split: dict[str, dict[str, int]] = {}
    for split in ("train", "val"):
        split_root = project_root / "outputs" / "features_frozen_v1" / split
        frame = pd.read_csv(split_root / f"features_{split}.csv", encoding="utf-8-sig")
        frame["split"] = split
        frames.append(frame)
        current_extra = read_extra_prediction_ids(split_root / f"{split}_spatial_mismatch_audit.csv")
        extra_ids.update(current_extra)
        per_split[split] = {
            "input_rows": int(len(frame)),
            "edge_instances": int(frame["touches_image_edge"].fillna(False).astype(bool).sum()),
            "spatial_extra_predictions": int(len(current_extra)),
        }

    all_features = pd.concat(frames, ignore_index=True)
    log = all_features[["image_id", "instance_id", "split", "processing_success", "touches_image_edge", "confidence"]].copy()
    log["is_spatial_extra_prediction"] = log["instance_id"].isin(extra_ids)
    log["human_qc_status"] = "batch_sample_review_accepted_by_user_2026-08-29"

    def exclusion_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if not bool(row["processing_success"]):
            reasons.append("processing_failed")
        if bool(row["touches_image_edge"]):
            reasons.append("touches_image_edge")
        if bool(row["is_spatial_extra_prediction"]):
            reasons.append("spatial_extra_prediction")
        return ";".join(reasons)

    log["exclusion_reason"] = log.apply(exclusion_reason, axis=1)
    log["include_in_analysis"] = log["exclusion_reason"].eq("")
    included_ids = set(log.loc[log["include_in_analysis"], "instance_id"])
    included = all_features.loc[all_features["instance_id"].isin(included_ids)].copy()
    included = included.sort_values(["split", "image_id", "instance_id"]).reset_index(drop=True)

    for split in ("train", "val"):
        per_split[split]["excluded_unique"] = int(
            ((log["split"] == split) & ~log["include_in_analysis"]).sum()
        )
        per_split[split]["included_rows"] = int(
            ((log["split"] == split) & log["include_in_analysis"]).sum()
        )

    summary: dict[str, object] = {
        "input_rows": int(len(all_features)),
        "included_rows": int(len(included)),
        "excluded_rows": int((~log["include_in_analysis"]).sum()),
        "per_split": per_split,
        "filter_policy": [
            "processing_success must be true",
            "exclude instances touching the image boundary",
            "exclude spatially unmatched extra predictions on count-mismatch images",
            "retain low-confidence instances unless another exclusion reason applies",
            "do not filter by feature magnitude, color, or damage-candidate ratios",
        ],
    }
    return included, log, summary


def descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce")
        q1 = float(values.quantile(0.25))
        q3 = float(values.quantile(0.75))
        rows.append(
            {
                "feature": feature,
                "unit": UNITS[feature],
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "min": float(values.min()),
                "P25": q1,
                "median": float(values.median()),
                "P75": q3,
                "max": float(values.max()),
                "IQR": q3 - q1,
                "zero_count": int((values == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": PALETTE["black"],
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_feature_figures(data: pd.DataFrame, hist_dir: Path, box_dir: Path) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    hist_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)
    for feature in FEATURES:
        values = pd.to_numeric(data[feature], errors="coerce").dropna().to_numpy(dtype=float)
        bin_edges = np.histogram_bin_edges(values, bins="fd")
        if len(bin_edges) < 3:
            bin_edges = np.linspace(float(values.min()), float(values.max()) + 1e-12, 11)

        fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
        ax.hist(values, bins=bin_edges, color=PALETTE["blue"], edgecolor=PALETTE["black"], linewidth=0.5)
        ax.axvline(np.median(values), color=PALETTE["orange"], linestyle="--", linewidth=1.5, label="Median")
        ax.set(title=f"Distribution of {feature}", xlabel=f"{feature} ({UNITS[feature]})", ylabel="Instance count")
        ax.legend(frameon=False)
        hist_path = hist_dir / f"{feature}_histogram.png"
        save_figure(fig, hist_path)

        fig, ax = plt.subplots(figsize=(7.2, 2.9), layout="constrained")
        ax.boxplot(
            values,
            orientation="horizontal",
            patch_artist=True,
            boxprops={"facecolor": PALETTE["blue"], "edgecolor": PALETTE["black"], "alpha": 0.75},
            medianprops={"color": PALETTE["orange"], "linewidth": 1.8},
            whiskerprops={"color": PALETTE["black"]},
            capprops={"color": PALETTE["black"]},
            flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": PALETTE["purple"], "markeredgecolor": "none", "alpha": 0.45},
        )
        ax.set(title=f"Boxplot of {feature}", xlabel=f"{feature} ({UNITS[feature]})", yticks=[])
        box_path = box_dir / f"{feature}_boxplot.png"
        save_figure(fig, box_path)

        manifest.extend(
            [
                {"file": hist_path.as_posix(), "type": "histogram", "feature": feature, "bin_method": "Freedman-Diaconis"},
                {"file": box_path.as_posix(), "type": "boxplot", "feature": feature, "whisker": "1.5 IQR"},
            ]
        )
    return manifest


def make_correlation_outputs(data: pd.DataFrame, output_dir: Path, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    matrix = data[FEATURES].corr(method="pearson")
    pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(FEATURES):
        for right in FEATURES[left_index + 1 :]:
            value = float(matrix.loc[left, right])
            if abs(value) >= threshold:
                pairs.append({"feature_1": left, "feature_2": right, "pearson_r": value, "abs_pearson_r": abs(value)})
    redundant = pd.DataFrame(pairs).sort_values("abs_pearson_r", ascending=False) if pairs else pd.DataFrame(
        columns=["feature_1", "feature_2", "pearson_r", "abs_pearson_r"]
    )

    fig, ax = plt.subplots(figsize=(12.5, 10.5), layout="constrained")
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(FEATURES)), FEATURES, rotation=55, ha="right")
    ax.set_yticks(range(len(FEATURES)), FEATURES)
    ax.set_title("Pearson correlation matrix")
    for row in range(len(FEATURES)):
        for col in range(len(FEATURES)):
            value = matrix.iloc[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(value) >= 0.55 else "black")
    colorbar = fig.colorbar(image, ax=ax, shrink=0.82)
    colorbar.set_label("Pearson r")
    correlation_path = output_dir / "correlation_matrix.png"
    save_figure(fig, correlation_path)
    return matrix, redundant, correlation_path


def make_pca_outputs(data: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], Path]:
    feature_values = data[FEATURES].astype(float)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(feature_values)
    pca = PCA(n_components=2, svd_solver="full")
    scores = pca.fit_transform(standardized)

    score_frame = data[["image_id", "instance_id", "split"]].copy()
    score_frame["PC1"] = scores[:, 0]
    score_frame["PC2"] = scores[:, 1]
    loading_frame = pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1_loading", "PC2_loading"]).reset_index(names="feature")
    metadata = {
        "features": FEATURES,
        "scaling": "StandardScaler fitted to all included train+val instances; test split excluded",
        "pca": "PCA(n_components=2, svd_solver='full')",
        "explained_variance_ratio": [float(value) for value in pca.explained_variance_ratio_],
        "cumulative_explained_variance": float(pca.explained_variance_ratio_.sum()),
        "scaler_mean": {feature: float(value) for feature, value in zip(FEATURES, scaler.mean_, strict=True)},
        "scaler_scale": {feature: float(value) for feature, value in zip(FEATURES, scaler.scale_, strict=True)},
    }

    fig, ax = plt.subplots(figsize=(8.2, 6.4), layout="constrained")
    styles = {
        "train": {"color": PALETTE["blue"], "marker": "o"},
        "val": {"color": PALETTE["orange"], "marker": "^"},
    }
    for split, style in styles.items():
        subset = score_frame["split"] == split
        ax.scatter(
            score_frame.loc[subset, "PC1"],
            score_frame.loc[subset, "PC2"],
            s=15,
            alpha=0.42,
            c=style["color"],
            marker=style["marker"],
            edgecolors="none",
            label=f"{split} (n={int(subset.sum())})",
        )
    ax.axhline(0, color=PALETTE["black"], linewidth=0.7, alpha=0.55)
    ax.axvline(0, color=PALETTE["black"], linewidth=0.7, alpha=0.55)
    ax.set(
        title="PCA of standardized interpretable features",
        xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)",
        ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)",
    )
    ax.legend(frameon=False)
    pca_path = output_dir / "pca_visualization.png"
    save_figure(fig, pca_path)
    return score_frame, loading_frame, metadata, pca_path


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    if args.input_features_csv is None:
        data, inclusion_log, run_summary = build_dataset(project_root)
        source_files = [
            "outputs/features_frozen_v1/train/features_train.csv",
            "outputs/features_frozen_v1/val/features_val.csv",
            "outputs/features_frozen_v1/train/train_spatial_mismatch_audit.csv",
            "outputs/features_frozen_v1/val/val_spatial_mismatch_audit.csv",
        ]
    else:
        input_path = args.input_features_csv
        if not input_path.is_absolute():
            input_path = project_root / input_path
        input_path = input_path.resolve()
        data, inclusion_log, run_summary = read_prefiltered_dataset(input_path)
        source_files = [str(input_path)]
    if data[FEATURES].isna().any().any():
        raise ValueError("Included data contain missing feature values.")
    if data["instance_id"].duplicated().any():
        raise ValueError("Included data contain duplicate instance_id values.")

    statistics = descriptive_statistics(data)
    data.to_csv(output_dir / "features.csv", index=False, encoding="utf-8-sig")
    inclusion_log.to_csv(output_dir / "qc_inclusion_log.csv", index=False, encoding="utf-8-sig")
    statistics.to_csv(output_dir / "feature_statistics.csv", index=False, encoding="utf-8-sig")

    figure_manifest = make_feature_figures(data, output_dir / "histograms", output_dir / "boxplots")
    correlation, redundant, correlation_path = make_correlation_outputs(data, output_dir, args.correlation_threshold)
    correlation.to_csv(output_dir / "correlation_matrix.csv", encoding="utf-8-sig")
    redundant.to_csv(output_dir / "high_correlation_pairs.csv", index=False, encoding="utf-8-sig")
    pca_scores, pca_loadings, pca_metadata, pca_path = make_pca_outputs(data, output_dir)
    pca_scores.to_csv(output_dir / "pca_scores.csv", index=False, encoding="utf-8-sig")
    pca_loadings.to_csv(output_dir / "pca_loadings.csv", index=False, encoding="utf-8-sig")
    (output_dir / "pca_metadata.json").write_text(json.dumps(pca_metadata, indent=2), encoding="utf-8")

    figure_manifest.extend(
        [
            {"file": correlation_path.as_posix(), "type": "correlation_heatmap", "scale": "Pearson r [-1,1]", "colormap": "RdBu_r"},
            {"file": pca_path.as_posix(), "type": "PCA scatter", "encoding": "split by color and marker"},
        ]
    )
    provenance = {
        "created_by": Path(__file__).name,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "matplotlib": mpl.__version__,
        "scikit_learn": sklearn.__version__,
        "random_seed": None,
        "source_files": source_files,
        "transformations": run_summary["filter_policy"],
        "figure_export": {"format": "PNG", "dpi": 300, "background": "white", "manifest": figure_manifest},
    }
    (output_dir / "figure_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(
            {
                **run_summary,
                "feature_count": len(FEATURES),
                "high_correlation_threshold": float(args.correlation_threshold),
                "high_correlation_pair_count": int(len(redundant)),
                "pca_explained_variance_ratio": pca_metadata["explained_variance_ratio"],
                "pca_cumulative_explained_variance": pca_metadata["cumulative_explained_variance"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads((output_dir / "analysis_summary.json").read_text(encoding="utf-8")), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
