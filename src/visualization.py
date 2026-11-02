"""Publication-quality plots and interpretable case visualizations."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from src.dataset import ImageMaskPair, load_pair
from src.preprocessing import robust_normalize
from src.segmentation import MorphologyConfig, segment

METRIC_LABELS = {
    "dice": "Dice",
    "iou": "IoU",
    "precision": "Precision",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
}


def _save_figure(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_method_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot mean headline metrics for all methods."""
    long = metrics.melt(
        id_vars=["method"],
        value_vars=list(METRIC_LABELS),
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].map(METRIC_LABELS)
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=long, x="metric", y="score", hue="method", errorbar="sd", ax=axis)
    axis.set(title="Classical skull-stripping method comparison", xlabel="Metric", ylabel="Score")
    # Leave headroom for standard-deviation bars even though scores themselves
    # are bounded by 1.0.
    axis.set_ylim(0, 1.08)
    axis.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "method_comparison.png")


def plot_metric_distribution(metrics: pd.DataFrame, metric: str, output_dir: Path) -> None:
    """Plot a bounded per-image metric distribution."""
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=metrics, x="method", y=metric, inner="box", cut=0, ax=axis)
    axis.set(
        title=f"Per-image {METRIC_LABELS[metric]} distribution",
        xlabel="Method",
        ylabel=METRIC_LABELS[metric],
    )
    axis.set_ylim(0, 1.02)
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / f"{metric}_distribution.png")


def plot_processing_time(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Compare mean processing time with standard-deviation error bars."""
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=metrics,
        x="method",
        y="processing_time_seconds",
        errorbar="sd",
        ax=axis,
    )
    axis.set(
        title="Processing time per image (mean ± standard deviation)",
        xlabel="Method",
        ylabel="Seconds",
    )
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "processing_time_comparison.png")


def plot_source_dice_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Compare Dice across the official dataset source families."""
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(
        data=metrics,
        x="dataset_source",
        y="dice",
        hue="method",
        errorbar="sd",
        ax=axis,
    )
    axis.set(
        title="Dice score by source dataset",
        xlabel="Source family encoded in scan identifier",
        ylabel="Dice",
    )
    axis.set_ylim(0, 1.08)
    axis.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "source_dice_comparison.png")


def plot_quality_warning_rates(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Show the fraction of predictions flagged by reference-free checks."""
    rates = (
        metrics.groupby("method", sort=False)["has_quality_warning"]
        .mean()
        .mul(100)
        .reset_index(name="warning_rate_percent")
    )
    figure, axis = plt.subplots(figsize=(9, 6))
    sns.barplot(data=rates, x="method", y="warning_rate_percent", ax=axis)
    for container in axis.containers:
        axis.bar_label(container, fmt="%.1f%%", padding=3)
    axis.set(
        title="Predictions flagged for manual review",
        xlabel="Method",
        ylabel="Flagged predictions (%)",
    )
    axis.set_ylim(0, max(100, float(rates["warning_rate_percent"].max()) * 1.15))
    axis.tick_params(axis="x", rotation=12)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "quality_warning_rates.png")


def plot_boundary_metric_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Compare boundary agreement and pixel-distance errors."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    specifications = (
        ("surface_dice", "Surface Dice (1-pixel tolerance)", "Higher is better"),
        ("hd95_mm", "HD95 (mm)", "Lower is better"),
        ("assd_mm", "ASSD (mm)", "Lower is better"),
    )
    for axis, (column, title, direction) in zip(axes, specifications, strict=True):
        sns.barplot(data=metrics, x="method", y=column, errorbar="sd", ax=axis)
        axis.set(title=f"{title}\n{direction}", xlabel="Method", ylabel=title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
        if column == "surface_dice":
            axis.set_ylim(0, 1.08)
    figure.suptitle("Boundary-quality comparison", fontsize=15)
    figure.tight_layout()
    _save_figure(figure, output_dir / "boundary_metric_comparison.png")


def _error_categories(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    categories = np.zeros(prediction.shape, dtype=np.uint8)
    categories[prediction & reference] = 1
    categories[prediction & ~reference] = 2
    categories[~prediction & reference] = 3
    return categories


def _draw_case_row(
    axes: np.ndarray,
    image: np.ndarray,
    reference: np.ndarray,
    prediction: np.ndarray,
    title: str,
) -> None:
    panels = axes.ravel()
    panels[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    panels[0].set_title(f"{title}\nNormalized MRI", fontsize=9)
    panels[1].imshow(reference, cmap="gray", vmin=0, vmax=1)
    panels[1].set_title("Ground-truth mask", fontsize=9)
    panels[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
    panels[2].set_title("Predicted mask", fontsize=9)

    panels[3].imshow(image, cmap="gray", vmin=0, vmax=1)
    if reference.any():
        panels[3].contour(reference, levels=[0.5], colors=["cyan"], linewidths=1.0)
    if prediction.any():
        panels[3].contour(prediction, levels=[0.5], colors=["magenta"], linewidths=1.0)
    panels[3].set_title("Overlay: GT=cyan, prediction=magenta", fontsize=9)

    error_map = _error_categories(prediction, reference)
    panels[4].imshow(error_map, cmap=ListedColormap(["#252525", "#2ca25f", "#f28e2b", "#6a51a3"]), vmin=0, vmax=3)
    panels[4].set_title("Error map", fontsize=9)
    legend = [
        Patch(color="#2ca25f", label="TP: correct brain"),
        Patch(color="#f28e2b", label="FP: outside reference"),
        Patch(color="#6a51a3", label="FN: missed brain"),
    ]
    panels[4].legend(handles=legend, fontsize=6, loc="lower center", framealpha=0.85)
    for axis in panels:
        axis.axis("off")


def _render_case(
    pair: ImageMaskPair,
    method: str,
    dice: float,
    iou: float,
    config: MorphologyConfig,
    normalize: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    image, reference = load_pair(pair)
    normalized = normalize(image)
    prediction, _ = segment(normalized, method, config)
    title = f"{pair.identifier} | {method} | Dice={dice:.3f}, IoU={iou:.3f}"
    return normalized, reference, prediction, title


def create_case_visualizations(
    metrics: pd.DataFrame,
    pairs: dict[str, ImageMaskPair],
    output_dir: Path,
    config: MorphologyConfig,
    lower_percentile: float,
    upper_percentile: float,
) -> str:
    """Create three best, three worst, and six individual five-panel figures."""
    method_means = metrics.groupby("method", sort=True)["dice"].mean()
    best_method = str(method_means.idxmax())
    ranked = metrics[metrics["method"] == best_method].sort_values(
        ["dice", "identifier"], ascending=[False, True]
    )
    best = ranked.head(3)
    worst = ranked.tail(3).sort_values(["dice", "identifier"], ascending=[True, True])
    case_dir = output_dir / "overlays"
    case_dir.mkdir(parents=True, exist_ok=True)

    normalize = lambda image: robust_normalize(  # noqa: E731 - compact configured callable
        image, lower_percentile, upper_percentile
    )
    for group_name, selection in (("best", best), ("worst", worst)):
        figure, axes = plt.subplots(len(selection), 5, figsize=(18, 3.8 * len(selection)), squeeze=False)
        for row_index, (_, record) in enumerate(selection.iterrows()):
            pair = pairs[str(record["identifier"])]
            rendered = _render_case(
                pair,
                best_method,
                float(record["dice"]),
                float(record["iou"]),
                config,
                normalize,
            )
            _draw_case_row(axes[row_index], *rendered)

            individual, individual_axes = plt.subplots(1, 5, figsize=(18, 4))
            _draw_case_row(individual_axes, *rendered)
            individual.suptitle(
                f"{group_name.title()} case ranked by Dice ({best_method})", fontsize=13
            )
            _save_figure(
                individual,
                case_dir / f"{group_name}_{row_index + 1}_{pair.identifier}.png",
            )
        figure.suptitle(
            f"{group_name.title()} cases ranked by Dice for {best_method}", fontsize=15
        )
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        _save_figure(figure, output_dir / f"{group_name}_cases.png")
    return best_method


def create_all_visualizations(
    metrics: pd.DataFrame,
    pairs: dict[str, ImageMaskPair],
    output_dir: Path,
    config: MorphologyConfig,
    lower_percentile: float,
    upper_percentile: float,
) -> str:
    """Create every required chart and return the best mean-Dice method."""
    sns.set_theme(style="whitegrid", context="notebook")
    plot_method_comparison(metrics, output_dir)
    plot_metric_distribution(metrics, "dice", output_dir)
    plot_metric_distribution(metrics, "iou", output_dir)
    plot_processing_time(metrics, output_dir)
    plot_source_dice_comparison(metrics, output_dir)
    plot_quality_warning_rates(metrics, output_dir)
    plot_boundary_metric_comparison(metrics, output_dir)
    return create_case_visualizations(
        metrics,
        pairs,
        output_dir,
        config,
        lower_percentile,
        upper_percentile,
    )

