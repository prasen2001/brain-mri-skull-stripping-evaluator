"""Aggregate charts and orthogonal review figures for 3D evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.preprocessing import robust_normalize
from src.volume_dataset import VolumePair, load_volume_pair
from src.volume_segmentation import VolumeMorphologyConfig, segment_volume


def _save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_volume_method_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Compare mean region metrics for complete-volume predictions."""
    columns = ["dice", "iou", "precision", "sensitivity", "specificity"]
    labels = {
        "dice": "Dice",
        "iou": "IoU",
        "precision": "Precision",
        "sensitivity": "Sensitivity",
        "specificity": "Specificity",
    }
    long = metrics.melt(
        id_vars="method", value_vars=columns, var_name="metric", value_name="score"
    )
    long["metric"] = long["metric"].map(labels)
    figure, axis = plt.subplots(figsize=(11, 6))
    sns.barplot(data=long, x="metric", y="score", hue="method", errorbar="sd", ax=axis)
    axis.set(
        title="3D classical skull-stripping comparison",
        xlabel="Metric",
        ylabel="Score",
        ylim=(0, 1.08),
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    _save(figure, output_dir / "method_comparison_3d.png")


def plot_volume_dice_distribution(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot per-volume Dice distributions for all methods."""
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.violinplot(data=metrics, x="method", y="dice", inner="box", cut=0, ax=axis)
    axis.set(
        title="Per-volume Dice distribution",
        xlabel="Method",
        ylabel="Dice",
        ylim=(0, 1.02),
    )
    axis.grid(axis="y", alpha=0.25)
    _save(figure, output_dir / "dice_distribution_3d.png")


def _slice(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Return a consistently displayed orthogonal slice."""
    return np.rot90(np.take(array, index, axis=axis))


def _error_map(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    categories = np.zeros(prediction.shape, dtype=np.uint8)
    categories[prediction & reference] = 1
    categories[prediction & ~reference] = 2
    categories[~prediction & reference] = 3
    return categories


def create_orthogonal_review(
    pair: VolumePair,
    dice: float,
    output_path: Path,
    config: VolumeMorphologyConfig,
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    """Render center sagittal, coronal, and axial views with labeled errors."""
    image, reference, _ = load_volume_pair(pair)
    normalized = robust_normalize(image, lower_percentile, upper_percentile)
    prediction, _ = segment_volume(normalized, "central_component", config)
    errors = _error_map(prediction, reference)
    orientations = ((0, "Sagittal"), (1, "Coronal"), (2, "Axial"))
    figure, axes = plt.subplots(3, 3, figsize=(13, 13))
    error_cmap = ListedColormap(["#252525", "#2ca25f", "#f28e2b", "#6a51a3"])
    for row, (axis_index, label) in enumerate(orientations):
        index = image.shape[axis_index] // 2
        image_slice = _slice(normalized, axis_index, index)
        reference_slice = _slice(reference, axis_index, index)
        prediction_slice = _slice(prediction, axis_index, index)
        axes[row, 0].imshow(image_slice, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"{label}: normalized MRI")
        axes[row, 1].imshow(image_slice, cmap="gray", vmin=0, vmax=1)
        if reference_slice.any():
            axes[row, 1].contour(
                reference_slice, levels=[0.5], colors=["cyan"], linewidths=1
            )
        if prediction_slice.any():
            axes[row, 1].contour(
                prediction_slice, levels=[0.5], colors=["magenta"], linewidths=1
            )
        axes[row, 1].set_title("Overlay: reference=cyan, prediction=magenta")
        axes[row, 2].imshow(
            _slice(errors, axis_index, index), cmap=error_cmap, vmin=0, vmax=3
        )
        axes[row, 2].set_title("Error map")
        for panel in axes[row]:
            panel.axis("off")
    axes[0, 2].legend(
        handles=[
            Patch(color="#2ca25f", label="TP: correct brain"),
            Patch(color="#f28e2b", label="FP: outside reference"),
            Patch(color="#6a51a3", label="FN: missed brain"),
        ],
        fontsize=8,
        loc="lower center",
    )
    figure.suptitle(
        f"3D central-component review: {pair.identifier} | Dice={dice:.3f}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    _save(figure, output_path)


def create_volume_visualizations(
    metrics: pd.DataFrame,
    pairs: dict[str, VolumePair],
    output_dir: Path,
    config: VolumeMorphologyConfig,
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    """Create aggregate plots and three best/worst orthogonal reviews."""
    sns.set_theme(style="whitegrid", context="notebook")
    plot_volume_method_comparison(metrics, output_dir)
    plot_volume_dice_distribution(metrics, output_dir)
    central = metrics[metrics["method"] == "central_component"].sort_values(
        ["dice", "identifier"], ascending=[False, True]
    )
    review_dir = output_dir / "orthogonal_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    selections = (
        ("best", central.head(3)),
        ("worst", central.tail(3).sort_values(["dice", "identifier"])),
    )
    for group, selection in selections:
        for rank, (_, row) in enumerate(selection.iterrows(), start=1):
            identifier = str(row["identifier"])
            create_orthogonal_review(
                pairs[identifier],
                float(row["dice"]),
                review_dir / f"{group}_{rank}_{identifier}.png",
                config,
                lower_percentile,
                upper_percentile,
            )
