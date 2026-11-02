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


def plot_modality_dice_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Compare Dice across modality tokens encoded in official identifiers."""
    order = sorted(metrics["modality"].unique())
    figure, axis = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=metrics,
        x="modality",
        y="dice",
        hue="method",
        order=order,
        errorbar="sd",
        ax=axis,
    )
    axis.set(
        title="Dice score by image modality",
        xlabel="Modality encoded in scan identifier",
        ylabel="Dice",
    )
    axis.set_ylim(0, 1.08)
    axis.legend(title="Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    axis.grid(axis="y", alpha=0.25)
    _save_figure(figure, output_dir / "modality_dice_comparison.png")


def plot_paired_dice_improvements(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Show per-scan Dice changes between progressive baseline stages."""
    pivot = metrics.pivot(index="identifier", columns="method", values="dice")
    changes = pd.DataFrame(
        {
            "Morphology minus Otsu": pivot["otsu_morphology"]
            - pivot["otsu_threshold"],
            "Central component minus morphology": pivot["central_component"]
            - pivot["otsu_morphology"],
        }
    ).melt(var_name="comparison", value_name="dice_change")
    figure, axis = plt.subplots(figsize=(10, 6))
    sns.violinplot(
        data=changes,
        x="comparison",
        y="dice_change",
        inner="box",
        cut=0,
        ax=axis,
    )
    axis.axhline(0, color="black", linestyle="--", linewidth=1, label="No change")
    axis.set(
        title="Paired per-scan Dice improvements",
        xlabel="Progressive method comparison",
        ylabel="Dice change (positive favors later method)",
    )
    axis.tick_params(axis="x", rotation=8)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    _save_figure(figure, output_dir / "paired_dice_improvement.png")


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


def create_failure_review(
    metrics: pd.DataFrame,
    pairs: dict[str, ImageMaskPair],
    output_dir: Path,
    config: MorphologyConfig,
    lower_percentile: float,
    upper_percentile: float,
    maximum_cases: int = 10,
) -> Path:
    """Create a ranked Markdown review with multi-method diagnostic figures."""
    central = metrics[metrics["method"] == "central_component"].copy()
    flagged = central[central["has_quality_warning"].astype(bool)].copy()
    flagged["review_priority"] = (
        flagged["quality_warning_count"] + flagged["method_disagreement"]
    )
    selection = flagged.sort_values(
        ["review_priority", "method_disagreement", "identifier"],
        ascending=[False, False, True],
    ).head(maximum_cases)
    review_dir = output_dir / "failure_review_cases"
    review_dir.mkdir(parents=True, exist_ok=True)
    markdown = [
        "# Ranked failure-review report",
        "",
        "Cases are ranked using prediction-only warning count and method disagreement. "
        "Ground-truth metrics are shown afterward for retrospective audit and do not "
        "influence the warning rules.",
        "",
        "| Rank | Identifier | Warnings | Disagreement | Area | Dice | Surface Dice | HD95 (mm) |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    normalize = lambda image: robust_normalize(  # noqa: E731
        image, lower_percentile, upper_percentile
    )
    for rank, (_, row) in enumerate(selection.iterrows(), start=1):
        identifier = str(row["identifier"])
        pair = pairs[identifier]
        image, reference = load_pair(pair)
        normalized = normalize(image)
        predictions = {
            method: segment(normalized, method, config)[0]
            for method in ("otsu_threshold", "otsu_morphology", "central_component")
        }
        central_prediction = predictions["central_component"]
        figure, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes[0, 0].imshow(normalized, cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("Normalized MRI")
        axes[0, 1].imshow(reference, cmap="gray", vmin=0, vmax=1)
        axes[0, 1].set_title("Ground-truth mask")
        axes[0, 2].imshow(normalized, cmap="gray", vmin=0, vmax=1)
        if reference.any():
            axes[0, 2].contour(reference, levels=[0.5], colors=["cyan"], linewidths=1)
        if central_prediction.any():
            axes[0, 2].contour(
                central_prediction, levels=[0.5], colors=["magenta"], linewidths=1
            )
        axes[0, 2].set_title("Central overlay: GT=cyan, prediction=magenta")
        axes[0, 3].imshow(
            _error_categories(central_prediction, reference),
            cmap=ListedColormap(["#252525", "#2ca25f", "#f28e2b", "#6a51a3"]),
            vmin=0,
            vmax=3,
        )
        axes[0, 3].set_title("Central error map: TP green, FP orange, FN purple")

        for axis, method in zip(
            axes[1, :3],
            ("otsu_threshold", "otsu_morphology", "central_component"),
            strict=True,
        ):
            method_row = metrics[
                (metrics["identifier"] == identifier) & (metrics["method"] == method)
            ].iloc[0]
            axis.imshow(predictions[method], cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"{method}\nDice={method_row['dice']:.3f}")
        axes[1, 3].axis("off")
        axes[1, 3].text(
            0,
            1,
            "\n".join(
                (
                    f"Warnings: {row['quality_warnings']}",
                    f"Disagreement: {row['method_disagreement']:.3f}",
                    f"Predicted area: {row['predicted_area_fraction']:.3f}",
                    f"Components: {int(row['component_count'])}",
                    f"Surface Dice: {row['surface_dice']:.3f}",
                    f"HD95: {row['hd95_mm']:.2f} mm",
                )
            ),
            va="top",
            fontsize=11,
        )
        for axis in axes.ravel()[:7]:
            axis.axis("off")
        figure.suptitle(f"Review rank {rank}: {identifier}", fontsize=15)
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        filename = f"{rank:02d}_{identifier}.png"
        _save_figure(figure, review_dir / filename)
        markdown.append(
            f"| {rank} | `{identifier}` | {row['quality_warnings']} | "
            f"{row['method_disagreement']:.3f} | {row['predicted_area_fraction']:.3f} | "
            f"{row['dice']:.3f} | {row['surface_dice']:.3f} | {row['hd95_mm']:.2f} |"
        )
        markdown.extend(("", f"![Review case {rank}](failure_review_cases/{filename})", ""))

    report_path = output_dir / "failure_review.md"
    report_path.write_text("\n".join(markdown), encoding="utf-8")
    return report_path


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
    plot_modality_dice_comparison(metrics, output_dir)
    plot_paired_dice_improvements(metrics, output_dir)
    plot_quality_warning_rates(metrics, output_dir)
    plot_boundary_metric_comparison(metrics, output_dir)
    best_method = create_case_visualizations(
        metrics,
        pairs,
        output_dir,
        config,
        lower_percentile,
        upper_percentile,
    )
    create_failure_review(
        metrics,
        pairs,
        output_dir,
        config,
        lower_percentile,
        upper_percentile,
    )
    return best_method

