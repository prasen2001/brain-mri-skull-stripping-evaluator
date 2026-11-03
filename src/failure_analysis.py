"""Retrospective, deterministic categorization of segmentation outcomes.

Unlike prediction-only quality warnings, this module uses reference-derived
metrics. Its categories explain observed errors after evaluation and must not be
presented as warnings that would be available for an unseen clinical image.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FailureAnalysisConfig:
    """Thresholds for descriptive, reference-based outcome categories."""

    acceptable_dice: float = 0.80
    severe_dice: float = 0.40
    critical_dice: float = 0.20
    precision_recall_margin: float = 0.15
    wrong_component_dice_drop: float = 0.10
    polarity_min_complement_dice: float = 0.60
    polarity_min_dice_gain: float = 0.25
    boundary_surface_dice: float = 0.20


DEFAULT_FAILURE_CONFIG = FailureAnalysisConfig()

PRIMARY_CATEGORY_ORDER = (
    "empty_prediction",
    "wrong_component_selected",
    "likely_polarity_failure",
    "fragmented_prediction",
    "border_leakage",
    "under_segmentation",
    "over_segmentation",
    "boundary_mismatch",
    "mixed_overlap_error",
    "acceptable",
)


def _complement_dice(row: pd.Series) -> float:
    """Calculate Dice that the exact inverse prediction would obtain."""
    numerator = 2.0 * float(row["fn"])
    denominator = numerator + float(row["tn"]) + float(row["tp"])
    return numerator / denominator if denominator else 1.0


def categorize_failures(
    metrics: pd.DataFrame,
    config: FailureAnalysisConfig = DEFAULT_FAILURE_CONFIG,
) -> pd.DataFrame:
    """Add severity and deterministic retrospective categories to metrics.

    Multiple reasons may apply. ``primary_failure_category`` uses a documented
    priority order so summaries remain mutually exclusive and reproducible.
    """
    required = {
        "identifier",
        "method",
        "tp",
        "tn",
        "fp",
        "fn",
        "dice",
        "precision",
        "sensitivity",
        "surface_dice",
        "quality_warnings",
    }
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Failure analysis is missing columns: {sorted(missing)}")

    result = metrics.copy()
    dice_by_method = result.pivot_table(
        index="identifier", columns="method", values="dice", aggfunc="first"
    )
    categories: list[str] = []
    all_reasons: list[str] = []
    severities: list[str] = []
    complement_dice_values: list[float] = []

    for _, row in result.iterrows():
        dice = float(row["dice"])
        precision = float(row["precision"])
        sensitivity = float(row["sensitivity"])
        warnings = set(str(row["quality_warnings"]).split(";"))
        complement_dice = _complement_dice(row)
        reasons: set[str] = set()

        if "empty_mask" in warnings:
            reasons.add("empty_prediction")
        if "fragmented_mask" in warnings or "many_components" in warnings:
            reasons.add("fragmented_prediction")
        if "excessive_border_contact" in warnings:
            reasons.add("border_leakage")

        identifier = str(row["identifier"])
        method = str(row["method"])
        if method == "central_component" and "otsu_morphology" in dice_by_method.columns:
            morphology_dice = dice_by_method.at[identifier, "otsu_morphology"]
            if (
                pd.notna(morphology_dice)
                and float(morphology_dice) - dice >= config.wrong_component_dice_drop
                and sensitivity < precision
            ):
                reasons.add("wrong_component_selected")

        if (
            method == "otsu_threshold"
            and complement_dice >= config.polarity_min_complement_dice
            and complement_dice - dice >= config.polarity_min_dice_gain
        ):
            reasons.add("likely_polarity_failure")

        if precision - sensitivity >= config.precision_recall_margin:
            reasons.add("under_segmentation")
        elif sensitivity - precision >= config.precision_recall_margin:
            reasons.add("over_segmentation")
        if (
            float(row["surface_dice"]) < config.boundary_surface_dice
            and config.severe_dice <= dice < config.acceptable_dice
        ):
            reasons.add("boundary_mismatch")

        if dice >= config.acceptable_dice and not reasons:
            reasons.add("acceptable")
        elif dice < config.acceptable_dice and not reasons:
            reasons.add("mixed_overlap_error")

        primary = next(category for category in PRIMARY_CATEGORY_ORDER if category in reasons)
        if dice < config.critical_dice:
            severity = "critical"
        elif dice < config.severe_dice:
            severity = "severe"
        elif dice < config.acceptable_dice:
            severity = "moderate"
        else:
            severity = "acceptable"

        categories.append(primary)
        all_reasons.append(";".join(sorted(reasons)))
        severities.append(severity)
        complement_dice_values.append(complement_dice)

    result["outcome_severity"] = severities
    result["primary_failure_category"] = categories
    result["failure_reasons"] = all_reasons
    result["complement_prediction_dice"] = complement_dice_values
    return result


def summarize_failure_categories(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize mutually exclusive primary categories for each method."""
    required = {"method", "primary_failure_category", "dice", "precision", "sensitivity"}
    missing = required - set(metrics.columns)
    if missing:
        raise ValueError(f"Failure summary is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    method_totals = metrics.groupby("method").size().to_dict()
    grouped = metrics.groupby(["method", "primary_failure_category"], sort=False)
    for (method, category), group in grouped:
        count = int(len(group))
        rows.append(
            {
                "method": method,
                "primary_failure_category": category,
                "n_images": count,
                "fraction_of_method": count / int(method_totals[method]),
                "dice_mean": float(group["dice"].mean()),
                "dice_median": float(group["dice"].median()),
                "precision_mean": float(group["precision"].mean()),
                "sensitivity_mean": float(group["sensitivity"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["method", "fraction_of_method", "primary_failure_category"],
        ascending=[True, False, True],
        ignore_index=True,
    )
