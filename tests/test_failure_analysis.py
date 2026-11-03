from __future__ import annotations

import pandas as pd
import pytest

from src.failure_analysis import categorize_failures, summarize_failure_categories


def _row(
    identifier: str,
    method: str,
    *,
    dice: float,
    precision: float,
    sensitivity: float,
    tp: int = 40,
    tn: int = 40,
    fp: int = 10,
    fn: int = 10,
    surface_dice: float = 0.5,
    warnings: str = "none",
) -> dict[str, object]:
    return {
        "identifier": identifier,
        "method": method,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "dice": dice,
        "precision": precision,
        "sensitivity": sensitivity,
        "surface_dice": surface_dice,
        "quality_warnings": warnings,
    }


def test_categories_distinguish_under_and_over_segmentation() -> None:
    metrics = pd.DataFrame(
        [
            _row("under", "otsu_morphology", dice=0.70, precision=0.95, sensitivity=0.60),
            _row("over", "otsu_morphology", dice=0.70, precision=0.55, sensitivity=0.90),
            _row("good", "otsu_morphology", dice=0.90, precision=0.90, sensitivity=0.90),
        ]
    )
    categorized = categorize_failures(metrics).set_index("identifier")
    assert categorized.at["under", "primary_failure_category"] == "under_segmentation"
    assert categorized.at["over", "primary_failure_category"] == "over_segmentation"
    assert categorized.at["good", "primary_failure_category"] == "acceptable"
    assert categorized.at["under", "outcome_severity"] == "moderate"


def test_wrong_component_uses_paired_morphology_result() -> None:
    metrics = pd.DataFrame(
        [
            _row("scan", "otsu_morphology", dice=0.85, precision=0.85, sensitivity=0.85),
            _row("scan", "central_component", dice=0.50, precision=0.90, sensitivity=0.35),
        ]
    )
    categorized = categorize_failures(metrics)
    central = categorized[categorized.method.eq("central_component")].iloc[0]
    assert central["primary_failure_category"] == "wrong_component_selected"
    assert "under_segmentation" in central["failure_reasons"]


def test_likely_polarity_failure_compares_with_inverse_prediction() -> None:
    metrics = pd.DataFrame(
        [
            _row(
                "inverse",
                "otsu_threshold",
                dice=0.0,
                precision=0.0,
                sensitivity=0.0,
                tp=0,
                tn=0,
                fp=50,
                fn=50,
                surface_dice=0.0,
            )
        ]
    )
    categorized = categorize_failures(metrics).iloc[0]
    assert categorized["complement_prediction_dice"] == 1.0
    assert categorized["primary_failure_category"] == "likely_polarity_failure"
    assert categorized["outcome_severity"] == "critical"


def test_prediction_only_geometry_warnings_map_to_retrospective_reasons() -> None:
    metrics = pd.DataFrame(
        [
            _row(
                "fragmented",
                "otsu_threshold",
                dice=0.5,
                precision=0.6,
                sensitivity=0.6,
                warnings="fragmented_mask;many_components",
            ),
            _row(
                "border",
                "otsu_threshold",
                dice=0.5,
                precision=0.6,
                sensitivity=0.6,
                warnings="excessive_border_contact",
            ),
        ]
    )
    categorized = categorize_failures(metrics).set_index("identifier")
    assert categorized.at["fragmented", "primary_failure_category"] == "fragmented_prediction"
    assert categorized.at["border", "primary_failure_category"] == "border_leakage"


def test_failure_summary_uses_mutually_exclusive_primary_category() -> None:
    metrics = pd.DataFrame(
        [
            _row("one", "otsu_threshold", dice=0.9, precision=0.9, sensitivity=0.9),
            _row("two", "otsu_threshold", dice=0.7, precision=0.95, sensitivity=0.6),
        ]
    )
    summary = summarize_failure_categories(categorize_failures(metrics))
    assert summary["n_images"].sum() == 2
    assert summary["fraction_of_method"].sum() == pytest.approx(1.0)


def test_missing_metric_columns_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        categorize_failures(pd.DataFrame({"identifier": ["scan"]}))
