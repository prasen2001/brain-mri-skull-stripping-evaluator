import pandas as pd
import pytest

from src.evaluate import (
    _bootstrap_confidence_intervals,
    _dataset_source,
    _quality_summary,
    _summarize_grouped,
)


def _metrics_frame() -> pd.DataFrame:
    rows = []
    for method, offset in (("first", 0.0), ("second", 0.1)):
        for index, identifier in enumerate(("ixi_t1_001", "fsm_t2_001")):
            score = 0.5 + 0.1 * index + offset
            rows.append(
                {
                    "identifier": identifier,
                    "dataset_source": _dataset_source(identifier),
                    "method": method,
                    "dice": score,
                    "iou": score,
                    "precision": score,
                    "sensitivity": score,
                    "specificity": score,
                    "pixel_accuracy": score,
                    "surface_dice": score,
                    "hd95_pixels": 2.0 - score,
                    "assd_pixels": 1.0 - score,
                    "processing_time_seconds": 0.01,
                    "has_quality_warning": index == 0,
                    "quality_warning_count": int(index == 0),
                    "quality_warnings": "area_too_small" if index == 0 else "none",
                }
            )
    return pd.DataFrame(rows)


def test_dataset_source_uses_identifier_prefix() -> None:
    assert _dataset_source("ixi_t1_001") == "IXI"
    assert _dataset_source("infant_t1_01") == "INFANT"


def test_source_summary_preserves_groups() -> None:
    summary = _summarize_grouped(_metrics_frame(), ["dataset_source", "method"])
    assert len(summary) == 4
    assert set(summary["dataset_source"]) == {"IXI", "FSM"}
    assert summary["n_images"].eq(1).all()


def test_bootstrap_confidence_intervals_are_deterministic() -> None:
    first = _bootstrap_confidence_intervals(_metrics_frame(), 100, 42)
    second = _bootstrap_confidence_intervals(_metrics_frame(), 100, 42)
    pd.testing.assert_frame_equal(first, second)
    dice = first[(first["method"] == "first") & (first["metric"] == "dice")].iloc[0]
    assert dice["mean"] == pytest.approx(0.55)
    assert dice["ci_95_lower"] <= dice["mean"] <= dice["ci_95_upper"]


def test_quality_summary_separates_warned_results() -> None:
    summary = _quality_summary(_metrics_frame())
    first = summary[summary["method"] == "first"].iloc[0]
    assert first["warning_rate"] == 0.5
    assert first["mean_dice_warned"] == 0.5
    assert first["mean_dice_not_warned"] == 0.6
    assert first["warning_type_counts"] == "area_too_small:1"
