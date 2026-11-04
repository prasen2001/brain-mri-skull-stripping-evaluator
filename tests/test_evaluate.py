import pandas as pd
import pytest

from src.evaluate import (
    _bootstrap_confidence_intervals,
    _dataset_source,
    _modality,
    _paired_method_comparisons,
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
                    "hd95_mm": 2.0 - score,
                    "assd_mm": 1.0 - score,
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


def test_modality_uses_official_identifier_token() -> None:
    assert _modality("ixi_t2_001") == "T2"
    assert _modality("qin_flair_01") == "FLAIR"
    assert _modality("asl_epi_101") == "ASL/EPI"
    assert _modality("fsm_qt1_02cd") == "qT1"


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


def test_paired_comparison_uses_scan_level_differences() -> None:
    frame = _metrics_frame()
    frame["method"] = frame["method"].replace(
        {"first": "otsu_threshold", "second": "otsu_morphology"}
    )
    comparisons = _paired_method_comparisons(frame, 100, 42)
    dice = comparisons[
        (comparisons["baseline_method"] == "otsu_threshold")
        & (comparisons["comparison_method"] == "otsu_morphology")
        & (comparisons["metric"] == "dice")
    ].iloc[0]
    assert dice["n_paired_scans"] == 2
    assert dice["mean_improvement"] == pytest.approx(0.1)
    assert dice["comparison_wins"] == 2
    assert dice["comparison_win_rate"] == 1.0
    assert dice["improvement_ci_95_lower"] == pytest.approx(0.1)
    assert dice["improvement_ci_95_upper"] == pytest.approx(0.1)


def test_paired_distance_improvement_treats_lower_as_better() -> None:
    frame = _metrics_frame()
    frame["method"] = frame["method"].replace(
        {"first": "otsu_threshold", "second": "otsu_morphology"}
    )
    frame.loc[frame.method == "otsu_morphology", "hd95_mm"] = 0.5
    comparisons = _paired_method_comparisons(frame, 50, 1)
    hd95 = comparisons[
        (comparisons["baseline_method"] == "otsu_threshold")
        & (comparisons["comparison_method"] == "otsu_morphology")
        & (comparisons["metric"] == "hd95_mm")
    ].iloc[0]
    assert hd95["mean_improvement"] > 0
    assert hd95["comparison_wins"] == 2


def test_paired_comparison_accepts_an_additional_method() -> None:
    frame = _metrics_frame()
    original = frame.copy()
    original["method"] = original["method"].replace(
        {"first": "otsu_threshold", "second": "otsu_morphology"}
    )
    spatial = frame[frame["method"] == "second"].copy()
    spatial["method"] = "spatial_prior"
    spatial["dice"] += 0.2
    combined = pd.concat((original, spatial), ignore_index=True)

    comparisons = _paired_method_comparisons(
        combined,
        50,
        42,
        ("otsu_threshold", "otsu_morphology", "spatial_prior"),
    )

    spatial_dice = comparisons[
        (comparisons["baseline_method"] == "otsu_morphology")
        & (comparisons["comparison_method"] == "spatial_prior")
        & (comparisons["metric"] == "dice")
    ].iloc[0]
    assert spatial_dice["mean_improvement"] == pytest.approx(0.2)
    assert spatial_dice["comparison_wins"] == 2
