import numpy as np
import pandas as pd
import pytest

from src.calibration import detection_metrics, rank_calibration_results


def test_detection_metrics_use_warning_as_positive_class() -> None:
    result = detection_metrics(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["specificity"] == 0.5
    assert result["f1"] == 0.5


def test_detection_metrics_reject_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        detection_metrics(np.array([True]), np.array([True, False]))


def test_calibration_ranking_prefers_f1_then_conservative_tie_breaker() -> None:
    frame = pd.DataFrame(
        [
            {"name": "many", "f1": 1.0, "recall": 1.0, "precision": 1.0, "specificity": 1.0, "warning_rate": 0.2, "distance_from_default": 0.1},
            {"name": "close", "f1": 1.0, "recall": 1.0, "precision": 1.0, "specificity": 1.0, "warning_rate": 0.1, "distance_from_default": 0.01},
            {"name": "worse", "f1": 0.8, "recall": 1.0, "precision": 0.7, "specificity": 0.7, "warning_rate": 0.1, "distance_from_default": 0.0},
        ]
    )
    ranked = rank_calibration_results(frame)
    assert ranked.iloc[0]["name"] == "close"
