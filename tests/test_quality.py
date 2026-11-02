import numpy as np
import pytest

from src.quality import (
    QualityConfig,
    mask_iou,
    mean_method_disagreement,
    prediction_diagnostics,
)


def test_centered_mask_has_stable_geometry() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    result = prediction_diagnostics(mask, method_disagreement=0.0)
    assert result["predicted_area_fraction"] == 0.25
    assert result["component_count"] == 1
    assert result["largest_component_fraction"] == 1.0
    assert result["centroid_distance_normalized"] == pytest.approx(0.0)
    assert result["has_quality_warning"] is False


def test_empty_and_tiny_masks_are_flagged() -> None:
    empty = np.zeros((20, 20), dtype=bool)
    assert prediction_diagnostics(empty, 0.0)["quality_warnings"] == "empty_mask"
    tiny = empty.copy()
    tiny[10, 10] = True
    assert "area_too_small" in prediction_diagnostics(tiny, 0.0)["quality_warnings"]


def test_fragmented_mask_and_disagreement_are_flagged() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:8, 4:8] = True
    mask[12:16, 12:16] = True
    config = QualityConfig(maximum_component_count=1, minimum_largest_component_fraction=0.8)
    result = prediction_diagnostics(mask, method_disagreement=0.8, config=config)
    assert "many_components" in result["quality_warnings"]
    assert "fragmented_mask" in result["quality_warnings"]
    assert "methods_disagree" in result["quality_warnings"]


def test_method_disagreement_uses_pairwise_iou() -> None:
    first = np.array([[1, 0], [0, 0]], dtype=bool)
    same = first.copy()
    different = np.array([[0, 1], [0, 0]], dtype=bool)
    assert mask_iou(first, same) == 1.0
    assert mask_iou(first, different) == 0.0
    assert mean_method_disagreement(first, [same, different]) == 0.5
