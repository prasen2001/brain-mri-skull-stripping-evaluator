import numpy as np
import pytest

from src.metrics import compute_binary_metrics, compute_boundary_metrics


def test_perfect_overlap() -> None:
    mask = np.array([[1, 0], [1, 0]], dtype=bool)
    result = compute_binary_metrics(mask, mask)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (2, 2, 0, 0)
    for metric in ("dice", "iou", "precision", "sensitivity", "specificity"):
        assert result[metric] == 1.0


def test_no_overlap() -> None:
    prediction = np.array([[1, 0], [0, 0]], dtype=bool)
    reference = np.array([[0, 1], [0, 0]], dtype=bool)
    result = compute_binary_metrics(prediction, reference)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (0, 2, 1, 1)
    assert result["dice"] == 0.0
    assert result["iou"] == 0.0
    assert result["precision"] == 0.0
    assert result["sensitivity"] == 0.0
    assert result["specificity"] == pytest.approx(2 / 3)


def test_partial_overlap() -> None:
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    reference = np.array([[1, 0], [1, 0]], dtype=bool)
    result = compute_binary_metrics(prediction, reference)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (1, 1, 1, 1)
    assert result["dice"] == 0.5
    assert result["iou"] == pytest.approx(1 / 3)
    assert result["precision"] == 0.5
    assert result["sensitivity"] == 0.5
    assert result["specificity"] == 0.5


def test_empty_prediction() -> None:
    prediction = np.zeros((2, 2), dtype=bool)
    reference = np.array([[1, 0], [0, 0]], dtype=bool)
    result = compute_binary_metrics(prediction, reference)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (0, 3, 0, 1)
    assert result["dice"] == 0.0
    assert result["precision"] == 1.0  # no predicted positives and therefore no FP
    assert result["sensitivity"] == 0.0


def test_empty_reference() -> None:
    prediction = np.array([[1, 0], [0, 0]], dtype=bool)
    reference = np.zeros((2, 2), dtype=bool)
    result = compute_binary_metrics(prediction, reference)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (0, 3, 1, 0)
    assert result["dice"] == 0.0
    assert result["precision"] == 0.0
    assert result["sensitivity"] == 1.0  # no actual positive pixels were missed
    assert result["specificity"] == 0.75


def test_both_masks_empty() -> None:
    empty = np.zeros((2, 2), dtype=bool)
    result = compute_binary_metrics(empty, empty)
    assert (result["tp"], result["tn"], result["fp"], result["fn"]) == (0, 4, 0, 0)
    for metric in ("dice", "iou", "precision", "sensitivity", "specificity"):
        assert result[metric] == 1.0


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="does not match"):
        compute_binary_metrics(np.zeros((2, 2)), np.zeros((3, 2)))


def test_boundary_metrics_for_identical_masks() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    result = compute_boundary_metrics(mask, mask)
    assert result == {
        "surface_dice": 1.0,
        "hd95_pixels": 0.0,
        "assd_pixels": 0.0,
        "hd95_mm": 0.0,
        "assd_mm": 0.0,
    }


def test_boundary_metrics_for_shifted_masks() -> None:
    prediction = np.zeros((10, 10), dtype=bool)
    reference = np.zeros((10, 10), dtype=bool)
    prediction[2:6, 2:6] = True
    reference[2:6, 4:8] = True
    result = compute_boundary_metrics(prediction, reference, tolerance_pixels=0)
    assert 0 < result["surface_dice"] < 1
    assert result["hd95_pixels"] > 0
    assert result["assd_pixels"] > 0


def test_boundary_metrics_respect_anisotropic_spacing() -> None:
    prediction = np.zeros((10, 10), dtype=bool)
    reference = np.zeros((10, 10), dtype=bool)
    prediction[4, 4] = True
    reference[5, 4] = True
    result = compute_boundary_metrics(
        prediction, reference, tolerance_pixels=0, spacing_mm=(2.5, 0.5)
    )
    assert result["hd95_pixels"] == 1.0
    assert result["assd_pixels"] == 1.0
    assert result["hd95_mm"] == 2.5
    assert result["assd_mm"] == 2.5


def test_boundary_metrics_for_empty_masks() -> None:
    empty = np.zeros((3, 4), dtype=bool)
    nonempty = empty.copy()
    nonempty[1, 1] = True
    assert compute_boundary_metrics(empty, empty)["surface_dice"] == 1.0
    result = compute_boundary_metrics(nonempty, empty)
    assert result["surface_dice"] == 0.0
    assert result["hd95_pixels"] == 5.0
    assert result["assd_pixels"] == 5.0

