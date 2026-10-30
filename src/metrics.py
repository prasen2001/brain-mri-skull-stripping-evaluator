"""Visible, dependency-light formulas for binary segmentation metrics."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

MetricValue: TypeAlias = int | float


def _safe_ratio(numerator: int, denominator: int, empty_value: float) -> float:
    """Divide counts while making the zero-denominator convention explicit."""
    return float(numerator / denominator) if denominator else empty_value


def compute_binary_metrics(
    prediction: np.ndarray, reference: np.ndarray
) -> dict[str, MetricValue]:
    """Compute confusion counts and common binary segmentation metrics.

    Both inputs are interpreted as boolean arrays and must share a shape.
    When both masks are empty, Dice and IoU are 1 (perfect agreement). A metric
    with no positive examples in its denominator is 1 when no corresponding
    error was made; this keeps the empty-case behavior deterministic.
    """
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match "
            f"reference shape {reference.shape}."
        )

    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    tp = int(np.count_nonzero(pred & ref))
    tn = int(np.count_nonzero(~pred & ~ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))

    pred_count = tp + fp
    ref_count = tp + fn
    union = tp + fp + fn
    total = tp + tn + fp + fn

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "dice": _safe_ratio(2 * tp, pred_count + ref_count, 1.0),
        "iou": _safe_ratio(tp, union, 1.0),
        "precision": _safe_ratio(tp, pred_count, 1.0),
        "sensitivity": _safe_ratio(tp, ref_count, 1.0),
        "specificity": _safe_ratio(tn, tn + fp, 1.0),
        "pixel_accuracy": _safe_ratio(tp + tn, total, 1.0),
    }

