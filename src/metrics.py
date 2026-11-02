"""Visible, dependency-light formulas for binary segmentation metrics."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from scipy import ndimage as ndi

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


def _surface(mask: np.ndarray) -> np.ndarray:
    """Return one-pixel-wide inner boundary pixels for a binary mask."""
    binary = np.asarray(mask, dtype=bool)
    if not binary.any():
        return np.zeros(binary.shape, dtype=bool)
    eroded = ndi.binary_erosion(
        binary, structure=np.ones((3, 3), dtype=bool), border_value=0
    )
    return binary & ~eroded


def compute_boundary_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    tolerance_pixels: float = 1.0,
    spacing_mm: tuple[float, float] = (1.0, 1.0),
) -> dict[str, float]:
    """Compute surface Dice, robust Hausdorff distance, and ASSD.

    Surface Dice measures the fraction of boundary pixels lying within the
    requested tolerance of the other boundary. HD95 is the 95th percentile of
    symmetric boundary distances, making it less sensitive than maximum
    Hausdorff distance to one extreme pixel. ASSD is the average symmetric
    surface distance. Pixel distances support image-array debugging; physical
    distances use NIfTI row/column spacing and are reported in millimeters.

    Two empty masks are a perfect boundary match. When exactly one mask is
    empty, overlap is zero and both distance metrics receive the image diagonal
    as a finite, explicit worst-case penalty.
    """
    if prediction.shape != reference.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match "
            f"reference shape {reference.shape}."
        )
    if tolerance_pixels < 0:
        raise ValueError("tolerance_pixels must be non-negative.")
    if len(spacing_mm) != 2 or any(
        not np.isfinite(value) or value <= 0 for value in spacing_mm
    ):
        raise ValueError("spacing_mm must contain two positive finite values.")

    pred_surface = _surface(np.asarray(prediction, dtype=bool))
    ref_surface = _surface(np.asarray(reference, dtype=bool))
    pred_count = int(pred_surface.sum())
    ref_count = int(ref_surface.sum())
    if pred_count == 0 and ref_count == 0:
        return {
            "surface_dice": 1.0,
            "hd95_pixels": 0.0,
            "assd_pixels": 0.0,
            "hd95_mm": 0.0,
            "assd_mm": 0.0,
        }
    if pred_count == 0 or ref_count == 0:
        diagonal_pixels = float(np.hypot(*prediction.shape))
        diagonal_mm = float(
            np.hypot(
                prediction.shape[0] * spacing_mm[0],
                prediction.shape[1] * spacing_mm[1],
            )
        )
        return {
            "surface_dice": 0.0,
            "hd95_pixels": diagonal_pixels,
            "assd_pixels": diagonal_pixels,
            "hd95_mm": diagonal_mm,
            "assd_mm": diagonal_mm,
        }

    distance_to_ref = ndi.distance_transform_edt(~ref_surface)
    distance_to_pred = ndi.distance_transform_edt(~pred_surface)
    pred_to_ref = distance_to_ref[pred_surface]
    ref_to_pred = distance_to_pred[ref_surface]
    symmetric_distances = np.concatenate((pred_to_ref, ref_to_pred))
    distance_to_ref_mm = ndi.distance_transform_edt(
        ~ref_surface, sampling=spacing_mm
    )
    distance_to_pred_mm = ndi.distance_transform_edt(
        ~pred_surface, sampling=spacing_mm
    )
    symmetric_distances_mm = np.concatenate(
        (distance_to_ref_mm[pred_surface], distance_to_pred_mm[ref_surface])
    )
    within_tolerance = int(np.count_nonzero(pred_to_ref <= tolerance_pixels))
    within_tolerance += int(np.count_nonzero(ref_to_pred <= tolerance_pixels))
    return {
        "surface_dice": float(within_tolerance / (pred_count + ref_count)),
        "hd95_pixels": float(np.percentile(symmetric_distances, 95)),
        "assd_pixels": float(symmetric_distances.mean()),
        "hd95_mm": float(np.percentile(symmetric_distances_mm, 95)),
        "assd_mm": float(symmetric_distances_mm.mean()),
    }

