"""Reference-free diagnostics for identifying suspicious predicted masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from scipy import ndimage as ndi

DiagnosticValue: TypeAlias = bool | int | float | str


@dataclass(frozen=True)
class QualityConfig:
    """Conservative geometry thresholds for prediction-only warnings."""

    minimum_area_fraction: float = 0.03
    maximum_area_fraction: float = 0.80
    maximum_centroid_distance: float = 0.55
    maximum_border_contact_fraction: float = 0.65
    maximum_component_count: int = 20
    minimum_largest_component_fraction: float = 0.80
    maximum_method_disagreement: float = 0.35


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    """Calculate mask IoU, treating two empty masks as agreement."""
    if first.shape != second.shape:
        raise ValueError(f"Mask shapes do not match: {first.shape} and {second.shape}.")
    first_bool = np.asarray(first, dtype=bool)
    second_bool = np.asarray(second, dtype=bool)
    union = int(np.count_nonzero(first_bool | second_bool))
    if union == 0:
        return 1.0
    return float(np.count_nonzero(first_bool & second_bool) / union)


def mean_method_disagreement(
    prediction: np.ndarray, other_predictions: list[np.ndarray]
) -> float:
    """Return one minus mean IoU with the other baseline predictions."""
    if not other_predictions:
        return 0.0
    return float(1.0 - np.mean([mask_iou(prediction, other) for other in other_predictions]))


def prediction_diagnostics(
    prediction: np.ndarray,
    method_disagreement: float,
    config: QualityConfig = QualityConfig(),
) -> dict[str, DiagnosticValue]:
    """Measure mask geometry and produce deterministic warning labels.

    These diagnostics use only predictions and image geometry. They never read
    the ground-truth mask, so they can also be used when a reference is absent.
    Warnings indicate that a result deserves inspection; they do not prove the
    result is wrong.
    """
    mask = np.asarray(prediction, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D prediction, received shape {mask.shape}.")
    total_pixels = int(mask.size)
    foreground_pixels = int(mask.sum())
    area_fraction = float(foreground_pixels / total_pixels) if total_pixels else 0.0

    labels, component_count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
    if component_count:
        component_areas = np.bincount(labels.ravel())[1:]
        largest_component_fraction = float(component_areas.max() / foreground_pixels)
        coordinates = np.argwhere(mask)
        centroid = coordinates.mean(axis=0)
        center = np.asarray([(mask.shape[0] - 1) / 2, (mask.shape[1] - 1) / 2])
        maximum_distance = float(np.linalg.norm(center)) or 1.0
        centroid_distance = float(np.linalg.norm(centroid - center) / maximum_distance)
    else:
        largest_component_fraction = 0.0
        centroid_distance = 1.0

    border = np.zeros(mask.shape, dtype=bool)
    if mask.size:
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
    border_pixels = int(border.sum())
    border_contact_fraction = (
        float(np.count_nonzero(mask & border) / border_pixels) if border_pixels else 0.0
    )

    warnings: list[str] = []
    if foreground_pixels == 0:
        warnings.append("empty_mask")
    else:
        if area_fraction < config.minimum_area_fraction:
            warnings.append("area_too_small")
        if area_fraction > config.maximum_area_fraction:
            warnings.append("area_too_large")
        if centroid_distance > config.maximum_centroid_distance:
            warnings.append("off_center")
        if border_contact_fraction > config.maximum_border_contact_fraction:
            warnings.append("excessive_border_contact")
        if component_count > config.maximum_component_count:
            warnings.append("many_components")
        if largest_component_fraction < config.minimum_largest_component_fraction:
            warnings.append("fragmented_mask")
    if method_disagreement > config.maximum_method_disagreement:
        warnings.append("methods_disagree")

    return {
        "predicted_area_fraction": area_fraction,
        "component_count": int(component_count),
        "largest_component_fraction": largest_component_fraction,
        "centroid_distance_normalized": centroid_distance,
        "border_contact_fraction": border_contact_fraction,
        "method_disagreement": float(method_disagreement),
        "quality_warning_count": len(warnings),
        "has_quality_warning": bool(warnings),
        "quality_warnings": ";".join(warnings) if warnings else "none",
    }
