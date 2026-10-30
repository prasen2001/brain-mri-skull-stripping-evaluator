"""Intensity preprocessing that does not alter source dataset files."""

from __future__ import annotations

import numpy as np


def robust_normalize(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """Clip finite intensities to per-image percentiles and scale to [0, 1].

    Percentile clipping limits the influence of a few extreme pixels. Constant
    images return zeros because they contain no intensity contrast to segment.
    Non-finite values are replaced safely before percentile calculation.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, received shape {image.shape}.")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")

    values = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(values.shape, dtype=np.float32)

    finite_values = values[finite]
    low, high = np.percentile(
        finite_values, [lower_percentile, upper_percentile]
    )
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros(values.shape, dtype=np.float32)

    clean = np.where(finite, values, low)
    normalized = (np.clip(clean, low, high) - low) / (high - low)
    return normalized.astype(np.float32, copy=False)

