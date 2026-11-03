from __future__ import annotations

import numpy as np

from src.preprocessing import robust_normalize
from src.volume_segmentation import (
    VolumeMorphologyConfig,
    central_component_volume,
    clean_volume_with_morphology,
    otsu_threshold_volume,
    select_largest_central_volume_component,
)


def test_volume_threshold_returns_binary_mask() -> None:
    volume = np.zeros((32, 34, 36), dtype=np.float32)
    volume[8:24, 8:26, 8:28] = 1.0
    mask, threshold = otsu_threshold_volume(volume)
    assert mask.shape == volume.shape
    assert mask.dtype == np.bool_
    assert 0 <= threshold <= 1


def test_3d_morphology_removes_island_and_fills_hole() -> None:
    mask = np.zeros((40, 40, 40), dtype=bool)
    mask[8:32, 8:32, 8:32] = True
    mask[18:22, 18:22, 18:22] = False
    mask[2, 2, 2] = True
    config = VolumeMorphologyConfig(
        opening_radius_fraction=0.025,
        closing_radius_fraction=0.025,
        minimum_component_volume_fraction=0.001,
    )
    cleaned = clean_volume_with_morphology(mask, config)
    assert cleaned[20, 20, 20]
    assert not cleaned[2, 2, 2]


def test_largest_central_volume_component_is_deterministic() -> None:
    mask = np.zeros((50, 50, 50), dtype=bool)
    mask[1:19, 1:19, 1:19] = True
    mask[20:37, 20:37, 20:37] = True
    first = select_largest_central_volume_component(mask, centrality_weight=0.75)
    second = select_largest_central_volume_component(mask, centrality_weight=0.75)
    assert np.array_equal(first, second)
    assert first[25, 25, 25]
    assert not first[5, 5, 5]


def test_constant_volume_does_not_crash_pipeline() -> None:
    volume = np.full((16, 18, 20), 7.0, dtype=np.float32)
    normalized = robust_normalize(volume)
    prediction, threshold = central_component_volume(normalized)
    assert not prediction.any()
    assert threshold == 0.0


def test_volume_polarity_handles_dark_center() -> None:
    volume = np.ones((32, 32, 32), dtype=np.float32)
    volume[8:24, 8:24, 8:24] = 0.0
    prediction, _ = otsu_threshold_volume(volume)
    assert prediction[16, 16, 16]
    assert not prediction[0, 0, 0]
