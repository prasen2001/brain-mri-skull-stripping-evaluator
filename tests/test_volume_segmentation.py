from __future__ import annotations

import numpy as np

from src.metrics import compute_binary_metrics
from src.preprocessing import robust_normalize
from src.volume_segmentation import (
    SPATIAL_PRIOR,
    SpatialPriorConfig,
    VolumeMorphologyConfig,
    central_component_volume,
    clean_volume_with_morphology,
    ellipsoidal_footprint,
    otsu_threshold_volume,
    segment_volume,
    select_largest_central_volume_component,
    spatial_prior_mask,
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


def test_physical_footprint_respects_anisotropic_spacing() -> None:
    footprint = ellipsoidal_footprint(2.0, (1.0, 1.0, 2.0))
    center = tuple(size // 2 for size in footprint.shape)
    assert footprint.shape == (5, 5, 3)
    assert footprint[center]
    assert footprint[center[0] + 2, center[1], center[2]]
    assert footprint[center[0], center[1], center[2] + 1]
    assert not footprint[center[0] + 2, center[1], center[2] + 1]


def test_spatial_prior_is_deterministic_and_excludes_inferior_volume() -> None:
    config = SpatialPriorConfig(inferior_exclusion_fraction=0.25)
    first = spatial_prior_mask((40, 50, 60), (1.0, 1.2, 2.0), config)
    second = spatial_prior_mask((40, 50, 60), (1.0, 1.2, 2.0), config)
    assert np.array_equal(first, second)
    assert not first[:, :, :15].any()
    assert first[20, 23, 37]
    assert not first[0, 0, -1]


def test_spatial_prior_reduces_synthetic_inferior_false_positives() -> None:
    shape = (64, 64, 64)
    coordinates = np.indices(shape)
    reference = (
        ((coordinates[0] - 32) / 15) ** 2
        + ((coordinates[1] - 29) / 17) ** 2
        + ((coordinates[2] - 40) / 15) ** 2
        <= 1
    )
    neck = np.zeros(shape, dtype=bool)
    neck[24:40, 22:42, 2:26] = True
    volume = np.zeros(shape, dtype=np.float32)
    volume[reference | neck] = 0.8

    baseline, _ = segment_volume(volume, "otsu_morphology")
    constrained, _ = segment_volume(
        volume, SPATIAL_PRIOR, spacing_mm=(1.0, 1.0, 1.0)
    )
    baseline_metrics = compute_binary_metrics(baseline, reference)
    constrained_metrics = compute_binary_metrics(constrained, reference)

    assert constrained_metrics["fp"] < baseline_metrics["fp"]
    assert constrained_metrics["dice"] > baseline_metrics["dice"]
    assert constrained[32, 29, 40]
