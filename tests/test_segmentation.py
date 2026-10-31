import numpy as np

from src.preprocessing import robust_normalize
from src.segmentation import (
    MorphologyConfig,
    central_component,
    clean_with_morphology,
    otsu_threshold,
    select_largest_central_component,
)


def test_threshold_returns_binary_mask() -> None:
    image = np.zeros((32, 32), dtype=np.float32)
    image[8:24, 8:24] = 1.0
    mask, threshold = otsu_threshold(image)
    assert mask.dtype == np.bool_
    assert mask.shape == image.shape
    assert 0 <= threshold <= 1


def test_morphology_removes_small_island() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    mask[2, 2] = True
    config = MorphologyConfig(
        opening_radius_fraction=0.015,
        closing_radius_fraction=0.015,
        minimum_component_area_fraction=0.005,
    )
    cleaned = clean_with_morphology(mask, config)
    assert cleaned[32, 32]
    assert not cleaned[2, 2]


def test_morphology_fills_hole() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:54, 10:54] = True
    mask[28:36, 28:36] = False
    cleaned = clean_with_morphology(mask)
    assert cleaned[32, 32]


def test_largest_central_component_is_deterministic() -> None:
    mask = np.zeros((60, 60), dtype=bool)
    mask[2:22, 2:22] = True  # area 400, far from center
    mask[23:42, 23:42] = True  # area 361, central and therefore preferred
    first = select_largest_central_component(mask, centrality_weight=0.75)
    second = select_largest_central_component(mask, centrality_weight=0.75)
    assert np.array_equal(first, second)
    assert first[30, 30]
    assert not first[5, 5]


def test_constant_image_does_not_crash_pipeline() -> None:
    image = np.full((32, 32), 7.0, dtype=np.float32)
    normalized = robust_normalize(image)
    prediction, threshold = central_component(normalized)
    assert np.all(normalized == 0)
    assert not prediction.any()
    assert threshold == 0.0


def test_polarity_handles_dark_center() -> None:
    image = np.ones((64, 64), dtype=np.float32)
    image[16:48, 16:48] = 0.0
    prediction, _ = otsu_threshold(image)
    assert prediction[32, 32]
    assert not prediction[0, 0]

