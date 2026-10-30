"""Three deterministic classical skull-stripping baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology

OTSU_THRESHOLD = "otsu_threshold"
OTSU_MORPHOLOGY = "otsu_morphology"
CENTRAL_COMPONENT = "central_component"
METHOD_NAMES = (OTSU_THRESHOLD, OTSU_MORPHOLOGY, CENTRAL_COMPONENT)


@dataclass(frozen=True)
class MorphologyConfig:
    """Scale-aware parameters shared by the morphology baselines."""

    opening_radius_fraction: float = 0.004
    closing_radius_fraction: float = 0.008
    minimum_component_area_fraction: float = 0.002
    centrality_weight: float = 0.75


def _radius(shape: tuple[int, int], fraction: float) -> int:
    return max(1, int(round(min(shape) * fraction)))


def _minimum_area(shape: tuple[int, int], fraction: float) -> int:
    return max(4, int(round(np.prod(shape) * fraction)))


def _polarity_score(mask: np.ndarray) -> float:
    """Score a foreground hypothesis without using the ground-truth mask.

    A plausible head/brain foreground occupies a meaningful but not nearly
    complete area, reaches the image center, and avoids covering most borders.
    This lets dark-on-bright images be inverted deterministically.
    """
    area_fraction = float(mask.mean())
    if area_fraction <= 0.001 or area_fraction >= 0.95:
        return -1e6

    h, w = mask.shape
    central = mask[h // 4 : max(h // 4 + 1, 3 * h // 4), w // 4 : max(w // 4 + 1, 3 * w // 4)]
    central_fraction = float(central.mean()) if central.size else 0.0
    border = np.concatenate((mask[0], mask[-1], mask[:, 0], mask[:, -1]))
    border_fraction = float(border.mean())
    # Broad area preference avoids favoring a tiny bright artifact.
    area_preference = 1.0 - min(abs(area_fraction - 0.30) / 0.30, 1.0)
    return 2.0 * central_fraction + area_preference - 1.5 * border_fraction


def otsu_threshold(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Otsu thresholding and choose the safer intensity polarity."""
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2D image, received shape {values.shape}.")
    if values.size == 0 or np.nanmax(values) <= np.nanmin(values):
        return np.zeros(values.shape, dtype=bool), float(np.nanmin(values) if values.size else 0.0)

    threshold = float(filters.threshold_otsu(values))
    bright = values > threshold
    dark = ~bright
    selected = bright if _polarity_score(bright) >= _polarity_score(dark) else dark
    return np.asarray(selected, dtype=bool), threshold


def clean_with_morphology(
    mask: np.ndarray, config: MorphologyConfig = MorphologyConfig()
) -> np.ndarray:
    """Open, close, fill holes, and remove small connected regions."""
    binary = np.asarray(mask, dtype=bool)
    opening = morphology.disk(_radius(binary.shape, config.opening_radius_fraction))
    closing = morphology.disk(_radius(binary.shape, config.closing_radius_fraction))
    cleaned = morphology.opening(binary, opening)
    cleaned = morphology.closing(cleaned, closing)
    cleaned = ndi.binary_fill_holes(cleaned)
    cleaned = morphology.remove_small_objects(
        cleaned, max_size=_minimum_area(binary.shape, config.minimum_component_area_fraction)
    )
    return np.asarray(cleaned, dtype=bool)


def otsu_morphology(
    image: np.ndarray, config: MorphologyConfig = MorphologyConfig()
) -> tuple[np.ndarray, float]:
    """Run Otsu then configurable scale-aware morphological cleanup."""
    initial, threshold = otsu_threshold(image)
    return clean_with_morphology(initial, config), threshold


def select_largest_central_component(
    mask: np.ndarray, centrality_weight: float = 0.75
) -> np.ndarray:
    """Select a large component whose centroid is close to the image center.

    Component score = area * (1 + weight * centrality), where centrality falls
    linearly from 1 at image center to 0 at a corner. Area remains dominant,
    but a similarly sized central component is preferred. Label order breaks an
    exact tie, making the heuristic deterministic.
    """
    binary = np.asarray(mask, dtype=bool)
    labels = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labels)
    if not regions:
        return np.zeros(binary.shape, dtype=bool)

    center = np.array([(binary.shape[0] - 1) / 2, (binary.shape[1] - 1) / 2])
    max_distance = float(np.linalg.norm(center)) or 1.0

    def score(region: measure._regionprops.RegionProperties) -> tuple[float, int, int]:
        distance = float(np.linalg.norm(np.asarray(region.centroid) - center))
        centrality = max(0.0, 1.0 - distance / max_distance)
        weighted = float(region.area) * (1.0 + centrality_weight * centrality)
        return weighted, int(region.area), -int(region.label)

    chosen = max(regions, key=score)
    return labels == chosen.label


def central_component(
    image: np.ndarray, config: MorphologyConfig = MorphologyConfig()
) -> tuple[np.ndarray, float]:
    """Run Method 2, select the largest central component, and clean it."""
    cleaned, threshold = otsu_morphology(image, config)
    selected = select_largest_central_component(cleaned, config.centrality_weight)
    selected = ndi.binary_fill_holes(selected)
    selected = morphology.closing(
        selected,
        morphology.disk(_radius(selected.shape, config.closing_radius_fraction)),
    )
    selected = ndi.binary_fill_holes(selected)
    return np.asarray(selected, dtype=bool), threshold


def segment(image: np.ndarray, method: str, config: MorphologyConfig) -> tuple[np.ndarray, float]:
    """Dispatch one of the three stable method names."""
    functions = {
        OTSU_THRESHOLD: lambda: otsu_threshold(image),
        OTSU_MORPHOLOGY: lambda: otsu_morphology(image, config),
        CENTRAL_COMPONENT: lambda: central_component(image, config),
    }
    if method not in functions:
        raise ValueError(f"Unknown method {method!r}; choose from {METHOD_NAMES}.")
    return functions[method]()

