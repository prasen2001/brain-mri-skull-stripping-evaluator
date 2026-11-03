"""Three deterministic classical baselines for complete 3D MRI volumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

from src.segmentation import CENTRAL_COMPONENT, METHOD_NAMES, OTSU_MORPHOLOGY, OTSU_THRESHOLD


@dataclass(frozen=True)
class VolumeMorphologyConfig:
    """Scale-aware parameters for the 3D classical baselines."""

    opening_radius_fraction: float = 0.008
    closing_radius_fraction: float = 0.004
    minimum_component_volume_fraction: float = 0.005
    centrality_weight: float = 0.75


DEFAULT_VOLUME_CONFIG = VolumeMorphologyConfig()


def _radius(shape: tuple[int, ...], fraction: float) -> int:
    return max(1, int(round(min(shape) * fraction)))


def _minimum_volume(shape: tuple[int, ...], fraction: float) -> int:
    return max(8, int(round(np.prod(shape) * fraction)))


def _polarity_score(mask: np.ndarray) -> float:
    """Score a 3D foreground hypothesis without consulting the reference."""
    area_fraction = float(mask.mean())
    if area_fraction <= 0.001 or area_fraction >= 0.95:
        return -1e6
    slices = tuple(
        slice(size // 4, max(size // 4 + 1, 3 * size // 4)) for size in mask.shape
    )
    central_fraction = float(mask[slices].mean())
    faces: list[np.ndarray] = []
    for axis in range(mask.ndim):
        lower: list[slice | int] = [slice(None)] * mask.ndim
        upper: list[slice | int] = [slice(None)] * mask.ndim
        lower[axis] = 0
        upper[axis] = -1
        faces.extend((mask[tuple(lower)].ravel(), mask[tuple(upper)].ravel()))
    border_fraction = float(np.concatenate(faces).mean())
    area_preference = 1.0 - min(abs(area_fraction - 0.20) / 0.20, 1.0)
    return 2.0 * central_fraction + area_preference - 1.5 * border_fraction


def otsu_threshold_volume(volume: np.ndarray) -> tuple[np.ndarray, float]:
    """Apply Otsu thresholding and select bright or dark 3D polarity."""
    values = np.asarray(volume, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected a 3D volume, received shape {values.shape}.")
    if values.size == 0 or np.nanmax(values) <= np.nanmin(values):
        threshold = float(np.nanmin(values) if values.size else 0.0)
        return np.zeros(values.shape, dtype=bool), threshold
    threshold = float(filters.threshold_otsu(values))
    bright = values > threshold
    dark = ~bright
    selected = bright if _polarity_score(bright) >= _polarity_score(dark) else dark
    return np.asarray(selected, dtype=bool), threshold


def clean_volume_with_morphology(
    mask: np.ndarray,
    config: VolumeMorphologyConfig = DEFAULT_VOLUME_CONFIG,
) -> np.ndarray:
    """Open, close, fill enclosed holes, and remove small 3D components."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Expected a 3D mask, received shape {binary.shape}.")
    opening = morphology.ball(_radius(binary.shape, config.opening_radius_fraction))
    closing = morphology.ball(_radius(binary.shape, config.closing_radius_fraction))
    cleaned = morphology.opening(binary, opening)
    cleaned = morphology.closing(cleaned, closing)
    cleaned = ndi.binary_fill_holes(cleaned)
    cleaned = morphology.remove_small_objects(
        cleaned,
        max_size=_minimum_volume(
            binary.shape, config.minimum_component_volume_fraction
        ),
        connectivity=3,
    )
    return np.asarray(cleaned, dtype=bool)


def otsu_morphology_volume(
    volume: np.ndarray,
    config: VolumeMorphologyConfig = DEFAULT_VOLUME_CONFIG,
) -> tuple[np.ndarray, float]:
    """Run 3D Otsu followed by scale-aware morphology."""
    initial, threshold = otsu_threshold_volume(volume)
    return clean_volume_with_morphology(initial, config), threshold


def select_largest_central_volume_component(
    mask: np.ndarray, centrality_weight: float = 0.75
) -> np.ndarray:
    """Select a large 26-connected component whose centroid is central."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Expected a 3D mask, received shape {binary.shape}.")
    structure = ndi.generate_binary_structure(3, 3)
    labels, component_count = ndi.label(binary, structure=structure)
    if component_count == 0:
        return np.zeros(binary.shape, dtype=bool)
    label_ids = np.arange(1, component_count + 1)
    areas = np.bincount(labels.ravel(), minlength=component_count + 1)[1:]
    centroids = ndi.center_of_mass(binary, labels, label_ids)
    center = (np.asarray(binary.shape, dtype=float) - 1) / 2
    maximum_distance = float(np.linalg.norm(center)) or 1.0

    scores: list[tuple[float, int, int]] = []
    for label, area, centroid in zip(label_ids, areas, centroids, strict=True):
        distance = float(np.linalg.norm(np.asarray(centroid) - center))
        centrality = max(0.0, 1.0 - distance / maximum_distance)
        weighted = float(area) * (1.0 + centrality_weight * centrality)
        scores.append((weighted, int(area), -int(label)))
    selected_label = int(label_ids[max(range(component_count), key=scores.__getitem__)])
    return labels == selected_label


def central_component_volume(
    volume: np.ndarray,
    config: VolumeMorphologyConfig = DEFAULT_VOLUME_CONFIG,
) -> tuple[np.ndarray, float]:
    """Run 3D morphology, retain one plausible component, and close it."""
    cleaned, threshold = otsu_morphology_volume(volume, config)
    selected = select_largest_central_volume_component(
        cleaned, config.centrality_weight
    )
    selected = ndi.binary_fill_holes(selected)
    selected = morphology.closing(
        selected,
        morphology.ball(_radius(selected.shape, config.closing_radius_fraction)),
    )
    return np.asarray(ndi.binary_fill_holes(selected), dtype=bool), threshold


def segment_volume(
    volume: np.ndarray,
    method: str,
    config: VolumeMorphologyConfig = DEFAULT_VOLUME_CONFIG,
) -> tuple[np.ndarray, float]:
    """Dispatch one of the same three stable method names in 3D."""
    functions = {
        OTSU_THRESHOLD: lambda: otsu_threshold_volume(volume),
        OTSU_MORPHOLOGY: lambda: otsu_morphology_volume(volume, config),
        CENTRAL_COMPONENT: lambda: central_component_volume(volume, config),
    }
    if method not in functions:
        raise ValueError(f"Unknown method {method!r}; choose from {METHOD_NAMES}.")
    return functions[method]()
