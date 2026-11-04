"""Deterministic classical baselines for complete 3D MRI volumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, morphology

from src.segmentation import CENTRAL_COMPONENT, METHOD_NAMES, OTSU_MORPHOLOGY, OTSU_THRESHOLD

SPATIAL_PRIOR = "spatial_prior"
VOLUME_METHOD_NAMES = (*METHOD_NAMES, SPATIAL_PRIOR)


@dataclass(frozen=True)
class VolumeMorphologyConfig:
    """Scale-aware parameters for the 3D classical baselines."""

    opening_radius_fraction: float = 0.008
    closing_radius_fraction: float = 0.004
    minimum_component_volume_fraction: float = 0.005
    centrality_weight: float = 0.75


@dataclass(frozen=True)
class SpatialPriorConfig:
    """Parameters for the optional exploratory anatomy-location baseline."""

    center_fraction: tuple[float, float, float] = (0.50, 0.46, 0.61)
    radius_fraction: tuple[float, float, float] = (0.39, 0.42, 0.34)
    inferior_exclusion_fraction: float = 0.25
    opening_radius_mm: float = 1.0
    closing_radius_mm: float = 2.0
    minimum_component_volume_ml: float = 1.0


DEFAULT_VOLUME_CONFIG = VolumeMorphologyConfig()
DEFAULT_SPATIAL_CONFIG = SpatialPriorConfig()


def _radius(shape: tuple[int, ...], fraction: float) -> int:
    return max(1, int(round(min(shape) * fraction)))


def _minimum_volume(shape: tuple[int, ...], fraction: float) -> int:
    return max(8, int(round(np.prod(shape) * fraction)))


def _validate_spacing(spacing_mm: tuple[float, float, float]) -> None:
    if len(spacing_mm) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in spacing_mm
    ):
        raise ValueError("spacing_mm must contain three positive finite values.")


def ellipsoidal_footprint(
    radius_mm: float, spacing_mm: tuple[float, float, float]
) -> np.ndarray:
    """Create a 3D morphology footprint with a physical-space radius."""
    _validate_spacing(spacing_mm)
    if not np.isfinite(radius_mm) or radius_mm < 0:
        raise ValueError("radius_mm must be a non-negative finite value.")
    if radius_mm == 0:
        return np.ones((1, 1, 1), dtype=bool)
    voxel_radii = np.maximum(
        1, np.ceil(radius_mm / np.asarray(spacing_mm, dtype=float)).astype(int)
    )
    grids = np.ogrid[
        tuple(slice(-radius, radius + 1) for radius in voxel_radii.tolist())
    ]
    squared_distance = np.zeros(
        tuple(int(2 * radius + 1) for radius in voxel_radii), dtype=np.float32
    )
    for grid, spacing in zip(grids, spacing_mm, strict=True):
        squared_distance += ((grid * spacing) / radius_mm) ** 2
    return squared_distance <= 1.0


def spatial_prior_mask(
    shape: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
    config: SpatialPriorConfig = DEFAULT_SPATIAL_CONFIG,
) -> np.ndarray:
    """Build a canonical-RAS ellipsoid that excludes inferior neck anatomy.

    The mask uses only array geometry and voxel spacing. It never reads image
    intensities or the reference mask. Fractions are measured across the full
    canonical volume: axis 0 is left-right, axis 1 posterior-anterior, and axis
    2 inferior-superior.
    """
    _validate_spacing(spacing_mm)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("shape must contain three positive dimensions.")
    center = np.asarray(config.center_fraction, dtype=float)
    radii = np.asarray(config.radius_fraction, dtype=float)
    if center.shape != (3,) or np.any(~np.isfinite(center)) or np.any(
        (center < 0) | (center > 1)
    ):
        raise ValueError("center_fraction must contain three values in [0, 1].")
    if radii.shape != (3,) or np.any(~np.isfinite(radii)) or np.any(
        (radii <= 0) | (radii > 1)
    ):
        raise ValueError("radius_fraction must contain three values in (0, 1].")
    if not 0 <= config.inferior_exclusion_fraction < 1:
        raise ValueError("inferior_exclusion_fraction must be in [0, 1).")

    distance = np.zeros(shape, dtype=np.float32)
    for axis, (size, spacing) in enumerate(zip(shape, spacing_mm, strict=True)):
        coordinate = np.arange(size, dtype=np.float32)
        coordinate = (coordinate - (size - 1) * center[axis]) * spacing
        physical_radius = max(spacing, (size - 1) * spacing * radii[axis])
        axis_shape = [1, 1, 1]
        axis_shape[axis] = size
        distance += (coordinate.reshape(axis_shape) / physical_radius) ** 2
    prior = distance <= 1.0
    inferior_stop = int(np.floor(shape[2] * config.inferior_exclusion_fraction))
    prior[:, :, :inferior_stop] = False
    return prior


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


def clean_volume_in_physical_space(
    mask: np.ndarray,
    spacing_mm: tuple[float, float, float],
    config: SpatialPriorConfig = DEFAULT_SPATIAL_CONFIG,
) -> np.ndarray:
    """Clean a mask using millimetre radii and a minimum volume in millilitres."""
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 3:
        raise ValueError(f"Expected a 3D mask, received shape {binary.shape}.")
    _validate_spacing(spacing_mm)
    if config.minimum_component_volume_ml < 0:
        raise ValueError("minimum_component_volume_ml must be non-negative.")
    opening = ellipsoidal_footprint(config.opening_radius_mm, spacing_mm)
    closing = ellipsoidal_footprint(config.closing_radius_mm, spacing_mm)
    cleaned = morphology.opening(binary, opening)
    cleaned = morphology.closing(cleaned, closing)
    minimum_voxels = max(
        1,
        int(
            np.ceil(
                config.minimum_component_volume_ml
                * 1000.0
                / float(np.prod(spacing_mm))
            )
        ),
    )
    cleaned = morphology.remove_small_objects(
        cleaned, max_size=minimum_voxels, connectivity=3
    )
    return np.asarray(ndi.binary_fill_holes(cleaned), dtype=bool)


def spatial_prior_volume(
    volume: np.ndarray,
    spacing_mm: tuple[float, float, float],
    morphology_config: VolumeMorphologyConfig = DEFAULT_VOLUME_CONFIG,
    spatial_config: SpatialPriorConfig = DEFAULT_SPATIAL_CONFIG,
) -> tuple[np.ndarray, float]:
    """Constrain the morphology baseline with a canonical spatial prior.

    This optional fourth method is intentionally simple and explainable. It
    starts from the unchanged Otsu-plus-morphology result, excludes voxels
    outside a configurable superior ellipsoid, and performs final cleanup in
    physical units. The ellipsoid is a coarse location prior, not an atlas.
    """
    cleaned, threshold = otsu_morphology_volume(volume, morphology_config)
    prior = spatial_prior_mask(cleaned.shape, spacing_mm, spatial_config)
    constrained = cleaned & prior
    return (
        clean_volume_in_physical_space(constrained, spacing_mm, spatial_config),
        threshold,
    )


def segment_volume(
    volume: np.ndarray,
    method: str,
    config: VolumeMorphologyConfig = DEFAULT_VOLUME_CONFIG,
    spacing_mm: tuple[float, float, float] = (1.0, 1.0, 1.0),
    spatial_config: SpatialPriorConfig = DEFAULT_SPATIAL_CONFIG,
) -> tuple[np.ndarray, float]:
    """Dispatch a stable 3D baseline or the optional spatial-prior method."""
    functions = {
        OTSU_THRESHOLD: lambda: otsu_threshold_volume(volume),
        OTSU_MORPHOLOGY: lambda: otsu_morphology_volume(volume, config),
        CENTRAL_COMPONENT: lambda: central_component_volume(volume, config),
        SPATIAL_PRIOR: lambda: spatial_prior_volume(
            volume, spacing_mm, config, spatial_config
        ),
    }
    if method not in functions:
        raise ValueError(
            f"Unknown method {method!r}; choose from {VOLUME_METHOD_NAMES}."
        )
    return functions[method]()
