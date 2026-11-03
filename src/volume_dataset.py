"""Discovery and validation for complete three-dimensional NFBS volumes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class VolumePair:
    """Paths for one full-head 3D image and aligned reference brain mask."""

    identifier: str
    image_path: Path
    mask_path: Path


def discover_nfbs_volumes(
    data_dir: Path | str,
) -> tuple[list[VolumePair], list[dict[str, str]]]:
    """Discover NFBS full-head/mask pairs in deterministic subject order."""
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"NFBS volume directory does not exist: {root}")
    pairs: list[VolumePair] = []
    invalid: list[dict[str, str]] = []
    for subject_dir in sorted(
        (path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name
    ):
        images = sorted(subject_dir.glob("*_T1w.nii.gz"))
        masks = sorted(subject_dir.glob("*_T1w_brainmask.nii.gz"))
        if len(images) != 1 or len(masks) != 1:
            invalid.append(
                {
                    "identifier": subject_dir.name,
                    "reason": (
                        "Expected one full-head T1 and one brain mask; "
                        f"found {len(images)} and {len(masks)}"
                    ),
                }
            )
            continue
        pairs.append(VolumePair(subject_dir.name, images[0], masks[0]))
    return pairs, invalid


def load_volume_pair(
    pair: VolumePair,
) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Load, canonicalize, binarize, and validate one complete 3D pair."""
    try:
        image_nifti = nib.as_closest_canonical(
            cast(nib.Nifti1Image, nib.load(pair.image_path))
        )
        mask_nifti = nib.as_closest_canonical(
            cast(nib.Nifti1Image, nib.load(pair.mask_path))
        )
        image = np.asanyarray(image_nifti.dataobj)
        mask = np.asanyarray(mask_nifti.dataobj)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{pair.identifier}: could not load NIfTI volumes: {exc}") from exc

    if image.ndim != 3 or mask.ndim != 3:
        raise ValueError(
            f"{pair.identifier}: expected 3D volumes, received {image.shape} and {mask.shape}."
        )
    if image.shape != mask.shape:
        raise ValueError(
            f"{pair.identifier}: image shape {image.shape} does not match mask shape {mask.shape}."
        )
    if image.size == 0 or not np.isfinite(image).any():
        raise ValueError(f"{pair.identifier}: image contains no usable finite voxels.")
    if not np.allclose(image_nifti.affine, mask_nifti.affine, atol=1e-4):
        raise ValueError(f"{pair.identifier}: canonical image and mask affines do not match.")

    spacing = tuple(float(value) for value in image_nifti.header.get_zooms()[:3])
    if len(spacing) != 3 or any(
        not np.isfinite(value) or value <= 0 for value in spacing
    ):
        raise ValueError(f"{pair.identifier}: invalid voxel spacing {spacing}.")
    return (
        np.asarray(image, dtype=np.float32),
        np.asarray(mask > 0, dtype=bool),
        (spacing[0], spacing[1], spacing[2]),
    )


def validate_volume_pairs(
    pairs: list[VolumePair],
) -> tuple[list[VolumePair], list[dict[str, str]]]:
    """Load-check each volume pair and return valid items plus reasons."""
    valid: list[VolumePair] = []
    invalid: list[dict[str, str]] = []
    for pair in pairs:
        try:
            load_volume_pair(pair)
            valid.append(pair)
        except ValueError as exc:
            invalid.append({"identifier": pair.identifier, "reason": str(exc)})
    return valid, invalid
