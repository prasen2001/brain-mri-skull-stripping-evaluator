"""Discovery and validation for the official SynthStrip 2D NIfTI layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class ImageMaskPair:
    """Paths and split metadata for one scan-level 2D image-mask pair."""

    identifier: str
    image_path: Path
    mask_path: Path
    subset: str


def _read_identifiers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _fallback_subset(identifier: str, seed: int) -> str:
    """Create a stable 20/80 scan-level split if official files are absent."""
    import hashlib

    digest = hashlib.sha256(f"{seed}:{identifier}".encode()).digest()
    return "development" if int.from_bytes(digest[:4], "big") % 5 == 0 else "test"


def discover_pairs(
    data_dir: Path | str,
    subset: str = "all",
    seed: int = 42,
) -> tuple[list[ImageMaskPair], list[dict[str, str]]]:
    """Discover pairs from actual directories and report invalid candidates.

    The official ``scans_valid.txt`` split maps to ``development``. If split
    files do not exist (useful for tests or another compatible dataset), a
    deterministic identifier-level split is used and should be documented as a
    limitation for any real experiment.
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    requested = "development" if subset in {"dev", "valid", "validation"} else subset
    if requested not in {"all", "development", "test"}:
        raise ValueError("subset must be one of: all, development, test")

    valid_ids = _read_identifiers(root / "scans_valid.txt")
    test_ids = _read_identifiers(root / "scans_test.txt")
    has_official_split = bool(valid_ids or test_ids)

    pairs: list[ImageMaskPair] = []
    invalid: list[dict[str, str]] = []
    candidates = sorted(
        (path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name
    )
    for scan_dir in candidates:
        image_path = scan_dir / "image.nii.gz"
        mask_path = scan_dir / "mask.nii.gz"
        if not image_path.exists() or not mask_path.exists():
            missing = []
            if not image_path.exists():
                missing.append("image.nii.gz")
            if not mask_path.exists():
                missing.append("mask.nii.gz")
            invalid.append(
                {"identifier": scan_dir.name, "reason": f"Missing {', '.join(missing)}"}
            )
            continue

        if has_official_split:
            if scan_dir.name in valid_ids:
                item_subset = "development"
            elif scan_dir.name in test_ids:
                item_subset = "test"
            else:
                invalid.append(
                    {
                        "identifier": scan_dir.name,
                        "reason": "Identifier is absent from official split files",
                    }
                )
                continue
        else:
            item_subset = _fallback_subset(scan_dir.name, seed)

        if requested == "all" or item_subset == requested:
            pairs.append(
                ImageMaskPair(
                    identifier=scan_dir.name,
                    image_path=image_path,
                    mask_path=mask_path,
                    subset=item_subset,
                )
            )
    return pairs, invalid


def _as_2d(array: np.ndarray, label: str, identifier: str) -> np.ndarray:
    """Remove only singleton axes; never silently discard true 3D data."""
    squeezed = np.squeeze(array)
    if squeezed.ndim != 2:
        raise ValueError(
            f"{identifier}: {label} is not 2D after removing singleton axes "
            f"(shape {array.shape})."
        )
    return squeezed


def load_pair(pair: ImageMaskPair) -> tuple[np.ndarray, np.ndarray]:
    """Load one NIfTI pair, binarize its mask, and validate spatial shape."""
    try:
        image = _as_2d(
            np.asanyarray(nib.load(pair.image_path).dataobj), "image", pair.identifier
        )
        mask = _as_2d(
            np.asanyarray(nib.load(pair.mask_path).dataobj), "mask", pair.identifier
        )
    except (OSError, ValueError) as exc:
        raise ValueError(f"{pair.identifier}: could not load NIfTI pair: {exc}") from exc

    if image.shape != mask.shape:
        raise ValueError(
            f"{pair.identifier}: image shape {image.shape} does not match "
            f"mask shape {mask.shape}."
        )
    if image.size == 0:
        raise ValueError(f"{pair.identifier}: image is empty.")
    if not np.isfinite(image).any():
        raise ValueError(f"{pair.identifier}: image contains no finite values.")
    return np.asarray(image, dtype=np.float32), np.asarray(mask > 0, dtype=bool)


def validate_pairs(
    pairs: Iterable[ImageMaskPair],
) -> tuple[list[ImageMaskPair], list[dict[str, str]]]:
    """Load-check every pair and return valid pairs plus clear skip reasons."""
    valid: list[ImageMaskPair] = []
    invalid: list[dict[str, str]] = []
    for pair in pairs:
        try:
            load_pair(pair)
            valid.append(pair)
        except ValueError as exc:
            invalid.append({"identifier": pair.identifier, "reason": str(exc)})
    return valid, invalid

