from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.volume_dataset import (
    discover_nfbs_volumes,
    load_volume_pair,
    validate_volume_pairs,
)


def _save_subject(
    root: Path,
    identifier: str,
    image: np.ndarray,
    mask: np.ndarray,
    affine: np.ndarray | None = None,
) -> None:
    subject = root / identifier
    subject.mkdir(parents=True)
    transform = np.eye(4) if affine is None else affine
    nib.save(
        nib.Nifti1Image(image, transform),
        subject / f"sub-{identifier}_ses-NFB3_T1w.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(mask, transform),
        subject / f"sub-{identifier}_ses-NFB3_T1w_brainmask.nii.gz",
    )


def test_volume_discovery_is_deterministic_and_loads_spacing(tmp_path: Path) -> None:
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    for identifier in ("B0002", "A0001"):
        _save_subject(
            tmp_path,
            identifier,
            np.ones((5, 6, 7), dtype=np.float32),
            np.ones((5, 6, 7), dtype=np.uint8),
            affine,
        )
    pairs, invalid = discover_nfbs_volumes(tmp_path)
    assert invalid == []
    assert [pair.identifier for pair in pairs] == ["A0001", "B0002"]
    image, mask, spacing = load_volume_pair(pairs[0])
    assert image.shape == mask.shape == (5, 6, 7)
    assert mask.dtype == np.bool_
    assert spacing == (1.0, 2.0, 3.0)


def test_missing_mask_is_reported(tmp_path: Path) -> None:
    subject = tmp_path / "A0001"
    subject.mkdir()
    nib.save(
        nib.Nifti1Image(np.ones((3, 4, 5)), np.eye(4)),
        subject / "sub-A0001_ses-NFB3_T1w.nii.gz",
    )
    pairs, invalid = discover_nfbs_volumes(tmp_path)
    assert pairs == []
    assert invalid[0]["identifier"] == "A0001"
    assert "found 1 and 0" in invalid[0]["reason"]


def test_mismatched_volume_shapes_are_invalid(tmp_path: Path) -> None:
    _save_subject(
        tmp_path,
        "A0001",
        np.ones((3, 4, 5), dtype=np.float32),
        np.ones((3, 4, 4), dtype=np.uint8),
    )
    pairs, _ = discover_nfbs_volumes(tmp_path)
    with pytest.raises(ValueError, match="does not match"):
        load_volume_pair(pairs[0])
    valid, invalid = validate_volume_pairs(pairs)
    assert valid == []
    assert "does not match" in invalid[0]["reason"]


def test_2d_input_is_rejected(tmp_path: Path) -> None:
    _save_subject(
        tmp_path,
        "A0001",
        np.ones((3, 4), dtype=np.float32),
        np.ones((3, 4), dtype=np.uint8),
    )
    pairs, _ = discover_nfbs_volumes(tmp_path)
    with pytest.raises(ValueError, match="expected 3D"):
        load_volume_pair(pairs[0])
