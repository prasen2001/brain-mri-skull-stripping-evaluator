from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from scripts.prepare_nfbs_external import prepare_nfbs_slices
from src.dataset import discover_pairs, load_pair_with_spacing


def test_prepare_nfbs_uses_geometry_selected_center_slice(tmp_path: Path) -> None:
    source = tmp_path / "source"
    subject = source / "A00000001"
    subject.mkdir(parents=True)
    image = np.zeros((5, 6, 7), dtype=np.float32)
    image[2, 1:5, 1:6] = 10
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[2, 2:5, 2:6] = 1
    affine = np.diag([1.0, 2.0, 3.0, 1.0])
    nib.save(
        nib.Nifti1Image(image, affine),
        subject / "sub-A00000001_ses-NFB3_T1w.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(mask, affine),
        subject / "sub-A00000001_ses-NFB3_T1w_brainmask.nii.gz",
    )
    output = tmp_path / "prepared"
    identifiers = prepare_nfbs_slices(source, output)
    assert identifiers == ["nfbs_t1_a00000001"]
    pairs, invalid = discover_pairs(output, "test")
    assert invalid == []
    loaded_image, loaded_mask, spacing = load_pair_with_spacing(pairs[0])
    assert loaded_image.shape == (6, 7)
    assert loaded_mask.any()
    assert spacing == (2.0, 3.0)
    assert "image geometry only" in (output / "EXTERNAL_DATASET_INFO.txt").read_text()
