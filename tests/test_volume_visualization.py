from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from matplotlib.axes import Axes

from src.volume_dataset import VolumePair
from src.volume_segmentation import (
    SPATIAL_PRIOR,
    SpatialPriorConfig,
    VolumeMorphologyConfig,
)
from src.volume_visualization import create_orthogonal_review


def _save_volume_pair(root: Path) -> VolumePair:
    image = np.zeros((32, 34, 36), dtype=np.float32)
    image[7:25, 7:27, 10:31] = 0.8
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[8:24, 8:26, 11:30] = 1
    image_path = root / "image.nii.gz"
    mask_path = root / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    return VolumePair("synthetic", image_path, mask_path)


def test_spatial_review_draws_prior_boundary(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pair = _save_volume_pair(tmp_path)
    original_contour = Axes.contour
    prior_contours: list[dict[str, Any]] = []

    def record_contour(self: Axes, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("colors") == ["#ffd92f"]:
            prior_contours.append(kwargs)
        return original_contour(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "contour", record_contour)
    output = tmp_path / "review.png"

    create_orthogonal_review(
        pair,
        SPATIAL_PRIOR,
        0.75,
        output,
        VolumeMorphologyConfig(),
        SpatialPriorConfig(),
        1.0,
        99.0,
    )

    assert output.is_file()
    assert len(prior_contours) == 3
    assert all(call["linestyles"] == ["--"] for call in prior_contours)
