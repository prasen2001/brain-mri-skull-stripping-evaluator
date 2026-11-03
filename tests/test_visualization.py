from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from src.dataset import ImageMaskPair
from src.segmentation import MorphologyConfig
from src.visualization import create_all_visualizations


def _save_pair(root: Path, identifier: str, offset: int) -> ImageMaskPair:
    image = np.zeros((40, 44, 1), dtype=np.float32)
    image[8 + offset : 32, 9:35, 0] = 0.7
    image[14:26, 15:29, 0] = 1.0
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[9 + offset : 31, 10:34, 0] = 1
    scan = root / identifier
    scan.mkdir(parents=True)
    image_path = scan / "image.nii.gz"
    mask_path = scan / "mask.nii.gz"
    nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
    nib.save(nib.Nifti1Image(mask, np.eye(4)), mask_path)
    return ImageMaskPair(identifier, image_path, mask_path, "test")


def _metrics(identifiers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    method_offsets = {
        "otsu_threshold": 0.00,
        "otsu_morphology": 0.04,
        "central_component": 0.08,
    }
    for scan_index, identifier in enumerate(identifiers):
        for method, method_offset in method_offsets.items():
            dice = 0.62 + 0.08 * scan_index + method_offset
            warned = method == "central_component" and scan_index == 0
            rows.append(
                {
                    "identifier": identifier,
                    "dataset_source": "SYNTHETIC",
                    "modality": "T1",
                    "method": method,
                    "dice": dice,
                    "iou": dice / (2 - dice),
                    "precision": min(1.0, dice + 0.05),
                    "sensitivity": max(0.0, dice - 0.05),
                    "specificity": min(1.0, dice + 0.10),
                    "processing_time_seconds": 0.001 + method_offset / 10,
                    "surface_dice": max(0.0, dice - 0.30),
                    "hd95_mm": 20.0 * (1 - dice),
                    "assd_mm": 5.0 * (1 - dice),
                    "has_quality_warning": warned,
                    "quality_warning_count": int(warned),
                    "quality_warnings": "methods_disagree" if warned else "none",
                    "method_disagreement": 0.5 if warned else 0.1,
                    "predicted_area_fraction": 0.2,
                    "component_count": 1,
                    "primary_failure_category": (
                        "under_segmentation" if warned else "acceptable"
                    ),
                    "failure_reasons": (
                        "under_segmentation" if warned else "acceptable"
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_create_all_visualizations_writes_complete_artifact_set(tmp_path: Path) -> None:
    identifiers = ["synthetic_t1_001", "synthetic_t1_002", "synthetic_t1_003"]
    pairs = {
        identifier: _save_pair(tmp_path / "data", identifier, index)
        for index, identifier in enumerate(identifiers)
    }
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    best_method = create_all_visualizations(
        _metrics(identifiers),
        pairs,
        output_dir,
        MorphologyConfig(),
        1.0,
        99.0,
    )
    assert best_method == "central_component"
    expected = {
        "method_comparison.png",
        "dice_distribution.png",
        "iou_distribution.png",
        "processing_time_comparison.png",
        "source_dice_comparison.png",
        "modality_dice_comparison.png",
        "paired_dice_improvement.png",
        "quality_warning_rates.png",
        "boundary_metric_comparison.png",
        "failure_category_distribution.png",
        "best_cases.png",
        "worst_cases.png",
        "failure_review.md",
    }
    assert expected <= {path.name for path in output_dir.iterdir() if path.is_file()}
    assert len(list((output_dir / "overlays").glob("*.png"))) == 6
    assert len(list((output_dir / "failure_review_cases").glob("*.png"))) == 1
    report = (output_dir / "failure_review.md").read_text(encoding="utf-8")
    assert "Retrospective category" in report
    assert "synthetic_t1_001" in report
