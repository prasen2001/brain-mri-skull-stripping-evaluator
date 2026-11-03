from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _save_pair(root: Path, identifier: str, offset: int) -> None:
    image = np.zeros((32, 32, 1), dtype=np.float32)
    image[6 + offset : 26, 7:25, 0] = 0.6
    image[10:22, 10:22, 0] = 1.0
    mask = np.zeros((32, 32, 1), dtype=np.uint8)
    mask[7 + offset : 25, 8:24, 0] = 1
    scan = root / identifier
    scan.mkdir(parents=True)
    nib.save(nib.Nifti1Image(image, np.eye(4)), scan / "image.nii.gz")
    nib.save(nib.Nifti1Image(mask, np.eye(4)), scan / "mask.nii.gz")


def test_complete_cli_on_synthetic_nifti_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "dataset"
    identifiers = ["synthetic_t1_001", "synthetic_t1_002", "synthetic_t2_003"]
    for offset, identifier in enumerate(identifiers):
        _save_pair(data_dir, identifier, offset)
    (data_dir / "scans_test.txt").write_text("\n".join(identifiers), encoding="utf-8")
    output_dir = tmp_path / "results"
    project_root = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    command = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--data-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--subset",
        "test",
        "--seed",
        "42",
        "--bootstrap-samples",
        "20",
    ]
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    required = (
        "per_image_metrics.csv",
        "summary_metrics.csv",
        "source_summary_metrics.csv",
        "modality_summary_metrics.csv",
        "bootstrap_confidence_intervals.csv",
        "paired_method_comparisons.csv",
        "quality_summary.csv",
        "failure_review.md",
        "failure_analysis.csv",
        "failure_category_summary.csv",
        "method_comparison.png",
        "failure_category_distribution.png",
        "modality_dice_comparison.png",
        "paired_dice_improvement.png",
    )
    for filename in required:
        assert (output_dir / filename).is_file(), filename
    metrics = pd.read_csv(output_dir / "per_image_metrics.csv")
    assert len(metrics) == len(identifiers) * 3
    assert {
        "hd95_mm",
        "assd_mm",
        "modality",
        "quality_warnings",
        "primary_failure_category",
        "failure_reasons",
    } <= set(metrics.columns)
