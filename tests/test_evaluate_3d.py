from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def _save_subject(root: Path, identifier: str, offset: int) -> None:
    image = np.zeros((28, 30, 32), dtype=np.float32)
    image[5 + offset : 23, 6:24, 7:25] = 0.7
    image[10:20, 11:21, 12:22] = 1.0
    mask = np.zeros_like(image, dtype=np.uint8)
    mask[6 + offset : 22, 7:23, 8:24] = 1
    subject = root / identifier
    subject.mkdir(parents=True)
    nib.save(
        nib.Nifti1Image(image, np.eye(4)),
        subject / f"sub-{identifier}_ses-NFB3_T1w.nii.gz",
    )
    nib.save(
        nib.Nifti1Image(mask, np.eye(4)),
        subject / f"sub-{identifier}_ses-NFB3_T1w_brainmask.nii.gz",
    )


def test_complete_3d_cli_on_synthetic_volumes(tmp_path: Path) -> None:
    data_dir = tmp_path / "volumes"
    for index, identifier in enumerate(("A0001", "A0002", "A0003")):
        _save_subject(data_dir, identifier, index)
    output_dir = tmp_path / "results"
    environment = os.environ.copy()
    environment["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluate_3d",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
            "--bootstrap-samples",
            "20",
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    expected = {
        "per_volume_metrics.csv",
        "summary_metrics.csv",
        "bootstrap_confidence_intervals.csv",
        "paired_method_comparisons.csv",
        "failure_category_summary.csv",
        "run_config.json",
        "method_comparison_3d.png",
        "dice_distribution_3d.png",
    }
    assert expected <= {path.name for path in output_dir.iterdir() if path.is_file()}
    metrics = pd.read_csv(output_dir / "per_volume_metrics.csv")
    assert len(metrics) == 9
    assert metrics["volume_shape"].eq("28x30x32").all()
    assert {"hd95_voxels", "hd95_mm", "primary_failure_category"} <= set(
        metrics.columns
    )
    run_config = json.loads((output_dir / "run_config.json").read_text())
    expected_best_method = metrics.groupby("method")["dice"].mean().idxmax()
    assert run_config["best_method_by_mean_dice"] == expected_best_method
    assert len(list((output_dir / "orthogonal_reviews").glob("*.png"))) == 6
