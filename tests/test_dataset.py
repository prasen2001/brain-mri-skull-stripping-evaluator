from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from src.dataset import discover_pairs, load_pair, load_pair_with_spacing, validate_pairs


def _save_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)


def _make_pair(root: Path, identifier: str, shape: tuple[int, ...] = (8, 9, 1)) -> None:
    _save_nifti(root / identifier / "image.nii.gz", np.ones(shape, dtype=np.float32))
    _save_nifti(root / identifier / "mask.nii.gz", np.ones(shape, dtype=np.uint8))


def test_pairing_and_deterministic_order(tmp_path: Path) -> None:
    _make_pair(tmp_path, "scan_b")
    _make_pair(tmp_path, "scan_a")
    (tmp_path / "scans_valid.txt").write_text("scan_b\n", encoding="utf-8")
    (tmp_path / "scans_test.txt").write_text("scan_a\n", encoding="utf-8")

    all_pairs, invalid = discover_pairs(tmp_path, "all")
    assert invalid == []
    assert [pair.identifier for pair in all_pairs] == ["scan_a", "scan_b"]
    assert [pair.subset for pair in all_pairs] == ["test", "development"]
    test_pairs, _ = discover_pairs(tmp_path, "test")
    assert [pair.identifier for pair in test_pairs] == ["scan_a"]


def test_missing_mask_is_reported(tmp_path: Path) -> None:
    _save_nifti(tmp_path / "scan_missing" / "image.nii.gz", np.ones((4, 4, 1)))
    pairs, invalid = discover_pairs(tmp_path, "all")
    assert pairs == []
    assert invalid[0]["identifier"] == "scan_missing"
    assert "mask.nii.gz" in invalid[0]["reason"]


def test_mismatched_shapes_are_rejected(tmp_path: Path) -> None:
    _save_nifti(tmp_path / "scan_bad" / "image.nii.gz", np.ones((5, 6, 1)))
    _save_nifti(tmp_path / "scan_bad" / "mask.nii.gz", np.ones((4, 6, 1)))
    pairs, _ = discover_pairs(tmp_path, "all")
    with pytest.raises(ValueError, match="does not match"):
        load_pair(pairs[0])
    valid, invalid = validate_pairs(pairs)
    assert valid == []
    assert "does not match" in invalid[0]["reason"]


def test_load_pair_binarizes_mask(tmp_path: Path) -> None:
    image = np.arange(12, dtype=np.float32).reshape(3, 4, 1)
    mask = np.array([0, 2] * 6, dtype=np.uint8).reshape(3, 4, 1)
    _save_nifti(tmp_path / "scan" / "image.nii.gz", image)
    _save_nifti(tmp_path / "scan" / "mask.nii.gz", mask)
    pairs, _ = discover_pairs(tmp_path, "all")
    loaded_image, loaded_mask = load_pair(pairs[0])
    assert loaded_image.shape == (3, 4)
    assert loaded_mask.dtype == np.bool_
    assert int(loaded_mask.sum()) == 6


def test_load_pair_reports_pixel_spacing(tmp_path: Path) -> None:
    image = nib.Nifti1Image(np.ones((3, 4, 1), dtype=np.float32), np.eye(4))
    image.header.set_zooms((2.0, 0.5, 3.0))
    mask = nib.Nifti1Image(np.ones((3, 4, 1), dtype=np.uint8), np.eye(4))
    scan = tmp_path / "scan"
    scan.mkdir()
    nib.save(image, scan / "image.nii.gz")
    nib.save(mask, scan / "mask.nii.gz")
    pairs, _ = discover_pairs(tmp_path, "all")
    _, _, spacing = load_pair_with_spacing(pairs[0])
    assert spacing == (2.0, 0.5)

