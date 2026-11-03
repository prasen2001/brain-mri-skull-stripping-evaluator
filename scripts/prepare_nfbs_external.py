"""Download, safely extract, and prepare the NFBS external 2D benchmark.

The source dataset contains 125 three-dimensional T1 volumes. This script
selects the center sagittal slice after canonical orientation, without using the
reference mask to choose a favorable slice. Original files remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import cast

import nibabel as nib
import numpy as np

NFBS_URL = (
    "https://fcp-indi.s3.amazonaws.com/data/Projects/RocklandSample/"
    "NFBS_Dataset.tar.gz"
)
NFBS_SHA256 = "fd616b9ea21aad3d0d4cd05a25a71f4a3c251140df38188e58e499254440d2cc"
NFBS_SIZE_BYTES = 1_751_464_473


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_archive(url: str, destination: Path) -> None:
    """Download the archive if it is not already present."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    temporary = destination.with_suffix(destination.suffix + ".partial")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_archive(path: Path) -> str:
    """Verify file size, pinned checksum, and tar readability."""
    if not path.is_file():
        raise FileNotFoundError(f"NFBS archive does not exist: {path}")
    if path.stat().st_size != NFBS_SIZE_BYTES:
        raise ValueError(
            f"NFBS archive size is {path.stat().st_size}, expected {NFBS_SIZE_BYTES}."
        )
    checksum = _sha256(path)
    if checksum != NFBS_SHA256:
        raise ValueError(f"NFBS checksum mismatch: received {checksum}.")
    if not tarfile.is_tarfile(path):
        raise ValueError(f"Downloaded NFBS file is not a readable tar archive: {path}")
    return checksum


def safe_extract(archive_path: Path, destination: Path) -> Path:
    """Extract regular files and directories while rejecting unsafe members."""
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if destination_root not in target.parents and target != destination_root:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.issym() or member.islnk() or not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported archive member: {member.name}")
        archive.extractall(destination, members=members, filter="data")
    extracted = destination / "NFBS_Dataset"
    if not extracted.is_dir():
        raise ValueError("NFBS_Dataset directory was absent after extraction.")
    return extracted


def _source_paths(subject_dir: Path) -> tuple[Path, Path]:
    identifier = subject_dir.name
    images = sorted(subject_dir.glob("*_T1w.nii.gz"))
    masks = sorted(subject_dir.glob("*_T1w_brainmask.nii.gz"))
    if len(images) != 1 or len(masks) != 1:
        raise ValueError(
            f"{identifier}: expected one full-head T1 and one brain mask; "
            f"found {len(images)} and {len(masks)}."
        )
    return images[0], masks[0]


def prepare_nfbs_slices(source_dir: Path, output_dir: Path) -> list[str]:
    """Create one canonical mid-sagittal image-mask pair per NFBS subject."""
    subject_dirs = sorted(path for path in source_dir.iterdir() if path.is_dir())
    if not subject_dirs:
        raise ValueError(f"No NFBS subject directories found in {source_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    identifiers: list[str] = []

    for subject_dir in subject_dirs:
        image_path, mask_path = _source_paths(subject_dir)
        image_nifti = nib.as_closest_canonical(cast(nib.Nifti1Image, nib.load(image_path)))
        mask_nifti = nib.as_closest_canonical(cast(nib.Nifti1Image, nib.load(mask_path)))
        image = np.asanyarray(image_nifti.dataobj)
        mask = np.asanyarray(mask_nifti.dataobj)
        if image.ndim != 3 or mask.ndim != 3 or image.shape != mask.shape:
            raise ValueError(
                f"{subject_dir.name}: incompatible image/mask shapes "
                f"{image.shape} and {mask.shape}."
            )
        sagittal_index = image.shape[0] // 2
        image_slice = np.asarray(image[sagittal_index, :, :], dtype=np.float32)
        mask_slice = np.asarray(mask[sagittal_index, :, :] > 0, dtype=np.uint8)
        if not np.isfinite(image_slice).any() or not mask_slice.any():
            raise ValueError(f"{subject_dir.name}: unusable canonical center slice.")

        identifier = f"nfbs_t1_{subject_dir.name.lower()}"
        scan_dir = output_dir / identifier
        scan_dir.mkdir(parents=True, exist_ok=True)
        zooms = image_nifti.header.get_zooms()
        affine = np.diag([float(zooms[1]), float(zooms[2]), 1.0, 1.0])
        nib.save(
            nib.Nifti1Image(image_slice[:, :, None], affine),
            scan_dir / "image.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(mask_slice[:, :, None], affine),
            scan_dir / "mask.nii.gz",
        )
        identifiers.append(identifier)

    (output_dir / "scans_test.txt").write_text(
        "\n".join(identifiers) + "\n", encoding="utf-8"
    )
    (output_dir / "EXTERNAL_DATASET_INFO.txt").write_text(
        "\n".join(
            (
                "Dataset: Neurofeedback Skull-stripped (NFBS) repository",
                f"Source: {NFBS_URL}",
                f"Archive SHA-256: {NFBS_SHA256}",
                f"Prepared subjects: {len(identifiers)}",
                "Preparation: closest-canonical orientation, center sagittal slice",
                "Slice selection uses image geometry only, never the reference mask.",
                "Citation: Puccio et al., GigaScience 2016;5:45.",
                "DOI: 10.1186/s13742-016-0150-5",
                "Use: separate external evaluation; never merge with SynthStrip results.",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return identifiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=Path("data/raw/NFBS_Dataset.tar.gz")
    )
    parser.add_argument(
        "--extract-dir", type=Path, default=Path("data/raw/nfbs_extracted")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed/nfbs_2d_external")
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    if not args.skip_download:
        download_archive(NFBS_URL, args.archive)
    checksum = verify_archive(args.archive)
    source = safe_extract(args.archive, args.extract_dir)
    identifiers = prepare_nfbs_slices(source, args.output_dir)
    print(f"Verified NFBS archive SHA-256: {checksum}")
    print(f"Prepared {len(identifiers)} external 2D pairs in {args.output_dir}")


if __name__ == "__main__":
    main()
