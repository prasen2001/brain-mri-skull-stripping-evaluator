"""Download, verify, and safely extract the official SynthStrip 2D v1.5 archive."""

from __future__ import annotations

import argparse
import hashlib
import tarfile
from collections.abc import Iterator
from pathlib import Path
from urllib.request import urlopen

URL = "https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/data/synthstrip_data_v1.5_2d.tar"
SHA256 = "69b3c3d9e37ef63b800137e25386f0cfb6851803916900923af1cedd4e0d55f4"


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_members(
    archive: tarfile.TarFile, destination: Path
) -> Iterator[tarfile.TarInfo]:
    destination = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if destination not in target.parents and target != destination:
            raise RuntimeError(f"Unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Archive links are not allowed: {member.name}")
        yield member


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.raw_dir / "synthstrip_data_v1.5_2d.tar"

    if not archive_path.exists() or checksum(archive_path) != SHA256:
        print(f"Downloading {URL}")
        with urlopen(URL) as response, archive_path.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    actual = checksum(archive_path)
    if actual != SHA256:
        raise RuntimeError(f"Checksum mismatch: expected {SHA256}, received {actual}")
    if not tarfile.is_tarfile(archive_path):
        raise RuntimeError(f"Downloaded file is not a valid tar archive: {archive_path}")

    with tarfile.open(archive_path) as archive:
        archive.extractall(
            args.processed_dir,
            members=safe_members(archive, args.processed_dir),
            filter="data",
        )
    print(f"Verified SHA-256 and extracted to {args.processed_dir / 'synthstrip_data_v1.5_2d'}")


if __name__ == "__main__":
    main()

