#!/usr/bin/env python3
"""Optional official-data download helper. Does not regenerate Waterbirds."""
import argparse
import hashlib
import shutil
import subprocess
import tarfile
from pathlib import Path


def fetch(url, target, expected_md5=None):
    if not target.exists():
        partial = target.with_suffix(target.suffix + ".part")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-",
                        "--output", str(partial), url], check=True)
        partial.replace(target)
    if expected_md5:
        digest = hashlib.md5()
        with target.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_md5:
            raise ValueError(f"Archive checksum mismatch; remove and re-download {target}")


def unpack(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")


def main():
    parser = argparse.ArgumentParser(description="Download official Waterbirds-95 and CUB segmentation masks")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--waterbirds-archive", help="Use an already downloaded official archive")
    parser.add_argument("--segmentation-archive", help="Use an already downloaded official segmentation archive")
    args = parser.parse_args()
    if shutil.which("curl") is None and not (args.waterbirds_archive and args.segmentation_archive):
        raise RuntimeError("Install curl or download the archives manually; see README.md")
    root = Path(args.data_dir).resolve()
    downloads = root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    waterbirds = Path(args.waterbirds_archive) if args.waterbirds_archive else downloads / "waterbird_complete95_forest2water2.tar.gz"
    masks = Path(args.segmentation_archive) if args.segmentation_archive else downloads / "segmentations.tgz"
    if args.waterbirds_archive and not waterbirds.is_file():
        raise FileNotFoundError(waterbirds)
    if args.segmentation_archive and not masks.is_file():
        raise FileNotFoundError(masks)
    fetch("https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz", waterbirds)
    fetch("https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1", masks,
          "4d47ba1228eae64f2fa547c47bc65255")
    unpack(waterbirds, root)
    unpack(masks, root / "CUB_200_2011")
    expected = root / "waterbird_complete95_forest2water2" / "metadata.csv"
    if not expected.exists():
        raise FileNotFoundError(f"Unexpected archive layout. Locate metadata.csv under {root} and set data_root accordingly.")
    if not (root / "CUB_200_2011" / "segmentations").is_dir():
        raise FileNotFoundError("Unexpected segmentation archive layout; set seg_root to the directory containing species folders")
    print(f"data_root: {expected.parent}\nseg_root: {root / 'CUB_200_2011' / 'segmentations'}")


if __name__ == "__main__":
    main()
