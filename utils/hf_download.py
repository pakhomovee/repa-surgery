#!/usr/bin/env python3
"""Download the CelebA dataset from the Hugging Face Hub and unzip it into data/.

The archive ``celeba.zip`` (repo ``pakhomovee/celeba``) extracts to a top-level
``celeba/`` folder holding ``img_align_celeba/*.jpg`` and
``annotations/list_attr_celeba.txt`` -- exactly the layout that
``datasets/celeba.py:CelebADataset`` expects at its ``root_dir``. So we extract
into ``<repo>/data/`` to produce ``<repo>/data/celeba/``.
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"

REPO_ID = "pakhomovee/celeba"
FILENAME = "celeba.zip"


def extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract ``zip_path`` into ``dest_dir``, skipping macOS ``__MACOSX`` junk."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.infolist() if not m.filename.startswith("__MACOSX/")]
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, dest_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory to extract into (default: <repo>/data). "
        "The archive creates a 'celeba/' subfolder here.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if data/celeba/img_align_celeba already exists.",
    )
    args = parser.parse_args()

    celeba_dir = args.data_dir / "celeba"
    images_dir = celeba_dir / "img_align_celeba"
    if images_dir.is_dir() and not args.force:
        print(f"CelebA already present at {celeba_dir}; skipping download. "
              "Use --force to re-extract.")
        return

    print(f"Downloading {FILENAME} from {REPO_ID} ...")
    zip_path = Path(
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    )
    print(f"Downloaded to {zip_path}")

    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting into {args.data_dir} ...")
    extract_zip(zip_path, args.data_dir)
    print(f"Done. CelebA available at {celeba_dir}")


if __name__ == "__main__":
    main()
