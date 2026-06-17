#!/usr/bin/env python3
"""
Export CelebA images from the CelebADataset to REPA-compatible image folders.

The output structure expected by REPA preprocessing (dataset_tools.py) is:
    <output_dir>/
        images/
            000000.jpg
            000001.jpg
            ...
        dataset.json   <- {"labels": [["000000.jpg", <class_id>], ...]}

After running this script, encode the images into VAE latents with:
    cd REPA/preprocessing
    python dataset_tools.py encode \\
        --source <output_dir> \\
        --dest <output_dir>/vae-sd \\
        --model-url stabilityai/sd-vae-ft-mse
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from torchvision import transforms
from tqdm import tqdm

# Allow importing celeba.py from the same scripts/ directory
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from celeba import CelebADataset, make_get_class  # noqa: E402

# ---------------------------------------------------------------------------
# Default selected attributes used for class conditioning.
# With 4 binary attributes we get 2^4 = 16 classes, which is a reasonable
# number for class-conditional generation on CelebA.
# ---------------------------------------------------------------------------
DEFAULT_ATTRS = ["Male", "Smiling", "Young", "Attractive"]


def export_celeba(
    root_dir: Path,
    output_dir: Path,
    selected_attrs: list[str],
    resolution: int,
) -> None:
    """Export CelebA images and labels into the REPA image-folder format."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.CenterCrop(178),   # CelebA standard centre-crop
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
    ])

    dataset = CelebADataset(root_dir=str(root_dir), transform=transform)

    if dataset.header is None:
        raise RuntimeError(
            "CelebADataset did not load an attribute header. "
            "Make sure list_attr_celeba.txt is present."
        )

    # Validate that every requested attribute is present in the header
    missing = [a for a in selected_attrs if a not in dataset.header]
    if missing:
        raise ValueError(
            f"The following attributes are not in the CelebA header: {missing}\n"
            f"Available attributes: {dataset.header}"
        )

    get_class = make_get_class(dataset.header, selected_attrs)

    labels: list[list] = []
    for idx in tqdm(range(len(dataset)), desc="Exporting CelebA"):
        img, info_dict = dataset[idx]
        rel_path = f"{idx:06d}.jpg"
        out_path = images_dir / rel_path
        if not out_path.exists():
            img.save(out_path, quality=95)

        # get_class expects a batch dimension on attributes; add it here
        import torch
        attrs_batched = info_dict["attributes"].unsqueeze(0)  # (1, 40)
        info_batched = {"attributes": attrs_batched}
        class_id = int(get_class(info_batched)[0])
        labels.append([rel_path, class_id])

    dataset_json_path = output_dir / "dataset.json"
    with dataset_json_path.open("w") as f:
        json.dump({"labels": labels}, f)

    num_classes = 2 ** len(selected_attrs)
    print(f"Exported {len(labels)} images to {output_dir}")
    print(f"Selected attributes: {selected_attrs}")
    print(f"Number of classes:   {num_classes}  (2^{len(selected_attrs)})")
    print(f"dataset.json written to {dataset_json_path}")
    print()
    print("Next step – encode VAE latents:")
    print(
        f"  cd REPA/preprocessing\n"
        f"  python dataset_tools.py encode \\\n"
        f"      --source {output_dir} \\\n"
        f"      --dest {output_dir}/vae-sd \\\n"
        f"      --model-url stabilityai/sd-vae-ft-mse"
    )
    print()
    print(f"Then train with --data-dir={output_dir} --num-classes={num_classes}")


def main() -> None:
    _repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Export CelebA to REPA image-folder format."
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "celeba",
        help="Root directory of the CelebA dataset (will auto-download if absent).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("REPA_ROOT", _repo_root)) / "data" / "celeba256",
        help="Destination directory for the exported REPA-compatible dataset.",
    )
    parser.add_argument(
        "--selected-attrs",
        nargs="+",
        default=DEFAULT_ATTRS,
        metavar="ATTR",
        help=(
            "CelebA binary attributes to use for class conditioning "
            f"(default: {DEFAULT_ATTRS}). "
            "The number of classes will be 2^len(selected_attrs)."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Target image resolution after centre-crop and resize (default: 256).",
    )
    args = parser.parse_args()

    export_celeba(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        selected_attrs=args.selected_attrs,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()