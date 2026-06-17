#!/usr/bin/env python3
"""Multi-GPU VAE-latent encoder for REPA datasets.

A faster, multi-GPU drop-in replacement for
``REPA/preprocessing/dataset_tools.py encode`` (which is single-GPU,
batch-size-1, and forbids distributed runs).

It reads an image folder produced by datasets/export_celeba.py (or REPA's
``convert``) and writes SD-VAE mean/std latents in the exact layout the REPA
``CustomDataset`` expects::

    <dest>/00000/img-mean-std-00000000.npy   # one (8, 32, 32) float32 array
    <dest>/dataset.json                       # {"labels": [[npy_path, class], ...]}

Images are sharded across the chosen GPUs; each GPU runs its own VAE in batches.

Label keying: source ``dataset.json`` keys are matched by image *basename*, so
both flat ("000000.jpg") and nested ("images/000000.jpg") label keys work. This
fixes the mismatch that otherwise silently drops all CelebA class labels.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Import REPA's StabilityVAEEncoder (pulls in torch_utils/dnnlib from there).
_PREPROCESSING = Path(__file__).resolve().parent.parent / "REPA" / "preprocessing"
if str(_PREPROCESSING) not in sys.path:
    sys.path.insert(0, str(_PREPROCESSING))
from encoders import StabilityVAEEncoder  # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(source: Path) -> list[Path]:
    """Return image paths under ``source``, sorted to match REPA's ordering."""
    files = [
        p for p in source.rglob("*")
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    ]
    # Sort by path string relative to source -> same order CustomDataset uses.
    return sorted(files, key=lambda p: str(p.relative_to(source)).replace("\\", "/"))


def load_label_map(source: Path) -> dict | None:
    """Load source dataset.json labels, keyed by image basename."""
    meta = source / "dataset.json"
    if not meta.is_file():
        return None
    data = json.loads(meta.read_text()).get("labels")
    if not data:
        return None
    return {os.path.basename(k): v for k, v in data}


def arch_fname(idx: int) -> str:
    """REPA feature filename for global index ``idx``."""
    s = f"{idx:08d}"
    return f"{s[:5]}/img-mean-std-{s}.npy"


class ShardDataset(Dataset):
    """Loads the rank's stride of images as uint8 CHW tensors."""

    def __init__(self, files: list[Path], indices: list[int]):
        self.files = files
        self.indices = indices  # global image indices this rank owns

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gidx = self.indices[i]
        img = PIL.Image.open(self.files[gidx]).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)  # HWC
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
        return gidx, tensor


def worker(rank: int, args, files, gpu_ids):
    gpu = gpu_ids[rank]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)

    indices = list(range(rank, len(files), len(gpu_ids)))  # strided shard
    encoder = StabilityVAEEncoder(vae_name=args.model_url, batch_size=args.batch_size)
    encoder.init(device)

    loader = DataLoader(
        ShardDataset(files, indices),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    label_map = load_label_map(Path(args.source))
    dest = Path(args.dest)
    local_labels: list[list] = []  # [[global_idx, npy_path, label], ...]

    show = rank == 0
    for gidx_batch, img_batch in tqdm(
        loader, desc=f"[gpu{gpu}] encode", disable=not show, total=len(loader)
    ):
        img_batch = img_batch.to(device, non_blocking=True)
        mean_std = encoder.encode_pixels(img_batch).to(torch.float32).cpu().numpy()
        for j, gidx in enumerate(gidx_batch.tolist()):
            rel = arch_fname(gidx)
            out_path = dest / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, mean_std[j])
            label = None
            if label_map is not None:
                label = label_map.get(os.path.basename(str(files[gidx])))
            local_labels.append([gidx, rel, label])

    # Each rank writes its label slice; main process merges in index order.
    (dest / f"_labels_rank{rank}.json").write_text(json.dumps(local_labels))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True,
                        help="Image folder (contains images/ and dataset.json).")
    parser.add_argument("--dest", required=True, help="Output dir for vae-sd latents.")
    parser.add_argument("--model-url", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated GPU ids, e.g. 0,1,2,3.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required for VAE encoding.")

    gpu_ids = [int(g) for g in args.gpus.split(",") if g != ""]
    source = Path(args.source)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    files = list_images(source)
    if not files:
        sys.exit(f"No images found under {source}")
    print(f"Encoding {len(files)} images on GPUs {gpu_ids} "
          f"(batch-size={args.batch_size}) -> {dest}")

    # Warm the HF cache once so workers don't race to download the VAE.
    print("Fetching VAE weights ...")
    StabilityVAEEncoder(vae_name=args.model_url, batch_size=1).init(torch.device("cpu"))

    if len(gpu_ids) == 1:
        worker(0, args, files, gpu_ids)
    else:
        mp.spawn(worker, args=(args, files, gpu_ids), nprocs=len(gpu_ids), join=True)

    # Merge per-rank label slices into the final dataset.json (sorted by index).
    merged: list[list] = []
    for rank in range(len(gpu_ids)):
        part = dest / f"_labels_rank{rank}.json"
        merged.extend(json.loads(part.read_text()))
        part.unlink()
    merged.sort(key=lambda r: r[0])  # by global image index
    labels = [[rel, lbl] for _idx, rel, lbl in merged]
    has_labels = all(lbl is not None for _rel, lbl in labels)
    metadata = {"labels": labels if has_labels else None}
    (dest / "dataset.json").write_text(json.dumps(metadata))
    print(f"Wrote {len(labels)} latents + dataset.json "
          f"(labels {'present' if has_labels else 'MISSING'}) to {dest}")


if __name__ == "__main__":
    main()
