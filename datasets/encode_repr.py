#!/usr/bin/env python3
"""Multi-GPU precompute of frozen-encoder (DINOv2) representations for REPA.

REPA aligns the DiT's noised-input hidden state with the encoder features of the
*clean* image. Those targets do not depend on the diffusion noise or the training
step, so they can be computed once and reused -- removing the per-step encoder
forward (and the raw-image read) from repa / haste / repa-sigma training.

Output mirrors the VAE-latent layout so REPA's CustomDataset can zip by index:
    <dest>/00000/img-repr-00000000.npy   # one (T, D) float16 array per image
    <dest>/meta.json                      # {"enc_type", "tokens", "dim", "count"}

Uses REPA's own load_encoders + preprocessing, so the features are identical
(fp16 rounding aside) to what training would compute on the fly.

Storage is large: DINOv2-B is 256x768 per image ~= 393 KB (fp16). Make sure the
destination disk has room (e.g. ~51 GB for ImageNet-100, ~80 GB for CelebA).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from tqdm import tqdm

_REPA_DIR = Path(__file__).resolve().parent.parent / "REPA"
if str(_REPA_DIR) not in sys.path:
    sys.path.insert(0, str(_REPA_DIR))
from utils import load_encoders  # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type):
    """Copied from REPA/train.py to keep this script decoupled (and identical)."""
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = x / 255.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x


def encoder_patch_tokens(encoder, encoder_type, x):
    """Match REPA's per-encoder token extraction (train.py loop)."""
    z = encoder.forward_features(x)
    if 'mocov3' in encoder_type:
        z = z[:, 1:]
    if 'dinov2' in encoder_type:
        z = z['x_norm_patchtokens']
    return z


def list_images(source: Path) -> list[Path]:
    files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_EXTS]
    return sorted(files, key=lambda p: str(p.relative_to(source)).replace("\\", "/"))


def arch_fname(idx: int) -> str:
    s = f"{idx:08d}"
    return f"{s[:5]}/img-repr-{s}.npy"


class ShardDataset(Dataset):
    def __init__(self, files, indices):
        self.files = files
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        gidx = self.indices[i]
        arr = np.asarray(PIL.Image.open(self.files[gidx]).convert("RGB"), dtype=np.uint8)
        return gidx, torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def worker(rank, args, files, gpu_ids):
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)
    encoders, encoder_types, _ = load_encoders(args.enc_type, device, args.resolution)
    encoder, etype = encoders[0], encoder_types[0]

    indices = list(range(rank, len(files), len(gpu_ids)))
    loader = DataLoader(ShardDataset(files, indices), batch_size=args.batch_size,
                        num_workers=args.num_workers, pin_memory=True)
    dest = Path(args.dest)
    show = rank == 0
    with torch.no_grad():
        for gidx_batch, imgs in tqdm(loader, desc=f"[gpu{gpu_ids[rank]}] repr",
                                     disable=not show, total=len(loader)):
            imgs = imgs.to(device, non_blocking=True).float()
            z = encoder_patch_tokens(encoder, etype, preprocess_raw_image(imgs, etype))
            z = z.to(torch.float16).cpu().numpy()
            for j, gidx in enumerate(gidx_batch.tolist()):
                out = dest / arch_fname(gidx)
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, z[j])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Image folder (e.g. <data>/images).")
    p.add_argument("--dest", required=True, help="Output dir for representations.")
    p.add_argument("--enc-type", default="dinov2-vit-b")
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--gpus", default="0")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA is required.")
    gpu_ids = [int(g) for g in args.gpus.split(",") if g != ""]
    files = list_images(Path(args.source))
    if not files:
        sys.exit(f"No images found under {args.source}")
    Path(args.dest).mkdir(parents=True, exist_ok=True)
    print(f"Encoding {len(files)} images with {args.enc_type} on GPUs {gpu_ids} -> {args.dest}")

    if len(gpu_ids) == 1:
        worker(0, args, files, gpu_ids)
    else:
        mp.spawn(worker, args=(args, files, gpu_ids), nprocs=len(gpu_ids), join=True)

    # Record token/dim shape so training can size the projectors without the encoder.
    sample = np.load(Path(args.dest) / arch_fname(0))
    meta = {"enc_type": args.enc_type, "tokens": int(sample.shape[0]),
            "dim": int(sample.shape[1]), "count": len(files)}
    (Path(args.dest) / "meta.json").write_text(json.dumps(meta))
    print(f"Done. tokens={meta['tokens']} dim={meta['dim']} -> {args.dest}/meta.json")


if __name__ == "__main__":
    main()
