#!/usr/bin/env python3
"""Check a trained SiT checkpoint for training-set memorization (2-GPU).

Pipeline:
  1. Generate N samples from the checkpoint (single GPU).
  2. Featurize all training images with DINOv2-B, sharded across --gpus.
  3. For each generated sample, find its nearest neighbours in the training set
     by cosine similarity in DINOv2 space.
  4. Write a grid PNG (each row: generated | NN1 | NN2 ...) and print similarity
     stats. High top-1 cosine (e.g. > --threshold) flags likely copies.

DINOv2 is a perceptual copy-detection-ish descriptor: near-duplicate crops of a
face score very high, semantically-different faces score low. Eyeball the grid
*and* the stats -- CelebA faces are visually similar, so use the grid to tell
"plausible new face" from "pixel-level copy of a specific training image".

Example:
    python training/memorization.py \
        --ckpt ../runs/celeba_sit-b_2_baseline/checkpoints/0020000.pt \
        --gpus 0,1 --num-samples 32 --topk 4
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

_REPA_DIR = Path(__file__).resolve().parent.parent / "REPA"
if str(_REPA_DIR) not in sys.path:
    sys.path.insert(0, str(_REPA_DIR))
from models.sit import SiT_models          # noqa: E402
from samplers import euler_sampler          # noqa: E402
from diffusers.models import AutoencoderKL   # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- #
# Generation (reconstruct model from checkpoint args; see sample.py).
# --------------------------------------------------------------------------- #
def infer_z_dims(state_dict: dict) -> list[int]:
    z_by_idx: dict[int, int] = {}
    pat = re.compile(r"^projectors\.(\d+)\.(\d+)\.weight$")
    for k, v in state_dict.items():
        m = pat.match(k)
        if m and (int(m.group(1)) not in z_by_idx or int(m.group(2)) >= 4):
            z_by_idx[int(m.group(1))] = v.shape[0]
    return [z_by_idx[i] for i in sorted(z_by_idx)]


@torch.no_grad()
def generate_samples(ckpt_path: Path, n: int, cfg_scale: float, num_steps: int,
                     weights: str, device: torch.device) -> torch.Tensor:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    state = ckpt[weights]
    latent_size = targs.resolution // 8
    model = SiT_models[targs.model](
        input_size=latent_size, num_classes=targs.num_classes,
        use_cfg=(targs.cfg_prob > 0), z_dims=infer_z_dims(state),
        encoder_depth=targs.encoder_depth,
        fused_attn=targs.fused_attn, qk_norm=targs.qk_norm,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    scale = torch.tensor([0.18215] * 4, device=device).view(1, 4, 1, 1)

    y = torch.arange(n, device=device) % targs.num_classes
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)
    latents = euler_sampler(
        model, xT, y, num_steps=num_steps, cfg_scale=cfg_scale,
        guidance_low=0.0, guidance_high=1.0, path_type=targs.path_type,
        num_classes=targs.num_classes,
    ).to(torch.float32)
    images = vae.decode(latents / scale).sample
    images = ((images + 1) / 2).clamp(0, 1).cpu()
    del model, vae
    torch.cuda.empty_cache()
    return images, targs, ckpt.get("steps", "na")


# --------------------------------------------------------------------------- #
# DINOv2 feature extraction (sharded across GPUs).
# --------------------------------------------------------------------------- #
class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def featurize(encoder, paths, device, batch_size, num_workers, desc, show):
    loader = DataLoader(ImagePathDataset(paths), batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True)
    feats = []
    for x in tqdm(loader, desc=desc, disable=not show):
        z = encoder.forward_features(x.to(device))["x_norm_clstoken"]
        feats.append(F.normalize(z.float(), dim=-1).cpu())
    return torch.cat(feats) if feats else torch.empty(0)


def list_train_images(images_dir: Path, max_train: int | None) -> list[Path]:
    files = sorted((p for p in images_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS),
                   key=lambda p: str(p.relative_to(images_dir)))
    return files[:max_train] if max_train else files


def feat_worker(rank, gpu_ids, train_files, batch_size, num_workers, outdir, gen_dir):
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                             verbose=False).to(device).eval()
    idxs = list(range(rank, len(train_files), len(gpu_ids)))
    feats = featurize(encoder, [train_files[i] for i in idxs], device,
                      batch_size, num_workers, f"[gpu{gpu_ids[rank]}] train", rank == 0)
    torch.save({"idx": idxs, "feat": feats}, outdir / f"train_rank{rank}.pt")
    if rank == 0:
        gen_files = sorted(gen_dir.glob("*.png"))
        gfeats = featurize(encoder, gen_files, device, batch_size, num_workers,
                           "[gpu] gen", True)
        torch.save({"feat": gfeats}, outdir / "gen.pt")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--gpus", default="0,1", help="Comma-separated GPU ids.")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Dataset dir with images/ (default: from checkpoint args).")
    p.add_argument("--num-samples", type=int, default=32)
    p.add_argument("--topk", type=int, default=4, help="Nearest neighbours per sample.")
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--weights", choices=["ema", "model"], default="ema")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap #train images searched (default: all).")
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Top-1 cosine above this flags likely memorization.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--grid-rows", type=int, default=16, help="Max sample rows in grid.")
    args = p.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",") if g != ""]
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")

    out_base = args.ckpt.resolve().parent.parent / "memorization"
    out_base.mkdir(parents=True, exist_ok=True)
    feat_dir = out_base / "_feat"
    feat_dir.mkdir(exist_ok=True)
    gen_dir = out_base / "_gen"
    gen_dir.mkdir(exist_ok=True)

    # 1. Generate samples on the first GPU.
    gen_device = torch.device(f"cuda:{gpu_ids[0]}")
    gen_images, targs, step = generate_samples(
        args.ckpt, args.num_samples, args.cfg_scale, args.num_steps,
        args.weights, gen_device)
    for i, img in enumerate(gen_images):
        save_image(img, gen_dir / f"{i:04d}.png")

    # 2. Resolve training images.
    data_dir = args.data_dir or (_REPA_DIR / targs.data_dir)
    images_dir = (data_dir / "images").resolve()
    train_files = list_train_images(images_dir, args.max_train)
    if not train_files:
        sys.exit(f"No training images under {images_dir}")
    print(f"Searching {len(train_files)} training images against "
          f"{len(gen_images)} samples on GPUs {gpu_ids}")

    # 3. Featurize (sharded across GPUs).
    if len(gpu_ids) == 1:
        feat_worker(0, gpu_ids, train_files, args.batch_size, args.num_workers,
                    feat_dir, gen_dir)
    else:
        mp.spawn(feat_worker,
                 args=(gpu_ids, train_files, args.batch_size, args.num_workers,
                       feat_dir, gen_dir),
                 nprocs=len(gpu_ids), join=True)

    # 4. Merge train features, nearest-neighbour search.
    dim = torch.load(feat_dir / "train_rank0.pt")["feat"].shape[1]
    train_feats = torch.empty(len(train_files), dim)
    for r in range(len(gpu_ids)):
        d = torch.load(feat_dir / f"train_rank{r}.pt")
        train_feats[d["idx"]] = d["feat"]
    gen_feats = torch.load(feat_dir / "gen.pt")["feat"]

    sims = gen_feats @ train_feats.T              # (Q, T), both L2-normalized
    top_vals, top_idx = sims.topk(args.topk, dim=1)
    top1 = top_vals[:, 0]

    print("\n=== Memorization report (DINOv2 cosine similarity) ===")
    print(f"top-1 similarity: mean={top1.mean():.3f} max={top1.max():.3f} "
          f"min={top1.min():.3f}")
    n_flag = int((top1 > args.threshold).sum())
    print(f"samples with top-1 > {args.threshold}: {n_flag}/{len(top1)} "
          f"{'<-- possible memorization' if n_flag else '(none)'}")
    worst = int(top1.argmax())
    print(f"closest pair: sample {worst} <-> {train_files[int(top_idx[worst,0])].name} "
          f"(cos={top1[worst]:.3f})")

    # 5. Grid: [generated | NN1 | NN2 ...] per row.
    to_img = transforms.Compose([transforms.Resize(160), transforms.CenterCrop(160),
                                 transforms.ToTensor()])
    rows = min(args.grid_rows, len(gen_images))
    order = torch.argsort(top1, descending=True)[:rows]  # most-suspicious first
    tiles = []
    for q in order.tolist():
        tiles.append(F.interpolate(gen_images[q:q+1], size=160, mode="bilinear",
                                   align_corners=False)[0])
        for j in range(args.topk):
            tiles.append(to_img(Image.open(train_files[int(top_idx[q, j])]).convert("RGB")))
    grid = torch.stack(tiles)

    if args.output is None:
        args.output = out_base / f"{str(step).zfill(7)}_nn.png"
    save_image(grid, args.output, nrow=1 + args.topk)
    print(f"\nGrid (left col = generated, sorted most-suspicious first) -> {args.output}")


if __name__ == "__main__":
    main()
