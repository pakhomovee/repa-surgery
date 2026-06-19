#!/usr/bin/env python3
"""Evaluate a run's checkpoints with KID (Kernel Inception Distance), 2-GPU.

KID is unbiased and stable with far fewer samples than FID, so it's practical to
sweep every checkpoint in a run.

Pipeline:
  1. Compute Inception features for a fixed set of real images ONCE (cached).
  2. For each checkpoint: generate samples, extract Inception features, compute
     KID against the cached real features.
  3. Write a step -> KID CSV (+ a curve PNG).

Generation and feature extraction are sharded across --gpus. Real features and
per-checkpoint fake features are cached on disk, so re-runs (e.g. after new
checkpoints land) only do the missing work.

KID uses the standard cubic polynomial-kernel MMD^2 estimator averaged over
random subsets (as in torch-fidelity / torchmetrics).

Example:
    python training/evaluate.py \
        --run-dir ../runs/celeba_sit-b_2_baseline \
        --gpus 0,1 --num-samples 10000 --num-real 10000
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

_REPA_DIR = Path(__file__).resolve().parent.parent / "REPA"
if str(_REPA_DIR) not in sys.path:
    sys.path.insert(0, str(_REPA_DIR))
from models.sit import SiT_models          # noqa: E402
from samplers import euler_sampler          # noqa: E402
from diffusers.models import AutoencoderKL   # noqa: E402

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# --------------------------------------------------------------------------- #
# Model reconstruction + generation (see sample.py).
# --------------------------------------------------------------------------- #
def infer_z_dims(state_dict: dict) -> list[int]:
    z_by_idx: dict[int, int] = {}
    pat = re.compile(r"^projectors\.(\d+)\.(\d+)\.weight$")
    for k, v in state_dict.items():
        m = pat.match(k)
        if m and (int(m.group(1)) not in z_by_idx or int(m.group(2)) >= 4):
            z_by_idx[int(m.group(1))] = v.shape[0]
    return [z_by_idx[i] for i in sorted(z_by_idx)]


def load_sit(ckpt_path: Path, weights: str, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    state = ckpt[weights]
    model = SiT_models[targs.model](
        input_size=targs.resolution // 8, num_classes=targs.num_classes,
        use_cfg=(targs.cfg_prob > 0), z_dims=infer_z_dims(state),
        encoder_depth=targs.encoder_depth,
        fused_attn=targs.fused_attn, qk_norm=targs.qk_norm,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, targs


@torch.no_grad()
def generate_images(model, vae, targs, n, cfg_scale, num_steps, gen_batch, device, seed):
    g = torch.Generator(device=device).manual_seed(seed)
    latent_size = targs.resolution // 8
    scale = torch.tensor([0.18215] * 4, device=device).view(1, 4, 1, 1)
    # Sampling runs the model in half precision (like training); without this it
    # falls back to fp32, which is several times slower (esp. on T4/A100).
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    out = []
    for i in tqdm(range(0, n, gen_batch)):
        b = min(gen_batch, n - i)
        y = (torch.arange(i, i + b, device=device) % targs.num_classes)
        xT = torch.randn((b, 4, latent_size, latent_size), device=device, generator=g)
        with torch.autocast("cuda", dtype=amp_dtype):
            lat = euler_sampler(
                model, xT, y, num_steps=num_steps, cfg_scale=cfg_scale,
                guidance_low=0.0, guidance_high=1.0, path_type=targs.path_type,
                num_classes=targs.num_classes,
            ).to(torch.float32)
            img = vae.decode(lat / scale).sample
        out.append(((img.float() + 1) / 2).clamp(0, 1).cpu())
    return torch.cat(out)


# --------------------------------------------------------------------------- #
# Inception features (canonical FID InceptionV3) + KID.
# --------------------------------------------------------------------------- #
def build_inception(device):
    from pytorch_fid.inception import InceptionV3
    idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    return InceptionV3([idx], resize_input=True, normalize_input=True).to(device).eval()


class _PathImages(Dataset):
    def __init__(self, paths):
        self.paths = paths
        self.tf = transforms.ToTensor()  # [0,1]; Inception resizes to 299 itself

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def _inception(inception, x, device):
    return inception(x.to(device))[0].squeeze(-1).squeeze(-1).cpu()


@torch.no_grad()
def features_from_paths(paths, inception, device, bs, nw, desc, show):
    loader = DataLoader(_PathImages(paths), batch_size=bs, num_workers=nw, pin_memory=True)
    return torch.cat([_inception(inception, x, device)
                      for x in tqdm(loader, desc=desc, disable=not show)])


@torch.no_grad()
def features_from_images(imgs, inception, device, bs):
    return torch.cat([_inception(inception, imgs[i:i + bs], device)
                      for i in range(0, len(imgs), bs)])


def _poly_kernel(X, Y):
    return (X @ Y.t() / X.shape[1] + 1.0).pow(3)


def _mmd2_unbiased(X, Y):
    m, n = X.shape[0], Y.shape[0]
    Kxx, Kyy, Kxy = _poly_kernel(X, X), _poly_kernel(Y, Y), _poly_kernel(X, Y)
    return ((Kxx.sum() - Kxx.diagonal().sum()) / (m * (m - 1))
            + (Kyy.sum() - Kyy.diagonal().sum()) / (n * (n - 1))
            - 2 * Kxy.mean())


def compute_kid(real, fake, subset_size, num_subsets, seed, device):
    real, fake = real.float().to(device), fake.float().to(device)
    m = min(subset_size, real.shape[0], fake.shape[0])
    g = torch.Generator(device=device).manual_seed(seed)
    vals = []
    for _ in range(num_subsets):
        xi = real[torch.randperm(real.shape[0], generator=g, device=device)[:m]]
        yi = fake[torch.randperm(fake.shape[0], generator=g, device=device)[:m]]
        vals.append(_mmd2_unbiased(xi, yi))
    vals = torch.stack(vals)
    return vals.mean().item(), vals.std().item()


# --------------------------------------------------------------------------- #
def step_of(ckpt: Path) -> int:
    return int(ckpt.stem)


def list_real(images_dir: Path, num_real: int | None) -> list[Path]:
    files = sorted((p for p in images_dir.rglob("*") if p.suffix.lower() in _IMG_EXTS),
                   key=lambda p: str(p.relative_to(images_dir)))
    return files[:num_real] if num_real else files


def worker(rank, gpu_ids, ckpts, real_files, args, eval_dir):
    device = torch.device(f"cuda:{gpu_ids[rank]}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    world = len(gpu_ids)
    inception = build_inception(device)

    # Real features (cached, computed once).
    real_path = eval_dir / f"real_rank{rank}.pt"
    if args.refresh_real or not real_path.exists():
        idxs = list(range(rank, len(real_files), world))
        feats = features_from_paths([real_files[i] for i in idxs], inception, device,
                                    args.batch_size, args.num_workers, "real", rank == 0)
        torch.save(feats, real_path)

    # Per-checkpoint fake features.
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    n_shard = args.num_samples // world + (1 if rank < args.num_samples % world else 0)
    for ck in ckpts:
        step = step_of(ck)
        fpath = eval_dir / f"fake_{step:07d}_rank{rank}.pt"
        if fpath.exists() and not args.refresh:
            continue
        model, targs = load_sit(ck, args.weights, device)
        imgs = generate_images(model, vae, targs, n_shard, args.cfg_scale,
                               args.num_steps, args.gen_batch, device,
                               seed=args.seed * 131 + rank * 977 + step % 997)
        feats = features_from_images(imgs, inception, device, args.batch_size)
        torch.save(feats, fpath)
        del model
        torch.cuda.empty_cache()
        if rank == 0:
            print(f"[gpu{gpu_ids[rank]}] step {step}: {args.num_samples} samples featurized")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Run dir; evaluates checkpoints/*.pt.")
    src.add_argument("--ckpt", type=Path, help="Single checkpoint to evaluate.")
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--num-samples", type=int, default=10000, help="Generated samples per ckpt.")
    p.add_argument("--num-real", type=int, default=10000, help="Real images for the reference.")
    p.add_argument("--num-steps", type=int, default=50, help="Sampler steps.")
    p.add_argument("--cfg-scale", type=float, default=1.5)
    p.add_argument("--weights", choices=["ema", "model"], default="ema")
    p.add_argument("--every", type=int, default=1, help="Evaluate every Nth checkpoint.")
    p.add_argument("--data-dir", type=Path, default=None, help="Real data dir (default: from ckpt).")
    p.add_argument("--batch-size", type=int, default=128, help="Inception batch size.")
    p.add_argument("--gen-batch", type=int, default=128,
                   help="Generation batch size (doubled internally when cfg-scale>1).")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--kid-subset-size", type=int, default=1000)
    p.add_argument("--kid-subsets", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--refresh", action="store_true", help="Recompute fake features.")
    p.add_argument("--refresh-real", action="store_true", help="Recompute real features.")
    p.add_argument("--output", type=Path, default=None, help="CSV path (default: <run>/eval/kid.csv).")
    args = p.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",") if g != ""]
    if not torch.cuda.is_available():
        sys.exit("CUDA required.")

    if args.ckpt:
        ckpts = [args.ckpt]
        run_dir = args.ckpt.resolve().parent.parent
    else:
        run_dir = args.run_dir.resolve()
        ckpts = sorted((run_dir / "checkpoints").glob("*.pt"), key=step_of)
        if not ckpts:
            sys.exit(f"No checkpoints under {run_dir}/checkpoints")
        ckpts = ckpts[:: args.every]
    print(f"Evaluating {len(ckpts)} checkpoint(s): "
          f"{[step_of(c) for c in ckpts]}")

    targs0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)["args"]
    data_dir = args.data_dir or (_REPA_DIR / targs0.data_dir)
    images_dir = (data_dir / "images").resolve()
    real_files = list_real(images_dir, args.num_real)
    if not real_files:
        sys.exit(f"No real images under {images_dir}")
    print(f"Reference: {len(real_files)} real images from {images_dir}")

    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    if len(gpu_ids) == 1:
        worker(0, gpu_ids, ckpts, real_files, args, eval_dir)
    else:
        mp.spawn(worker, args=(gpu_ids, ckpts, real_files, args, eval_dir),
                 nprocs=len(gpu_ids), join=True)

    # Merge features + compute KID per checkpoint.
    device = torch.device(f"cuda:{gpu_ids[0]}")
    real = torch.cat([torch.load(eval_dir / f"real_rank{r}.pt") for r in range(len(gpu_ids))])
    rows = []
    print("\n=== KID (lower is better) ===")
    for ck in ckpts:
        step = step_of(ck)
        fake = torch.cat([torch.load(eval_dir / f"fake_{step:07d}_rank{r}.pt")
                          for r in range(len(gpu_ids))])
        mean, std = compute_kid(real, fake, args.kid_subset_size, args.kid_subsets,
                                args.seed, device)
        rows.append((step, mean, std))
        print(f"  step {step:>8}: KID = {mean:.6f} ± {std:.6f}")

    out = args.output or (eval_dir / "kid.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "kid_mean", "kid_std"])
        w.writerows(rows)
    print(f"\nWrote {out}")
    best = min(rows, key=lambda r: r[1])
    print(f"Best: step {best[0]} (KID {best[1]:.6f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        steps = [r[0] for r in rows]
        means = [r[1] for r in rows]
        stds = [r[2] for r in rows]
        plt.figure(figsize=(7, 4))
        plt.errorbar(steps, means, yerr=stds, marker="o", capsize=3)
        plt.xlabel("training step")
        plt.ylabel("KID")
        plt.title(f"KID vs step — {run_dir.name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        curve = eval_dir / "kid_curve.png"
        plt.savefig(curve, dpi=120)
        print(f"Wrote {curve}")
    except Exception as e:  # plotting is optional
        print(f"(skipped curve plot: {e})")


if __name__ == "__main__":
    main()
