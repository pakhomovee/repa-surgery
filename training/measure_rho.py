#!/usr/bin/env python3
"""Post-hoc gradient-conflict diagnostics from existing checkpoints (no retrain).

Loads each checkpoint in a run, takes one fixed batch, and measures
cos(g_diff, g_repa) per noise-level bin -- the offline twin of train.py's
--log-grad-conflict. Writes <run>/rho.csv, then render it with
results/analysis/analyze_rho.py.

Works for ANY alignment-mode checkpoint (repa / haste / repa-PCGrad / precond):
 rho is computed from the diffusion vs. alignment LOSS gradients, not from the
combined update, so the method that produced the weights is irrelevant. Baseline
has no projectors -> nothing to measure.

The same batch (and noise stratification) is reused across checkpoints, so the
only thing that changes is the model weights.

Example:
    python training/measure_rho.py --run-dir runs/celeba_sit-b_2_repa-PCGrad --gpu 0
    python results/analysis/analyze_rho.py runs/celeba_sit-b_2_repa-PCGrad
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_REPA_DIR = Path(__file__).resolve().parent.parent / "REPA"
if str(_REPA_DIR) not in sys.path:
    sys.path.insert(0, str(_REPA_DIR))
from models.sit import SiT_models          # noqa: E402
from loss import SILoss                      # noqa: E402
from dataset import CustomDataset            # noqa: E402
from utils import load_encoders              # noqa: E402

# --- small helpers copied from train.py so this stays decoupled (no wandb) --- #
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type):
    res = x.shape[-1]
    if 'clip' in enc_type:
        x = x / 255.
        x = torch.nn.functional.interpolate(x, 224 * (res // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.)
    elif 'dinov2' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.)
        x = torch.nn.functional.interpolate(x, 224 * (res // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.)
    elif 'jepa' in enc_type:
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x / 255.)
        x = torch.nn.functional.interpolate(x, 224 * (res // 256), mode='bicubic')
    return x


def encoder_tokens(encoder, etype, x):
    z = encoder.forward_features(x)
    if 'mocov3' in etype:
        z = z[:, 1:]
    if 'dinov2' in etype:
        z = z['x_norm_patchtokens']
    return z


def sample_posterior(moments, scale, bias):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    return z * scale + bias


def infer_z_dims(state_dict):
    z = {}
    pat = re.compile(r"^projectors\.(\d+)\.(\d+)\.weight$")
    for k, v in state_dict.items():
        m = pat.match(k)
        if m and (int(m.group(1)) not in z or int(m.group(2)) >= 4):
            z[int(m.group(1))] = v.shape[0]
    return [z[i] for i in sorted(z)]


def measure(model, loss_fn, x, model_kwargs, zs, params, num_bins, device):
    """Per-noise-bin cos(g_diff, g_repa) via autograd over a shared forward."""
    B = x.shape[0]
    bin_idx = torch.arange(B, device=device) * num_bins // B
    t = (bin_idx + torch.rand(B, device=device)) / num_bins
    dn, pr, _ = loss_fn.per_sample(model, x, model_kwargs, zs=zs, time_input=t)
    rows = []
    for k in range(num_bins):
        m = bin_idx == k
        if int(m.sum()) < 2:
            continue
        gd = torch.autograd.grad(dn[m].mean(), params, retain_graph=True, allow_unused=True)
        gr = torch.autograd.grad(pr[m].mean(), params, retain_graph=True, allow_unused=True)
        gd = [g if g is not None else torch.zeros_like(p) for g, p in zip(gd, params)]
        gr = [g if g is not None else torch.zeros_like(p) for g, p in zip(gr, params)]
        dot = sum((a * b).sum() for a, b in zip(gd, gr))
        nd = sum((a * a).sum() for a in gd).sqrt()
        nr = sum((b * b).sum() for b in gr).sqrt()
        rows.append({"t_lo": k / num_bins, "t_hi": (k + 1) / num_bins,
                     "t_center": (k + 0.5) / num_bins,
                     "rho": (dot / (nd * nr + 1e-12)).item(),
                     "g_diff_norm": nd.item(), "g_repa_norm": nr.item(),
                     "n": int(m.sum())})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--run-dir", type=Path, help="Run dir; measures checkpoints/*.pt.")
    src.add_argument("--ckpt", type=Path, help="Single checkpoint.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--data-dir", type=Path, default=None, help="Default: from checkpoint args.")
    p.add_argument("--repr-dir", type=str, default=None,
                   help="Use precomputed features instead of the on-the-fly encoder.")
    p.add_argument("--enc-type", default="dinov2-vit-b", help="On-the-fly encoder (no --repr-dir).")
    p.add_argument("--weights", choices=["model", "ema"], default="model",
                   help="model = the live training weights (the conflict the run saw).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-bins", type=int, default=8)
    p.add_argument("--every", type=int, default=1, help="Measure every Nth checkpoint.")
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, default=None, help="Default: <run>/rho.csv.")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if args.ckpt:
        ckpts = [args.ckpt]
        run_dir = args.ckpt.resolve().parent.parent
    else:
        run_dir = args.run_dir.resolve()
        ckpts = sorted((run_dir / "checkpoints").glob("*.pt"), key=lambda c: int(c.stem))[:: args.every]
        if not ckpts:
            sys.exit(f"No checkpoints under {run_dir}/checkpoints")

    targs0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)["args"]
    data_dir = args.data_dir or (_REPA_DIR / targs0.data_dir)
    scale = torch.tensor([0.18215] * 4, device=device).view(1, 4, 1, 1)
    bias = torch.zeros(4, device=device).view(1, 4, 1, 1)
    loss_fn = SILoss(prediction="v", path_type=targs0.path_type, weighting="uniform")

    # One fixed batch, reused across all checkpoints.
    torch.manual_seed(args.seed)
    ds = (CustomDataset(str(data_dir), repr_dir=args.repr_dir) if args.repr_dir
          else CustomDataset(str(data_dir), load_raw=True))
    g = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, generator=g, drop_last=True)
    first, latent, label = next(iter(loader))
    latent, label = latent.to(device), label.to(device)
    x = sample_posterior(latent, scale, bias)
    model_kwargs = dict(y=label)

    if args.repr_dir:
        zs = [first.to(device).float()]
        print(f"Using precomputed targets from {args.repr_dir}")
    else:
        encoders, etypes, _ = load_encoders(args.enc_type, device, targs0.resolution)
        with torch.no_grad():
            zs = [encoder_tokens(encoders[0], etypes[0],
                                 preprocess_raw_image(first.to(device).float(), etypes[0]))]
        print(f"Computed {args.enc_type} targets on the fly")

    out = args.output or (run_dir / "rho.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "t_lo", "t_hi", "t_center", "rho", "g_diff_norm", "g_repa_norm", "n"])
        for ck in ckpts:
            step = int(ck.stem)
            blob = torch.load(ck, map_location="cpu", weights_only=False)
            state = blob[args.weights]
            z_dims = infer_z_dims(state)
            if not z_dims:
                print(f"  step {step}: no projectors (baseline?) -- skipped")
                continue
            ta = blob["args"]
            model = SiT_models[ta.model](
                input_size=ta.resolution // 8, num_classes=ta.num_classes,
                use_cfg=(ta.cfg_prob > 0), z_dims=z_dims, encoder_depth=ta.encoder_depth,
                fused_attn=ta.fused_attn, qk_norm=ta.qk_norm).to(device)
            model.load_state_dict(state)
            model.eval()  # deterministic (no cfg label-drop); LayerNorm same as train
            params = [p for p in model.parameters() if p.requires_grad]
            torch.manual_seed(args.seed)  # same noise/bins per checkpoint
            rows = measure(model, loss_fn, x, model_kwargs, zs, params, args.num_bins, device)
            for r in rows:
                w.writerow([step, r["t_lo"], r["t_hi"], r["t_center"], r["rho"],
                            r["g_diff_norm"], r["g_repa_norm"], r["n"]])
            neg = sum(r["rho"] < 0 for r in rows)
            print(f"  step {step:>8}: {neg}/{len(rows)} bins conflict (rho<0)")
            del model
            torch.cuda.empty_cache()
    print(f"\nWrote {out}\n-> python results/analysis/analyze_rho.py {run_dir}")


if __name__ == "__main__":
    main()
