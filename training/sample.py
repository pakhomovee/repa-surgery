#!/usr/bin/env python3
"""Sample a grid of images from a trained SiT checkpoint to eyeball quality.

Single-GPU, writes a PNG grid to disk (handy on headless AutoDL boxes -- scp the
png and look at it). Works for any --mode the pipeline trained (baseline / repa /
haste / repa-sigma): the model is rebuilt from the args stored in the checkpoint,
and the projector shapes (z_dims) are inferred from the weights, so baseline
(no projectors) and repa load alike.

Example:
    python training/sample.py \
        --ckpt ../runs/celeba_sit-b_2_baseline/checkpoints/0020000.pt \
        --num-samples 64 --cfg-scale 1.5
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

# Import REPA internals (models, samplers).
_REPA_DIR = Path(__file__).resolve().parent.parent / "REPA"
if str(_REPA_DIR) not in sys.path:
    sys.path.insert(0, str(_REPA_DIR))
from models.sit import SiT_models          # noqa: E402
from samplers import euler_sampler, euler_maruyama_sampler  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


def infer_z_dims(state_dict: dict) -> list[int]:
    """Recover the projector output dims from a SiT state dict.

    build_mlp ends in a Linear at index 4, so projectors.<i>.4.weight has shape
    (z_dim, projector_dim). No projector keys => baseline (empty list).
    """
    z_by_idx: dict[int, int] = {}
    pat = re.compile(r"^projectors\.(\d+)\.(\d+)\.weight$")
    for k, v in state_dict.items():
        m = pat.match(k)
        if m:
            i, layer = int(m.group(1)), int(m.group(2))
            # keep the deepest linear layer's output dim for this projector
            if i not in z_by_idx or layer >= 4:
                z_by_idx[i] = v.shape[0]
    return [z_by_idx[i] for i in sorted(z_by_idx)]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, type=Path, help="Path to a .pt checkpoint.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG (default: <ckpt-dir>/../samples/<step>.png).")
    p.add_argument("--num-samples", type=int, default=64)
    p.add_argument("--cfg-scale", type=float, default=1.5, help="1.0 disables CFG.")
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--mode", choices=["ode", "sde"], default="ode",
                   help="ode=euler, sde=euler-maruyama.")
    p.add_argument("--weights", choices=["ema", "model"], default="ema")
    p.add_argument("--random-labels", action="store_true",
                   help="Random class labels (default: evenly spread over classes).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt["args"]  # argparse.Namespace saved at train time
    state = ckpt[args.weights]
    step = ckpt.get("steps", "na")
    z_dims = infer_z_dims(state)
    num_classes = targs.num_classes
    latent_size = targs.resolution // 8

    print(f"ckpt={args.ckpt} step={step} model={targs.model} "
          f"num_classes={num_classes} z_dims={z_dims} weights={args.weights}")

    model = SiT_models[targs.model](
        input_size=latent_size,
        num_classes=num_classes,
        use_cfg=(targs.cfg_prob > 0),
        z_dims=z_dims,
        encoder_depth=targs.encoder_depth,
        fused_attn=targs.fused_attn,
        qk_norm=targs.qk_norm,
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    latents_scale = torch.tensor([0.18215] * 4, device=device).view(1, 4, 1, 1)
    latents_bias = torch.zeros(4, device=device).view(1, 4, 1, 1)

    n = args.num_samples
    if args.random_labels:
        y = torch.randint(0, num_classes, (n,), device=device)
    else:
        y = (torch.arange(n, device=device) % num_classes)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)

    sampler = euler_maruyama_sampler if args.mode == "sde" else euler_sampler
    latents = sampler(
        model, xT, y,
        num_steps=args.num_steps,
        cfg_scale=args.cfg_scale,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type=targs.path_type,
        num_classes=num_classes,
    ).to(torch.float32)

    images = vae.decode((latents - latents_bias) / latents_scale).sample
    images = ((images + 1) / 2).clamp(0, 1)

    if args.output is None:
        out_dir = args.ckpt.resolve().parent.parent / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / f"{str(step).zfill(7)}_cfg{args.cfg_scale}_{args.mode}.png"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nrow = int(n ** 0.5) if int(n ** 0.5) ** 2 == n else min(num_classes, n)
    save_image(images, args.output, nrow=nrow)
    print(f"Saved {n} samples -> {args.output}")


if __name__ == "__main__":
    main()
