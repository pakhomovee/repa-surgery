import argparse
import copy
from copy import deepcopy
import logging
import os
from pathlib import Path
from collections import OrderedDict
import json

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from models.sit import SiT_models
from loss import SILoss
from utils import load_encoders

from dataset import CustomDataset
from diffusers.models import AutoencoderKL
# import wandb_utils
import wandb
import math
from torchvision.utils import make_grid
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize

logger = get_logger(__name__)

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def _allreduce_mean(tensors, world_size):
    """Average a list of per-parameter gradient tensors across DDP ranks.

    Used by REPA-PCGrad, which computes gradients via torch.autograd.grad and
    therefore bypasses DDP's automatic reduction. Flattens into a single
    buffer so the whole list is reduced in one all-reduce.
    """
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    flat = _flatten_dense_tensors(tensors)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size
    return list(_unflatten_dense_tensors(flat, tensors))


def preprocess_raw_image(x, enc_type):
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


def array2grid(x):
    nrow = round(math.sqrt(x.size(0)))
    x = make_grid(x.clamp(0, 1), nrow=nrow, value_range=(0, 1))
    x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    return x


@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    device = moments.device
    
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z * latents_scale + latents_bias) 
    return z 


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)
    return logger


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):    
    # set accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
        )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if args.grad_surgery:
        # Manual two-gradient assembly is incompatible with fp16's GradScaler,
        # and the simple per-step surgery below assumes no grad accumulation.
        if args.mixed_precision == "fp16":
            raise ValueError(
                "--grad-surgery (REPA-PCGrad) requires --mixed-precision=bf16 (or no); "
                "fp16 uses a GradScaler that conflicts with manual gradients."
            )
        if args.gradient_accumulation_steps != 1:
            raise ValueError("--grad-surgery requires --gradient-accumulation-steps=1.")
    elif args.precond:
        raise ValueError("--precond only applies to --grad-surgery (REPA-PCGrad).")

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        save_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(save_dir, exist_ok=True)
        args_dict = vars(args)
        # Save to a JSON file
        json_dir = os.path.join(save_dir, "args.json")
        with open(json_dir, 'w') as f:
            json.dump(args_dict, f, indent=4)
        checkpoint_dir = f"{save_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(save_dir)
        logger.info(f"Experiment directory created at {save_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False    
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)
    
    # Create model:
    assert args.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.resolution // 8

    # Representation-alignment target source, in priority order:
    #   1. --repr-dir: precomputed encoder features (no encoder loaded here; the
    #      per-step encoder forward + raw-image read are skipped entirely).
    #   2. --enc-type != none: load the encoder, compute features on the fly.
    #   3. baseline: no alignment, no projectors.
    use_repr = args.repr_dir is not None
    if use_repr:
        with open(os.path.join(args.repr_dir, "meta.json")) as f:
            repr_meta = json.load(f)
        encoders, encoder_types, architectures = [], [], []
        z_dims = [repr_meta["dim"]]
    elif args.enc_type not in (None, "None", "none"):
        encoders, encoder_types, architectures = load_encoders(
            args.enc_type, device, args.resolution
            )
        z_dims = [encoder.embed_dim for encoder in encoders]
    else:
        # No encoders => no projectors at all (empty list). This avoids DDP
        # "parameter did not receive grad" errors from unused projector weights.
        encoders, encoder_types, architectures = [], [], []
        z_dims = []
    alignment_enabled = use_repr or len(encoders) > 0
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg = (args.cfg_prob > 0),
        z_dims = z_dims,
        encoder_depth=args.encoder_depth,
        **block_kwargs
    )

    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(device)
    requires_grad(ema, False)
    
    latents_scale = torch.tensor(
        [0.18215, 0.18215, 0.18215, 0.18215]
        ).view(1, 4, 1, 1).to(device)
    latents_bias = torch.tensor(
        [0., 0., 0., 0.]
        ).view(1, 4, 1, 1).to(device)

    # create loss function
    loss_fn = SILoss(
        prediction=args.prediction,
        path_type=args.path_type, 
        encoders=encoders,
        accelerator=accelerator,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        weighting=args.weighting
    )
    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )    
    
    # Setup data. The dataloader is (re)built "by kind" so we can drop the
    # encoder inputs when alignment is off:
    #   align_repr -> precomputed features in the image slot (use_repr)
    #   align_raw  -> raw images (on-the-fly encoder)
    #   plain      -> latents only (baseline, or HASTE after termination)
    local_batch_size = int(args.batch_size // accelerator.num_processes)

    def make_dataloader(kind):
        if kind == "align_repr":
            ds = CustomDataset(args.data_dir, repr_dir=args.repr_dir)
        elif kind == "align_raw":
            ds = CustomDataset(args.data_dir, load_raw=True)
        else:  # plain
            ds = CustomDataset(args.data_dir, load_raw=False)
        dl = DataLoader(ds, batch_size=local_batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        return accelerator.prepare(dl), len(ds)

    def loader_kind(step):
        active = alignment_enabled and (
            args.alignment_end_step < 0 or step < args.alignment_end_step)
        if not active:
            return "plain"
        return "align_repr" if use_repr else "align_raw"

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # resume:
    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) +'.pt'
        ckpt = torch.load(
            f'{os.path.join(args.output_dir, args.exp_name)}/checkpoints/{ckpt_name}',
            map_location='cpu',
            )
        model.load_state_dict(ckpt['model'])
        ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['opt'])
        global_step = ckpt['steps']

    model, optimizer = accelerator.prepare(model, optimizer)
    current_kind = loader_kind(global_step)
    train_dataloader, dataset_len = make_dataloader(current_kind)
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {dataset_len:,} samples ({args.data_dir}); "
                    f"dataloader kind: {current_kind}")

    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(
            project_name="REPA", 
            config=tracker_config,
            init_kwargs={
                "wandb": {"name": f"{args.exp_name}"}
            },
        )
        
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Labels to condition the model with (feel free to change):
    sample_batch_size = 64 // accelerator.num_processes
    _first, gt_xs, _ = next(iter(train_dataloader))
    if current_kind == "align_raw":
        assert _first.shape[-1] == args.resolution
    gt_xs = gt_xs[:sample_batch_size]
    gt_xs = sample_posterior(
        gt_xs.to(device), latents_scale=latents_scale, latents_bias=latents_bias
        )
    ys = torch.randint(args.num_classes, size=(sample_batch_size,), device=device)
    ys = ys.to(device)
    # Create sampling noise:
    n = ys.size(0)
    xT = torch.randn((n, 4, latent_size, latent_size), device=device)

    # REPA-PCGrad: running EMA of the (synced) diffusion gradient, kept as a
    # list of per-parameter tensors. Lazily initialized on the first step.
    grad_ema = None

    for epoch in range(args.epochs):
        # Switch the dataloader when alignment turns off (HASTE termination), so
        # the post-termination phase stops loading encoder inputs entirely.
        kind = loader_kind(global_step)
        if kind != current_kind:
            if accelerator.is_main_process:
                logger.info(f"step {global_step}: dataloader '{current_kind}' -> '{kind}'")
            train_dataloader, _ = make_dataloader(kind)
            current_kind = kind
        align_now = kind != "plain"
        model.train()
        for first, x, y in train_dataloader:
            x = x.squeeze(dim=1).to(device)
            y = y.to(device)
            if args.legacy:
                # In our early experiments, we accidentally apply label dropping twice:
                # once in train.py and once in sit.py.
                # We keep this option for exact reproducibility with previous runs.
                drop_ids = torch.rand(y.shape[0], device=y.device) < args.cfg_prob
                labels = torch.where(drop_ids, args.num_classes, y)
            else:
                labels = y
            with torch.no_grad():
                x = sample_posterior(x, latents_scale=latents_scale, latents_bias=latents_bias)
                zs = []
                if align_now and use_repr:
                    # Precomputed encoder features (already (B, T, D)).
                    zs = [first.to(device)]
                elif align_now:
                    raw_image = first.to(device)
                    with accelerator.autocast():
                        for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                            raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                            z = encoder.forward_features(raw_image_)
                            if 'mocov3' in encoder_type: z = z[:, 1:]
                            if 'dinov2' in encoder_type: z = z['x_norm_patchtokens']
                            zs.append(z)

            # HASTE: disable the alignment loss after the termination step.
            proj_coeff = args.proj_coeff
            if args.alignment_end_step >= 0 and global_step >= args.alignment_end_step:
                proj_coeff = 0.0

            if args.grad_surgery:
                # REPA-PCGrad: PCGrad-style surgery between the diffusion gradient
                # (g_diff) and the alignment gradient (g_repa), using an EMA of
                # g_diff as the stable reference direction.
                model_kwargs = dict(y=labels)
                loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                loss_mean = loss.mean()
                proj_loss_mean = proj_loss.mean()
                params = [p for p in model.parameters() if p.requires_grad]

                # g_diff via the normal DDP backward, so its all-reduce overlaps
                # with the backward pass (the expensive bit). The 0 * proj term
                # keeps the projector params in the autograd graph so DDP does
                # not flag them as unused. g_diff lands (synced) in .grad.
                accelerator.backward(loss_mean + 0.0 * proj_loss_mean, retain_graph=True)

                # EMA of the synced diffusion gradient now sitting in .grad.
                grads = [p.grad for p in params]
                if grad_ema is None:
                    grad_ema = [g.detach().clone() for g in grads]
                else:
                    torch._foreach_mul_(grad_ema, args.grad_ema_decay)
                    torch._foreach_add_(grad_ema, grads, alpha=1 - args.grad_ema_decay)

                # g_repa via autograd over the (partial) alignment graph -- only
                # the encoder-depth blocks + projectors get a gradient. Local, so
                # all-reduce just that subset (much smaller than the full model).
                g_repa = torch.autograd.grad(
                    proj_loss_mean * proj_coeff, params, allow_unused=True)
                idx = [i for i, g in enumerate(g_repa) if g is not None]
                repa_grads = [g_repa[i] for i in idx]
                if accelerator.num_processes > 1 and repa_grads:
                    repa_grads = _allreduce_mean(repa_grads, accelerator.num_processes)

                # PCGrad: project out the part of g_repa that conflicts with the
                # EMA diffusion direction (only when the dot product is negative).
                # clamp(dot, max=0) => scale is 0 when there is no conflict, which
                # avoids a second device sync / branch.
                if repa_grads:
                    if args.precond:
                        # Diagonal preconditioner P = 1/(sqrt(v_hat)+eps) from
                        # AdamW's bias-corrected 2nd moment, so the conflict test
                        # is in the whitened (~diffusion-curvature) metric rather
                        # than Euclidean. P(p) returns None before the optimizer's
                        # first step (no state yet) => fall back to P=1. Computed
                        # inline so no full-model copy of P is ever held.
                        opt_state = getattr(optimizer, "optimizer", optimizer).state
                        beta2 = args.adam_beta2

                        def precond(p):
                            st = opt_state.get(p)
                            if not st or "exp_avg_sq" not in st:
                                return None
                            step_t = st["step"]
                            step_v = step_t.item() if torch.is_tensor(step_t) else step_t
                            bc = 1.0 - beta2 ** step_v
                            v_hat = st["exp_avg_sq"] / max(bc, 1e-12)
                            return v_hat.sqrt().add_(args.precond_eps).reciprocal_()

                        def pmul(t, p):
                            P = precond(p)
                            return t if P is None else t * P

                        # <g_repa, g_ema>_P over the alignment subset; <g_ema, g_ema>_P over all.
                        dot = sum((pmul(g, params[i]) * grad_ema[i]).sum()
                                  for g, i in zip(repa_grads, idx))
                        ema_sq = sum((pmul(ge, params[k]) * ge).sum()
                                     for k, ge in enumerate(grad_ema)) + 1e-12
                    else:
                        dot = sum((g * grad_ema[i]).sum() for g, i in zip(repa_grads, idx))
                        ema_sq = torch.stack(torch._foreach_norm(grad_ema)).pow(2).sum() + 1e-12
                    scale = (torch.clamp(dot, max=0.0) / ema_sq).item()
                    # grad <- g_diff - scale * g_ema (+ g_repa on its subset)
                    torch._foreach_add_(grads, grad_ema, alpha=-scale)
                    torch._foreach_add_([grads[i] for i in idx], repa_grads)

                grad_norm = accelerator.clip_grad_norm_(params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_ema(ema, model)
            else:
                with accelerator.accumulate(model):
                    model_kwargs = dict(y=labels)
                    loss, proj_loss = loss_fn(model, x, model_kwargs, zs=zs)
                    loss_mean = loss.mean()
                    proj_loss_mean = proj_loss.mean()
                    loss = loss_mean + proj_loss_mean * proj_coeff

                    ## optimization
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = model.parameters()
                        grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        update_ema(ema, model) # change ema function
            
            ### enter
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1                
            if global_step % args.checkpointing_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                        "steps": global_step,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

            if (global_step == 1 or (global_step % args.sampling_steps == 0 and global_step > 0)):
                from samplers import euler_sampler
                with torch.no_grad():
                    samples = euler_sampler(
                        model, 
                        xT, 
                        ys,
                        num_steps=50, 
                        cfg_scale=4.0,
                        guidance_low=0.,
                        guidance_high=1.,
                        path_type=args.path_type,
                        heun=False,
                        num_classes=args.num_classes,
                    ).to(torch.float32)
                    samples = vae.decode((samples -  latents_bias) / latents_scale).sample
                    gt_samples = vae.decode((gt_xs - latents_bias) / latents_scale).sample
                    samples = (samples + 1) / 2.
                    gt_samples = (gt_samples + 1) / 2.
                out_samples = accelerator.gather(samples.to(torch.float32))
                gt_samples = accelerator.gather(gt_samples.to(torch.float32))
                accelerator.log({"samples": wandb.Image(array2grid(out_samples)),
                                 "gt_samples": wandb.Image(array2grid(gt_samples))})
                logging.info("Generating EMA samples done.")

            logs = {
                "loss": accelerator.gather(loss_mean).mean().detach().item(), 
                "proj_loss": accelerator.gather(proj_loss_mean).mean().detach().item(),
                "grad_norm": accelerator.gather(grad_norm).mean().detach().item()
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training")

    # logging:
    parser.add_argument("--output-dir", type=str, default="exps")
    parser.add_argument("--exp-name", type=str, required=True)
    parser.add_argument("--logging-dir", type=str, default="logs")
    parser.add_argument("--report-to", type=str, default="wandb")
    parser.add_argument("--sampling-steps", type=int, default=10000)
    parser.add_argument("--resume-step", type=int, default=0)

    # model
    parser.add_argument("--model", type=str)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--encoder-depth", type=int, default=8)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm",  action=argparse.BooleanOptionalAction, default=False)

    # dataset
    parser.add_argument("--data-dir", type=str, default="../data/imagenet256")
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--batch-size", type=int, default=256)

    # precision
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])

    # optimization
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-train-steps", type=int, default=400000)
    parser.add_argument("--checkpointing-steps", type=int, default=50000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam-beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam-weight-decay", type=float, default=0., help="Weight decay to use.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")

    # seed
    parser.add_argument("--seed", type=int, default=0)

    # cpu
    parser.add_argument("--num-workers", type=int, default=4)

    # loss
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--prediction", type=str, default="v", choices=["v"]) # currently we only support v-prediction
    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--enc-type", type=str, default='dinov2-vit-b')
    # Precomputed encoder representations (datasets/encode_repr.py). When set, the
    # encoder is not loaded and per-step encoder forwards + raw-image reads are
    # skipped; the stored features are used as the alignment targets.
    parser.add_argument("--repr-dir", type=str, default=None)
    parser.add_argument("--proj-coeff", type=float, default=0.5)
    parser.add_argument("--weighting", default="uniform", type=str, help="Max gradient norm.")
    parser.add_argument("--legacy", action=argparse.BooleanOptionalAction, default=False)

    # HASTE: stage-wise termination of representation alignment.
    # After this many optimizer steps, the alignment (proj) loss is disabled
    # and training continues on the pure denoising loss. -1 => never (plain REPA).
    parser.add_argument("--alignment-end-step", type=int, default=-1)

    # REPA-PCGrad: PCGrad-style gradient surgery between the diffusion and
    # alignment gradients, using an EMA of the diffusion gradient as the stable
    # reference direction. Requires bf16 (fp16 GradScaler is incompatible with
    # manually-assembled gradients).
    parser.add_argument("--grad-surgery", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grad-ema-decay", type=float, default=0.99)
    # Preconditioned surgery: measure the diffusion/alignment conflict in the
    # metric induced by Adam's 2nd-moment (a diagonal whitening of the diffusion
    # curvature) instead of the plain Euclidean inner product. Free: reuses the
    # exp_avg_sq AdamW already stores.
    parser.add_argument("--precond", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--precond-eps", type=float, default=1e-8)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
        
    return args

if __name__ == "__main__":
    args = parse_args()
    
    main(args)
