# Training pipeline

One entry point — [`train.sh`](train.sh) — drives REPA training for any dataset.
It sets up the environment, optionally builds the dataset, then launches
`REPA/train.py` via `accelerate launch`.

```
training/
  train.sh                # single entry point
  lib/common.sh           # env setup helpers (AutoDL, conda, deps)
  envs/<dataset>.env       # per-dataset config + prepare_dataset()
  requirements-extra.txt   # deps beyond REPA/requirements.txt
```

## Quick start (CelebA)

```bash
# Prepare data once (export images + labels, then encode VAE latents) and train:
training/train.sh --dataset celeba --gpus 0,1,2,3 --prepare

# Subsequent runs (data already built):
training/train.sh --dataset celeba --gpus 0,1,2,3 \
  --model SiT-B/2 --batch-size 256 --num-workers 4 --checkpointing-steps 10000

# Plain SiT baseline (no DINOv2 alignment):
training/train.sh --dataset celeba --gpus 0 --mode baseline
```

## Training modes

`--mode` selects the training variant (reflected in the auto exp-name, e.g.
`celeba_sit-b_2_haste`):

| mode | what it does |
|------|--------------|
| `repa` (default) | Representation alignment with the env's `ENC_TYPE`/`PROJ_COEFF` (DINOv2, 0.5). |
| `baseline` (alias `--baseline`) | Plain SiT, `enc-type=none, proj-coeff=0`. |
| `haste` | REPA alignment until `HASTE_END_STEP` (default 100k), then pure diffusion loss. Implements the "stage-wise termination" of *REPA Works Until It Doesn't* (arXiv 2505.16792). |
| `repa-sigma` | PCGrad-style gradient surgery: keeps an EMA of the diffusion gradient as a stable reference; when the alignment gradient is anticorrelated with it (`⟨g_ema,g_repa⟩<0`), the conflicting component is projected out before the update. **Forces bf16** (fp16's GradScaler is incompatible with manually-assembled gradients). |

Mode knobs live in the dataset env: `HASTE_END_STEP` (haste) and
`GRAD_EMA_DECAY` (repa-sigma, default 0.99).

Implementation note: `haste` and `repa-sigma` are backed by new `train.py`
flags (`--alignment-end-step`, `--grad-surgery`/`--grad-ema-decay`). `repa-sigma`
gets `g_diff` from a normal DDP backward (so its all-reduce overlaps with the
backward) and `g_repa` from `torch.autograd.grad` over the partial alignment
graph (only that subset is all-reduced), then assembles `.grad` with fused
`_foreach` ops — so it needs `--gradient-accumulation-steps=1`. It runs ~1.2×
the cost of plain `repa` (the irreducible extra is the second, partial backward).

`--dry-run` prints the exact `accelerate` command without launching.

## Tunable parameters

CLI flags override the dataset env defaults: `--gpus`, `--model`,
`--batch-size`, `--num-workers`, `--checkpointing-steps`, `--max-train-steps`,
`--exp-name`. `--gpus 0,1,2,3` both sets `CUDA_VISIBLE_DEVICES` and derives the
number of `accelerate` processes. Anything after `--` is passed verbatim to
`train.py`. Run `training/train.sh --help` for the full list.

## Inspecting samples

Training only logs sample grids at step 1, and with `report-to=tensorboard`
those images are dropped (the TB tracker logs scalars only). To eyeball quality
from a checkpoint, use [`sample.py`](sample.py) — it writes a PNG grid to disk:

```bash
python training/sample.py \
  --ckpt ../runs/celeba_sit-b_2_baseline/checkpoints/0020000.pt \
  --num-samples 64 --cfg-scale 1.5
# -> ../runs/celeba_sit-b_2_baseline/samples/0020000_cfg1.5_ode.png  (scp & view)
```

It rebuilds the model from the args saved in the checkpoint and infers the
projector shapes from the weights, so it works for any mode (baseline has no
projectors, repa does) without extra flags. Uses EMA weights by default
(`--weights model` for the raw model); labels are spread evenly across classes
(`--random-labels` to randomize); `--mode sde` switches euler→euler-maruyama.

## Checking for memorization

[`memorization.py`](memorization.py) checks whether a checkpoint copies training
images: it generates samples, featurizes the whole training set with DINOv2-B
(**sharded across the `--gpus`**), finds each sample's nearest neighbours by
cosine similarity, and writes a `generated | NN1 | NN2 …` grid plus stats:

```bash
python training/memorization.py \
  --ckpt ../runs/celeba_sit-b_2_baseline/checkpoints/0020000.pt \
  --gpus 0,1 --num-samples 32 --topk 4
# -> ../runs/.../memorization/0020000_nn.png  + a printed top-1 similarity report
```

Rows are sorted most-suspicious-first; a high top-1 cosine (> `--threshold`,
default 0.95) flags a likely copy. CelebA faces look alike, so read the grid
*with* the stats — judge "new plausible face" vs "pixel-level copy of a specific
training image". Use `--max-train` to subsample the search for a quick pass.

## Per-dataset envs

Each dataset is one file in [`envs/`](envs/) holding its `DATA_DIR`,
`NUM_CLASSES`, `RESOLUTION`, default model / REPA settings, default training
scale, and a `prepare_dataset()` function. To add a dataset (e.g. another
face/scene set), copy [`envs/celeba.env`](envs/celeba.env), adjust the values,
and implement `prepare_dataset()`. [`envs/imagenet.env`](envs/imagenet.env) is a
ready template for the ImageNet path.

## Why no precomputed DINOv2 features

The dataset only stores **raw images** + **VAE latents**. REPA applies the
representation encoder (`--enc-type=dinov2-vit-b`) *on-the-fly during training*
(`load_encoders` + `preprocess_raw_image` over the raw uint8 images), so the
encoder choice stays a training-time knob. Precomputing DINOv2 would freeze
`enc-type` into the dataset and save little, so the export step deliberately
skips it.

## AutoDL

If `/etc/network_turbo` exists it is sourced automatically to accelerate
HuggingFace / torch.hub / GitHub downloads. HF traffic is routed through
`HF_ENDPOINT=https://hf-mirror.com` with the Xet backend disabled
(`HF_HUB_DISABLE_XET=1`) — both are set unconditionally and can be overridden by
exporting them yourself. By default the pipeline uses the
machine's **current Python** (AutoDL base env) and just `pip install`s the deps —
it does **not** create a conda env, since the AutoDL conda mirror often fails
behind the turbo proxy. To use a conda env anyway, set `REPA_CONDA_ENV=<name>`
(activated, created only if missing). Use `REPA_SKIP_INSTALL=1` or `--skip-setup`
to skip dependency install entirely.
