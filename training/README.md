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

## Training mode

`--mode repa` (default) trains with representation alignment using the encoder
configured in the dataset env (`ENC_TYPE`/`PROJ_COEFF`). `--mode baseline`
(alias: `--baseline`) trains plain SiT with `enc-type=none, proj-coeff=0`. The
mode is reflected in the auto exp-name (`..._dinov2-vit-b` vs `..._baseline`).

`--dry-run` prints the exact `accelerate` command without launching.

## Tunable parameters

CLI flags override the dataset env defaults: `--gpus`, `--model`,
`--batch-size`, `--num-workers`, `--checkpointing-steps`, `--max-train-steps`,
`--exp-name`. `--gpus 0,1,2,3` both sets `CUDA_VISIBLE_DEVICES` and derives the
number of `accelerate` processes. Anything after `--` is passed verbatim to
`train.py`. Run `training/train.sh --help` for the full list.

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
