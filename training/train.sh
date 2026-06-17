#!/usr/bin/env bash
#
# Single entry point for REPA training.
#
# It (1) sets up the environment (AutoDL network turbo + python deps),
#    (2) loads a per-dataset env from training/envs/<dataset>.env,
#    (3) optionally prepares the dataset (export + VAE encode),
#    (4) launches REPA/train.py via `accelerate launch`.
#
# Usage:
#   training/train.sh --dataset celeba [options] [-- <extra train.py args>]
#
# Options (all optional except --dataset):
#   -d, --dataset NAME        env to load: training/envs/NAME.env   [required]
#       --gpus IDS            comma-separated GPU ids, e.g. 0,1,2,3 (default: 0)
#       --model NAME          SiT model, e.g. SiT-B/2, SiT-XL/2
#       --batch-size N        global batch size
#       --num-workers N       dataloader workers per process
#       --checkpointing-steps N
#       --max-train-steps N
#       --exp-name NAME       run name (default: auto from dataset/model/enc)
#       --mode MODE           "repa" (encoder alignment, default) or "baseline"
#                             (plain SiT: enc-type=none, proj-coeff=0)
#       --baseline            shortcut for --mode baseline
#       --prepare             run dataset prep before training
#       --skip-setup          skip env + dependency setup
#       --dry-run             print the launch command and exit
#   --                        pass everything after this verbatim to train.py
#
# Env vars: REPA_CONDA_ENV (opt-in conda env; default: use current python),
#           REPA_SKIP_INSTALL=1 (skip pip install),
#           MAIN_PROCESS_PORT (default 29521), WANDB_MODE (default offline).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

# ---- Defaults / CLI state --------------------------------------------------
DATASET=""
GPUS=""
MODEL=""
BATCH_SIZE=""
NUM_WORKERS=""
CHECKPOINTING_STEPS=""
MAX_TRAIN_STEPS=""
EXP_NAME=""
MODE="repa"
DO_PREPARE=0
SKIP_SETUP=0
DRY_RUN=0
EXTRA_ARGS=()

# ---- Arg parsing -----------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--dataset)            DATASET="$2"; shift 2 ;;
    --gpus)                  GPUS="$2"; shift 2 ;;
    --model)                 MODEL="$2"; shift 2 ;;
    --batch-size)            BATCH_SIZE="$2"; shift 2 ;;
    --num-workers)           NUM_WORKERS="$2"; shift 2 ;;
    --checkpointing-steps)   CHECKPOINTING_STEPS="$2"; shift 2 ;;
    --max-train-steps)       MAX_TRAIN_STEPS="$2"; shift 2 ;;
    --exp-name)              EXP_NAME="$2"; shift 2 ;;
    --mode)                  MODE="$2"; shift 2 ;;
    --baseline)              MODE="baseline"; shift ;;
    --prepare)               DO_PREPARE=1; shift ;;
    --skip-setup)            SKIP_SETUP=1; shift ;;
    --dry-run)               DRY_RUN=1; shift ;;
    --)                      shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help)               sed -n '2,40p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)                       die "Unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$DATASET" ]] || die "Missing required --dataset (e.g. --dataset celeba)"
ENV_FILE="$SCRIPT_DIR/envs/${DATASET}.env"
[[ -f "$ENV_FILE" ]] || die "No env file: $ENV_FILE"

case "$MODE" in
  repa|baseline) ;;
  *) die "Invalid --mode '$MODE' (expected 'repa' or 'baseline')" ;;
esac

# ---- Environment setup -----------------------------------------------------
# HF env vars are set unconditionally (they only affect downloads).
setup_hf_env
if [[ "$SKIP_SETUP" == "0" ]]; then
  setup_autodl_network
  setup_python_env
else
  log "--skip-setup: not touching network/python env"
fi

# ---- Load dataset env ------------------------------------------------------
log "Loading dataset env: $ENV_FILE"
# shellcheck disable=SC1090
source "$ENV_FILE"

# Resolve GPUs early so dataset prep (multi-GPU VAE encode) can use them.
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
NUM_PROCESSES="$(count_gpus "$GPUS")"
export GPUS

# ---- Dataset preparation ---------------------------------------------------
if [[ "$DO_PREPARE" == "1" ]]; then
  if declare -F prepare_dataset >/dev/null; then
    log "Preparing dataset '$DATASET'"
    prepare_dataset
  else
    warn "Env '$DATASET' defines no prepare_dataset(); skipping prep"
  fi
fi

# ---- Resolve effective settings (CLI > env default) ------------------------
MODEL="${MODEL:-$DEFAULT_MODEL}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-$DEFAULT_CHECKPOINTING_STEPS}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-$DEFAULT_MAX_TRAIN_STEPS}"

# baseline mode => disable encoder alignment; repa mode keeps the env's enc.
if [[ "$MODE" == "baseline" ]]; then
  ENC_TYPE="none"
  PROJ_COEFF=0.0
fi

# Auto exp-name, e.g. celeba_sit_b2_dinov2-vit-b  /  celeba_sit_b2_baseline
if [[ -z "$EXP_NAME" ]]; then
  _m="$(echo "$MODEL" | tr 'A-Z/' 'a-z_' | tr -d ' ')"
  if [[ "$ENC_TYPE" == "none" ]]; then _tag="baseline"; else _tag="$ENC_TYPE"; fi
  EXP_NAME="${DATASET}_${_m}_${_tag}"
fi

# ---- Build launch command --------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

log "dataset=$DATASET model=$MODEL gpus=$GPUS (procs=$NUM_PROCESSES) bs=$BATCH_SIZE"
log "enc-type=$ENC_TYPE proj-coeff=$PROJ_COEFF exp-name=$EXP_NAME"

CMD=(
  accelerate launch
    --num_processes "$NUM_PROCESSES"
    --mixed_precision "fp16"
    --main_process_port "${MAIN_PROCESS_PORT:-29521}"
  train.py
    --report-to="${REPORT_TO:-tensorboard}"
    --allow-tf32
    --mixed-precision="fp16"
    --seed=0
    --path-type="$PATH_TYPE"
    --prediction="$PREDICTION"
    --weighting="$WEIGHTING"
    --model="$MODEL"
    --num-classes="$NUM_CLASSES"
    --enc-type="$ENC_TYPE"
    --proj-coeff="$PROJ_COEFF"
    --encoder-depth="$ENCODER_DEPTH"
    --output-dir="../runs"
    --logging-dir="${LOGGING_DIR:-logs}"
    --exp-name="$EXP_NAME"
    --data-dir="$DATA_DIR"
    --resolution="$RESOLUTION"
    --batch-size="$BATCH_SIZE"
    --max-train-steps="$MAX_TRAIN_STEPS"
    --checkpointing-steps="$CHECKPOINTING_STEPS"
    --sampling-steps="${DEFAULT_SAMPLING_STEPS}"
    --num-workers="$NUM_WORKERS"
)
CMD+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run -- command (from $REPA_DIR):"
  printf '  %q' "${CMD[@]}"; echo
  exit 0
fi

log "Launching from $REPA_DIR"
cd "$REPA_DIR"
exec "${CMD[@]}"
