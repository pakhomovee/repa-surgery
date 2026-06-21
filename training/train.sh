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
#       --mode MODE           one of:
#                               repa       encoder alignment (default)
#                               baseline   plain SiT (enc-type=none, proj-coeff=0)
#                               haste      alignment until HASTE_END_STEP, then
#                                          pure diffusion (REPA early-stopping)
#                               repa-PCGrad PCGrad gradient surgery (forces bf16)
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
  repa|baseline|haste|repa-PCGrad) ;;
  *) die "Invalid --mode '$MODE' (expected 'repa', 'baseline', 'haste', or 'repa-PCGrad')" ;;
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

# Capture the dataset's encoder + precomputed-representation paths *before* any
# baseline override of ENC_TYPE. REPR_DIR is REPA-relative (passed to train.py);
# REPR_PATH is absolute (existence checks / precompute source).
REPR_ENC="$ENC_TYPE"
REPR_DIR="${DATA_DIR}/repr-${REPR_ENC}"
REPR_PATH="${REPO_ROOT}/data/$(basename "$DATA_DIR")/repr-${REPR_ENC}"

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

# ---- Optional: precompute encoder representations (opt-in; ~50-80 GB) -------
# Removes the per-step encoder forward + raw-image read from repa/haste/repa-PCGrad.
# Once present, training auto-uses it (see --repr-dir wiring below).
if [[ "${PRECOMPUTE_REPR:-0}" == "1" && "$MODE" != "baseline" && ! -f "${REPR_PATH}/meta.json" ]]; then
  log "Precomputing ${REPR_ENC} representations -> ${REPR_PATH}"
  python "${REPO_ROOT}/datasets/encode_repr.py" \
    --source "${REPO_ROOT}/data/$(basename "$DATA_DIR")/images" \
    --dest "${REPR_PATH}" \
    --enc-type "${REPR_ENC}" \
    --resolution "${RESOLUTION}" \
    --gpus "${GPUS}" \
    --batch-size "${ENCODE_BATCH_SIZE:-64}" \
    --num-workers "${ENCODE_NUM_WORKERS:-8}"
fi

# ---- Resolve effective settings (CLI > env default) ------------------------
MODEL="${MODEL:-$DEFAULT_MODEL}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
NUM_WORKERS="${NUM_WORKERS:-$DEFAULT_NUM_WORKERS}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-$DEFAULT_CHECKPOINTING_STEPS}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-$DEFAULT_MAX_TRAIN_STEPS}"

# ---- Mode-specific configuration -------------------------------------------
# baseline: no alignment.  repa: alignment from env.  haste: alignment until a
# termination step, then pure diffusion.  repa-PCGrad: PCGrad gradient surgery.
PRECISION="fp16"
MODE_ARGS=()
case "$MODE" in
  baseline)
    ENC_TYPE="none"; PROJ_COEFF=0.0 ;;
  repa)
    ;;
  haste)
    MODE_ARGS+=("--alignment-end-step=${HASTE_END_STEP:-100000}") ;;
  repa-PCGrad)
    # Manual two-gradient surgery is incompatible with fp16's GradScaler.
    PRECISION="bf16"
    MODE_ARGS+=("--grad-surgery" "--grad-ema-decay=${GRAD_EMA_DECAY:-0.99}")
    # PRECOND=1 measures the conflict in Adam's whitened metric (see --precond).
    [[ "${PRECOND:-0}" == "1" ]] && MODE_ARGS+=("--precond") ;;
esac

# Auto-use precomputed encoder representations for alignment modes when present
# (set NO_REPR=1 to force the on-the-fly encoder instead).
if [[ "$MODE" != "baseline" && "${NO_REPR:-0}" != "1" && -f "${REPR_PATH}/meta.json" ]]; then
  MODE_ARGS+=("--repr-dir=${REPR_DIR}")
  log "using precomputed representations: ${REPR_PATH}"
fi

# Auto exp-name, e.g. celeba_sit-b_2_repa  /  celeba_sit-b_2_haste
if [[ -z "$EXP_NAME" ]]; then
  _m="$(echo "$MODEL" | tr 'A-Z/' 'a-z_' | tr -d ' ')"
  EXP_NAME="${DATASET}_${_m}_${MODE}"
fi

# ---- Build launch command --------------------------------------------------
export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

log "dataset=$DATASET model=$MODEL gpus=$GPUS (procs=$NUM_PROCESSES) bs=$BATCH_SIZE"
log "mode=$MODE enc-type=$ENC_TYPE proj-coeff=$PROJ_COEFF precision=$PRECISION exp-name=$EXP_NAME"

CMD=(
  accelerate launch
    --num_processes "$NUM_PROCESSES"
    --mixed_precision "$PRECISION"
    --main_process_port "${MAIN_PROCESS_PORT:-29521}"
  train.py
    --report-to="${REPORT_TO:-tensorboard}"
    --allow-tf32
    --mixed-precision="$PRECISION"
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
CMD+=(${MODE_ARGS[@]+"${MODE_ARGS[@]}"})
CMD+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

if [[ "$DRY_RUN" == "1" ]]; then
  log "Dry run -- command (from $REPA_DIR):"
  printf '  %q' "${CMD[@]}"; echo
  exit 0
fi

log "Launching from $REPA_DIR"
cd "$REPA_DIR"
exec "${CMD[@]}"
