#!/usr/bin/env bash
#
# Evaluate every run under runs/ with IDENTICAL sampler settings.
#
# KID/FID are only comparable across runs when the sampler settings match, so
# this script resolves one set of knobs, prints them, and applies them to every
# run it finds. Each run is scored by training/evaluate.py (which caches real and
# per-checkpoint fake features, so re-runs only do missing work), and its
# eval/kid.csv is collected into results/<dataset>/kid_<mode>.csv for
# results/analysis/analyze_kid.py.
#
# Usage:
#   training/eval_all.sh [options] [-- <extra evaluate.py args>]
#
# Options (all optional):
#       --runs-dir DIR      where the runs live            (default: <repo>/runs)
#       --gpus IDS          comma-separated GPU ids        (default: 0,1)
#       --num-samples N     generated samples per ckpt     (default: 10000)
#       --num-real N        real images in the reference   (default: 10000)
#       --gen-batch N       generation batch size          (default: 250)
#       --num-steps N       sampler steps                  (default: 50)
#       --cfg-scale F       classifier-free guidance       (default: 1.5)
#       --weights ema|model which weights to sample        (default: ema)
#       --every N           evaluate every Nth checkpoint  (default: 1)
#       --compile           torch.compile the model (one warmup per rank)
#       --include REGEX     only runs whose name matches
#       --exclude REGEX     skip runs whose name matches   (default: ^timing)
#       --collect DIR       collect CSVs here (default: results/<dataset>)
#       --no-collect        do not copy kid.csv anywhere
#       --dry-run           print what would run and exit
#   --                      pass everything after this verbatim to evaluate.py
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/lib/common.sh"

RUNS_DIR="$REPO_ROOT/runs"
GPUS="0,1"
NUM_SAMPLES=10000
NUM_REAL=10000
GEN_BATCH=250
NUM_STEPS=50
CFG_SCALE=1.5
WEIGHTS="ema"
EVERY=1
COMPILE=0
INCLUDE=""
EXCLUDE="^timing"
COLLECT=""
DO_COLLECT=1
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs-dir)     RUNS_DIR="$2"; shift 2 ;;
    --gpus)         GPUS="$2"; shift 2 ;;
    --num-samples)  NUM_SAMPLES="$2"; shift 2 ;;
    --num-real)     NUM_REAL="$2"; shift 2 ;;
    --gen-batch)    GEN_BATCH="$2"; shift 2 ;;
    --num-steps)    NUM_STEPS="$2"; shift 2 ;;
    --cfg-scale)    CFG_SCALE="$2"; shift 2 ;;
    --weights)      WEIGHTS="$2"; shift 2 ;;
    --every)        EVERY="$2"; shift 2 ;;
    --compile)      COMPILE=1; shift ;;
    --include)      INCLUDE="$2"; shift 2 ;;
    --exclude)      EXCLUDE="$2"; shift 2 ;;
    --collect)      COLLECT="$2"; shift 2 ;;
    --no-collect)   DO_COLLECT=0; shift ;;
    --dry-run)      DRY_RUN=1; shift ;;
    --)             shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help)      sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)              die "Unknown option: $1 (use --help)" ;;
  esac
done

[[ -d "$RUNS_DIR" ]] || die "No runs dir: $RUNS_DIR"

# Inception + SD-VAE weights come from HF/torch.hub on first use.
setup_hf_env
setup_autodl_network

# ---- Collect the runs worth evaluating -------------------------------------
RUNS=()
for d in "$RUNS_DIR"/*/; do
  name="$(basename "$d")"
  compgen -G "$d/checkpoints/*.pt" >/dev/null || { warn "skip $name (no checkpoints)"; continue; }
  [[ -n "$INCLUDE" && ! "$name" =~ $INCLUDE ]] && continue
  [[ -n "$EXCLUDE" && "$name" =~ $EXCLUDE ]] && { warn "skip $name (--exclude)"; continue; }
  RUNS+=("$name")
done
[[ ${#RUNS[@]} -gt 0 ]] || die "No runs with checkpoints under $RUNS_DIR"

# A ragged final batch costs an extra compile per checkpoint (new input shape).
NUM_GPUS="$(count_gpus "$GPUS")"
SHARD=$(( NUM_SAMPLES / NUM_GPUS ))
if [[ "$COMPILE" == "1" && $(( SHARD % GEN_BATCH )) -ne 0 ]]; then
  warn "--gen-batch $GEN_BATCH does not divide the $SHARD-sample per-GPU shard;"
  warn "pick a divisor of $SHARD to avoid an extra compile per checkpoint."
fi

log "runs dir : $RUNS_DIR"
log "runs     : ${RUNS[*]}"
log "settings : samples=$NUM_SAMPLES real=$NUM_REAL steps=$NUM_STEPS cfg=$CFG_SCALE"
log "           weights=$WEIGHTS every=$EVERY gen-batch=$GEN_BATCH gpus=$GPUS compile=$COMPILE"

EVAL_ARGS=(
  --gpus "$GPUS"
  --num-samples "$NUM_SAMPLES"
  --num-real "$NUM_REAL"
  --num-steps "$NUM_STEPS"
  --cfg-scale "$CFG_SCALE"
  --weights "$WEIGHTS"
  --every "$EVERY"
  --gen-batch "$GEN_BATCH"
)
[[ "$COMPILE" == "1" ]] && EVAL_ARGS+=(--compile)
EVAL_ARGS+=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})

if [[ "$DRY_RUN" == "1" ]]; then
  for name in "${RUNS[@]}"; do
    printf '  python %q --run-dir %q' "$SCRIPT_DIR/evaluate.py" "$RUNS_DIR/$name"
    printf ' %q' "${EVAL_ARGS[@]}"; echo
  done
  exit 0
fi

# ---- Evaluate ---------------------------------------------------------------
OK=(); FAILED=()
for name in "${RUNS[@]}"; do
  run="$RUNS_DIR/$name"
  log "=== evaluating $name ==="
  mkdir -p "$run/eval"
  if python "$SCRIPT_DIR/evaluate.py" --run-dir "$run" "${EVAL_ARGS[@]}" \
       2>&1 | tee "$run/eval/eval.log"; then
    OK+=("$name")
  else
    warn "FAILED: $name (see $run/eval/eval.log)"
    FAILED+=("$name")
    continue
  fi

  # runs are named <dataset>_<model>_<mode>, e.g. imagenet100_sit-b_2_repa
  if [[ "$DO_COLLECT" == "1" && -f "$run/eval/kid.csv" ]]; then
    dest="${COLLECT:-$REPO_ROOT/results/${name%%_*}}"
    mkdir -p "$dest"
    cp "$run/eval/kid.csv" "$dest/kid_${name##*_}.csv"
    log "collected -> $dest/kid_${name##*_}.csv"
  fi
done

# ---- Summary ----------------------------------------------------------------
echo
log "=== summary (best checkpoint per run) ==="
python - "$RUNS_DIR" ${OK[@]+"${OK[@]}"} <<'PY'
import csv, sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
print(f"  {'run':<40} {'best step':>10} {'KID x10^3':>12} {'FID':>9}")
for name in sys.argv[2:]:
    csv_path = runs_dir / name / "eval" / "kid.csv"
    if not csv_path.exists():
        print(f"  {name:<40} {'-':>10} {'no kid.csv':>12}")
        continue
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        print(f"  {name:<40} {'-':>10} {'empty':>12}")
        continue
    best = min(rows, key=lambda r: float(r["kid_mean"]))
    print(f"  {name:<40} {best['step']:>10} "
          f"{float(best['kid_x1e3']):>12.3f} {float(best['fid']):>9.3f}")
PY

if [[ ${#FAILED[@]} -gt 0 ]]; then
  warn "failed runs: ${FAILED[*]}"
  exit 1
fi
log "done: ${#OK[@]} run(s) evaluated"
