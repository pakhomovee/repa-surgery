# Shared helpers for the REPA training pipeline. Sourced by train.sh.
# Not meant to be executed directly.

# Resolve the repo root (parent of this lib/ -> training/ -> repo root).
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$_LIB_DIR/../.." && pwd)"
REPA_DIR="$REPO_ROOT/REPA"

log()  { printf '\033[1;34m[train]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[train]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[train]\033[0m %s\n' "$*" >&2; exit 1; }

# --- AutoDL network acceleration -------------------------------------------
# On AutoDL GPU machines, sourcing /etc/network_turbo proxies outbound traffic
# (HuggingFace / torch.hub / github) through a fast mirror. No-op elsewhere.
setup_autodl_network() {
  if [[ -f /etc/network_turbo ]]; then
    log "AutoDL detected: sourcing /etc/network_turbo"
    # shellcheck disable=SC1091
    source /etc/network_turbo
  fi
}

# --- Hugging Face download settings ----------------------------------------
# Route HF traffic through the hf-mirror.com mirror (fast + reachable from CN
# boxes) and disable the Xet chunked-transfer backend, whose CAS server returns
# 401s through the network_turbo proxy. Honors any pre-set HF_ENDPOINT/HF_TOKEN.
# Always run (even with --skip-setup) since it only sets env vars.
setup_hf_env() {
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  log "HF_ENDPOINT=$HF_ENDPOINT HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET"
}

# --- Python environment -----------------------------------------------------
# By default uses the currently active interpreter (AutoDL images already ship
# a working CUDA-enabled Python base env) and just pip-installs the deps.
#
# Conda is OPT-IN: set REPA_CONDA_ENV=<name> to activate that env, creating it
# only if missing. We deliberately do NOT auto-create a conda env, because on
# AutoDL the configured conda mirror often fails behind the network_turbo proxy.
#
# Skip dependency install with REPA_SKIP_INSTALL=1.
setup_python_env() {
  local env_name="${REPA_CONDA_ENV:-}"

  if [[ -n "$env_name" ]]; then
    command -v conda >/dev/null 2>&1 || die "REPA_CONDA_ENV=$env_name set but conda not found"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -qE "^\s*${env_name}\s"; then
      log "Creating conda env '$env_name' (python=3.10)"
      conda create -n "$env_name" python=3.10 -y
    fi
    log "Activating conda env '$env_name'"
    conda activate "$env_name"
  else
    log "Using current python: $(command -v python) ($(python --version 2>&1))"
  fi

  if [[ "${REPA_SKIP_INSTALL:-0}" != "1" ]]; then
    log "Installing dependencies (set REPA_SKIP_INSTALL=1 to skip)"
    pip install -q -r "$REPA_DIR/requirements.txt"
    pip install -q -r "$_LIB_DIR/../requirements-extra.txt"
  else
    log "REPA_SKIP_INSTALL=1: skipping dependency install"
  fi
}

# Count comma-separated GPU ids: "0,1,2,3" -> 4
count_gpus() {
  local ids="$1"
  awk -F',' '{print NF}' <<<"$ids"
}
