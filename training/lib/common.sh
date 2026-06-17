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

# --- Python environment -----------------------------------------------------
# Activates conda env $REPA_ENV_NAME (default "repa"), creating it if conda is
# available and the env is missing. Falls back to the current interpreter.
# Installs REPA + pipeline deps unless REPA_SKIP_INSTALL=1.
setup_python_env() {
  local env_name="${REPA_ENV_NAME:-repa}"

  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    if ! conda env list | grep -qE "^\s*${env_name}\s"; then
      log "Creating conda env '$env_name' (python=3.9)"
      conda create -n "$env_name" python=3.9 -y
    fi
    log "Activating conda env '$env_name'"
    conda activate "$env_name"
  else
    warn "conda not found; using current python: $(command -v python)"
  fi

  if [[ "${REPA_SKIP_INSTALL:-0}" != "1" ]]; then
    log "Installing dependencies"
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
