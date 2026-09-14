#!/bin/bash
# ==============================================================================
# Server & Environment Configuration
#
# Two site profiles. The site is auto-detected; override with
#     export SIONNA_SITE=explorer      (or: pomplun)
# Any individual value can also be overridden by exporting it before sourcing.
#
# Sourced by run_pipeline_gpu.sh, run_pipeline_cpu.sh and submit.sh.
#
# NOTE: these are deliberately SIONNA_* and not SLURM_*. SLURM reads several SLURM_*
# variables as *input* (SLURM_ACCOUNT, SLURM_PARTITION, SLURM_MEM_PER_NODE, ...), and
# inside a job allocation they are already set to the parent job's values. Using our
# own namespace keeps the two from fighting.
# ==============================================================================

# repo root, so SIONNA_VENV can default to <repo>/.venv regardless of cwd
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

_sionna_detect_site() {
    [ -n "$SIONNA_SITE" ] && { echo "$SIONNA_SITE"; return; }
    local h; h="$(hostname -f 2>/dev/null || hostname)"
    case "$h" in
        *explorer*|*northeastern*) echo "explorer"; return ;;
    esac
    if command -v sinfo >/dev/null 2>&1; then
        local parts; parts="$(sinfo -h -o '%P' 2>/dev/null | tr -d '*' | sort -u)"
        echo "$parts" | grep -qx 'pomplun'  && { echo "pomplun";  return; }
        echo "$parts" | grep -qx 'gpu'      && { echo "explorer"; return; }
    fi
    echo "unknown"
}

SIONNA_SITE="$(_sionna_detect_site)"

case "$SIONNA_SITE" in

  explorer)
    # Northeastern Explorer (Mariona). The home quota is small and already full, so
    # BOTH miniconda and the dataset live under /projects/ipl_lab. The separate
    # miniconda in $HOME (grpo-motion etc.) is left alone -- do not use it here.
    export SIONNA_PROJ="${SIONNA_PROJ:-/projects/ipl_lab/$USER}"
    # A venv in the repo, not a conda env. Everything here is pip-installable, and
    # `source .venv/bin/activate` behaves identically in a login shell and a batch
    # job -- unlike `conda activate`, which needs a shell hook that is often absent
    # inside SLURM. miniconda under $SIONNA_PROJ is only the interpreter source.
    export SIONNA_VENV="${SIONNA_VENV:-$SCRIPT_DIR/.venv}"
    export CONDA_ENV_NAME="${CONDA_ENV_NAME:-sionna}"
    export CONDA_SH="${CONDA_SH:-$SIONNA_PROJ/miniconda3/etc/profile.d/conda.sh}"
    export CONDA_BIN_DIR="${CONDA_BIN_DIR:-$SIONNA_PROJ/miniconda3/envs/sionna/bin}"
    # the repo itself already lives on /projects, so repo-relative is fine and avoids
    # a second copy of the path
    export SIONNA_DATASET_ROOT="${SIONNA_DATASET_ROOT:-./datasets}"
    export SIONNA_PARTITION="${SIONNA_PARTITION:-gpu}"
    export SIONNA_ACCOUNT="${SIONNA_ACCOUNT:-}"          # Explorer does not need one
    export SIONNA_GPU_GRES="${SIONNA_GPU_GRES:-gpu:a100:1}"   # or gpu:h200:1
    export SIONNA_GPU_PARTITION="${SIONNA_GPU_PARTITION:-gpu}"
    export SIONNA_CPU_PARTITION="${SIONNA_CPU_PARTITION:-short}"
    ;;

  pomplun)
    # Original configuration from Luis. Left exactly as pulled.
    export CONDA_ENV_NAME="${CONDA_ENV_NAME:-sionna}"
    export CONDA_SH="${CONDA_SH:-}"
    export CONDA_BIN_DIR="${CONDA_BIN_DIR:-/pomplun/share_home/l.gonzalezgudino001/.conda/envs/sionna/bin}"
    export SIONNA_DATASET_ROOT="${SIONNA_DATASET_ROOT:-./datasets}"
    export SIONNA_PARTITION="${SIONNA_PARTITION:-pomplun}"
    export SIONNA_ACCOUNT="${SIONNA_ACCOUNT:-cs_tales.imbiriba}"
    export SIONNA_GPU_GRES="${SIONNA_GPU_GRES:-gpu:1}"
    export SIONNA_GPU_PARTITION="${SIONNA_GPU_PARTITION:-pomplun}"
    export SIONNA_CPU_PARTITION="${SIONNA_CPU_PARTITION:-pomplun}"
    ;;

  *)
    echo "[server_env] WARNING: unknown site '$SIONNA_SITE'. Set SIONNA_SITE=explorer" >&2
    echo "[server_env]          or SIONNA_SITE=pomplun, or export the values by hand." >&2
    export CONDA_ENV_NAME="${CONDA_ENV_NAME:-sionna}"
    export CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    export CONDA_BIN_DIR="${CONDA_BIN_DIR:-$HOME/miniconda3/envs/sionna/bin}"
    export SIONNA_DATASET_ROOT="${SIONNA_DATASET_ROOT:-./datasets}"
    export SIONNA_PARTITION="${SIONNA_PARTITION:-gpu}"
    export SIONNA_ACCOUNT="${SIONNA_ACCOUNT:-}"
    export SIONNA_GPU_GRES="${SIONNA_GPU_GRES:-gpu:1}"
    export SIONNA_GPU_PARTITION="${SIONNA_GPU_PARTITION:-gpu}"
    export SIONNA_CPU_PARTITION="${SIONNA_CPU_PARTITION:-$SIONNA_PARTITION}"
    ;;
esac

# --- keep third-party tools off $HOME -------------------------------------------
# Nothing in this repo writes to $HOME, but several dependencies do by default. On
# Explorer the per-user home quota is small and frequently full, and the failures are
# obscure: conda reports solver errors, drjit silently disables its cache, matplotlib
# warns about fonts, and Claude Code cannot persist its OAuth token (401 loops).
if [ -n "$SIONNA_PROJ" ] && [ -d "$SIONNA_PROJ" ]; then
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SIONNA_PROJ/.cache}"
    export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SIONNA_PROJ/pip-cache}"
    export MPLCONFIGDIR="${MPLCONFIGDIR:-$SIONNA_PROJ/.cache/matplotlib}"
    export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$SIONNA_PROJ/conda-pkgs}"
    mkdir -p "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" 2>/dev/null || true
fi
# drjit/mitsuba need libLLVM for the CPU backend; it is present on Explorer but not
# on the loader path. Without it `import sionna.rt` fails on CPU nodes.
if [ -z "$DRJIT_LIBLLVM_PATH" ]; then
    for _l in /usr/lib64/libLLVM.so* /usr/lib/x86_64-linux-gnu/libLLVM.so*; do
        [ -e "$_l" ] && { export DRJIT_LIBLLVM_PATH="$_l"; break; }
    done
fi
# NOTE: drjit's kernel cache path is derived from $HOME and has no env override. When
# home is full it prints `could not use "~/.drjit"` and disables the disk cache. That
# is a compile-speed hit only, not a correctness problem -- the real fix is freeing
# home space, not an environment variable.

# Shared across sites
export MITSUBA_VARIANT="${MITSUBA_VARIANT:-cuda_ad_mono_polarized}"
export SIONNA_GPU_MEM="${SIONNA_GPU_MEM:-16G}"
export SIONNA_CPU_MEM="${SIONNA_CPU_MEM:-32G}"
export SIONNA_GPU_TIME="${SIONNA_GPU_TIME:-04:00:00}"
export SIONNA_CPU_TIME="${SIONNA_CPU_TIME:-08:00:00}"
export SIONNA_SITE

# Activate the project environment. A venv is tried first: `source bin/activate` is
# a plain shell script and works the same in an interactive shell and a batch job,
# whereas `conda activate` needs conda's shell hook, which is frequently not sourced
# inside SLURM even when it works when you log in. Conda is kept as a fallback for
# sites that use it.
sionna_activate_env() {
    if [ -n "$SIONNA_VENV" ] && [ -f "$SIONNA_VENV/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$SIONNA_VENV/bin/activate" && return 0
    fi
    if [ -n "$CONDA_SH" ] && [ -f "$CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$CONDA_SH" && conda activate "${CONDA_ENV_NAME:-sionna}" && return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        conda activate "${CONDA_ENV_NAME:-sionna}" 2>/dev/null && return 0
    fi
    if [ -d "$CONDA_BIN_DIR" ]; then
        export PATH="$CONDA_BIN_DIR:$PATH" && return 0
    fi
    echo "[server_env] ERROR: no environment found." >&2
    echo "[server_env]   venv : $SIONNA_VENV/bin/activate" >&2
    echo "[server_env]   conda: $CONDA_SH  (env '${CONDA_ENV_NAME}')" >&2
    return 1
}

# backwards-compatible alias
sionna_activate_conda() { sionna_activate_env "$@"; }
