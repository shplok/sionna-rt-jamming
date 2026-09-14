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
    # Northeastern Explorer (Mariona). miniconda in $HOME, project storage under
    # /projects/ipl_lab -- the dataset is ~73 GB and must NOT land on the home quota.
    export CONDA_ENV_NAME="${CONDA_ENV_NAME:-sionna}"
    export CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
    export CONDA_BIN_DIR="${CONDA_BIN_DIR:-$HOME/miniconda3/envs/sionna/bin}"
    export SIONNA_DATASET_ROOT="${SIONNA_DATASET_ROOT:-/projects/ipl_lab/$USER/sionna-rt-jamming/datasets}"
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

# Shared across sites
export MITSUBA_VARIANT="${MITSUBA_VARIANT:-cuda_ad_mono_polarized}"
export SIONNA_GPU_MEM="${SIONNA_GPU_MEM:-16G}"
export SIONNA_CPU_MEM="${SIONNA_CPU_MEM:-32G}"
export SIONNA_GPU_TIME="${SIONNA_GPU_TIME:-04:00:00}"
export SIONNA_CPU_TIME="${SIONNA_CPU_TIME:-08:00:00}"
export SIONNA_SITE

# Activate the environment the way the caller's shell can actually do it. `conda
# activate` only works after conda's shell hook has been sourced, which is not
# guaranteed inside a batch job even when it works interactively.
sionna_activate_conda() {
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
    echo "[server_env] ERROR: could not activate '${CONDA_ENV_NAME}'." >&2
    echo "[server_env]   tried CONDA_SH=$CONDA_SH" >&2
    echo "[server_env]   tried CONDA_BIN_DIR=$CONDA_BIN_DIR" >&2
    return 1
}

# Activate the environment the way the caller's shell can actually do it. `conda
# activate` only works after conda's shell hook has been sourced, which is not
# guaranteed inside a batch job even when it works interactively.
sionna_activate_conda() {
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
    echo "[server_env] ERROR: could not activate '${CONDA_ENV_NAME}'." >&2
    echo "[server_env]   tried CONDA_SH=$CONDA_SH" >&2
    echo "[server_env]   tried CONDA_BIN_DIR=$CONDA_BIN_DIR" >&2
    return 1
}
