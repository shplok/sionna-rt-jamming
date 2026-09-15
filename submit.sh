#!/bin/bash
# ==============================================================================
# Submit the pipeline, picking the SLURM settings for whichever cluster you are on.
#
#   ./submit.sh show         # print the resolved settings and exit
#   ./submit.sh verify       # read-only preflight: partitions, env, GPU, data
#   ./submit.sh smoke        # 20-position simulate_static test
#   ./submit.sh gpu          # stage A: ray tracing            (needs a GPU)
#   ./submit.sh cpu          # stage B: everything after       (no GPU)
#
# Safe to run from inside an srun allocation: sbatch is just a client, and the
# submitted job queues independently. Inherited SLURM_* job variables are stripped
# before submitting so the child does not pick up this shell's memory/ntasks.
#
# #SBATCH directives inside a job script are static comments and cannot be made
# conditional, so this wrapper passes the site-specific values on the sbatch command
# line, where they override the in-file defaults.
#
# Override anything by exporting it first, e.g.
#   SIONNA_GPU_GRES=gpu:h200:1 ./submit.sh gpu
# ==============================================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./server_env.sh

ACTION="${1:-show}"
mkdir -p logs

_acct() { [ -n "$SIONNA_ACCOUNT" ] && echo "--account=$SIONNA_ACCOUNT"; }

# Inside an allocation, SLURM_* describe THIS job and some are read as input by
# sbatch/srun. Strip them so a submitted job gets only what we pass explicitly.
_clean() {
    env $(env | grep -oE '^(SLURM|SBATCH)_[A-Z0-9_]+' | sed 's/^/-u /' | tr '\n' ' ') "$@"
}

IN_ALLOC=""
[ -n "$SLURM_JOB_ID" ] && IN_ALLOC="yes"

case "$ACTION" in

  show)
    echo "site               : $SIONNA_SITE"
    echo "conda env / bin    : $CONDA_ENV_NAME  ($CONDA_BIN_DIR)"
    echo "mitsuba variant    : $MITSUBA_VARIANT"
    echo "GPU  partition     : $SIONNA_GPU_PARTITION"
    echo "GPU  gres          : $SIONNA_GPU_GRES"
    echo "GPU  mem / time    : $SIONNA_GPU_MEM / $SIONNA_GPU_TIME"
    echo "CPU  partition     : $SIONNA_CPU_PARTITION"
    echo "CPU  mem / time    : $SIONNA_CPU_MEM / $SIONNA_CPU_TIME"
    echo "account            : ${SIONNA_ACCOUNT:-<none>}"
    echo
    echo "inside an allocation : ${IN_ALLOC:-no}${SLURM_JOB_ID:+  (job $SLURM_JOB_ID)}"
    echo
    echo "If the partitions look wrong:  sinfo -o '%P %G %l'"
    ;;

  verify)
    # Read-only preflight. Nothing here writes to the dataset.
    echo "--- site ---";          "$0" show
    echo; echo "--- partitions available ---"
    sinfo -o '%P %G %l %D' 2>/dev/null | head -20 || echo "sinfo unavailable"
    echo; echo "--- does the configured GPU partition exist? ---"
    if sinfo -h -p "$SIONNA_GPU_PARTITION" -o '%P %G' 2>/dev/null | grep -q .; then
        echo "  OK   $SIONNA_GPU_PARTITION"
    else
        echo "  BAD  partition '$SIONNA_GPU_PARTITION' not found -- set SIONNA_GPU_PARTITION"
    fi
    if sinfo -h -p "$SIONNA_CPU_PARTITION" -o '%P' 2>/dev/null | grep -q .; then
        echo "  OK   $SIONNA_CPU_PARTITION"
    else
        echo "  BAD  partition '$SIONNA_CPU_PARTITION' not found -- set SIONNA_CPU_PARTITION"
    fi
    echo; echo "--- your associations ---"
    sacctmgr -n show assoc user="$USER" format=account,partition 2>/dev/null | head || echo "  n/a"
    echo; echo "--- GPU visible here? ---"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
        || echo "  no GPU in this shell (fine if you are on a login node)"
    echo; echo "--- input data ---"
    for d in ./data/NYC3KM_585751_4512036/simple_OSM_scene.xml ./data/NYC3KM_585751_4512036/mesh; do
        [ -e "$d" ] && echo "  OK   $d" || echo "  BAD  missing $d"
    done
    n=$(ls ./datasets/batch_simulation_nyc/single_trajectory_jammers/traj_*.npy 2>/dev/null | wc -l)
    echo "  $n trajectory .npy files present (54 expected; 0 is fine, stage A1 makes them)"
    echo; echo "--- python environment ---"
    if [ -f "$SIONNA_VENV/bin/activate" ]; then
        echo "  OK   venv $SIONNA_VENV"
        base=$(grep -E '^home' "$SIONNA_VENV/pyvenv.cfg" 2>/dev/null | cut -d= -f2- | xargs)
        echo "       interpreter source: ${base:-?}"
        case "$base" in
          "$HOME"*) echo "  WARN base interpreter is under \$HOME -- the venv breaks if that"
                    echo "       install is removed. Prefer one under /projects." ;;
        esac
    else
        echo "  BAD  no venv at $SIONNA_VENV -- create it with:"
        echo "       <py311>/bin/python3.11 -m venv $SIONNA_VENV"
    fi
    if sionna_activate_env >/dev/null 2>&1; then
        echo "  OK   activated -> $(command -v python) ($(python -V 2>&1))"
        # these must run AFTER activation, or they report ModuleNotFoundError for a
        # perfectly good venv
        python -c "import numpy, scipy, trimesh, matplotlib; print('  OK   core: numpy', numpy.__version__)" \
            2>&1 | tail -1
        python - <<'PYCHK' 2>&1 | tail -3
import os, sys
try:
    import mitsuba as mi, drjit
    print("  OK   mitsuba", mi.__version__, "| drjit", drjit.__version__)
except Exception as e:
    print("  BAD  mitsuba/drjit:", e); sys.exit(0)
try:
    import sionna.rt
    print("  OK   sionna.rt imports")
except Exception as e:
    print("  BAD  sionna.rt:", str(e)[:200])
    if "libLLVM" in str(e):
        print("       -> DRJIT_LIBLLVM_PATH is unset or wrong. server_env.sh globs for it;")
        print("          check with: ls /usr/lib64/libLLVM.so*")
PYCHK
    else
        echo "  BAD  could not activate any environment"
    fi
    echo "  DRJIT_LIBLLVM_PATH=${DRJIT_LIBLLVM_PATH:-<unset>}"

    echo; echo "--- home quota (conda and pip fail obscurely when this is full) ---"
    if command -v check-quota >/dev/null 2>&1; then
        check-quota "$HOME" 2>&1 | sed 's/^/       /' | head -8
        echo "       (check-quota only answers from a compute node, not a login node)"
    else
        { quota -s 2>/dev/null || lfs quota -h -u "$USER" "$HOME" 2>/dev/null; } \
            | tail -3 | sed 's/^/       /' || echo "       (no quota tool found)"
    fi
    hu=$(du -sh "$HOME" 2>/dev/null | cut -f1); echo "  \$HOME usage: ${hu:-?}"
    if ! ( : > "$HOME/.sionna_write_test" ) 2>/dev/null; then
        echo "  BAD  \$HOME is NOT writable -- quota is full. conda/pip will fail with"
        echo "       'Errno 122 Disk quota exceeded'. Run: conda clean --all -y"
    else
        rm -f "$HOME/.sionna_write_test"; echo "  OK   \$HOME writable"
    fi

    echo; echo "--- dataset storage ---"
    echo "  root: $SIONNA_DATASET_ROOT"
    if mkdir -p "$SIONNA_DATASET_ROOT" 2>/dev/null && [ -w "$SIONNA_DATASET_ROOT" ]; then
        avail=$(df -BG --output=avail "$SIONNA_DATASET_ROOT" 2>/dev/null | tail -1 | tr -dc '0-9')
        [ -z "$avail" ] && avail=$(df -g "$SIONNA_DATASET_ROOT" 2>/dev/null | tail -1 | awk '{print $4}')
        echo "  OK   writable, ${avail:-?} GB free  (the run needs ~80 GB)"
        if [ -n "$avail" ] && [ "$avail" -lt 100 ] 2>/dev/null; then
            echo "  WARN under 100 GB free -- the dataset is ~73 GB plus slack"
        fi
        case "$SIONNA_DATASET_ROOT" in
          "$HOME"*|./*) echo "  WARN this looks like home or the repo. Home quotas are usually"
                        echo "       far smaller than 73 GB -- point SIONNA_DATASET_ROOT at"
                        echo "       project storage instead." ;;
        esac
    else
        echo "  BAD  not writable: $SIONNA_DATASET_ROOT"
    fi
    df -h "$SIONNA_DATASET_ROOT" 2>/dev/null | tail -1 | sed 's/^/       /'
    ;;

  gpu)
    echo "[submit] $SIONNA_SITE -> $SIONNA_GPU_PARTITION / $SIONNA_GPU_GRES"
    [ -n "$IN_ALLOC" ] && echo "[submit] submitting from inside job $SLURM_JOB_ID (fine)"
    _clean sbatch --partition="$SIONNA_GPU_PARTITION" \
           --gres="$SIONNA_GPU_GRES" \
           --mem="$SIONNA_GPU_MEM" \
           --time="$SIONNA_GPU_TIME" \
           $(_acct) \
           run_pipeline_gpu.sh
    ;;

  cpu)
    echo "[submit] $SIONNA_SITE -> $SIONNA_CPU_PARTITION (no GPU)"
    [ -n "$IN_ALLOC" ] && echo "[submit] submitting from inside job $SLURM_JOB_ID (fine)"
    _clean sbatch --partition="$SIONNA_CPU_PARTITION" \
           --mem="$SIONNA_CPU_MEM" \
           --time="$SIONNA_CPU_TIME" \
           $(_acct) \
           run_pipeline_cpu.sh
    ;;

  smoke)
    # 20 positions is enough to prove simulate_static works end to end: it exercises
    # scene loading, the solver call, the memmap write and the progress checkpoint.
    DS="./datasets/batch_simulation_smoke"
    if [ -n "$IN_ALLOC" ]; then
        echo "[submit] already inside job $SLURM_JOB_ID -- running the smoke test here"
        RUNNER=(bash -c)
    else
        echo "[submit] smoke test via srun on $SIONNA_GPU_PARTITION / $SIONNA_GPU_GRES"
        RUNNER=(srun --partition="$SIONNA_GPU_PARTITION" --gres="$SIONNA_GPU_GRES"
                --nodes=1 --ntasks=1 --mem="$SIONNA_GPU_MEM" --time=00:30:00 $(_acct)
                bash -c)
    fi
    "${RUNNER[@]}" "
            source ~/.bashrc
            conda activate ${CONDA_ENV_NAME:-sionna} 2>/dev/null || export PATH=\"$CONDA_BIN_DIR:\$PATH\"
            export MITSUBA_VARIANT=$MITSUBA_VARIANT
            set -e
            python main_batch_cluster.py --action generate_static --dataset-dir $DS \
                --n-static 20 --map-bounds-b 1500 --cell-size 10 10
            python main_batch_cluster.py --action simulate_static --dataset-dir $DS \
                --map-bounds-b 1500 --cell-size 10 10 --samples-per-tx 10000000 --max-depth 80
            python - <<'PY'
import numpy as np
a = np.load('$DS/single_static_jammers/watts.npy', mmap_mode='r')
print('shape', a.shape, a.dtype)
m = float(np.asarray(a[0]).max())
print('max watts in map 0:', m)
assert a.shape == (20, 300, 300), a.shape
assert m > 0, 'map is empty - the solver returned nothing'
print('SMOKE TEST PASSED')
PY
         "
    echo "[submit] remove the scratch dataset with:  rm -rf $DS"
    ;;

  *)
    echo "usage: ./submit.sh {show|verify|smoke|gpu|cpu}" >&2
    exit 2
    ;;
esac
