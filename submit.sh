#!/bin/bash
# ==============================================================================
# Submit the pipeline, picking the SLURM settings for whichever cluster you are on.
#
#   ./submit.sh gpu          # stage A: ray tracing            (needs a GPU)
#   ./submit.sh cpu          # stage B: everything after       (no GPU)
#   ./submit.sh smoke        # 20-position GPU smoke test, interactive
#   ./submit.sh show         # print the resolved settings and exit
#
# #SBATCH directives inside a job script are static comments and cannot be made
# conditional, so this wrapper passes the site-specific values on the sbatch command
# line, where they override the in-file defaults.
#
# Override anything by exporting it first, e.g.
#   SLURM_GPU_GRES=gpu:h200:1 ./submit.sh gpu
# ==============================================================================
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./server_env.sh

ACTION="${1:-show}"
mkdir -p logs

_acct() { [ -n "$SLURM_ACCOUNT" ] && echo "--account=$SLURM_ACCOUNT"; }

case "$ACTION" in

  show)
    echo "site               : $SIONNA_SITE"
    echo "conda env / bin    : $CONDA_ENV_NAME  ($CONDA_BIN_DIR)"
    echo "mitsuba variant    : $MITSUBA_VARIANT"
    echo "GPU  partition     : $SLURM_GPU_PARTITION"
    echo "GPU  gres          : $SLURM_GPU_GRES"
    echo "GPU  mem / time    : $SLURM_GPU_MEM / $SLURM_GPU_TIME"
    echo "CPU  partition     : $SLURM_CPU_PARTITION"
    echo "CPU  mem / time    : $SLURM_CPU_MEM / $SLURM_CPU_TIME"
    echo "account            : ${SLURM_ACCOUNT:-<none>}"
    echo
    echo "If the partitions look wrong:  sinfo -o '%P %G %l'"
    ;;

  gpu)
    echo "[submit] $SIONNA_SITE -> $SLURM_GPU_PARTITION / $SLURM_GPU_GRES"
    sbatch --partition="$SLURM_GPU_PARTITION" \
           --gres="$SLURM_GPU_GRES" \
           --mem="$SLURM_GPU_MEM" \
           --time="$SLURM_GPU_TIME" \
           $(_acct) \
           run_pipeline_gpu.sh
    ;;

  cpu)
    echo "[submit] $SIONNA_SITE -> $SLURM_CPU_PARTITION (no GPU)"
    sbatch --partition="$SLURM_CPU_PARTITION" \
           --mem="$SLURM_CPU_MEM" \
           --time="$SLURM_CPU_TIME" \
           $(_acct) \
           run_pipeline_cpu.sh
    ;;

  smoke)
    # 20 positions is enough to prove simulate_static works end to end: it exercises
    # scene loading, the solver call, the memmap write and the progress checkpoint.
    DS="./datasets/smoke_batch_simulation_test"
    echo "[submit] interactive smoke test on $SLURM_GPU_PARTITION / $SLURM_GPU_GRES"
    srun --partition="$SLURM_GPU_PARTITION" --gres="$SLURM_GPU_GRES" \
         --nodes=1 --ntasks=1 --mem="$SLURM_GPU_MEM" --time=00:30:00 $(_acct) \
         bash -c "
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
    echo "usage: ./submit.sh {gpu|cpu|smoke|show}" >&2
    exit 2
    ;;
esac
