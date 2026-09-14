#!/bin/bash
#SBATCH --job-name=sionna_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/gpu_%j.out
#SBATCH --error=logs/gpu_%j.err

#
# Partition, account, gres, mem and time are supplied by ./submit.sh, which picks
# them per site from server_env.sh. The #SBATCH lines below are fallback defaults
# for a bare `sbatch run_pipeline_gpu.sh`; sbatch command-line flags override them.
# ==============================================================================
# Stage A - everything that needs a GPU.  Submit:  sbatch run_pipeline_gpu.sh
#
# Ray tracing only. ~1.5 h for the defaults below. When this finishes, run
# run_pipeline_cpu.sh on a CPU partition - none of the remaining work touches
# the GPU, so holding a GPU allocation for it would waste the reservation.
# ==============================================================================
DATASET_DIR="./datasets/batch_simulation_nyc"
SCENE_DIR="./data/NYC3KM_585751_4512036"
CELL_SIZE=10                # 3000 m / 10 m = 300 x 300, same cell as DeepMTL
MAP_BOUNDS_B=1500
N_STATIC=20000              # static jammer positions; 0.235 s each => ~78 min

set -e
mkdir -p logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/server_env.sh" ] && source "${SCRIPT_DIR}/server_env.sh"
source ~/.bashrc
conda activate "${CONDA_ENV_NAME:-sionna}" 2>/dev/null || [ -z "$CONDA_BIN_DIR" ] || export PATH="${CONDA_BIN_DIR}:$PATH"
export MITSUBA_VARIANT="${MITSUBA_VARIANT:-cuda_ad_mono_polarized}"

echo "=== GPU check ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print('variant:', mi.variant())"

COMMON="--dataset-dir $DATASET_DIR --scene-path $SCENE_DIR/simple_OSM_scene.xml \
        --mesh-dir $SCENE_DIR/mesh --map-bounds-b $MAP_BOUNDS_B \
        --cell-size $CELL_SIZE $CELL_SIZE --seed 42"

# --- tracking branch ----------------------------------------------------------
echo -e "\n[A1/4] trajectories (CPU, seconds)"
python main_batch_cluster.py --action generate $COMMON \
    --durations 30 60 90 --velocities 3 9 15 --count-per-combo 6 \
    --mode straight --max-overlap-ratio 0.75 --proximity-threshold 20.0 \
    --z-height 1.5 --time-step 1.0

echo -e "\n[A2/4] trajectory radio maps (GPU, ~13 min)"
# NOTE: resumes by filename and does NOT check the grid. If you change CELL_SIZE,
# delete $DATASET_DIR/single_trajectory_jammers/radio_maps first.
python main_batch_cluster.py --action simulate_bases $COMMON \
    --samples-per-tx 10000000 --max-depth 80 --power-dbw 10.0

# --- detection branch ---------------------------------------------------------
echo -e "\n[A3/4] static positions (CPU, seconds)"
python main_batch_cluster.py --action generate_static $COMMON \
    --n-static $N_STATIC --z-height 1.5

echo -e "\n[A4/4] static radio maps (GPU, ~78 min at N=20000)"
python main_batch_cluster.py --action simulate_static $COMMON \
    --samples-per-tx 10000000 --max-depth 80 --power-dbw 10.0

echo -e "\n=== GPU stage done: $(date) ==="
echo "Next: sbatch run_pipeline_cpu.sh   (no GPU needed)"
