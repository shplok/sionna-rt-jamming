#!/bin/bash
#SBATCH --job-name=sionna_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/cpu_%j.out
#SBATCH --error=logs/cpu_%j.err

#
# Partition, account, gres, mem and time are supplied by ./submit.sh, which picks
# them per site from server_env.sh. The #SBATCH lines below are fallback defaults
# for a bare `sbatch run_pipeline_cpu.sh`; sbatch command-line flags override them.
# ==============================================================================
# Stage B - everything after ray tracing.  Submit:  sbatch run_pipeline_cpu.sh
#
# NO GPU. Pure NumPy over the precomputed maps, so request a CPU partition.
# Memory is the binding resource, not compute: the tracking aggregation holds one
# trajectory pool in RAM (~1.4 GB) and the static one memmaps a 7.2 GB library.
#
# Run run_pipeline_gpu.sh first.
# ==============================================================================
SCENE_DIR="./data/NYC3KM_585751_4512036"
CELL_SIZE=10
MAP_BOUNDS_B=1500

# tracking scenarios
N_TRAIN=4000
N_VAL=500
N_TEST=500

# detector samples
N_STATIC_TRAIN=70000
N_STATIC_VAL=15000
N_STATIC_TEST=15000
MIN_JAMMERS=0               # 0 => noise-only negatives included
MAX_JAMMERS=10
MEAS_NOISE_VAR=1.0          # Gaussian dB noise; INR = 10*log10(10/var) = 10 dB
DENSITIES="2 4 6 8 10"      # sensor densities, percent
PRECISION="float16"

set -e
mkdir -p logs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "${SCRIPT_DIR}/server_env.sh" ] && source "${SCRIPT_DIR}/server_env.sh"
# after server_env.sh, so SIONNA_DATASET_ROOT is known
DATASET_DIR="${DATASET_DIR:-${SIONNA_DATASET_ROOT:-./datasets}/batch_simulation_nyc}"
sionna_activate_env || exit 1
echo "python: $(command -v python)"

echo -e "\n[B1/4] splits: trajectory pools, scenarios, static positions, sensors"
python scripts/make_splits.py \
    --dataset-dir "$DATASET_DIR" --mesh-dir "$SCENE_DIR/mesh" \
    --cell-size "$CELL_SIZE" --map-bounds-b "$MAP_BOUNDS_B" \
    --n-train "$N_TRAIN" --n-val "$N_VAL" --n-test "$N_TEST" \
    --n-static-train "$N_STATIC_TRAIN" --n-static-val "$N_STATIC_VAL" \
    --n-static-test "$N_STATIC_TEST" \
    --min-jammers "$MIN_JAMMERS" --max-jammers "$MAX_JAMMERS" \
    --k-balance stratified --densities $DENSITIES --street-only --seed 42

echo -e "\n[B2/4] labels"
python scripts/make_labels.py --dataset-dir "$DATASET_DIR"

echo -e "\n[B3/4] aggregation"
COMMON="--dataset-dir $DATASET_DIR --map-bounds-b $MAP_BOUNDS_B \
        --cell-size $CELL_SIZE $CELL_SIZE --meas-noise-var $MEAS_NOISE_VAR \
        --precision $PRECISION"
echo "  tracking scenarios (~46 GB)"
python main_batch_cluster.py --action aggregate $COMMON
echo "  detector samples (~18 GB)"
python main_batch_cluster.py --action aggregate_static $COMMON

echo -e "\n[B4/4] validating the generated dataset"
python scripts/validate_dataset.py --dataset-dir "$DATASET_DIR"

echo -e "\n=== CPU stage done: $(date) ==="
echo "Dataset root: $DATASET_DIR"
echo "  single_trajectory_jammers/  54 trajectories + per-frame maps"
echo "  single_static_jammers/      static positions + maps"
echo "  multi_trajectory_jammers/   tracking scenarios"
echo "  multi_static_jammers/       detector samples"
echo "  splits.json  labels/  sensors/"
