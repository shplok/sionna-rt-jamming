#!/bin/bash
#SBATCH --job-name=sionna_pipeline
#SBATCH --partition=pomplun
#SBATCH --account=cs_tales.imbiriba
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err

# ==============================================================================
# Sionna RT Jamming - Unified Non-Interactive Pipeline (SLURM + GPU)
# 
# ==============================================================================
# Experiment Parameters (Easily modify here)
# ==============================================================================
ACTION="all"                # all | generate (trajectories only) | plot | simulate
NUM_COMBINATIONS=10         # Number of multi-jammer combinations to aggregate (10)
MIN_JAMMERS=2               # Minimum number of jammers per combination
MAX_JAMMERS=10              # Maximum number of jammers per combination
SKIP_GIF=""                 # GIF animation: "" (generate GIFs) | "--skip-gif" (skip GIFs to save time and disk space)
PRECISION="float16"         # Precision for aggregated maps: float16 | float32 | float64

set -e  # Stop script if any step fails

echo "=================================================================="
echo "Sionna RT Jamming - Unified Pipeline"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Action: $ACTION | Combos: $NUM_COMBINATIONS (Jammers: $MIN_JAMMERS-$MAX_JAMMERS)"
echo "=================================================================="

# Create directory for SLURM logs
mkdir -p logs

# 0. Load server/environment settings (editable in server_env.sh or server_config.json)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SCRIPT_DIR}/server_env.sh" ]; then
    source "${SCRIPT_DIR}/server_env.sh"
fi

source ~/.bashrc
conda activate "${CONDA_ENV_NAME:-sionna}" 2>/dev/null || [ -z "$CONDA_BIN_DIR" ] || export PATH="${CONDA_BIN_DIR}:$PATH"

export MITSUBA_VARIANT="${MITSUBA_VARIANT:-cuda_ad_mono_polarized}"

echo -e "\n[0/3] Verifying GPU environment..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print('Active Mitsuba variant:', mi.variant())"

# ------------------------------------------------------------------------------
# Run full pipeline: Generation + 2D Visualization + GPU Simulation
# ------------------------------------------------------------------------------
echo -e "\n=================================================================="
echo "Running main_no_interactive.py with action=$ACTION..."
echo "=================================================================="
python main_no_interactive.py \
    --action "$ACTION" \
    --mesh-dir ./data/NYC3KM_585751_4512036/mesh \
    --scene-path ./data/NYC3KM_585751_4512036/simple_OSM_scene.xml \
    --output-dir ./datasets/nyc_single_jammers \
    --sim-output-dir ./datasets/simulation_results_nyc \
    --durations 30 60 90 \
    --velocities 3 9 15 \
    --count-per-combo 6 \
    --max-overlap-ratio 0.75 \
    --proximity-threshold 20.0 \
    --mode straight \
    --map-bounds-b 1500 \
    --cell-size 8 8 \
    --samples-per-tx 10000000 \
    --max-depth 80 \
    --num-combinations "$NUM_COMBINATIONS" \
    --min-jammers "$MIN_JAMMERS" \
    --max-jammers "$MAX_JAMMERS" \
    --precision "$PRECISION" \
    $SKIP_GIF

echo -e "\n=================================================================="
echo "Full pipeline completed successfully at: $(date)"
echo "Trajectory results in: ./datasets/nyc_single_jammers"
echo "Simulation results in:   ./datasets/simulation_results_nyc"
echo "=================================================================="
