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
ACTION="all"                # all | generate (trajectories only) | plot | simulate_bases | aggregate | simulate
MODE="curved"               # curved (new mode) | straight (original mode)
NUM_COMBINATIONS=100        # Number of multi-jammer combinations to aggregate (100 requested)
MIN_JAMMERS=2               # Minimum number of jammers per combination
MAX_JAMMERS=5               # Maximum number of jammers per combination (requested max: 5)
SKIP_GIF="--skip-gif"       # GIF animation: "" (generate GIFs) | "--skip-gif" (skip GIFs to save time and disk space)
PRECISION="float16"         # Precision for aggregated maps: float16 | float32 | float64

# Storage locations (auto-detected by paths_config.py or overridden by env vars)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_DIR"

OUTPUT_DIR="${OUTPUT_DIR:-$(python3 -c "import sys; sys.path.insert(0, '$REPO_DIR'); import paths_config; print(paths_config.get_trajectories_dir())" 2>/dev/null || echo "$REPO_DIR/datasets/nyc_${MODE}_jammers")}"
SIM_OUTPUT_DIR="${SIM_OUTPUT_DIR:-$(python3 -c "import sys; sys.path.insert(0, '$REPO_DIR'); import paths_config; print(paths_config.get_sim_results_dir())" 2>/dev/null || echo "$REPO_DIR/datasets/simulation_results_nyc")}"

set -e  # Stop script if any step fails

echo "=================================================================="
echo "Sionna RT Jamming - Unified Pipeline (TMLR)"
echo "Host: $(hostname)"
echo "Start time: $(date)"
echo "Action: $ACTION | Mode: $MODE | Combos: $NUM_COMBINATIONS (Jammers: $MIN_JAMMERS-$MAX_JAMMERS)"
echo "Trajectories output dir: $OUTPUT_DIR"
echo "Simulation output dir:   $SIM_OUTPUT_DIR"
echo "=================================================================="

# Create directory for SLURM logs and output directories
mkdir -p logs
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SIM_OUTPUT_DIR"

# 0. Load server/environment settings (editable in server_env.sh or server_config.json)
if [ -f "${REPO_DIR}/server_env.sh" ]; then
    source "${REPO_DIR}/server_env.sh"
elif [ -f "${SCRIPT_DIR}/server_env.sh" ]; then
    source "${SCRIPT_DIR}/server_env.sh"
fi

source ~/.bashrc
conda activate "${CONDA_ENV_NAME:-sionna}" 2>/dev/null || [ -z "$CONDA_BIN_DIR" ] || export PATH="${CONDA_BIN_DIR}:$PATH"

PYTHON_EXEC="python"
if ! command -v python &> /dev/null; then
    PYTHON_EXEC="${CONDA_BIN_DIR}/python"
fi

export MITSUBA_VARIANT="${MITSUBA_VARIANT:-cuda_ad_mono_polarized}"

echo -e "\n[0/3] Verifying GPU environment..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
"$PYTHON_EXEC" -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print('Active Mitsuba variant:', mi.variant())"

# ------------------------------------------------------------------------------
# Run full pipeline: Generation + 2D Visualization + GPU Simulation
# ------------------------------------------------------------------------------
echo -e "\n=================================================================="
echo "Running main_batch_cluster.py with action=$ACTION, mode=$MODE..."
echo "=================================================================="
"$PYTHON_EXEC" main_batch_cluster.py \
    --action "$ACTION" \
    --mode "$MODE" \
    --mesh-dir ./data/NYC3KM_585751_4512036/mesh \
    --scene-path ./data/NYC3KM_585751_4512036/simple_OSM_scene.xml \
    --output-dir "$OUTPUT_DIR" \
    --sim-output-dir "$SIM_OUTPUT_DIR" \
    --durations 30 60 90 \
    --velocities 3 9 15 \
    --count-per-combo 6 \
    --max-overlap-ratio 0.75 \
    --proximity-threshold 20.0 \
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
echo "Trajectory results in: $OUTPUT_DIR"
echo "Simulation results in:   $SIM_OUTPUT_DIR"
echo "=================================================================="

