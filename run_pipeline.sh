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
# Parámetros del Experimento (Modifica fácilmente aquí)
# ==============================================================================
ACTION="all"                # all (todo) | generate (solo trayectorias) | plot | simulate
NUM_COMBINATIONS=10         # Número de combinaciones multi-jammer a agregar (10)
MIN_JAMMERS=2               # Mínimo de jammers por combinación
MAX_JAMMERS=10              # Máximo de jammers por combinación
SAVE_INDIVIDUAL=""          # Solo guardar combination_summary.json y rss_aggregated.npy
SKIP_GIF=""                 # Generar animación GIF para cada combinación

set -e  # Detener el script si algún paso falla

echo "=================================================================="
echo "Sionna RT Jamming - Unified Pipeline"
echo "Host: $(hostname)"
echo "Fecha de inicio: $(date)"
echo "Action: $ACTION | Combos: $NUM_COMBINATIONS (Jammers: $MIN_JAMMERS-$MAX_JAMMERS)"
echo "=================================================================="

# Crear carpeta para logs de SLURM
mkdir -p logs

# 0. Configurar entorno Python / Conda y variante Mitsuba para GPU
source ~/.bashrc
conda activate sionna 2>/dev/null || export PATH="/pomplun/share_home/l.gonzalezgudino001/.conda/envs/sionna/bin:$PATH"

export MITSUBA_VARIANT="cuda_ad_mono_polarized"

echo -e "\n[0/3] Verificando entorno GPU..."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print('Mitsuba variant activa:', mi.variant())"

# ------------------------------------------------------------------------------
# Ejecutar pipeline completo: Generación + Visualización 2D + Simulación GPU
# ------------------------------------------------------------------------------
echo -e "\n=================================================================="
echo "Ejecutando main_no_interactive.py con action=$ACTION..."
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
    $SAVE_INDIVIDUAL \
    $SKIP_GIF

echo -e "\n=================================================================="
echo "Pipeline completo finalizado exitosamente a las: $(date)"
echo "Resultados de trayectorias en: ./datasets/nyc_single_jammers"
echo "Resultados de simulación en:   ./datasets/simulation_results_nyc"
echo "=================================================================="
