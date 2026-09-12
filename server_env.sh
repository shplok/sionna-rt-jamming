#!/bin/bash
# ==============================================================================
# Server & Environment Configuration
# Modify these paths when running on a different server or HPC cluster.
# ==============================================================================

# Name of the Conda environment
export CONDA_ENV_NAME="sionna"

# Absolute path to the Conda environment bin directory (used as fallback if 'conda activate' fails)
export CONDA_BIN_DIR="/pomplun/share_home/l.gonzalezgudino001/.conda/envs/sionna/bin"

# Mitsuba GPU variant
export MITSUBA_VARIANT="cuda_ad_mono_polarized"

# SLURM Cluster Settings (for reference or manual sbatch submission)
export SLURM_PARTITION="pomplun"
export SLURM_ACCOUNT="cs_tales.imbiriba"
