# Sionna RT Jamming

Generates time-series radio maps (RSS) for GNSS jammer scenarios over urban ray-tracing
scenes with [Sionna RT](https://nvlabs.github.io/sionna/rt/).

---

## Publications & Workflows

This repository hosts codebase and pipelines for two research initiatives:

### 1. `TAES/` (IEEE Transactions on Aerospace and Electronic Systems)
Contains the comprehensive dataset generation pipeline and documentation for deep learning **detection** (DeepMTL-style static snapshots) and **tracking** (PMBM multi-jammer sequences):
- Documentation & Dataset Guide: [`TAES/README.md`](TAES/README.md)
- SLURM Pipelines: [`run_pipeline_gpu.sh`](run_pipeline_gpu.sh) (ray tracing) and [`run_pipeline_cpu.sh`](run_pipeline_cpu.sh) (aggregation)
- Dataset scripts: `scripts/batch_cluster/` (`make_splits.py`, `make_labels.py`, `validate_dataset.py`)

### 2. `TMLR/` (Transactions on Machine Learning Research)
Contains tools and pipelines for **intermittent jammer simulations** (periodic pulsing, random bursts, round-robin, and custom deactivation schedules) and unified SLURM execution:
- Intermittent Generator: [`TMLR/generate_intermittent_simulations.py`](TMLR/generate_intermittent_simulations.py)
- Unified SLURM Execution Script: [`TMLR/run_pipeline.sh`](TMLR/run_pipeline.sh)

---

## Core Entry Points

| Script | Environment | Description |
|---|---|---|
| [`main_batch_cluster.py`](main_batch_cluster.py) | **GPU Cluster** | Unified headless CLI entry point: detection snapshots, straight/curved trajectories, ray tracing, and scenario aggregation. |
| [`main_interactive_local.py`](main_interactive_local.py) | **Local Laptop** | Interactive GUI path planner and single-scenario visualizer. |
| [`paths_config.py`](paths_config.py) | Multi-Site | Centralized environment and high-capacity scratch storage auto-detection (`pomplun`, `explorer`, `local`). |
