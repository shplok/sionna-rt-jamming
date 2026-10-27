# Sionna RT Jamming Framework

A high-performance simulation and dataset generation framework for GNSS Jammer Radio Frequency Interference (RFI) in complex urban environments using **[Sionna RT](https://nvlabs.github.io/sionna/rt/)** and **Mitsuba 3**.

---

## 1. Key Concept & Core Principles

Ray tracing in dense 3D urban meshes is computationally intensive; adding incoherent signal power is not. Radio frequency interference from independent jammers adds incoherently in linear Watts:

$$P_{\text{agg}}(t, x, y) = \sum_{k=1}^K P_k(t, x, y)$$

After spatial summation, the composite received signal strength (RSS) is converted to dBW with a thermal noise floor:

$$\text{RSS}_{\text{dBW}}(t, x, y) = 10 \log_{10}\left(P_{\text{agg}}(t, x, y) + P_{\text{noise}}\right), \quad P_{\text{noise}} = 8 \times 10^{-15}\text{ W}$$

This enables **instantaneous scenario generation**: ray-trace each single jammer base path once on a GPU, then sum any multi-jammer combination ($K \ge 1$) in pure NumPy on CPU.

### Radio-Map Cleaning (Despiking)
Due to Monte-Carlo ray tracing with grazing angles at pedestrian antenna heights ($z = 1.5\text{ m}$), raw solvers occasionally produce isolated high-power spikes or non-finite values. By default, every single-jammer map is traced across 3 distinct seeds (`--clean-seeds 42 7 123`) and cleaned:
1. **Per-cell median** across seeds.
2. **Despiking**: cells $> 20\text{ dB}$ higher than all 8 neighbours (outside a $30\text{ m}$ transmitter exclusion zone) are replaced by the neighbour median.
3. **Power cap**: values exceeding the jammer transmit power ($10\text{ dBW} = 10\text{ W}$) are capped.
4. **Non-finite repair**: any NaN/inf is replaced with neighbour medians.
Cleaning settings and provenance are recorded in `map_cleaning.json`.

---

## 2. Repository Structure

```
sionna_rt_jamming/
├── main_batch_cluster.py         # Unified headless CLI pipeline (Cluster & Local batch)
├── main_interactive_local.py     # Interactive GUI planner and single-scenario visualizer (Laptop)
├── paths_config.py               # Auto-detection of compute cluster (Pomplun, Explorer, Local) & scratch paths
├── server_env.sh                 # Environment profiles (SLURM accounts, partitions, conda paths)
├── submit.sh                     # Site-aware job submission wrapper (./submit.sh {show,gpu,cpu,smoke})
├── run_pipeline_gpu.sh           # Stage A SLURM job (GPU ray-tracing only)
├── run_pipeline_cpu.sh           # Stage B SLURM job (CPU aggregation & splits generation)
│
├── TAES/                         # TAES publication workspace & detailed dataset guide
│   └── README.md                 # Full dataset generation documentation (DeepMTL / PMBM)
│
├── TMLR/                         # TMLR publication workspace (Intermittent & dynamic tracking)
│   └── generate_intermittent_simulations.py  # Generates intermittent (pulsing/burst) datasets
│
├── core/                         # Simulation core
│   ├── engine.py                 # MotionEngine (scene transmitter updates, collision checks)
│   ├── strategies.py             # Path strategies (Math modeling, Waypoints, GraphNav)
│   └── trajectory_generator.py   # Corridor & curved trajectory generator (kinematics, overlap filter)
│
├── utils/                        # Utilities
│   ├── map_cleaning.py           # Multi-seed median, despiking, and power-capping
│   ├── scene_objects.py          # Antenna arrays and cached building footprint extractors
│   ├── plotter.py                # Visualizations (2D trajectory maps, RSS contact sheets, animated GIFs)
│   └── jammer_config.py          # State persistence
│
├── scripts/
│   ├── batch_cluster/            # Dataset tools: make_splits.py, make_labels.py, validate_dataset.py, preview_dataset.py
│   └── interactive_local/        # Local visualization tools: visualize_paths.py
│
└── data/NYC3KM_585751_4512036/   # 3D urban model (OSM scene XML + building meshes)
```

---

## 3. Available Modes

### A. Tracking Mode (Dynamic Trajectories)
Jammers traverse continuous spatial paths through the urban street network:
- **Straight Corridors (`--mode straight`)**: Generates 54 collision-free linear trajectories across combinations of durations (30, 60, 90 s) and velocities (3, 9, 15 m/s) with mutual overlap $\le 75\%$.
- **Curved Trajectories (`--mode curved`)**: Generates trajectories with realistic cornering and turns along street intersections using shortest-path and spline kinematics.
- Multi-jammer combinations can be generated:
  - **Random Aggregation**: (`--num-combinations N --min-jammers A --max-jammers B`) creates $N$ multi-jammer scenarios on the fly with optional animated GIFs (`--skip-gif` to omit).
  - **Splits-Based Aggregation**: derives exact multi-jammer tracking scenarios from `splits.json`.

### B. Detection Mode (Static Snapshots)
Single-snapshot training (e.g., DeepMTL-style localization):
- Static positions are continuously jittered inside street cells (avoiding cell centers so sub-cell regression targets remain non-degenerate).
- Generated via `--action generate_static`, simulated on GPU via `--action simulate_static`, and aggregated via `--action aggregate_static`.

### C. Intermittent Mode (`TMLR/`)
Simulates jammers that turn on/off or change duty cycles:
- Controlled by `TMLR/generate_intermittent_simulations.py`.
- Supports patterns: **Periodic** (fixed on/off steps), **Staggered** (multi-jammer phase-shifted pulsing), **Random Bursts**, **Round-Robin**, or **Custom JSON intervals**.

### D. Interactive GUI Mode (`main_interactive_local.py`)
- Designed for local exploration on a laptop without SLURM.
- Allows interactive clicking of waypoints, testing trajectory planning strategies, and rendering sample scenarios on CPU (`llvm_ad_mono_polarized`).

---

## 4. Output Format

All data is structured under the designated output directory (configured by `--dataset-dir` or auto-detected by `paths_config.py`):

```
output_dir/
├── single_trajectory_jammers/          # Base library: 1 jammer per trajectory
│   ├── traj_dur30s_vel03mps_00.npy     # Trajectory coordinates: (T, 3) in [x, y, z]
│   ├── trajectories_manifest.json      # Metadata (start/end pos, headings, distance, kinematics)
│   └── radio_maps/
│       ├── rm_watts_traj_*.npy         # Single-jammer power in Watts: (T, H, W), float32
│       ├── rm_dbw_traj_*.npy           # Single-jammer power in dBW: (T, H, W), float32
│       ├── map_cleaning.json           # Provenance of despiking / cleaning
│       └── radio_maps_manifest.json
│
├── multi_trajectory_jammers/           # Multi-jammer scenarios
│   ├── combo_0000_k03/ (or train/val/test splits)
│   │   ├── rss_aggregated.npy          # Aggregated RSS array: (T, H, W), float16
│   │   ├── combination_summary.json    # Scenario metadata (jammer IDs, padding, min/max dBW)
│   │   └── jammer_animation.gif        # (Optional) Visual animated GIF
│   └── combinations_manifest.json
│
├── single_static_jammers/              # Static snapshot library
│   ├── positions.npy                   # Continuous coordinates: (N, 3), float32
│   ├── watts.npy                       # Memory-mapped Watts library: (N, H, W), float32
│   └── map_cleaning.json
│
├── multi_static_jammers/               # Multi-jammer detection samples
│   ├── train/, val/, test/
│   │   ├── rss.npy                     # (N_samples, H, W), float16
│   │   └── labels.npz                  # Ground-truth positions and counts
│   └── map_cleaning.json
│
└── labels/                             # Per-frame ground truth jammer coordinates
    ├── train_labels.npy
    ├── val_labels.npy
    └── test_labels.npy
```

### Grid & Precision Specifications
| Property | Value | Notes |
|---|---|---|
| **Coverage Area** | $3000\text{ m} \times 3000\text{ m}$ | Half-width $b = 1500\text{ m}$ ($[-1500, 1500]$) |
| **Grid Resolution** | $10\text{ m} \times 10\text{ m} \rightarrow 300 \times 300$ | Default (matches DeepMTL grid). $8\text{ m} \rightarrow 375 \times 375$ also supported |
| **Array Precision** | `float16` (`fp16`) | Default for aggregated maps (halves disk space and I/O); `float32` optional |
| **Carrier Frequency** | $1.57542\text{ GHz}$ | GPS L1 |
| **Transmitter Power** | $10\text{ dBW}$ ($10\text{ W}$, $40\text{ dBm}$) | Altitude $z = 1.5\text{ m}$ |

---

## 5. The Two-Stage SLURM Pipeline ("Doble .sh")

A complete dataset requires extensive GPU ray tracing as well as heavy CPU aggregation:
- **Ray tracing** requires high-end CUDA GPUs (e.g., A100 / H200) with OptiX hardware acceleration.
- **Aggregation, split generation, and labeling** require **no GPU**, but demand significant CPU memory (32–64 GB RAM) for large NumPy array operations.

Holding an expensive GPU allocation during CPU-bound aggregation wastes cluster quota. Therefore, the pipeline is decoupled into **two sequential SLURM scripts**:

```
 [Stage A: GPU Job]                           [Stage B: CPU Job]
 sbatch run_pipeline_gpu.sh                   sbatch run_pipeline_cpu.sh
 ─────────────────────────                    ──────────────────────────
 1. generate_static  [CPU, ~5s]               1. make_splits.py      [CPU, ~2m]
 2. simulate_static  [GPU, ~4h]               2. make_labels.py      [CPU, ~1m]
 3. generate         [CPU, ~5s]       ───>    3. aggregate_static    [CPU, ~20m]
 4. simulate_bases   [GPU, ~40m]              4. aggregate           [CPU, ~30m]
                                              5. validate_dataset.py [CPU, ~1m]
```

### Running on the Cluster

#### Option 1: Using `submit.sh` (Recommended)
`submit.sh` auto-detects the cluster site (`pomplun` vs `explorer`), injects partition/account settings from `server_env.sh`, and validates paths:

```bash
# 1. Inspect environment and settings
./submit.sh show

# 2. Submit Stage A (GPU ray-tracing only)
./submit.sh gpu

# 3. Once Stage A finishes successfully, submit Stage B (CPU aggregation only)
./submit.sh cpu
```

#### Option 2: Direct SLURM Submissions
```bash
sbatch run_pipeline_gpu.sh
# Wait until completion, then:
sbatch run_pipeline_cpu.sh
```

#### Option 3: Standalone Tracking / Ad-hoc Generation (Direct CLI / Interactive GPU)
For generating custom multi-jammer tracking datasets (e.g., curved trajectories with random combinations) without running the full two-stage split pipeline:
```bash
# Direct execution (or within an interactive GPU session via srun):
python main_batch_cluster.py \
    --action all \
    --mode curved \
    --durations 30 60 90 \
    --velocities 3 9 15 \
    --count-per-combo 6 \
    --num-combinations 100 \
    --min-jammers 2 \
    --max-jammers 5 \
    --skip-gif
```
