# Sionna RT Jamming Dataset Generator

Generates time-series radio maps (RSS) for GNSS jammer scenarios over urban ray-tracing
scenes with [Sionna RT](https://nvlabs.github.io/sionna/rt/), for training **detection**
(DeepMTL-style) and **tracking** (PMBM) models.

Two entry points, for two different machines:

| | Entry point | Where | What it is for |
|---|---|---|---|
| **Interactive** | `main_interactive_local.py` | **your laptop** | Design jammer paths in a GUI, simulate a handful of jammers. Exploration and figures. |
| **Batch** | `main_batch_cluster.py` | **GPU cluster only** | Generate the full dataset headless, via `sbatch run_pipeline.sh`. |

**The batch pipeline is cluster-only by design.** Ray tracing the base maps needs a CUDA
GPU, and the finished dataset is ~46 GB — neither fits on a laptop. Only the interactive
entry point is meant to run locally.

The dataset itself is documented separately, in
[`datasets/batch_simulation_nyc/README.md`](datasets/batch_simulation_nyc/README.md) — design
decisions, file formats, labels, noise model. **Read that one before generating data.**

### Which environment do I need?

| | `requirements_local.txt` | `requirements_cluster.txt` |
|---|---|---|
| Machine | laptop, CPU only | Linux + NVIDIA GPU |
| Runs | `main_interactive_local.py` | `main_batch_cluster.py`, `run_pipeline.sh` |
| Contents | the six packages the repo imports | full `pip freeze` of the working env |
| Mitsuba variant | `llvm_ad_mono_polarized` | `cuda_ad_mono_polarized` |
| Why not the other | — | pins `torch==2.6.0+cu124` and `nvidia-*` wheels, `linux_x86_64` only |

Both lists include `trimesh`, `numpy`, `scipy`, `matplotlib`, `mitsuba` and `sionna-rt`;
they install fine on macOS. The split exists purely because of the CUDA pins.

---

## The key idea

Ray tracing is expensive; adding power is not. Jammers are incoherent sources, so total
received power is the **sum in watts** — which means a K-jammer scene is a NumPy add over
precomputed maps, not a new ray trace.

So every stage is either **[GPU] ray trace one jammer at a time**, or **[CPU] sum them**:

```
            [GPU]  ray trace                    [CPU]  sum K of them
TRACKING    54 trajectories, one map/frame  ->  5 000 scenarios      (46 GB)
DETECTION   20 000 static positions         ->  100 000 samples      (18 GB)
```

Two branches because the two models want different things. The tracker needs sequences. The
detector is a **single-snapshot** model — motion is irrelevant to it, and sampling frames
from trajectories would confine every training jammer to 54 straight corridors with adjacent
frames correlated at r = 0.884. Independent static positions cost the same GPU time and
scatter over the whole street network.

It also keeps the two models honest: the detector never sees a trajectory frame, so the
detections it feeds to the tracker are **out-of-sample by construction**.

**Total: ~1.5 h of GPU, then everything else on CPU.** About 73 GB of output.

---

## What runs where

| | Machine | GPU? | Stages |
|---|---|---|---|
| `main_interactive_local.py` | laptop | no | GUI planning, a handful of jammers |
| `run_pipeline_gpu.sh` | cluster | **yes** | `generate`, `simulate_bases`, `generate_static`, `simulate_static` |
| `run_pipeline_cpu.sh` | cluster | no | splits, labels, `aggregate`, `aggregate_static`, validation |

Submitting them as two jobs matters: ray tracing takes ~1.5 h, the CPU work takes longer, and
holding a GPU reservation through the CPU half would waste it.

---

## Repository layout

| Path | Role |
|---|---|
| `main_interactive_local.py` | **Local** entry point: GUI path planning + simulation |
| `main_batch_cluster.py` | **Cluster** entry point: CLI, all dataset stages |
| `requirements_local.txt` | Laptop / CPU deps — what the repo actually imports |
| `requirements_cluster.txt` | Cluster deps — full `pip freeze`, Linux + CUDA 12.4 only |
| `submit.sh` | Site-aware wrapper: `./submit.sh {show,verify,smoke,gpu,cpu}` |
| `run_pipeline_gpu.sh` | SLURM job — **ray tracing only**, needs a GPU |
| `run_pipeline_cpu.sh` | SLURM job — everything after, **no GPU**, 64 GB RAM |
| `run_pipeline.sh` | Luis's original single-job script (`eff898c4`), kept for reference |
| `server_env.sh` | Per-site SLURM + conda profiles (`explorer`, `pomplun`) |
| `scripts/make_splits.py` | Trajectory pools, scenarios, static position splits, sensors → `splits.json` |
| `scripts/validate_dataset.py` | Post-generation checks; non-zero exit on failure |
| `scripts/make_labels.py` | Per-frame jammer positions → `labels/` |
| `config.py` | Dataclasses for planner / strategy settings |
| `core/engine.py` | Collision checks, trajectory sync, scene TX updates |
| `core/strategies.py` | Path generation (Math, Waypoint, GraphNav) |
| `core/trajectory_generator.py` | Headless straight-corridor trajectory generation |
| `ui/` | Tkinter menus and planners |
| `utils/scene_objects.py` | Antenna arrays, mesh bbox extraction (cached) |
| `utils/plotter.py` | RSS animation GIF |
| `utils/jammer_config.py` | Persists edited initial positions back to `main_interactive_local.py` |
| `visualize_paths.py` | Viewer for saved path `.npy` files |
| `visualize_aggregated.py` | Viewer for RSS cubes — **stale**, defaults point at removed paths |
| `data/<scene>/` | `simple_OSM_scene.xml` + `mesh/*.ply` |
| `datasets/` | Generated outputs (**gitignored**) |

---

## A. Interactive, on your laptop

### Install

```bash
conda create -n sionna python=3.11
conda activate sionna
pip install --upgrade pip setuptools wheel
pip install -r requirements_local.txt
```

Then force the CPU Mitsuba variant — the code defaults to CUDA and only falls back on
failure, so setting this explicitly avoids a confusing warning on every run:

```bash
export MITSUBA_VARIANT=llvm_ad_mono_polarized
```

Verify:

```bash
python -c "import mitsuba as mi; mi.set_variant('llvm_ad_mono_polarized'); \
           import sionna.rt; print('OK:', mi.variant())"
```

### Run

```bash
conda activate sionna
python main_interactive_local.py
```

1. **Select mode** — Individual or Batch.
2. **Individual** — per jammer, choose a strategy (Math Modeling or Waypoint), time step and
   padding mode, then plan on the map. In the Math planner use **Change initial jammer
   position** *before* adding any segment; the clicked position is written back into
   `main_interactive_local.py`.
3. **Batch** — set graph parameters, build the navigation graph, generate `N` paths.
4. The tool saves trajectories, runs the radio-map simulation (slow on CPU), and writes to
   `datasets/<DATASET_NAME>/`.

### Configure

Edit the constants at the top of `main_interactive_local.py`:

| Setting | Description |
|---|---|
| `SCENE_PATH` | Mitsuba scene XML |
| `MESHES_PATH` | Folder with building `.ply` meshes |
| `OUTPUT_DIR` / `DATASET_NAME` | Output location |
| `FREQ_HZ` | Carrier frequency (GPS L1: `1.57542e9`) |
| `Z_HEIGHT` | Jammer altitude (m) |
| `GLOBAL_TX_POWER_DBW` | TX power, converted to dBm for Sionna |
| `initial_jammers_config` | `name`, `initial_position`, `power_dbm`, `color` per jammer |
| `b` / `map_bounds` | Square region `[-b, b]` — currently **750**, i.e. the central 1.5 km |
| `cell_size` | Grid resolution, currently `(8, 8)` m |

> **⚠️ Known mismatch, deliberately left open.** This script uses `b = 750` and
> `cell_size = (8, 8)` while pointing at the 3 km scene, so it covers the central 1.5 km at
> 8 m — a 188 × 188 grid. The batch pipeline uses ±1500 m and 10 m cells → 300 × 300.
>
> **Its output is therefore not comparable with `datasets/batch_simulation_*`** and must not
> be mixed into training or into figures that also use batch data. Fine for interactive
> exploration, which is all this entry point is for.
>
> Aligning is two lines (`b = 1500`, `cell_size = (10, 10)`), flagged with a `TODO` at the
> top of `main()`. Tracked in the dataset README's status list.

### Output

```
datasets/<DATASET_NAME>/
  path_Jammer1.npy       # (T, 3) x, y, z per timestep
  meta_Jammer1.txt       # strategy, distance, duration, segments
  rss_aggregated.npy     # (T, H, W) combined RSS in dBW
  rss_Jammer1.npy        # (T, H, W) per-jammer RSS in dBW
  jammer_animation.gif
```

Optional viewer:

```bash
python visualize_paths.py --folder ./datasets/<NAME> --meshes ./data/NYC3KM_585751_4512036/mesh
```

The first run reads ~7.5 k building meshes; `gather_bboxes` caches the result to a `.pkl` in
the mesh directory, so later runs are instant.

---

## B. Batch, on the cluster

The whole dataset is built here. Two SLURM jobs: ray tracing (GPU), then everything else
(CPU). Follow the steps below in order.

### Submitting on the cluster

`./submit.sh` picks the SLURM settings for whichever cluster you are on. Two profiles live
in `server_env.sh`:

| | `explorer` | `pomplun` |
|---|---|---|
| Whose | Mariona (Northeastern Explorer) | Luis's original, left as pulled |
| GPU partition / gres | `gpu` / `gpu:a100:1` | `pomplun` / `gpu:1` |
| CPU partition | `short` | `pomplun` |
| Account | none | `cs_tales.imbiriba` |
| Conda hook | `$HOME/miniconda3/etc/profile.d/conda.sh` | `/pomplun/share_home/l.gonzalezgudino001/...` |
| Dataset root | `/projects/ipl_lab/$USER/sionna-rt-jamming/datasets` | `./datasets` |

The site is detected from the hostname, falling back to which partitions `sinfo` reports.
Force it with `export SIONNA_SITE=explorer` (or `pomplun`).

**Step 1 — preflight.**

```bash
./submit.sh show       # resolved settings
./submit.sh verify     # read-only: partitions, associations, GPU, python env, input data
```

`verify` checks that the configured partitions actually exist, that `mitsuba` loads the CUDA
variant, and that the scene data is present. Nothing it does writes to the dataset.

In particular **`short` is a guess for the Explorer CPU partition** — if `verify` flags it,
`export SIONNA_CPU_PARTITION=<yours>`. Every value is overridable:

```bash
SIONNA_GPU_GRES=gpu:h200:1 ./submit.sh gpu     # h200 instead of a100
```

Config lives in the `SIONNA_*` namespace, deliberately not `SLURM_*` — SLURM reads several
`SLURM_*` variables as *input*, and inside an allocation they already hold the parent job's
values.

**Working from inside an `srun`?** That is the recommended way: grab an interactive GPU
shell, run `verify` and `smoke` there, then submit the real jobs with `./submit.sh gpu` /
`cpu`. `sbatch` works fine from inside an allocation — the submitted job queues
independently. `submit.sh` strips the inherited `SLURM_*`/`SBATCH_*` job variables before
submitting so the child does not inherit this shell's memory or task count, and `smoke`
detects the allocation and runs in place rather than nesting an `srun`.

**Step 2 — environment (once).**

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n sionna python=3.11 && conda activate sionna
pip install --upgrade pip setuptools wheel
pip install -r requirements_cluster.txt
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print(mi.variant())"
```

`conda activate` only works after conda's shell hook has been sourced, which is **not**
guaranteed inside a batch job even when it works in your login shell. The pipeline scripts
call `sionna_activate_conda` from `server_env.sh`, which sources `$CONDA_SH` first, falls
back to `conda` on `PATH`, then to `$CONDA_BIN_DIR`, and fails loudly rather than silently
running against the wrong Python.

**⚠️ Where the dataset lands.** The run writes **~73 GB**, which will blow a typical home
quota. The `explorer` profile therefore points `SIONNA_DATASET_ROOT` at
`/projects/ipl_lab/$USER/sionna-rt-jamming/datasets`, not at the repo. `./submit.sh verify`
checks that it is writable, reports free space, and warns if the root looks like `$HOME` or
the repo. Override with `export SIONNA_DATASET_ROOT=/your/project/space`.

**Step 3 — smoke test the stage that has never run.** `simulate_static` is new code that has
never touched a GPU. 20 positions exercises scene loading, the solver call, the memmap write
and the checkpoint — about a minute. Needs a GPU, so run it from an interactive shell:

```bash
srun --partition=gpu --gres=gpu:a100:1 --nodes=1 --ntasks=1 --mem=16G --time=00:30:00 --pty bash
```

```bash
./submit.sh smoke
```

Expect `shape (20, 300, 300) float32`, a positive max, and `SMOKE TEST PASSED`. Then
`rm -rf ./datasets/batch_simulation_smoke`. You can stay in that shell and submit steps 4–5
from it.

**Step 4 — the GPU job** (~1.5 h): trajectories, their radio maps, static positions, their
radio maps.

```bash
./submit.sh gpu
squeue -u $USER
tail -f logs/gpu_<jobid>.out
```

**Step 5 — the CPU job**, only after step 4 finishes. No GPU; writes ~64 GB.

```bash
./submit.sh cpu
tail -f logs/cpu_<jobid>.out
```

It ends with `validate_dataset.py`, which exits non-zero on failure — so a bad dataset fails
the job rather than sitting there looking finished.

**Step 6 — confirm.**

```bash
python scripts/validate_dataset.py --dataset-dir ./datasets/batch_simulation_nyc
du -sh datasets/batch_simulation_nyc/*
```

#### If something goes wrong

| Symptom | Cause |
|---|---|
| `sbatch: error: invalid account` / `invalid partition` | Wrong profile — run `./submit.sh show` and compare against `sinfo -o '%P %G %l'` |
| `Could not set variant 'cuda_ad_mono_polarized'` | No GPU on the node, or CPU-only Mitsuba — it falls back to LLVM and runs ~50× slower, so kill it |
| `simulate_bases` finishes instantly | It resumes by filename and does **not** check the grid. Delete `single_trajectory_jammers/radio_maps/` and resubmit |
| `Base maps are 375x375 but splits.json expects 300x300` | Stale library — same fix as above |
| `splits.json has no 'static' section` | `make_splits.py` ran before `generate_static`. Re-run it |
| GPU job preempted mid-`simulate_static` | Harmless — it checkpoints every 50 positions to `watts.progress.json`. Just resubmit |

#### Timing and footprint

| | Wall | Output |
|---|---|---|
| GPU job | ~1.5 h (4 h requested) | ~8.4 GB of base libraries |
| CPU job | ~1–2 h, mostly I/O (8 h requested) | ~64 GB of scenarios and samples |

---

### Dataset layout

Batch datasets live under `datasets/` and are always named **`batch_simulation_<tag>`**.
`--dataset-dir` is the only path you pass; everything else derives from it:

```
datasets/batch_simulation_<tag>/     <- --dataset-dir
├── single_trajectory_jammers/   54 trajectories + per-frame maps    [GPU]
├── multi_trajectory_jammers/    tracking scenarios, train/val/test  [CPU]
├── single_static_jammers/       positions.npy + watts.npy           [GPU]
├── multi_static_jammers/        detector samples, train/val/test    [CPU]
├── splits.json  ·  labels/  ·  sensors/
```

### What the two jobs actually run

**Only stages A2 and A4 need the GPU.**

```bash
sbatch run_pipeline_gpu.sh     # A1 trajectories       [CPU, seconds]
                               # A2 trajectory maps    [GPU, ~13 min]
                               # A3 static positions   [CPU, seconds]
                               # A4 static maps        [GPU, ~78 min]

sbatch run_pipeline_cpu.sh     # B1 splits.json        [CPU]
                               # B2 labels             [CPU]
                               # B3 aggregate + aggregate_static  [CPU, ~64 GB out]
                               # B4 validate_dataset.py           [CPU]
```

Tunables live in the parameter block at the top of each script. The same thing by hand, if you want to run stages individually:

```bash
DS=./datasets/batch_simulation_nyc
SCENE=./data/NYC3KM_585751_4512036
COMMON="--dataset-dir $DS --scene-path $SCENE/simple_OSM_scene.xml \
        --mesh-dir $SCENE/mesh --map-bounds-b 1500 --cell-size 10 10 --seed 42"

# --- GPU node ---
python main_batch_cluster.py --action generate        $COMMON \
    --durations 30 60 90 --velocities 3 9 15 --count-per-combo 6
python main_batch_cluster.py --action simulate_bases  $COMMON \
    --samples-per-tx 10000000 --max-depth 80 --power-dbw 10.0
python main_batch_cluster.py --action generate_static $COMMON --n-static 20000
python main_batch_cluster.py --action simulate_static $COMMON \
    --samples-per-tx 10000000 --max-depth 80 --power-dbw 10.0

# --- CPU node ---
python scripts/make_splits.py --dataset-dir $DS --mesh-dir $SCENE/mesh \
    --cell-size 10 --map-bounds-b 1500 \
    --n-train 4000 --n-val 500 --n-test 500 \
    --n-static-train 70000 --n-static-val 15000 --n-static-test 15000 \
    --min-jammers 0 --max-jammers 10 \
    --k-balance stratified --densities 2 4 6 8 10 --street-only --seed 42
python scripts/make_labels.py --dataset-dir $DS
python main_batch_cluster.py --action aggregate        --dataset-dir $DS \
    --map-bounds-b 1500 --cell-size 10 10 --meas-noise-var 1.0 --precision float16
python main_batch_cluster.py --action aggregate_static --dataset-dir $DS \
    --map-bounds-b 1500 --cell-size 10 10 --meas-noise-var 1.0 --precision float16
python scripts/validate_dataset.py --dataset-dir $DS
```

**Ordering matters in one place:** `make_splits.py` reads
`single_static_jammers/positions.npy` to plan the detector samples, so it must run *after*
`generate_static`. If that file is missing it skips the detection branch with a warning
rather than failing.

### Watch out for

- **`--skip-gif` is not optional at scale.** GIFs are ~25 MB per scenario and would roughly
  double the dataset.
- **`simulate_bases` resumes by filename only** and does *not* check the grid. If you change
  `--cell-size`, **delete `single_trajectory_jammers/radio_maps/` first** or it will silently
  keep the old maps. `traj_*.npy` is grid-independent and should be kept. (`simulate_static`
  and both aggregate stages *do* check, and refuse to run against a mismatched library.)
- **`simulate_static` checkpoints every 50 positions** to `watts.progress.json` and resumes,
  so a preempted 78-minute job does not start over.
- **RAM, not compute, is the CPU-side constraint.** `aggregate` holds one trajectory pool
  (~1.4 GB); `aggregate_static` memmaps the 7.2 GB static library.
- **`datasets/` is gitignored.** Nothing generated here is version-controlled; move results
  off the node yourself.

---

## C. What the downstream repo consumes

### Detector (DeepMTL-style) — from `multi_static_jammers/`

| File | Contents |
|---|---|
| `{split}/rss.npy` | `(n, 300, 300)` float16 dBW — **dense**, mask to sensors yourself |
| `{split}/labels.npz` | per (sample, jammer): `x, y` in metres, plus `col, row, dx, dy` |
| `{split}/meta.npz` | `num_jammers`, `position_ids`, `sensor_density_pct`, `num_sensors`, `sensor_seed`, `noise_seed` |
| `splits.json` | `sensor_cells` (street cells) and the per-sample specs |

Preprocessing, all cheap and all pure functions of the above:

1. **Sensor masking** — take the sample's cells from `sensors/sensors_{split}.npz`, gather
   `rss.npy` there, floor the rest. This is what turns our dense map into DeepMTL's sparse
   observation matrix.
2. **Normalisation** — subtract the noise floor `N`, divide by `−N/2`: empty cells 0, sensor
   cells in (0, 2].
3. **Gaussian targets** — from the continuous `x, y`: amplitude 10, σ = 0.9, 5 × 5 support.
   A model hyperparameter, which is why it is not baked into the data.
4. **YOLO labels** — `(class, x, y, w, h)`, class 1, w = h = 5.

### Tracker (PMBM / neural-enhanced) — from `multi_trajectory_jammers/`

`{split}/{scenario_id}/rss_aggregated.npy` is `(T, 300, 300)` float16, with `labels.npz` and
`scenario_summary.json` beside it. Same masking and normalisation, applied per frame.

### The one methodological rule

Train the detector on `multi_static_jammers`, then run it over `multi_trajectory_jammers` to
produce the detections the tracker's measurement model learns from. Because the two libraries
are disjoint, those detections are **out-of-sample by construction** — no cross-fitting
needed, and the false-positive statistics the model learns are the ones it will meet at test
time. Do not train the tracker on detections from scenarios the detector was fitted to.

Reasoning and file formats in full: the
[dataset README](datasets/batch_simulation_nyc/README.md).

---

## Importing new scenes

> **Owner: Luis.** Placeholder for the OSM → Sionna workflow.

Each scene is a folder under `data/`:

```
data/<SceneName>/
  simple_OSM_scene.xml
  mesh/*.ply
```

`mesh/` must stay next to the XML, with the relative paths inside the XML kept consistent.
Point `--scene-path` / `--mesh-dir` (or `SCENE_PATH` / `MESHES_PATH` in `main_interactive_local.py`) at them.

For a larger area, **keep the 10 m cell size** and let the grid grow — a 6 km scene is
600 × 600. Changing the cell size instead would break the physical scale the model learned.
