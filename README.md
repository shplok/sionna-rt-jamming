# Sionna RT Jamming Dataset Generator

Generates time-series radio maps (RSS) for GNSS jammer scenarios over urban ray-tracing
scenes with [Sionna RT](https://nvlabs.github.io/sionna/rt/), for training **detection**
(DeepMTL-style) and **tracking** (PMBM) models.

Two entry points, for two different machines:

| | Entry point | Where | What it is for |
|---|---|---|---|
| **Interactive** | `main_interactive_local.py` | **your laptop** | Design jammer paths in a GUI, simulate a handful of jammers. Exploration and figures. |
| **Batch** | `main_batch_cluster.py` | **GPU cluster only** | Generate the full dataset headless, via `./submit.sh gpu` then `./submit.sh cpu`. |

*Headless* means no screen and nobody clicking: a compute node has no display, so anything
that runs there must write files instead of opening windows. In matplotlib terms that is the
`Agg` backend; GUI backends like `TkAgg` fail on a node with
`TclError: no display name and no $DISPLAY environment variable`.

**The batch pipeline is cluster-only by design.** Ray tracing the base maps — the
single-jammer maps every K-jammer scene is summed from — wants a CUDA GPU, and the finished
dataset is ~96 GB. Neither fits on a laptop. Only the interactive entry point is meant to run
locally.

(*To be precise: CUDA is a throughput requirement, not a functional one. Both entry points try
`cuda_ad_mono_polarized` and fall back to `llvm_ad_mono_polarized`, which is numerically
correct — just ~94× slower, 22 s per trace versus 0.235 s. Batch needs 23 240 traces: ~1.5 h
on an A100, ~140 h on CPU. Interactive needs a handful at the same cost each, so CPU is the
intended local path, not a downgrade.*)

The dataset itself is documented separately, in
[`datasets/batch_simulation_nyc/README.md`](datasets/batch_simulation_nyc/README.md) — design
decisions, file formats, labels, noise model. **Read that one before generating data.**

### Which environment do I need?

| | `requirements_local.txt` | `requirements_cluster.txt` |
|---|---|---|
| Machine | laptop, CPU only | Linux + NVIDIA GPU |
| Runs | `main_interactive_local.py` | `main_batch_cluster.py`, `run_pipeline_gpu.sh`, `run_pipeline_cpu.sh` |
| Contents | the seven packages the repo imports | full `pip freeze` of the working env |
| Mitsuba variant | `llvm_ad_mono_polarized` | `cuda_ad_mono_polarized` |

Both lists include `trimesh`, `numpy`, `scipy`, `matplotlib`, `mitsuba`, `drjit` and
`sionna-rt` at identical versions.

**Why two files.** `requirements_cluster.txt` is a raw `pip freeze` — 111 lines of everything
installed in the working env, including `torch`, `torchvision` and 13 `nvidia-*` CUDA wheels.
`requirements_local.txt` is hand-written: only the seven packages the repo imports.

**The difference is one-directional.** The cluster file will *not* install on a laptop — its
CUDA wheels are published for Linux + NVIDIA only. The local file installs and runs fine on
the cluster, since mitsuba's Linux package already contains the GPU version. So: local is the
portable one, cluster is the exactly-reproducible one.

**Two caveats.** The packages that block the cluster file are ones the repo never uses —
there is no `import torch` anywhere; they are there only because `pip freeze` captures the
whole env. And neither file suffices on Explorer: both pin stock `drjit==1.5.0`, which
crashes on its GPUs and needs the patched build described below.

---

## The key idea

Ray tracing is expensive; adding power is not. Jammers are incoherent sources, so total
received power is the **sum in watts** — which means a K-jammer scene is a NumPy add over
precomputed maps, not a new ray trace.

So every stage is either **[GPU] ray trace one jammer at a time**, or **[CPU] sum them**:

```
            [GPU]  ray trace                    [CPU]  sum K of them
TRACKING    54 trajectories, one map/frame  ->  5 000 scenarios      (69 GB)
DETECTION   20 000 static positions         ->  100 000 samples      (18 GB)
```

Two branches because the two models want different things. The tracker needs sequences. The
detector is a **single-snapshot** model — motion is irrelevant to it, and sampling frames
from trajectories would confine every training jammer to 54 straight corridors with adjacent
frames correlated at r = 0.884. Independent static positions cost the same GPU time and
scatter over the whole street network.

(*`r = 0.884` is the Pearson correlation between the flattened dBW map at frame `t` and frame
`t+1`, averaged over all 3 186 adjacent pairs — i.e. the next frame is ~88 % linearly
predictable from the current one, so two adjacent frames are nowhere near two independent
training samples. Note it is unit-dependent: the same data in linear watts gives r = 0.19,
because ~94 % of the dBW map sits flat at the noise floor and those identical dead cells
dominate the correlation, whereas in watts the floor is ≈ 0 and only the moving peak counts.
The unit-free version of the argument is displacement: at 1 fps the jammer advances 0.3, 0.9
and 1.5 cells per frame at 3, 9 and 15 m/s — at 3 m/s it does not even leave its own cell.*)

It also keeps the two models honest: the detector never sees a trajectory frame, so the
detections it feeds to the tracker are **out-of-sample by construction**.

**Total: ~1.5 h of GPU, then everything else on CPU.** About 96 GB of output.

---

## What runs where

| | Machine | GPU? | Stages |
|---|---|---|---|
| `main_interactive_local.py` | laptop | no | GUI planning, a handful of jammers |
| `run_pipeline_gpu.sh` | cluster | **2 of 4 stages** | `generate` [CPU], `simulate_bases` [**GPU**], `generate_static` [CPU], `simulate_static` [**GPU**] |
| `run_pipeline_cpu.sh` | cluster | no | splits, labels, `aggregate`, `aggregate_static`, validation |

The stages come in pairs: **`generate` decides *where* the jammers are, `simulate_bases`
computes *what the radio field looks like* there.** `generate` is pure geometry — it lays out
the 54 trajectories and writes `traj_*.npy`, no radio involved, CPU-seconds.
`simulate_bases` then ray traces one radio map per frame of those trajectories, which is the
expensive GPU part. `generate_static` / `simulate_static` are the same division of labour for
the static positions.

So **only the two `simulate_*` stages actually need the GPU.** The `generate_*` stages never
import Sionna or Mitsuba — those imports are local to `run_base_simulations()` and
`run_static_simulation()` — and they run fine on a login node in seconds. They sit in the GPU
job purely because each must precede its `simulate_*` partner and costs nothing to include.

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
| `utils/plotter.py` | Plotting library — RSS animation GIF, single panels, contact sheets |
| `utils/jammer_config.py` | Persists edited initial positions back to `main_interactive_local.py` |
| `visualize_paths.py` | Tk GUI browser for saved path `.npy` files — **laptop only** |
| `scripts/preview_dataset.py` | Headless contact sheets of the generated dataset, one panel per K |
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

Force the CPU Mitsuba variant (the code defaults to CUDA and only falls back on failure):

```bash
export MITSUBA_VARIANT=llvm_ad_mono_polarized
python -c "import mitsuba as mi; mi.set_variant('llvm_ad_mono_polarized'); \
           import sionna.rt; print('OK:', mi.variant())"
```

### Run

```bash
conda activate sionna
python main_interactive_local.py
```

1. **Individual mode** — choose a strategy (Math Modeling or Waypoint), time step, and padding mode, then plan on the map. In the Math planner, use **Change initial jammer position** *before* adding any segment — it writes the clicked position back into `main_interactive_local.py`.
2. **Batch mode** — set graph parameters, build the navigation graph, generate `N` paths.
3. Saves trajectories, runs simulation (slow on CPU), writes to `datasets/<DATASET_NAME>/`.

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

> **⚠️ Known mismatch.** `b = 750`, `cell_size = (8, 8)` → 188 × 188 grid covering the central 1.5 km at 8 m. The batch pipeline uses ±1500 m / 10 m → 300 × 300. **Output is not comparable with `datasets/batch_simulation_*`**; do not mix into training or figures. Fine for interactive exploration only. Fix is two lines (`b = 1500`, `cell_size = (10, 10)`), flagged with a TODO in `main()`.

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

---

## Plotting

Three pieces, split by *what kind of thing they are* rather than by which branch they serve.

| | What it is | Backend | Runs where |
|---|---|---|---|
| `utils/plotter.py` | library of plotting functions — take arrays, write figures | — | anywhere |
| `scripts/preview_dataset.py` | CLI that knows the dataset layout | `Agg` | **cluster or laptop** |
| `visualize_paths.py` | Tk GUI application | `TkAgg` | **laptop only** |

`utils/plotter.py` holds the drawing and nothing else — no CLI, no knowledge of where files
live:

| Function | Draws |
|---|---|
| `create_jammer_animation(rss_list, paths_dict, ...)` | RSS cube → animated GIF. Used by the interactive run and by `--gif` below |
| `plot_rss_panel(ax, rss, extent, ...)` | one RSS frame into an existing axis |
| `plot_rss_sheet(panels, extent, ...)` | grid of panels → one PNG, shared colour scale |
| `draw_buildings(ax, buildings)` | grey building overlay, shared by all of the above |

### Previewing the generated dataset

`scripts/preview_dataset.py` writes contact sheets with **one panel per jammer count
K = 0…10**, in the same style as the interactive GIF: viridis RSS, grey buildings, white
ground-truth markers, one shared colour scale per sheet so brightness is comparable across
panels.

```bash
# both branches, K = 0..10, from the val split
python scripts/preview_dataset.py --dataset-dir ./datasets/batch_simulation_nyc

# detector samples only
python scripts/preview_dataset.py --branch static

# tracking scenarios, plus one GIF per previewed scenario
python scripts/preview_dataset.py --branch trajectory --gif
```

Writes to `<dataset-dir>/previews/` (inside `datasets/`, so gitignored):

```
previews/
  static_val_by_k.png        11 detector samples,   K = 0..10
  trajectory_val_by_k.png    11 tracking scenarios, K = 0..10
  <scenario_id>.gif          only with --gif
```

| Flag | Default | Notes |
|---|---|---|
| `--branch` | `both` | `static` \| `trajectory` \| `both` |
| `--split` | `val` | `val` is the fast choice; `train` works the same way |
| `--max-k` | `10` | previews `K = 0..max-k`, so 11 panels by default |
| `--frame-frac` | `0.5` | which frame of each scenario to show, as a fraction of its duration |
| `--mesh-dir` | NYC meshes | pass `--mesh-dir ''` to skip the building overlay — at 3 km the 7 552 footprints are visually heavy |
| `--gif` | off | **~20–27 MB per scenario**, so ~250 MB for all 11. Sheets only, unless you need the motion |
| `--vmin` / `--vmax` | `-145` / `0` | dBW colour limits |

Picking by K is cheap in both branches: static samples carry `num_jammers` in `meta.npz`, and
trajectory scenarios encode it in the directory name (`val_00007_k03`).

**Sanity check it doubles as.** A correct sheet has exactly K markers per panel, every marker
sitting on a bright spot, and a uniformly dark K=0 panel (noise floor only). If markers land
off the peaks, the row/col → x/y convention has been broken somewhere: `rss[row, col]` with
`x = -1500 + (col + frac) · 10` and `y` likewise from `row`, drawn with `origin='lower'`.

The first run reads ~7.5 k building meshes; `gather_bboxes` caches the result to a `.pkl` in
the mesh directory, so later runs are instant.

---

## B. Batch, on the cluster

The whole dataset is built here. Two SLURM jobs: ray tracing (GPU), then everything else
(CPU). Follow the steps below in order.

### ⚠️ Northeastern Explorer: the home quota will bite you

Explorer gives each user a small quota on `/home` (separate from the lab's 35 TB on `/projects`). A single miniconda install is ~4.3 GB. When it fills, failures are **misleading**:

| What you see | What it actually is |
|---|---|
| `CondaValueError: solver backend (libmamba) not recognized` | conda cannot write `~/.condarc` |
| `conda create --prefix /projects/...` fails with `Errno 122` | conda writes lockfiles and its index cache to `~/.conda` **regardless of `--prefix`** |
| `jit_init(): could not use "~/.drjit"` | drjit kernel cache disabled |
| `Could not save font_manager cache` | matplotlib font cache |
| **Claude Code 401 loops after `/login` succeeds** | it cannot persist the refreshed OAuth token to `~/.claude/` |

`df -h /home` shows the filesystem (110 TB free), **not your quota**. Test writability:

```bash
touch ~/.quota_test && rm ~/.quota_test && echo "home OK" || echo "HOME FULL"
```

**Rules for this project on Explorer**

1. **Everything on `/projects/ipl_lab`** — repo, miniconda, venv, dataset. Nothing on `/home`.
2. **`server_env.sh` redirects what it can** (`XDG_CACHE_HOME`, `PIP_CACHE_DIR`, `MPLCONFIGDIR`, `CONDA_PKGS_DIRS`) automatically when `SIONNA_PROJ` is set.
3. **conda's internal writes can't be redirected.** Override `HOME` per command:
   ```bash
   mkdir -p /projects/ipl_lab/$USER/conda-home
   HOME=/projects/ipl_lab/$USER/conda-home bash Miniconda3-py311_*-Linux-x86_64.sh -b -p /projects/ipl_lab/miniconda3
   ```
4. **drjit kernel cache** is derived from `$HOME` and can't be overridden. If home fills, it disables the disk cache (costs compile time, not correctness). Only fix: free space.
5. **Keep some home space free** — Claude Code, SLURM, and your shell need it. `conda clean --all -y` recovers 10–20 GB from the download cache without touching environments.

Working layout:

```
/projects/ipl_lab/
├── miniconda3/                     Python 3.11 interpreter source only
└── jaramillocivill.m/
    ├── conda-home/                 HOME override target for conda commands
    ├── pip-cache/  .cache/
    └── sionna-rt-jamming/
        ├── .venv/                  ~6 GB, gitignored
        └── datasets/               ~96 GB, gitignored
```

### ⚠️ Explorer needs a patched drjit (OptiX PTX bug)

**Symptom.** `--action simulate_bases` or `simulate_static` aborts with `SIGABRT`
(exit 134) a second or two into the first `RadioMapSolver` call:

```
New backend is missing implementation for PTX intrinsic optix.ptx.copysign.f32
jit_optix_compile(): optixModuleGetCompilationState() indicates that the compilation
did not complete successfully. State: 0x2363
```

**Cause.** drjit 1.5.0 emits a `copysign.f32` PTX instruction that OptiX translates to
`optix.ptx.copysign.f32`. Support landed in **NVIDIA driver 572.46**; Explorer's A100 and
H200 nodes are on 570.86.15 — too old. The patch becomes unnecessary if Explorer updates
to ≥ 572.46 (but is harmless to leave in).

**Fix.** `patches/drjit-core-copysign-optix.patch` replaces the instruction with
equivalent bit manipulation — mask the magnitude, mask the sign, OR them. It is
**bit-exact**, not an approximation: identical results including NaN payloads, since
`copysign` is pure bit shuffling either way. The 7.2 GB static library and 2.2 GB
trajectory library were produced with it and are numerically sound.

```bash
cd /projects/ipl_lab/$USER
git clone --depth 1 --branch v1.5.0 --recurse-submodules     https://github.com/mitsuba-renderer/drjit.git drjit-src
cd drjit-src/ext/drjit-core
git apply /projects/ipl_lab/$USER/sionna-rt-jamming/patches/drjit-core-copysign-optix.patch
cd /projects/ipl_lab/$USER/drjit-src

module load cmake/3.30.2 cuda/12.8.0
source /projects/ipl_lab/$USER/sionna-rt-jamming/.venv/bin/activate
pip wheel . -w /tmp/drjit-wheel --no-deps
pip install --force-reinstall /tmp/drjit-wheel/drjit-1.5.0-cp311-cp311-linux_x86_64.whl
```

Verify on a GPU node — it must report `cuda_ad_mono_polarized`, not the LLVM fallback:

```bash
srun --partition=gpu --gres=gpu:a100:1 --mem=8G --time=00:10:00      ./scripts/smoke_test.sh
```

**Do not "fix" this by changing package versions.** sionna-rt 2.1.0 hard-pins
`drjit==1.5.0` and `mitsuba==3.9.1`, so downgrading either breaks the install.

<details>
<summary>Approaches that do not work (so nobody repeats them)</summary>

| Attempt | Why it fails |
|---|---|
| Downgrade to mitsuba 3.6 / drjit 1.0.1 | sionna-rt 2.1.0 hard-pins drjit 1.5.0 and mitsuba 3.9.1; pip resolves back or the install breaks |
| `dr.set_flag(dr.JitFlag.ShaderExecutionReordering, False)` | The intrinsic is not SER-specific. Tested: still aborts. |
| `LD_PRELOAD` shim to rewrite the PTX | drjit `dlopen`s OptiX and resolves symbols via `dlsym`, so `LD_PRELOAD` interposition never sees the call — `RTLD_NEXT` returns NULL and the shim segfaults (exit 139) |

CPU/LLVM fallback is numerically correct but measured 22 s per position — ~120 h for the static library.

</details>

### Submitting on the cluster

`./submit.sh` picks SLURM settings per site. Two profiles in `server_env.sh`:

| | `explorer` | `pomplun` |
|---|---|---|
| GPU partition / gres | `gpu` / `gpu:a100:1` | `pomplun` / `gpu:1` |
| CPU partition | `short` | `pomplun` |
| Account | none | `cs_tales.imbiriba` |
| Dataset root | `/projects/ipl_lab/$USER/sionna-rt-jamming/datasets` | `./datasets` |

Site is auto-detected from hostname; force with `export SIONNA_SITE=explorer` (or `pomplun`).

**Step 1 — preflight.**

```bash
./submit.sh show       # resolved settings
./submit.sh verify     # read-only: partitions, associations, GPU, python env, input data
```

`verify` checks partitions exist, mitsuba loads the CUDA variant, and scene data is present. It writes nothing.

`short` is the default Explorer CPU partition — if `verify` flags it, `export SIONNA_CPU_PARTITION=<yours>`. Every value is overridable:

```bash
SIONNA_GPU_GRES=gpu:h200:1 ./submit.sh gpu     # H200 instead of A100
```

Config is in the `SIONNA_*` namespace (not `SLURM_*` — inside an allocation those already hold the parent job's values).

**Recommended workflow:** grab an interactive GPU shell, run `verify` and `smoke` there, then submit with `./submit.sh gpu` / `cpu`. `sbatch` works fine from inside an allocation. `submit.sh` strips inherited `SLURM_*`/`SBATCH_*` variables so the child doesn't inherit this shell's memory or task count.

**Step 2 — environment (once).**

```bash
source /projects/ipl_lab/$USER/miniconda3/etc/profile.d/conda.sh
conda create -n sionna python=3.11 && conda activate sionna
pip install --upgrade pip setuptools wheel
pip install -r requirements_cluster.txt
python -c "import mitsuba as mi; mi.set_variant('cuda_ad_mono_polarized'); print(mi.variant())"
```

`conda activate` requires the shell hook to be sourced first — not guaranteed in batch jobs. Pipeline scripts call `sionna_activate_env` (aliased as `sionna_activate_conda` for backwards compatibility) from `server_env.sh`, which tries the venv first, falls back to conda, and fails loudly rather than silently using the wrong Python.

**⚠️ Where the dataset lands.** The run writes **~96 GB**, which will blow a typical home
quota. The `explorer` profile therefore points `SIONNA_DATASET_ROOT` at
`/projects/ipl_lab/$USER/sionna-rt-jamming/datasets`, not at the repo. `./submit.sh verify`
checks that it is writable, reports free space, and warns if the root looks like `$HOME` or
the repo. Override with `export SIONNA_DATASET_ROOT=/your/project/space`.

**Step 3 — smoke test.** Exercises scene loading, solver call, memmap write, and checkpoint (~1 min). Run from an interactive GPU shell:

```bash
srun --partition=gpu --gres=gpu:a100:1 --nodes=1 --ntasks=1 --mem=16G --time=00:30:00 --pty bash
./submit.sh smoke
```

Expect `shape (20, 300, 300) float32`, positive max, `SMOKE TEST PASSED`. Then `rm -rf ./datasets/batch_simulation_smoke`. You can submit steps 4–5 from that same shell.

**Step 4 — GPU job** (~1.5 h): trajectories + radio maps, static positions + radio maps.

```bash
./submit.sh gpu
tail -f logs/gpu_<jobid>.out
```

**Step 5 — CPU job**, only after step 4 finishes. No GPU; writes ~64 GB.

```bash
./submit.sh cpu
tail -f logs/cpu_<jobid>.out
```

Ends with `validate_dataset.py` (non-zero exit on failure).

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

Tunables are at the top of each script. To run stages individually:

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

**`--max-depth 80`** is the ray-tracing bounce limit: how many times a ray may reflect off
buildings before it is dropped. Low values only capture near line-of-sight paths; 80 lets
energy bounce deep into shadowed side streets and urban canyons, which is the multipath that
makes this data worth ray tracing rather than modelling with a path-loss formula. It costs
little here because runtime is dominated by `--samples-per-tx`.

**Ordering:** `make_splits.py` reads `single_static_jammers/positions.npy`, so it must run *after* `generate_static`. Missing file → detection branch silently skipped with a warning.

### Watch out for

- **`--skip-gif` is mandatory at scale** — GIFs are ~25 MB each and would roughly double the dataset.
- **`simulate_bases` resumes by filename only** — it does *not* check the grid. If you change `--cell-size`, **delete `single_trajectory_jammers/radio_maps/` first** or stale maps survive silently. (`traj_*.npy` is grid-independent; keep it. `simulate_static` and both aggregate stages *do* check and refuse mismatched libraries.)
- **`simulate_static` checkpoints every 50 positions** to `watts.progress.json` — a preempted job resumes rather than restarting.
- **RAM is the CPU-side constraint:** `aggregate` holds one pool (~1.4 GB); `aggregate_static` memmaps the 7.2 GB static library.
- **`datasets/` is gitignored** — move results off the node yourself.

---

## C. What the downstream repo consumes

### Detector (DeepMTL-style) — from `multi_static_jammers/`

| File | Contents |
|---|---|
| `{split}/rss.npy` | `(n, 300, 300)` float16 dBW — **dense**, mask to sensors yourself |
| `{split}/labels.npz` | per (sample, jammer): `x, y` in metres, plus `col, row, dx, dy` |
| `{split}/meta.npz` | `num_jammers`, `position_ids`, `sensor_density_pct`, `num_sensors`, `sensor_seed`, `noise_seed` |
| `splits.json` | `sensor_cells` (street cells) and the per-sample specs |

**`noise_seed`** is the seed for that sample's measurement-noise draw. The ray tracer is
deterministic, so the stored map is the same every time; the σ = 1 dB Gaussian added on top
(`--meas-noise-var 1.0`, in dB) is not. Recording the seed per sample means any sample's exact
noise realisation can be reproduced, and that two front ends can be handed byte-identical
measurements. `sensor_seed` plays the same role for which cells become sensors.

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

Train the detector on `multi_static_jammers`, then run it over `multi_trajectory_jammers`. Because the two libraries are disjoint, detections fed to the tracker are **out-of-sample by construction** — no cross-fitting, and false-positive statistics are honest. Do not train the tracker on detections from scenarios the detector was fitted to.

Full rationale and file formats: [dataset README](datasets/batch_simulation_nyc/README.md).

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
