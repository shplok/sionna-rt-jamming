# Sionna RT Jamming Dataset Generator

Interactive tool for planning jammer trajectories over urban ray-tracing scenes and generating time-series radio maps (RSS) with [Sionna RT](https://nvlabs.github.io/sionna/rt/).

The pipeline loads a Mitsuba XML scene, lets you design one or more jammer paths through a GUI, runs `RadioMapSolver` at each timestep, and writes NumPy datasets plus a summary GIF.

---

## How it works

```text
main.py
  │
  ├─ Load scene (XML) + building meshes (PLY bboxes)
  ├─ MotionEngine (collision checks, trajectory storage)
  │
  ├─ Menu: Individual or Batch mode
  │
  ├─ Path planning (per jammer or batch)
  │     Math Modeling  → kinematic segments (const vel / accel / turn)
  │     Waypoint       → click waypoints + smoothing
  │     GraphNav       → random graph paths (batch only)
  │
  ├─ Synchronize trajectories (padding so all paths share length T)
  ├─ Save paths + metadata → datasets/<name>/
  │
  └─ RadioMapSolver loop (T steps)
        Move transmitters → compute RSS grid → save .npy + GIF
```

**Individual mode** configures paths for each jammer listed in `main.py` (`Jammer1`, `Jammer2`, …), one after another. The first jammer sets the global time step `dt`; later jammers reuse it.

**Batch mode** generates `N` random graph-navigation paths and treats them as simultaneous jammers in one simulation.

---

## Repository layout


| Path                     | Role                                                    |
| ------------------------ | ------------------------------------------------------- |
| `main.py`                | Entry point: scene paths, jammer config, simulation     |
| `config.py`              | Dataclasses for planner / strategy settings             |
| `core/engine.py`         | Collision checks, trajectory sync, scene TX updates     |
| `core/strategies.py`     | Path generation (Math, Waypoint, GraphNav)              |
| `ui/`                    | Tkinter menus and planners                              |
| `utils/scene_objects.py` | Antenna arrays, mesh bbox extraction                    |
| `utils/plotter.py`       | RSS animation GIF                                       |
| `utils/jammer_config.py` | Persists edited initial positions back to `main.py`     |
| `data/<scene>/`          | `simple_OSM_scene.xml` + `mesh/*.ply`                   |
| `datasets/`              | Generated outputs (gitignored)                          |
| `data/sionna_osm/`       | Submodule: OSM → Mitsuba scene tooling (see map import) |
| `visualize_paths.py`     | Optional viewer for saved path `.npy` files             |


---

## Environment setup

Versions are pinned from the working `**sionna`** conda environment.

```bash
conda create -n sionna python=3.12
conda activate sionna
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Verify:

```python
import mitsuba as mi
import drjit as dr
import sionna.rt as rt
import numpy as np

mi.set_variant("llvm_ad_mono_polarized")  # same as main.py
print("Mitsuba OK:", mi.variant())
print("Sionna RT OK")
```

`main.py` sets `mi.set_variant("llvm_ad_mono_polarized")`. If you have a CUDA GPU and a matching Mitsuba build, you can switch to a `cuda_ad_*` variant for faster solves.

---

## Running the pipeline

From the repo root:

```bash
conda activate sionna
python main.py
```

1. **Select mode** — Individual or Batch.
2. **Per jammer (individual)** — Choose strategy (Math Modeling or Waypoint), time step, padding mode, then plan in the map GUI.
  - **Math planner:** use **Change initial jammer position** before adding any segment (click map; position is saved to `main.py`).
3. **Batch** — Set graph parameters, build the navigation graph in the GUI, then generate `N` paths.
4. After planning, the tool **saves trajectories**, runs the **radio map simulation** (can take a long time), and writes outputs under `datasets/<DATASET_NAME>/`.

---

## Configuration (`main.py`)

Edit these constants before running:


| Setting                  | Description                                                                  |
| ------------------------ | ---------------------------------------------------------------------------- |
| `SCENE_PATH`             | Mitsuba scene XML (e.g. `./data/NYC3KM_585751_4512036/simple_OSM_scene.xml`) |
| `MESHES_PATH`            | Folder with building `.ply` meshes (for collision / plotting)                |
| `OUTPUT_DIR`             | Base output folder (default `./datasets`)                                    |
| `DATASET_NAME`           | Subfolder name for this run                                                  |
| `FREQ_HZ`                | Carrier frequency (default GPS L1: `1.57542e9`)                              |
| `Z_HEIGHT`               | Jammer altitude (m)                                                          |
| `GLOBAL_TX_POWER_DBW`    | TX power; converted to dBm for Sionna                                        |
| `initial_jammers_config` | List of jammers: `name`, `initial_position`, `power_dbm`, `color`            |
| `map_bounds`             | `b` → square region `[-b, b]` in x and y                                     |
| `cell_size`              | Radio map grid resolution `(8, 8)` m                                         |


Example jammer entry:

```python
{
    "name": "Jammer1",
    "initial_position": np.array([-595.8, 55.2, Z_HEIGHT]),
    "power_dbm": GLOBAL_TX_POWER_DBM,
    "color": [1.0, 0.0, 0.0],
}
```

---

## Generated datasets

After a successful run, outputs live in:

```text
datasets/<DATASET_NAME>/
  path_Jammer1.npy          # (T, 3) — x, y, z per timestep
  meta_Jammer1.txt          # strategy, distance, duration, segment count
  rss_aggregated.npy        # (T, H, W) — combined RSS in dBW (+ noise floor)
  rss_Jammer1.npy           # (T, H, W) — per-jammer RSS in dBW
  jammer_animation.gif      # aggregated RSS over time
```

- **T** = number of timesteps (longest path after padding).
- **H, W** = radio map grid size from `cell_size` and `map_bounds`.
- Aggregated RSS sums all transmitters in watts, adds noise floor `8e-15` W, then converts to dBW.

`datasets/` is in `.gitignore`; copy results elsewhere if you need to version them.

### Optional: visualize paths only

```bash
conda activate sionna
python visualize_paths.py
# or explicitly:
python visualize_paths.py --folder ./datasets/NYC_3jammer --meshes ./data/NYC3KM_585751_4512036/mesh
```

Loads `path_*.npy` files (not RSS arrays). The first run may take a minute while ~7k building meshes are read for the map background.

---

## Importing new maps

> **Owner: Luis** — this section is a placeholder for the OSM → Sionna scene workflow.

### Expected layout

Each scene should be a folder under `data/`:

```text
data/<SceneName>/
  simple_OSM_scene.xml    # Mitsuba scene (or equivalent)
  mesh/
    *.ply                 # Building meshes (same paths referenced by XML)
```

`main.py` must point `SCENE_PATH` and `MESHES_PATH` at these paths. The `mesh/` directory must stay **next to** the XML file with consistent relative paths inside the XML.