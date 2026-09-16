# Batch Simulation — NYC 3 km

Dataset for GNSS jammer **detection** (DeepMTL-style) and **tracking** (PMBM), built with
Sionna RT on the `NYC3KM_585751_4512036` scene.

Batch datasets live in `datasets/` and are named **`batch_simulation_<tag>`**. Every one has
the same internal layout, so `--dataset-dir` is the only path the pipeline needs.

There are **two branches**, because detection and tracking want different data. Each has a
*single*-jammer base library (ray traced, GPU) and a *multi*-jammer combination built from it
(NumPy, CPU):

```
datasets/batch_simulation_nyc/          <- --dataset-dir
├── README.md
├── splits.json                  pools, scenarios, static positions, sample specs
├── labels/                      per-frame jammer positions
├── sensors/                     per-scenario sensor layouts
│
│   ── TRACKING ───────────────────────────────────────────────────────────
├── single_trajectory_jammers/   54 trajectories, one map per frame     [GPU]
│   ├── traj_*.npy / .json
│   └── radio_maps/
├── multi_trajectory_jammers/    5000 scenarios                         [CPU]
│   └── {train,val,test}/<scenario_id>/{rss_aggregated.npy, scenario_summary.json, labels.npz}
│
│   ── DETECTION ──────────────────────────────────────────────────────────
├── single_static_jammers/       N static positions, one map each       [GPU]
│   ├── positions.npy            (N, 3) float64, continuous metres
│   └── watts.npy                (N, 300, 300) float32, memmap-able
└── multi_static_jammers/        100 000 detector samples               [CPU]
    └── {train,val,test}/{rss.npy, labels.npz, meta.npz}
```

**Why the detector gets its own library.** DeepMTL is a single-snapshot model — motion is irrelevant. Sampling from trajectories would confine every training jammer to 54 corridors with adjacent frames correlated at r = 0.884. Independent static positions cost the same 0.235 s of GPU each and scatter over the whole street network. It also removes a methodological trap: the detector never sees a trajectory frame, so detections fed to the tracker are **out-of-sample by construction**.

> **Dataset fully generated.** GPU job 10357699 (2026-09-15, ~1.9 h) produced all 54 trajectory maps and 20 000 static maps. CPU job 10361058 (2026-09-15) produced all scenarios, samples, splits, and labels — 13/13 validation checks passed.

---

## Design decisions

| | Decision | Why |
|---|---|---|
| Grid | **300 × 300 @ 10 m**, bounds ±1500 m | Same cell size as DeepMTL, so their metrics in metres and their blob/anchor hyperparameters transfer directly |
| Tracking scenarios | **4000 / 500 / 500** = 5000, from 34/10/10 trajectories | |
| Static positions | **20 000**, split **14 000 / 3 000 / 3 000** at random | 78 min GPU, 7.2 GB; random split because samples combine K of them under fresh noise and sensors |
| Detector samples | **70 000 / 15 000 / 15 000** = 100 000 | Matches DeepMTL's dataset scale |
| Trajectory pools | **34 / 10 / 10**, disjoint, stratified over duration × velocity | Bases are reused in hundreds of scenarios; sharing one across splits leaks geometry and multipath |
| K (jammers) | **0–10, stratified** (equal count per K) | A uniform draw over subsets yields ≤ 1 noise-only scenario per split; stratified gives 46 |
| RSS storage | **full dense grid**, every cell | Sensor sub-sampling is preprocessing, so density can change without regenerating |
| Sensor placement | **street cells only** (30.9 % of the grid), fresh layout per scenario, min-spacing relocation | Indoor GNSS receivers are not useful sensors; no train/test partition, which would bias the splits |
| Sensor density | **2/4/6/8/10 %**, drawn per scenario | One model over mixed density, evaluated stratified, reproduces DeepMTL's density curve |
| TX power | **fixed 10 dBW** — settled, not varied | Free to add later without re-tracing (see [Part 6](#free-augmentation-tx-power)); out of scope for this paper |
| Measurement noise | **Gaussian in dB**, var 1.0 (σ = 1 dB, INR 10 dB) | Matches the sensor-noise model and INR sweep from the federated localization work |
| Precision | **float16** on disk | Quantisation inflates noise by 0.07 % at σ = 1 dB — irrelevant |

### Fixed scene parameters

| Parameter | Value |
|---|---|
| Scene | `data/NYC3KM_585751_4512036/simple_OSM_scene.xml` |
| Map bounds | x, y ∈ [−1500, +1500] m |
| Jammer altitude | z = 1.5 m |
| Carrier | 1.57542 GHz (GPS L1) |
| Noise floor | 8 × 10⁻¹⁵ W (= −140.97 dBW), constant |
| Frame rate | 1 fps (`time_step = 1.0 s`) |
| Ray tracing | `max_depth=80`, `samples_per_tx=1e7`, diffraction + edge diffraction + refraction on |
| Seed | 42 |

---

## Part 1 — The 54 single-jammer trajectories

A 3 × 3 sweep, 6 trajectories per cell:

| | 3 m/s | 9 m/s | 15 m/s |
|---|---|---|---|
| **30 s** (30 frames) | 6 | 6 | 6 |
| **60 s** (60 frames) | 6 | 6 | 6 |
| **90 s** (90 frames) | 6 | 6 | 6 |

Named `traj_dur{DURATION}s_vel{VELOCITY:02d}mps_{INDEX:02d}`.

`core/trajectory_generator.py`, driven by `run_generation()`:

1. Combinations sorted **descending by travel distance** — longest paths (90 s @ 15 m/s = 1335 m) claim corridors first.
2. `find_straight_street_corridors()` rejection-samples start + heading; keeps only collision-free paths (`min_separation=25 m`, up to 40 000 attempts).
3. Candidates rejected if overlap with any accepted trajectory exceeds `max_overlap_ratio=0.75` within `proximity_threshold=20 m`.
4. Straight-line, constant velocity, zero acceleration. `inclusive_end=False` → 30 s = **30 frames**, not 31.

**Files** — `traj_*.npy` is `(T, 3)` **float64**, continuous XYZ in metres: the ground truth
for every label downstream. `traj_*.json` carries `start_position`, `end_position`,
`heading_degrees`, `unit_direction`, `velocity_mps`, `total_distance_meters`,
`is_collision_free`, `kinematics`.

## Part 2 — The 54 base radio maps  `[GPU]`

Each transmitter is stepped along its path and `RadioMapSolver` runs **once per frame**.
This is the only stage that needs a GPU: **3240 frames in 762 s ≈ 12.7 min** measured
(~0.24 s/frame). Cost is dominated by `samples_per_tx`, not grid size, so 10 m costs the
same as 8 m.

| File | Shape / dtype | Notes |
|---|---|---|
| `rm_watts_{traj_id}.npy` | `(T, 300, 300)` float32 | **Linear watts — the one that matters** |
| `rm_dbw_{traj_id}.npy` | `(T, 300, 300)` float32 | `10·log10(watts + 8e-15)`, derived |
| `rm_{traj_id}.json` | — | `num_steps`, `power_dbw`, min/max dBW, `sim_time_seconds` |
| `radio_maps_manifest.json` | — | index over all 54 |

**Why watts.** Jammers are incoherent — power sums linearly in watts, not in dBW. Storing watts reduces a K-jammer scenario to a NumPy add over precomputed maps instead of a new ray trace.

**⚠️ Two base maps contain `+inf` cells** (zero-distance singularity where the TX sits exactly
on a grid point). Sanitize on load:

```python
if not np.all(np.isfinite(w)):
    fm = np.isfinite(w)
    w = np.nan_to_num(w, posinf=float(w[fm].max()), neginf=0.0, nan=0.0)
```

These 54 maps are also **3240 labelled single-jammer frames** — usable as K=1 data directly.

---

## Part 2b — The static base library  `[GPU]`

The detection branch's equivalent of Part 2: one ray-traced map per **static** jammer
position, instead of one per trajectory frame.

**Positions** (`--action generate_static`, CPU, seconds). N cells drawn **with replacement** from street cells, each jittered continuously inside its cell:

```python
x = -1500 + (col + rng.random()) * 10.0      # never the cell centre
```

The jitter is not cosmetic: without it `dx = dy = 5.0` for every sample and the sub-cell regression target is constant — DeepMTL's second stage has nothing to learn. It also rules out "just ray trace all 27 788 street cells" as a shortcut.

**Maps** (`--action simulate_static`, **GPU**). One `RadioMapSolver` call per position, at
the measured 0.235 s each:

| N | GPU | `watts.npy` | street coverage |
|---|---|---|---|
| 5 000 | 20 min | 1.8 GB | 18 % |
| 10 000 | 39 min | 3.6 GB | 36 % |
| **20 000 (default)** | **78 min** | **7.2 GB** | **72 %** |

Stored as **one `(N, 300, 300)` float32 array**, not N files. 20 000 separate `.npy` files
would burn 20 000 inodes — cluster filesystems have quotas — and make the random access the
combination step needs awkward. A single array is `mmap_mode='r'` and O(1) by index. The
stage checkpoints every 50 positions to `watts.progress.json` and resumes.

N buys **position diversity only** — a sample is "sum K maps + noise" (~1 ms), so 20 000 positions comfortably support 100 000 samples.

## Part 2c — Detector samples  `[CPU]`

`--action aggregate_static`. Each sample is one snapshot:

1. **K** drawn stratified over 0…10 — equal counts per K, so ~9 % are noise-only negatives
2. **K positions** drawn from that split's position pool
3. summed in watts, then `10·log10(ΣP + 8e-15)`
4. **σ noise** added in dB from the sample's own `noise_seed` (Part 4)
5. **sensors**: a density from {2, 4, 6, 8, 10} %, a layout from the sample's `sensor_seed`

Written per split as three files, again to keep the inode count sane:

| File | Contents |
|---|---|
| `rss.npy` | `(n, 300, 300)` float16 — dense, mask to sensors downstream |
| `labels.npz` | one row per (sample, jammer): `x, y` in metres + `col, row, dx, dy` |
| `meta.npz` | `num_jammers`, `position_ids` (+offsets), `sensor_density_pct`, `num_sensors`, `sensor_seed`, `noise_seed` |

100 000 samples ≈ **18 GB in 12 files**, versus 300 000 inodes for DeepMTL's
directory-per-sample layout.

### Splitting the static positions

Split at N = 20 000: **14 000 / 3 000 / 3 000**, drawn at random. A sample only ever combines positions from its own split.

The split is **not** spatially separated. A detector sample is a sum of K fields under a per-sample density, sensor layout, and noise draw — no configuration repeats across splits, so there is nothing to memorise at the level the model sees. This matches the DeepMTL baseline, which trains and tests over the same area.

**What the test set measures:** generalisation to unseen *configurations* (counts, positions, densities, layouts, noise) — not to an unseen part of the city (that would require a spatial partition). State the claim the first way.

---

## Part 3 — Tracking scenarios  `[CPU]`

### Splits

| Split | Pool | Scenarios | Subset space | Snapshots @ 10/scenario |
|---|---|---|---|---|
| train | 34 trajectories | 4000 | 208 791 332 | 40 000 |
| val | 10 trajectories | 500 | 1024 | 5 000 |
| test | 10 trajectories | 500 | 1024 | 5 000 |
| **total** | **54** | **5000** | | **50 000** |

Pools are stratified over the 3 × 3 duration × velocity grid and **disjoint** — scenarios only combine trajectories from their own pool. This is the single most important rule: with mean K ≈ 5, each base appears in hundreds of scenarios, so a shared trajectory leaks geometry and multipath into evaluation.

### Why K is stratified

C(10,0) = 1 — uniform sampling over subsets gives at most **one** noise-only scenario per split. One negative in 500 is useless for a detector.

`--k-balance stratified` (default) allocates equally across K = 0…10, letting subsets repeat where the space is smaller than the quota. **Repeats are still distinct**: each scenario draws its own noise, sensor density, and sensor layout.

| | scenarios | distinct subsets | per K |
|---|---|---|---|
| train | 4000 | 3307 | 364 (363 for K ≥ 7) |
| val | 500 | 339 | 46 (45 for K ≥ 5) |
| test | 500 | 339 | 46 (45 for K ≥ 5) |

That is **46 noise-only scenarios per evaluation split**. `--k-balance subset` restores
uniform-over-subsets if wanted.

### Length and padding

Scenario length = max over its jammers. Shorter jammers are held at their final position:

```python
padded = np.vstack([watts, np.tile(watts[-1:], (max_steps - len(watts), 1, 1))])
```

`original_steps` and `padded_steps` recorded per jammer. **A padded jammer is still transmitting** — positive label throughout. K=0 scenarios sample their length from the pool's trajectory lengths.

### Disk

A 90-frame cube at 300 × 300 float16 is **16.2 MB**. Measured totals for 4000/500/500:
**257 700 frames**, 71.6 h of simulated time.

| | full cubes | 10 sampled frames/scenario |
|---|---|---|
| 5000 scenarios | **~69 GB** | **~9 GB** (50 000 snapshots) |

Tracking needs the full cubes; detection needs only the sampled frames. Write them as two
sets rather than slicing 69 GB at train time. Always pass `--skip-gif` — GIFs are ~25 MB
each and would roughly double everything.

---

## Part 4 — Aggregation: RSS + noise

### The physics

```
P_total(t,i,j) = sum_k P_k(t,i,j)
RSS_dBW        = 10 * log10(P_total + N0) + eps
```

**Noise floor** `N0 = 8e-15 W` — added *before* the log so zero-power cells give −140.97 dBW instead of −∞. Not noise; eliminates the need for −inf workarounds.

**Measurement noise** `eps` — random draw *after* the log, necessary because Sionna RT is deterministic (same positions → byte-identical maps). Without it every K=0 scenario is a flat plane and "is any pixel ≠ −140.97 dBW?" trivially detects jammers.

```python
agg_dbw += rng.normal(0.0, np.sqrt(meas_noise_var), size=agg_dbw.shape)
```

Gaussian in dB = log-normal in power — the shadowing form from log-distance models (and DeepMTL's setup), not thermal receiver noise.

With Ptx = 10 dBW, INR = 10·log₁₀(Ptx / `meas_noise_var`):

| `--meas-noise-var` | σ | INR |
|---|---|---|
| 10 | 3.162 dB | 0 dB |
| 10/√10 ≈ 3.162 | 1.778 dB | 5 dB |
| **1.0 (default)** | **1.000 dB** | **10 dB** |
| 0.1 | 0.316 dB | 20 dB |

`--meas-noise-var 0` disables it. `meas_noise_var_db` and `seed` are recorded in each
`combination_summary.json`.

---

## Part 5 — Labels

### Positions are stored in continuous metres

`traj_*.npy` is float64 XYZ. Labels keep it that way; cell index and sub-cell offset are derived:

```python
col_f = (x - x0) / cell      # continuous, e.g. 111.8723
col   = int(np.floor(col_f)) # cell index      111
dx    = (col_f - col) * cell # offset in m     8.723
```

Anchor + offset: predict the cell as classification, `(dx, dy)` as bounded regression, recover metric position exactly. Quantising to cell centres would inject **mean 4.01 m, p95 5.82 m** of error — the same magnitude as DeepMTL's full reported error (~1–5 m).

### Output

`labels_{split}.npz` — one row per (scenario, frame, jammer): `x, y, z` metres, `col, row, dx, dy`, `velocity_mps`, `heading_deg`, `is_padded`. Measured: **2.18 M rows, ~10 MB compressed**. Built from `splits.json` + `traj_*.npy` only — not the radio maps — so labels can be regenerated before aggregation and independently.

---

## Part 6 — Sensors and DeepMTL

Reference: Zhan, Ghaderibaneh, Sahu, Gupta, *"DeepMTL: Deep Learning Based Multiple
Transmitter Localization"*, IEEE WoWMoM 2021. Two stages: **sen2peak** (sensor readings →
Gaussian blobs at TX locations) then **YOLOv3-cust** (blobs → coordinates).

| | DeepMTL | This dataset |
|---|---|---|
| Area | 1 km × 1 km | 3 km × 3 km |
| Grid | 100 × 100 @ 10 m | 300 × 300 **@ 10 m (same cell)** |
| Sensor coverage | sparse, 2–10 % (default 6 %) | sparse, 2–10 % drawn per scenario |
| Transmitters | 1–10 (default 5) | 0–10 |
| TX power | random, 0–5 dBm | fixed, 40 dBm |
| Propagation | log-distance + Gaussian shadowing, or SPLAT! | Sionna RT (ray traced) |
| Transmitters move | no | **yes, trajectories** |
| Dataset size | 100 000 train / 20 000 test | 70 000 / 15 000 / 15 000 (detector) |

### Dense on disk, sparse at training time

DeepMTL's input holds readings only at sensor locations (6 % density → 94 % empty). Our maps are dense: every cell has a ray-traced value. **Stored dense; sub-sampled to sensors at preprocessing time** using `splits.json`. Sensor density and placement stay free parameters that can change without regenerating 69 GB.

What `splits.json` fixes:

- **`sensor_cells`** — every placeable (street) cell, **shared by all splits**. No spatial partition: DeepMTL's source reuses one fixed layout per `(grid_length, density, seed)` for both train and test; we do the same.
- **`sensors/sensors_{split}.npz`** — actual layouts: `cells` (flat, row-major) + `offsets` per scenario, `density_pct`, `spacing_radius`. Stored (not regenerated from seed) so the relocation pass doesn't need reimplementing in the training repo. ~47 MB for 5000 scenarios.
- **Per scenario** — `sensor_density_pct`, `num_sensors`, `sensor_seed`, `noise_seed`.

### Min-spacing relocation

Uniform draws clump. DeepMTL uses a greedy claim-and-relocate pass (5 rounds): each sensor claims a `(2r+1)²` block; sensors landing in a claimed block are re-drawn. We do the same (`--relocation-passes`, default 5) with two deviations:

- **Best effort:** their code raises when free cells run out. Ours leaves un-relocatable sensors in place (count is exact).
- **Radius is computed, not tabulated:** their tabulated radii (4 cells at 200 sensors → 2 at 1000) assume a full grid. Placement is street-only here — **30.9 % of cells** — so in-street density is ~3× nominal and those radii are unreachable.

Measured on the NYC scene, the computed radius beats their table precisely because
over-claiming flags more sensors than there are free cells to move them to:

| density | sensors | r | mean NN | % within 2 cells |
|---|---|---|---|---|
| 2 % | 1800 | 1 | 3.49 | **0 %** |
| 4 % | 3600 | 1 | 2.66 | **0 %** |
| 6 % | 5400 | 1 | 2.33 | **0 %** |
| 8 % | 7200 | 1 | 2.11 | 7 % |
| 10 % | 9000 | 1 | 1.71 | 39 % |

(Uniform placement gives 33 % / 67 % / 85 % within 2 cells at 2/6/10 %. DeepMTL's own r=2 at
10 % leaves 81 %.) The 10 % case is geometrically capped: 9000 sensors in 27 788 street cells
is 32 % occupancy, so some crowding is unavoidable.

Sensors are **street-only** — indoor receivers are useless for GNSS. Density is a percentage of **total grid cells** (comparable to DeepMTL's figure); placement is restricted to street cells. `--no-street-only` reverts to uniform.

### Their on-disk layout vs. ours

Read from their source, not the paper. One DeepMTL sample:

```
data/{dataset}/{folder:06d}/{i}.npy       input  - float32 (grid_length, grid_length)
                            {i}.target    label  - float32 CONTINUOUS tx coordinates
                            {i}.power     label  - float32 per-tx powers
                            {i}.json      meta   - {"0": {"location": (x, y), "gain": g}, ...}
                                                   plus per-sensor rssi, coords rounded to 3dp
```

| | DeepMTL | Ours |
|---|---|---|
| Sample unit | **one snapshot** | **one scenario = T frames** |
| Input | `{i}.npy`, sparse, sensors only | `rss_aggregated.npy` `(T,300,300)` float16, **dense** |
| Label | `{i}.target`, continuous x, y | `labels.npz`, continuous x, y + derived `col,row,dx,dy` |
| Power | `{i}.power`, varies 0–5 dBm | fixed 40 dBm, not stored per jammer |
| Gaussian target | built at train time | same — built downstream |
| Sensor layout | one fixed layout per (grid, density, seed), reused for train and test | fresh layout per scenario, stored in `sensors/` |
| Jammer count | 1–10 | 0–10 |

Two differences matter for wiring into their model:

1. **Time series vs. snapshot.** Flatten: each `(scenario, frame)` pair → one DeepMTL sample (50 000 at 10 frames/scenario). Cubes stay un-flattened on disk for the PMBM.
2. **Dense vs. sparse.** Mask with the stored sensor cells to reproduce their sparse input.

Everything else aligns: same 10 m cell, continuous coordinates, Gaussian targets built at train time.

### Do downstream, in the training repo

- **Sparse masking + normalisation** — gather dense map at sensor cells, floor the rest. Subtract `N`, divide by `−N/2` → empty cells = 0, sensor cells ∈ (0, 2].
- **Gaussian blob targets** — model hyperparameter. DeepMTL: amplitude 10, σ = 0.9, 5 × 5 support, centred on continuous TX location.
- **YOLO labels** — `(class=1, x, y, w=5, h=5)`.
- **Resize / tiling** — DeepMTL resizes 100 × 100 → 416 × 416, 3 channels (416 = YOLOv3 default, multiple of 32). Our grid is 300 × 300 at the same 10 m cell; for a larger scene keep cell size at 10 m and **tile** (overlapping windows, NMS across seams; overlap must exceed blob size).

### Free augmentation: TX power

DeepMTL randomises TX power; ours is fixed. Adding it requires **no new ray tracing** — power scales linearly in watts:

```python
scaled = base_watts * 10 ** ((P_new_dBm - 40.0) / 10.0)   # per jammer, before summing
```

Settled: fixed for this paper. Noted because it costs nothing to add later.

### Not covered by DeepMTL

- **K = 0.** Their transmitter count starts at 1; noise-only scenarios are our addition, so
  there is no baseline to compare against.
- **Motion.** DeepMTL localises static transmitters from one snapshot. Our scenarios are
  trajectories, which is what also feeds the PMBM — but only the per-frame detection task maps
  onto DeepMTL directly.

---

## Part 7 — Running it

Everything derives from `--dataset-dir`. Only the two `simulate_*` stages need a GPU.

| Stage | Action | Where | Cost |
|---|---|---|---|
| 1 | `generate` | CPU | seconds |
| 2 | `simulate_bases` | **GPU** | ~13 min |
| 3 | `generate_static` | CPU | seconds |
| 4 | `simulate_static` | **GPU** | ~78 min at N=20 000 |
| 5 | `scripts/make_splits.py` | CPU | ~2 min |
| 6 | `scripts/make_labels.py` | CPU | seconds |
| 7 | `aggregate` | CPU | ~69 GB out |
| 8 | `aggregate_static` | CPU | ~18 GB out |
| 9 | `scripts/validate_dataset.py` | CPU | minutes |

On the cluster that is two jobs, so the GPU reservation is released as soon as ray tracing
finishes:

```bash
./submit.sh gpu     # stages 1-4, needs a GPU, ~1.5 h
./submit.sh cpu     # stages 5-9, no GPU
```

`make_splits.py` must run **after** both `simulate_*` stages: it reads
`single_static_jammers/positions.npy` to plan the detector samples, and skips the whole
detection branch with a warning if that file is missing.

`submit.sh` picks partition, gres, and account per site from `server_env.sh` (since `#SBATCH` directives can't be conditional). `run_pipeline.sh` is Luis's original single-job script (`eff898c4`), kept as reference — it predates the two-branch layout.

### Memory, not compute, is the CPU constraint

- `aggregate`: holds one trajectory pool (~1.4 GB for the 34-trajectory train pool)
- `aggregate_static`: memmaps the 7.2 GB static library, touches K maps per sample

### Verified

Tracking branch, against the real base maps:

- Trajectory pools disjoint, union = 54, all 9 duration × velocity strata in every split
- No scenario draws a trajectory outside its own pool
- Scenario ids match `splits.json`; K=0 cubes carry σ = 1.00 dB, not flat
- Labels round-trip to 4.5 × 10⁻¹³ m; all inside the grid
- The grid guard fires: a 10 m `splits.json` against an 8 m library raises

Detection branch, end-to-end on CPU (ray tracing stubbed, so the maps were synthetic but
every shape, dtype and index path is real):

- `generate_static` → 400 positions, all on street cells, continuously jittered
- `make_splits` → position split disjoint across train/val/test
- `aggregate_static` → **K histogram exactly uniform 0…10**, all five densities present,
  label rows = Σ K, positions drawn only from the split's own pool
- K=0 samples measure σ = 0.999 dB — noise-only, not flat
- Sub-cell offsets span `dx ∈ [0.054, 9.970] m` — the regression target is non-degenerate,
  which is the whole point of the jitter

---

## Status

**Code: complete.** Every stage is written and every CPU stage has been executed end-to-end.

| | |
|---|---|
| ✅ | `generate`, `plot`, `simulate_bases`, `aggregate` — tracking branch |
| ✅ | `generate_static`, `simulate_static`, `aggregate_static` — detection branch |
| ✅ | `scripts/make_splits.py`, `make_labels.py`, `validate_dataset.py` |
| ✅ | `run_pipeline_gpu.sh` / `run_pipeline_cpu.sh` — GPU and CPU jobs separated |
| ✅ | Settled: 10 m grid, 20 000 positions → 14k/3k/3k, 70k/15k/15k samples, K 0–10 stratified, fixed TX power, σ = 1 dB, random position split |

**Base libraries: generated.** GPU job `10357699` on Explorer, 1 h 53 m, exit 0.

| | |
|---|---|
| `single_trajectory_jammers/` | 54 trajectories + 54 radio maps, **2.2 GB** |
| `single_static_jammers/` | `watts.npy` `(20000, 300, 300)` float32 = **7.2 GB** exactly; 20 000 positions over 14 305 distinct street cells |
| Throughput | 3.55 positions/s, 94 min for the static library |
| Variant | `cuda_ad_mono_polarized` — real CUDA, via the patched drjit (see repo README) |

- [x] `./submit.sh cpu` — job 10361058 on Explorer, completed 2026-09-15, 13/13 validation checks passed; ~96 GB total on disk

**Deferred, flagged in place.**

- [x] `visualize_aggregated.py` — deleted; replaced by `scripts/preview_dataset.py`
- [ ] `main_interactive_local.py` — `b = 750`, `cell_size = (8, 8)`, so its output is **not**
      comparable with this dataset (TODO at the top of its `main()`)

Downstream, in the training repo: sparse masking, Gaussian blob targets, YOLO labels,
resize/tiling.
