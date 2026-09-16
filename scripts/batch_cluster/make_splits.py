#!/usr/bin/env python3
"""
Build splits.json for a batch_simulation_<tag> dataset.

Three things are decided here, once, and written to disk so they are reproducible
and auditable:

1. Trajectory pools   - the 54 base trajectories partitioned into train/val/test,
                        stratified over the 3x3 duration x velocity grid.
2. Scenario subsets   - which base trajectories make up each scenario.
                        Train samples randomly; val/test enumerate their pool's
                        full subset space and take a seeded slice of it.
3. Static positions   - the detector's base positions split into train/val/test, plus
                        one spec per detector sample (K, density, seeds). Only written
                        if single_static_jammers/positions.npy exists.
4. Sensor draws       - one density and one seed per scenario, drawn independently
                        from the shared pool of placeable (street) cells. No
                        train/test partition of the grid: that would give each split a
                        different spatial sample of the city and bias the sets.

Nothing here touches the radio maps, so it runs in seconds on CPU. The only
optional dependency is trimesh, needed when --street-only is set (the default) to
exclude cells that fall inside buildings.

Layout assumed (paths derived from --dataset-dir):

    datasets/batch_simulation_<tag>/
        single_trajectory_jammers/   traj_*.npy + radio_maps/      [tracking base]
        single_static_jammers/       positions.npy + watts.npy     [detection base]
        multi_trajectory_jammers/    tracking scenarios
        multi_static_jammers/        detector samples
        splits.json                  <- this script
        labels/  sensors/            <- scripts/batch_cluster/make_labels.py, this script

Usage:
    python scripts/batch_cluster/make_splits.py \
        --dataset-dir ./datasets/batch_simulation_nyc \
        --mesh-dir ./data/NYC3KM_585751_4512036/mesh
"""

import argparse
import itertools
import json
import os
import re
import sys

import math

import numpy as np

TRAJ_RE = re.compile(r"traj_dur(\d+)s_vel(\d+)mps_(\d+)$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="./datasets/batch_simulation_nyc",
                   help="Dataset root, by convention ./datasets/batch_simulation_<tag>. "
                        "--traj-dir and --out are derived from it unless given.")
    p.add_argument("--traj-dir", default=None,
                   help="Override for the single-jammer folder "
                        "(default: <dataset-dir>/single_trajectory_jammers).")
    p.add_argument("--mesh-dir", default="./data/NYC3KM_585751_4512036/mesh",
                   help="Building meshes, used for --street-only sensor placement.")
    p.add_argument("--out", default=None,
                   help="Override for splits.json (default: <dataset-dir>/splits.json).")

    p.add_argument("--n-train", type=int, default=4000, help="Training scenarios (default: 4000).")
    p.add_argument("--n-val", type=int, default=500, help="Validation scenarios (default: 500).")
    p.add_argument("--n-test", type=int, default=500, help="Test scenarios (default: 500).")
    p.add_argument("--pool-train", type=int, default=34, help="Trajectories in the train pool (default: 34).")
    p.add_argument("--pool-val", type=int, default=10)
    p.add_argument("--pool-test", type=int, default=10)
    p.add_argument("--min-jammers", type=int, default=0, help="Minimum K (default: 0, noise-only allowed).")
    p.add_argument("--max-jammers", type=int, default=10)
    p.add_argument("--k-balance", choices=["stratified", "subset"], default="stratified",
                   help="stratified (default): equal scenarios per K, repeating subsets where the "
                        "space is small - guarantees a usable number of K=0 negatives. "
                        "subset: uniform over distinct subsets, which follows C(n,K) and yields "
                        "at most ONE K=0 scenario per split.")

    p.add_argument("--map-bounds-b", type=float, default=1500.0, help="Half-width in metres (default: 1500).")
    p.add_argument("--cell-size", type=float, default=10.0, help="Grid cell size in metres (default: 10).")
    p.add_argument("--densities", type=float, nargs="+", default=[2.0, 4.0, 6.0, 8.0, 10.0],
                   help="Sensor densities in percent (default: 2 4 6 8 10).")
    p.add_argument("--street-only", dest="street_only", action="store_true", default=True,
                   help="Exclude cells inside buildings when placing sensors (default).")
    p.add_argument("--no-street-only", dest="street_only", action="store_false",
                   help="Place sensors uniformly over all cells, as DeepMTL does.")

    p.add_argument("--static-dir", default=None,
                   help="Default: <dataset-dir>/single_static_jammers. If positions.npy "
                        "exists there, detector sample specs are added to splits.json.")
    p.add_argument("--n-static-train", type=int, default=70000)
    p.add_argument("--n-static-val", type=int, default=15000)
    p.add_argument("--n-static-test", type=int, default=15000)
    p.add_argument("--static-split-frac", type=float, nargs=3, default=[0.70, 0.15, 0.15],
                   help="Fraction of static POSITIONS given to train/val/test (default: "
                        ".70 .15 .15 -> 14000/3000/3000 at N=20000). Positions are split "
                        "first; samples only ever combine positions from their own split.")
    p.add_argument("--relocation-passes", type=int, default=5,
                   help="Min-spacing relocation passes per sensor layout, as in DeepMTL "
                        "(default: 5). 0 = plain uniform placement.")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    root = os.path.normpath(a.dataset_dir)
    if not os.path.basename(root).startswith("batch_simulation_"):
        print(f"[warn] dataset dir '{os.path.basename(root)}' does not follow the "
              f"batch_simulation_<tag> convention; continuing anyway.")
    if a.traj_dir is None:
        a.traj_dir = os.path.join(root, "single_trajectory_jammers")
    if a.out is None:
        a.out = os.path.join(root, "splits.json")
    if a.static_dir is None:
        a.static_dir = os.path.join(root, "single_static_jammers")
    return a


# -----------------------------------------------------------------------------
# 1. Trajectory pools
# -----------------------------------------------------------------------------
def build_pools(traj_dir, sizes, rng):
    """Partition the 54 trajectories, stratified over duration x velocity."""
    ids = sorted(
        os.path.splitext(f)[0] for f in os.listdir(traj_dir)
        if f.startswith("traj_") and f.endswith(".npy")
    )
    if not ids:
        raise FileNotFoundError(f"No traj_*.npy found in {traj_dir}")

    # group by (duration, velocity)
    cells = {}
    for tid in ids:
        m = TRAJ_RE.match(tid)
        if not m:
            raise ValueError(f"Unexpected trajectory name: {tid}")
        dur, vel = int(m.group(1)), int(m.group(2))
        cells.setdefault((dur, vel), []).append(tid)

    n_total = sum(sizes.values())
    if len(ids) != n_total:
        raise ValueError(f"Pool sizes sum to {n_total} but found {len(ids)} trajectories")

    # Deal round-robin out of each stratum so every split covers all kinematics.
    # Order the splits by how many slots they still need, so the remainders after
    # an even per-cell deal land where they are needed rather than always in train.
    pools = {k: [] for k in sizes}
    remaining = dict(sizes)
    for key in sorted(cells):
        members = list(cells[key])
        rng.shuffle(members)
        for tid in members:
            # give to whichever split is furthest from its quota, proportionally
            pick = max(remaining, key=lambda s: (remaining[s] / sizes[s] if sizes[s] else -1,
                                                 remaining[s]))
            pools[pick].append(tid)
            remaining[pick] -= 1

    for split, want in sizes.items():
        got = len(pools[split])
        if got != want:
            raise AssertionError(f"{split}: wanted {want} trajectories, allocated {got}")
        pools[split].sort()

    # no trajectory may appear in two pools
    seen = set()
    for split, members in pools.items():
        overlap = seen & set(members)
        if overlap:
            raise AssertionError(f"{split} overlaps an earlier pool: {sorted(overlap)}")
        seen |= set(members)

    return pools, cells


# -----------------------------------------------------------------------------
# 2. Scenario subsets
# -----------------------------------------------------------------------------
def _subsets_for_k(pool_size, K, want, rng):
    """`want` subsets of size K, distinct where the space allows, repeating when it does not.

    Low K has very few distinct subsets - K=0 has exactly one, the empty set. Repeats
    are still useful samples because each scenario gets its own measurement-noise draw,
    so two scenarios over the same jammers are not identical maps.
    """
    space = 1
    for i in range(K):
        space = space * (pool_size - i) // (i + 1)

    if space <= want:                                   # take all, then repeat to fill
        allsubs = list(itertools.combinations(range(pool_size), K))
        out = list(allsubs)
        while len(out) < want:
            out.extend(allsubs[:want - len(out)])
        return out[:want]

    seen, out = set(), []                               # sample distinct
    while len(out) < want:
        idx = tuple(sorted(rng.choice(pool_size, size=K, replace=False))) if K else ()
        if idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return out


def stratified_scenarios(pool, n_scen, k_min, k_max, rng):
    """Equal count per K, so K=0 negatives and high-K scenarios are both represented."""
    k_hi = min(k_max, len(pool))
    ks = list(range(k_min, k_hi + 1))
    base, extra = divmod(n_scen, len(ks))
    want = {K: base + (1 if i < extra else 0) for i, K in enumerate(ks)}

    out = []
    for K in ks:
        for idx in _subsets_for_k(len(pool), K, want[K], rng):
            out.append([pool[i] for i in idx])
    rng.shuffle(out)
    return out


def subset_uniform_scenarios(pool, n_scen, k_min, k_max, rng):
    """Uniform over distinct subsets. Follows C(n,K), so mid-K dominates and K=0 appears
    at most once."""
    k_hi = min(k_max, len(pool))
    subsets = [
        c for K in range(k_min, k_hi + 1)
        for c in itertools.combinations(range(len(pool)), K)
    ] if len(pool) <= 20 else None

    if subsets is not None:
        if n_scen > len(subsets):
            raise ValueError(
                f"Asked for {n_scen} scenarios but the pool of {len(pool)} only admits "
                f"{len(subsets)} distinct subsets for K in [{k_min},{k_hi}]."
            )
        order = rng.permutation(len(subsets))[:n_scen]
        return [[pool[i] for i in subsets[j]] for j in order]

    seen, out = set(), []                               # pool too large to enumerate
    while len(out) < n_scen:
        K = int(rng.integers(k_min, k_hi + 1))
        idx = tuple(sorted(rng.choice(len(pool), size=K, replace=False))) if K else ()
        if idx in seen:
            continue
        seen.add(idx)
        out.append([pool[i] for i in idx])
    return out


# -----------------------------------------------------------------------------
# 3. Sensor coordinates
# -----------------------------------------------------------------------------
def building_mask(mesh_dir, n_cells, b, cell_size):
    """True where a cell centre falls inside a building footprint."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from utils.scene_objects import gather_bboxes

    obstacles = gather_bboxes(mesh_dir, footprints=False, use_cache=True)
    mask = np.zeros((n_cells, n_cells), dtype=bool)
    centres = -b + (np.arange(n_cells) + 0.5) * cell_size  # metres, cell centres

    for ob in obstacles:
        lo, hi = ob["min"], ob["max"]
        c0 = np.searchsorted(centres, lo[0], side="left")
        c1 = np.searchsorted(centres, hi[0], side="right")
        r0 = np.searchsorted(centres, lo[1], side="left")
        r1 = np.searchsorted(centres, hi[1], side="right")
        if c1 > c0 and r1 > r0:
            mask[r0:r1, c0:c1] = True
    return mask


def _relocate_once(sel, placeable_mask, n_cells, radius, rng):
    """One greedy claim-and-relocate pass, after DeepMTL's GenerateSensors.relocate_sensors.

    Walk the sensors in order: if a sensor falls in a neighbourhood already claimed by an
    earlier one it is flagged, otherwise it claims its own (2r+1)^2 block. Flagged sensors
    are then re-drawn from the cells no block covers.

    Two deviations from the original, both forced by confining sensors to streets:
      - best effort. DeepMTL does random.sample(available, len(need)), which raises when
        there are not enough free cells. Here any sensor that cannot be moved stays put,
        so the sensor count is always preserved exactly.
      - the caller caps the radius (see _spacing_radius); DeepMTL's own radii assume the
        full grid is available.
    """
    occupied = np.zeros((n_cells, n_cells), dtype=bool)
    keep, need = [], []
    rows, cols = np.divmod(sel, n_cells)
    for s, r, c in zip(sel, rows, cols):
        if occupied[r, c]:
            need.append(s)
            continue
        keep.append(s)
        occupied[max(0, r - radius):r + radius + 1,
                 max(0, c - radius):c + radius + 1] = True
    if not need:
        return np.asarray(keep, dtype=np.int64)

    avail = np.flatnonzero(placeable_mask & ~occupied.ravel())
    take = min(len(avail), len(need))
    moved = rng.choice(avail, size=take, replace=False) if take else np.empty(0, np.int64)
    return np.concatenate([np.asarray(keep, dtype=np.int64),
                           np.asarray(moved, dtype=np.int64),
                           np.asarray(need[take:], dtype=np.int64)])


def _spacing_radius(n_sensors, n_placeable):
    """Largest radius whose (2r+1)^2 blocks could tile the placeable cells, floored at 1.

    DeepMTL picks the radius from a fixed table (4 cells at 200 sensors down to 2 at 1000)
    because its sensors may sit anywhere in the grid. Ours are confined to streets - about
    31 % of the cells here - so the in-street density is ~3x the nominal figure and those
    radii cannot be met. Measured on this scene, the capped radius actually spaces sensors
    better than DeepMTL's: at 10 % density r=1 leaves 39 % of sensors within 2 cells of a
    neighbour, where their r=2 leaves 81 %, because over-claiming flags more sensors than
    there are free cells to move them to.
    """
    return max(1, int((math.sqrt(n_placeable / max(n_sensors, 1)) - 1) // 2))


def draw_sensors(placeable, placeable_mask, n_cells, n_sensors, seed, passes=5):
    """Sensor layout for one scenario: uniform draw, then `passes` relocation passes."""
    rng = np.random.default_rng(seed)
    sel = rng.choice(placeable, size=n_sensors, replace=False)
    radius = _spacing_radius(n_sensors, len(placeable))
    for _ in range(passes):
        sel = _relocate_once(sel, placeable_mask, n_cells, radius, rng)
    return np.sort(sel), radius


def assign_scenario_sensors(scenarios, placeable, densities, n_cells, rng):
    """Give every scenario its own density and its own sensor-draw seed.

    Sensors are drawn independently per scenario from the SAME pool of placeable cells
    for every split - there is no train/val/test partition of the grid. Partitioning
    would hand each split a different spatial sample of the city, biasing the sets;
    with an independent draw per scenario, no sensor layout is ever reused and the
    splits stay statistically identical in geometry.

    (Checked against DeepMTL's own source, github.com/caitaozhan/deeplearning-localization:
    it has no train/test sensor partition either. It goes further the other way and reuses
    one fixed layout per (grid, density, seed) for both sets, loaded from
    data/sensors/{grid_length}-{sensor_density}-{random_seed}. Drawing per scenario gives
    strictly more variety than that.)

    Storing a seed rather than the cell list keeps splits.json small: a 10 % draw is 9000
    cells, and 5000 scenarios would be 45 M integers. Preprocessing regenerates the exact
    set with:

        pool = np.array(splits["sensor_cells"])
        sel  = pool[np.random.default_rng(seed).choice(len(pool), n, replace=False)]
    """
    total_cells = n_cells * n_cells
    counts = {d: int(round(total_cells * d / 100.0)) for d in densities}
    biggest = max(counts.values())
    if biggest > len(placeable):
        raise ValueError(
            f"The highest density ({max(densities):g}%) needs {biggest} sensor cells but "
            f"only {len(placeable)} are placeable. Lower --densities, or pass "
            f"--no-street-only to place sensors over the full grid."
        )

    for split, scens in scenarios.items():
        for sc in scens:
            d = float(rng.choice(densities))
            sc["sensor_density_pct"] = d
            sc["num_sensors"] = counts[d]
            sc["sensor_seed"] = int(rng.integers(0, 2**31 - 1))
            sc["noise_seed"] = int(rng.integers(0, 2**31 - 1))
    return counts


def build_sensor_layouts(scenarios, placeable, n_cells, out_dir, passes=5):
    """Draw and persist each scenario's sensor cells.

    The layouts are written out rather than left to be regenerated downstream. The seed
    alone is no longer enough now that a relocation pass is involved - reproducing a
    layout would mean reimplementing the algorithm in the training repo. Storing them is
    cheap next to the RSS cubes.
    """
    placeable = np.asarray(placeable, dtype=np.int64)
    mask = np.zeros(n_cells * n_cells, dtype=bool)
    mask[placeable] = True
    os.makedirs(out_dir, exist_ok=True)

    for split, scens in scenarios.items():
        flat, offsets, radii = [], [0], []
        for sc in scens:
            sel, radius = draw_sensors(placeable, mask, n_cells,
                                       sc["num_sensors"], sc["sensor_seed"], passes)
            sc["spacing_radius_cells"] = int(radius)
            flat.append(sel.astype(np.int32))
            offsets.append(offsets[-1] + len(sel))
            radii.append(radius)
        path = os.path.join(out_dir, f"sensors_{split}.npz")
        np.savez_compressed(
            path,
            cells=np.concatenate(flat) if flat else np.empty(0, np.int32),
            offsets=np.asarray(offsets, dtype=np.int64),
            density_pct=np.asarray([sc["sensor_density_pct"] for sc in scens], np.float32),
            spacing_radius=np.asarray(radii, np.int16),
        )
        print(f"  {split}: {len(scens)} layouts -> {os.path.getsize(path)/1e6:.1f} MB "
              f"({path})")


def split_static_positions(pos, fracs, rng):
    """Partition the static positions into train/val/test at random.

    Deliberately NOT spatially separated. Two positions a few metres apart do produce
    similar single-jammer fields, but a detector sample is never a single field: it is a
    sum of K of them, at a density drawn per sample, read through a sensor layout drawn
    per sample, under noise drawn per sample. No configuration is ever repeated across
    splits, so there is nothing to memorise at the sample level.

    This also matches the DeepMTL baseline, which trains and tests over the same 1 km area
    with transmitters placed at random for both. The claim it supports is generalisation to
    unseen *configurations*, not to unseen city regions - a spatial split would be needed
    for the latter, at the cost of train and test covering different building geometry.
    """
    n = len(pos)
    order = rng.permutation(n)
    n_tr = int(round(fracs[0] * n))
    n_va = int(round(fracs[1] * n))
    return {"train": np.sort(order[:n_tr]),
            "val": np.sort(order[n_tr:n_tr + n_va]),
            "test": np.sort(order[n_tr + n_va:])}


def build_static_samples(pos_split, n_samples, k_min, k_max, densities, n_cells, rng):
    """Detector sample specs: stratified K, a density, and seeds. One dict per sample."""
    total_cells = n_cells * n_cells
    counts = {d: int(round(total_cells * d / 100.0)) for d in densities}
    out = {}
    for split, ids in pos_split.items():
        n = n_samples[split]
        ks = list(range(k_min, k_max + 1))
        base, extra = divmod(n, len(ks))
        want = {K: base + (1 if i < extra else 0) for i, K in enumerate(ks)}
        samples = []
        for K in ks:
            for _ in range(want[K]):
                sel = rng.choice(ids, size=min(K, len(ids)), replace=False) if K else []
                d = float(rng.choice(densities))
                samples.append({
                    "position_ids": [int(v) for v in sel],
                    "num_jammers": int(len(sel)),
                    "sensor_density_pct": d,
                    "num_sensors": counts[d],
                    "sensor_seed": int(rng.integers(0, 2**31 - 1)),
                    "noise_seed": int(rng.integers(0, 2**31 - 1)),
                })
        rng.shuffle(samples)
        out[split] = samples
    return out


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    n_cells = int(round(2 * args.map_bounds_b / args.cell_size))
    print(f"Grid: {n_cells} x {n_cells} cells @ {args.cell_size:g} m "
          f"(bounds +/-{args.map_bounds_b:g} m)")

    sizes = {"train": args.pool_train, "val": args.pool_val, "test": args.pool_test}
    pools, cells = build_pools(args.traj_dir, sizes, rng)
    print(f"\nTrajectory pools ({len(cells)} duration x velocity strata):")
    for s in ("train", "val", "test"):
        print(f"  {s:<5} {len(pools[s]):>3} trajectories")

    n_scen = {"train": args.n_train, "val": args.n_val, "test": args.n_test}
    gen = stratified_scenarios if args.k_balance == "stratified" else subset_uniform_scenarios
    scenarios = {}
    print(f"\nScenario sampling: {args.k_balance}")
    for split in ("train", "val", "test"):
        scenarios[split] = gen(pools[split], n_scen[split], args.min_jammers, args.max_jammers, rng)
        ks = [len(s) for s in scenarios[split]]
        uniq = len({tuple(s) for s in scenarios[split]})
        hist = {K: ks.count(K) for K in sorted(set(ks))}
        print(f"  {split}: {len(scenarios[split])} scenarios, {uniq} distinct subsets, "
              f"mean K {np.mean(ks):.2f}")
        print(f"    per-K: {hist}")

    # sensors: placeable cells -> disjoint per-split pools -> per-scenario draw
    if args.street_only:
        print(f"\nBuilding mask from {args.mesh_dir} ...")
        bmask = building_mask(args.mesh_dir, n_cells, args.map_bounds_b, args.cell_size)
        free = np.flatnonzero(~bmask.ravel())
        print(f"  {len(free)} of {n_cells*n_cells} cells are outside buildings "
              f"({100*len(free)/(n_cells*n_cells):.1f}%)")
    else:
        free = np.arange(n_cells * n_cells)
        print(f"\nSensors placed uniformly over all {len(free)} cells.")

    placeable = [int(i) for i in free]
    scen_docs = {s: [{"scenario_id": f"{s}_{i:05d}_k{len(m):02d}", "jammers": m}
                     for i, m in enumerate(scenarios[s])]
                 for s in scenarios}
    counts = assign_scenario_sensors(scen_docs, placeable, args.densities, n_cells, rng)

    print(f"Sensor cells available (shared by all splits): {len(placeable)} "
          f"({100*len(placeable)/(n_cells*n_cells):.1f}% of the grid)")
    print("Sensors per density:", ", ".join(f"{d:g}%={n}" for d, n in sorted(counts.items())))
    print(f"Drawing sensor layouts ({args.relocation_passes} relocation passes)...")
    build_sensor_layouts(scen_docs, placeable, n_cells,
                         os.path.join(os.path.dirname(os.path.abspath(args.out)), "sensors"),
                         args.relocation_passes)
    for s in scen_docs:
        ds = [sc["sensor_density_pct"] for sc in scen_docs[s]]
        print(f"  {s}: density drawn per scenario, "
              f"{ {d: ds.count(d) for d in sorted(set(ds))} }")

    doc = {
        "seed": args.seed,
        "grid": {"n_cells": n_cells, "cell_size_m": args.cell_size,
                 "map_bounds_b_m": args.map_bounds_b,
                 "origin_m": [-args.map_bounds_b, -args.map_bounds_b]},
        "k_range": [args.min_jammers, args.max_jammers],
        "k_balance": args.k_balance,
        "sensor_placement": "street_only" if args.street_only else "uniform",
        "sensor_densities_pct": sorted(args.densities),
        "sensor_cell_index": "flat row-major: cell = row * n_cells + col",
        "sensor_draw": ("per scenario, independent, no train/test partition: uniform draw "
                        "from sensor_cells then min-spacing relocation. Layouts are stored "
                        "in sensors/sensors_{split}.npz (cells + offsets)."),
        "relocation_passes": args.relocation_passes,
        "trajectory_pools": pools,
        "sensor_cells": placeable,
        "scenarios": scen_docs,
    }
    # ---- detector branch: only if the static library has been generated ----------
    pos_path = os.path.join(args.static_dir, "positions.npy")
    if os.path.exists(pos_path):
        pos = np.load(pos_path)
        print(f"\nStatic library: {len(pos)} positions from {pos_path}")
        assign = split_static_positions(pos, args.static_split_frac, rng)
        print("  position split (random): "
              + ", ".join(f"{k}={len(v)}" for k, v in assign.items()))
        n_samples = {"train": args.n_static_train, "val": args.n_static_val,
                     "test": args.n_static_test}
        samples = build_static_samples(assign, n_samples, args.min_jammers,
                                       args.max_jammers, args.densities, n_cells, rng)
        for k, v in samples.items():
            ks = [x["num_jammers"] for x in v]
            print(f"  {k}: {len(v)} samples, K {min(ks)}..{max(ks)}, "
                  f"{ks.count(0)} noise-only")
        doc["static"] = {
            "n_positions": int(len(pos)),
            "split_fractions": list(args.static_split_frac),
            "split_mode": "random",
            "position_split": {k: [int(i) for i in v] for k, v in assign.items()},
            "samples": samples,
        }
    else:
        print(f"\n[skip] no {pos_path} - detector samples not planned. "
              f"Run --action generate_static first, then re-run this script.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"\nWrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
