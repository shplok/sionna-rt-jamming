#!/usr/bin/env python3
"""
Build per-frame jammer labels for every scenario in splits.json.

Labels are stored in **continuous metres** (float64, straight from the trajectory
files) with the grid cell and sub-cell offset carried alongside as derived fields.
Quantising to a cell index would throw away up to half a cell of precision, which is
exactly the precision DeepMTL's second stage is trying to recover.

Runs on CPU in seconds, needs only splits.json and the traj_*.npy files - not the
radio maps. So labels can be built before, or independently of, aggregation.

Written twice, on purpose:

    labels/labels_{split}.npz              one file per split, flat arrays. This is what
                                           a DataLoader should read - 5000 tiny files
                                           would be far slower than three.
    multi_trajectory_jammers/{split}/{id}/labels.npz
                                           the same rows for that one scenario, so a
                                           scenario directory is self-contained and can
                                           be inspected or copied on its own.

The per-scenario copies total ~10 MB, so keeping both costs nothing.

Usage:
    python scripts/batch_cluster/make_labels.py --dataset-dir ./datasets/batch_simulation_nyc
"""

import argparse
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="./datasets/batch_simulation_nyc",
                   help="Dataset root; --splits, --traj-dir and --out-dir derive from it.")
    p.add_argument("--splits", default=None, help="Default: <dataset-dir>/splits.json")
    p.add_argument("--traj-dir", default=None,
                   help="Default: <dataset-dir>/single_trajectory_jammers")
    p.add_argument("--out-dir", default=None, help="Default: <dataset-dir>/labels")
    p.add_argument("--scenarios-dir", default=None,
                   help="Default: <dataset-dir>/multi_trajectory_jammers")
    p.add_argument("--no-per-scenario", dest="per_scenario", action="store_false", default=True,
                   help="Skip writing a labels.npz inside each scenario directory.")
    p.add_argument("--splits-to-build", nargs="+", default=["train", "val", "test"])
    a = p.parse_args()
    root = os.path.normpath(a.dataset_dir)
    if a.splits is None:
        a.splits = os.path.join(root, "splits.json")
    if a.traj_dir is None:
        a.traj_dir = os.path.join(root, "single_trajectory_jammers")
    if a.out_dir is None:
        a.out_dir = os.path.join(root, "labels")
    if a.scenarios_dir is None:
        a.scenarios_dir = os.path.join(root, "multi_trajectory_jammers")
    return a


def load_trajectories(traj_dir, ids):
    """traj_id -> (positions (T,3) float64, metadata dict)."""
    out = {}
    for tid in sorted(ids):
        npy = os.path.join(traj_dir, f"{tid}.npy")
        jsn = os.path.join(traj_dir, f"{tid}.json")
        if not os.path.exists(npy):
            raise FileNotFoundError(f"Missing trajectory: {npy}")
        meta = {}
        if os.path.exists(jsn):
            with open(jsn, "r", encoding="utf-8") as f:
                meta = json.load(f)
        out[tid] = (np.load(npy), meta)
    return out


def build_split(scenarios, trajs, grid):
    """Flat per-(scenario, frame, jammer) label rows."""
    n_cells = grid["n_cells"]
    cell = grid["cell_size_m"]
    x0, y0 = grid["origin_m"]

    cols = {k: [] for k in ("scenario_idx", "frame", "jammer_idx", "x", "y", "z",
                            "col", "row", "dx", "dy", "velocity_mps",
                            "heading_deg", "is_padded")}
    index = []

    for s_idx, scen in enumerate(scenarios):
        jids = scen["jammers"]
        K = len(jids)
        # scenario length = max over its jammers; K=0 scenarios still need a length,
        # so fall back to the longest trajectory in the pool
        if K:
            n_frames = max(len(trajs[t][0]) for t in jids)
        else:
            n_frames = max(len(p) for p, _ in trajs.values())

        index.append({"scenario_id": scen["scenario_id"], "num_jammers": K,
                      "total_steps": int(n_frames), "jammers": jids})

        for j_idx, tid in enumerate(jids):
            pos, meta = trajs[tid]
            T = len(pos)
            for t in range(n_frames):
                # padded frames hold the final position - the jammer is still
                # transmitting there, so it stays a positive label
                tc = min(t, T - 1)
                x, y, z = pos[tc]
                cf = (x - x0) / cell
                rf = (y - y0) / cell
                c = int(np.floor(cf))
                r = int(np.floor(rf))
                cols["scenario_idx"].append(s_idx)
                cols["frame"].append(t)
                cols["jammer_idx"].append(j_idx)
                cols["x"].append(x)
                cols["y"].append(y)
                cols["z"].append(z)
                cols["col"].append(c)
                cols["row"].append(r)
                cols["dx"].append((cf - c) * cell)
                cols["dy"].append((rf - r) * cell)
                cols["velocity_mps"].append(float(meta.get("velocity_mps", np.nan)))
                cols["heading_deg"].append(float(meta.get("heading_degrees", np.nan)))
                cols["is_padded"].append(t >= T)

    dtypes = {"scenario_idx": np.int32, "frame": np.int16, "jammer_idx": np.int8,
              "col": np.int16, "row": np.int16, "is_padded": bool}
    arrays = {k: np.asarray(v, dtype=dtypes.get(k, np.float64)) for k, v in cols.items()}

    # every label must land inside the grid
    if len(arrays["col"]):
        oob = ((arrays["col"] < 0) | (arrays["col"] >= n_cells) |
               (arrays["row"] < 0) | (arrays["row"] >= n_cells))
        if oob.any():
            raise ValueError(f"{int(oob.sum())} labels fall outside the {n_cells}x{n_cells} grid")
    return arrays, index


def main():
    args = parse_args()
    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    grid = splits["grid"]
    print(f"Grid: {grid['n_cells']}x{grid['n_cells']} @ {grid['cell_size_m']:g} m, "
          f"origin {grid['origin_m']}")
    os.makedirs(args.out_dir, exist_ok=True)

    for split in args.splits_to_build:
        scenarios = splits["scenarios"][split]
        used = {t for s in scenarios for t in s["jammers"]}
        trajs = load_trajectories(args.traj_dir, used | set(splits["trajectory_pools"][split]))

        arrays, index = build_split(scenarios, trajs, grid)
        npz = os.path.join(args.out_dir, f"labels_{split}.npz")
        np.savez_compressed(npz, **arrays)
        with open(os.path.join(args.out_dir, f"labels_{split}.json"), "w", encoding="utf-8") as f:
            json.dump({"split": split, "grid": grid, "num_scenarios": len(scenarios),
                       "num_rows": int(len(arrays["frame"])), "scenarios": index}, f, indent=2)

        # per-scenario copies, so each scenario dir stands alone
        n_written = 0
        if args.per_scenario:
            sidx = arrays["scenario_idx"]
            for s_i, meta in enumerate(index):
                sel = sidx == s_i
                sc_dir = os.path.join(args.scenarios_dir, split, meta["scenario_id"])
                os.makedirs(sc_dir, exist_ok=True)
                np.savez_compressed(os.path.join(sc_dir, "labels.npz"),
                                    **{k: v[sel] for k, v in arrays.items()})
                n_written += 1

        frames = sum(s["total_steps"] for s in index)
        print(f"  {split:<5} {len(scenarios):>5} scenarios, {frames:>7} frames, "
              f"{len(arrays['frame']):>8} label rows -> {os.path.getsize(npz)/1e6:.1f} MB")
        if len(arrays["dx"]):
            print(f"        sub-cell offset: dx in [{arrays['dx'].min():.3f}, "
                  f"{arrays['dx'].max():.3f}] m, padded rows {int(arrays['is_padded'].sum())}")
        if args.per_scenario:
            print(f"        + {n_written} per-scenario labels.npz")

    print(f"\nWrote labels to {args.out_dir}")


if __name__ == "__main__":
    main()
