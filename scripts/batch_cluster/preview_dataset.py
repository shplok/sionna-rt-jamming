#!/usr/bin/env python3
"""
Contact sheets of the generated dataset, one panel per jammer count K.

Headless (Agg backend), so it runs on a compute node with no display. Panels use the
same style as the interactive GIF: viridis RSS, grey buildings, white ground-truth
markers, one shared colour scale per sheet.

    # both branches, K = 0..10, from the val split
    python scripts/batch_cluster/preview_dataset.py --dataset-dir ./datasets/batch_simulation_nyc

    # detector samples only, and also write a GIF per trajectory scenario
    python scripts/batch_cluster/preview_dataset.py --branch static
    python scripts/batch_cluster/preview_dataset.py --branch trajectory --gif
"""

import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from utils.plotter import (create_jammer_animation, plot_dataset_stats,
                           plot_density_ladder, plot_rss_3d, plot_rss_sheet)
from utils.scene_objects import gather_bboxes


def load_splits(dataset_dir):
    with open(os.path.join(dataset_dir, "splits.json")) as f:
        return json.load(f)


def street_mask_from_splits(splits):
    """(n, n) bool, True where a cell is street. sensor_cells is the placeable list."""
    n = int(splits["grid"]["n_cells"])
    mask = np.zeros(n * n, dtype=bool)
    mask[np.asarray(splits["sensor_cells"], dtype=np.int64)] = True
    return mask.reshape(n, n)


def static_sensor_cells(splits, sample_spec):
    """
    Reproduce a detector sample's sensor layout.

    Only the tracking scenarios get their layouts written to sensors/*.npz;
    static samples store num_sensors + sensor_seed and are redrawn on demand
    (make_splits.py calls build_sensor_layouts with the scenarios only).
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_splits import draw_sensors

    n = int(splits["grid"]["n_cells"])
    placeable = np.asarray(splits["sensor_cells"], dtype=np.int64)
    mask = np.zeros(n * n, dtype=bool)   # flat: _relocate_once ANDs it with occupied.ravel()
    mask[placeable] = True
    cells, _ = draw_sensors(placeable, mask, n,
                            int(sample_spec["num_sensors"]),
                            int(sample_spec["sensor_seed"]),
                            int(splits.get("relocation_passes", 5)))
    return cells


def build_extent(b):
    return [-b, b, -b, b]


def load_buildings(mesh_dir):
    if not mesh_dir:
        return []
    try:
        return gather_bboxes(mesh_dir, footprints=False, use_cache=True)
    except FileNotFoundError:
        print(f"  no mesh dir at {mesh_dir}, drawing without buildings")
        return []


def static_pick(split_dir, ks):
    """(K, sample index) for the first detector sample at each K."""
    meta = np.load(os.path.join(split_dir, "meta.npz"))
    num_jammers = meta["num_jammers"]

    picked = []
    for k in ks:
        hits = np.where(num_jammers == k)[0]
        if len(hits) == 0:
            print(f"  K={k}: no sample in this split, skipped")
            continue
        picked.append((k, int(hits[0])))
    return picked


def static_sample(split_dir, i):
    """RSS frame, ground-truth xy and sensor metadata for one detector sample."""
    meta = np.load(os.path.join(split_dir, "meta.npz"))
    labels = np.load(os.path.join(split_dir, "labels.npz"))
    rss = np.load(os.path.join(split_dir, "rss.npy"), mmap_mode="r")

    s, e = meta["position_offsets"][i], meta["position_offsets"][i + 1]
    return {
        "rss": np.asarray(rss[i]).astype(np.float32),
        "jammers": np.column_stack([labels["x"][s:e], labels["y"][s:e]]),
        "density": float(meta["sensor_density_pct"][i]),
        "num_sensors": int(meta["num_sensors"][i]),
    }


def static_panels(split_dir, picks):
    panels = []
    for k, i in picks:
        d = static_sample(split_dir, i)
        panels.append({
            "rss": d["rss"],
            "jammers": d["jammers"],
            "title": f"K={k}  sample {i}  ({d['density']:g}% sensors)",
        })
    return panels


def trajectory_scenarios(split_dir, ks):
    """One scenario directory per K. K is encoded in the directory name."""
    picked = []
    for k in ks:
        hits = sorted(glob.glob(os.path.join(split_dir, f"*_k{k:02d}")))
        if not hits:
            print(f"  K={k}: no scenario in this split, skipped")
            continue
        picked.append((k, hits[0]))
    return picked


def trajectory_panels(scenarios, frame_frac):
    panels = []
    for k, d in scenarios:
        cube = np.load(os.path.join(d, "rss_aggregated.npy"), mmap_mode="r")
        labels = np.load(os.path.join(d, "labels.npz"))

        t = min(int(frame_frac * cube.shape[0]), cube.shape[0] - 1)
        at_t = labels["frame"] == t
        panels.append({
            "rss": np.asarray(cube[t]).astype(np.float32),
            "jammers": np.column_stack([labels["x"][at_t], labels["y"][at_t]]),
            "title": f"K={k}  {os.path.basename(d)}  frame {t}/{cube.shape[0] - 1}",
        })
    return panels


def static_3d(split_dir, picks, splits, street, out_dir, split, vmin, vmax, elev, azim):
    """One PNG per K: street plan on the floor, this sample's sensor readings above it."""
    specs = splits["static"]["samples"][split]
    for k, i in picks:
        d = static_sample(split_dir, i)
        cells = static_sensor_cells(splits, specs[i])
        plot_rss_3d(
            d["rss"], cells, street, splits["grid"], jammers=d["jammers"],
            vmin=vmin, vmax=vmax, elev=elev, azim=azim,
            title=(f"Detector sample {i} - K={k} jammers, "
                   f"{d['density']:g}% sensor density ({len(cells)} sensors)"),
            filename=os.path.join(out_dir, f"static_{split}_k{k:02d}_3d.png"),
        )


def trajectory_3d(scenarios, splits, street, dataset_dir, out_dir, split, frame_frac,
                  vmin, vmax, elev, azim):
    """One PNG per K, using the scenario's own stored sensor layout."""
    sens = np.load(os.path.join(dataset_dir, "sensors", f"sensors_{split}.npz"))
    cells_all, offsets, dens = sens["cells"], sens["offsets"], sens["density_pct"]

    for k, d in scenarios:
        cube = np.load(os.path.join(d, "rss_aggregated.npy"), mmap_mode="r")
        labels = np.load(os.path.join(d, "labels.npz"))
        name = os.path.basename(d)

        # scenario_id is <split>_<index>_k<K>, and sensors/*.npz is in that index order
        idx = int(name.split("_")[1])
        cells = cells_all[offsets[idx]:offsets[idx + 1]]

        t = min(int(frame_frac * cube.shape[0]), cube.shape[0] - 1)
        at_t = labels["frame"] == t
        plot_rss_3d(
            np.asarray(cube[t]).astype(np.float32), cells, street, splits["grid"],
            jammers=np.column_stack([labels["x"][at_t], labels["y"][at_t]]),
            vmin=vmin, vmax=vmax, elev=elev, azim=azim,
            title=(f"{name} - K={k} jammers, frame {t}/{cube.shape[0] - 1}, "
                   f"{dens[idx]:g}% sensor density ({len(cells)} sensors)"),
            filename=os.path.join(out_dir, f"trajectory_{split}_k{k:02d}_3d.png"),
        )


def trajectory_gifs(scenarios, buildings, out_dir, b, vmin, vmax, fps):
    for k, d in scenarios:
        cube = np.load(os.path.join(d, "rss_aggregated.npy"), mmap_mode="r")
        labels = np.load(os.path.join(d, "labels.npz"))

        paths = {}
        for j in np.unique(labels["jammer_idx"]):
            m = labels["jammer_idx"] == j
            order = np.argsort(labels["frame"][m])
            paths[f"jammer{int(j)}"] = np.column_stack(
                [labels["x"][m][order], labels["y"][m][order]]
            )

        name = os.path.basename(d)
        create_jammer_animation(
            rss_list=[np.asarray(cube[t]).astype(np.float32) for t in range(cube.shape[0])],
            paths_dict=paths,
            buildings=buildings,
            map_size=(2 * b, 2 * b),
            map_center=(0.0, 0.0),
            vmin=vmin,
            vmax=vmax,
            filename=os.path.join(out_dir, f"{name}.gif"),
            fps=fps,
            cbar_label="RSS (dBW)",
        )


def density_ladder(split_dir, splits, street, out_dir, split, k, vmin, vmax):
    """One jammer configuration rendered at every rung of the density ladder.

    The ladder is the dataset's single most consequential design choice, and it is
    invisible in the per-K contact sheets because each sample there carries whatever
    density it happened to be assigned. Here the configuration is held fixed and only
    the density varies, which is the comparison the ladder rationale actually makes.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_splits import draw_sensors

    meta = np.load(os.path.join(split_dir, "meta.npz"))
    hits = np.where(meta["num_jammers"] == k)[0]
    if len(hits) == 0:
        print(f"  density ladder: no K={k} sample in {split}, skipped")
        return
    i = int(hits[0])
    d = static_sample(split_dir, i)

    n = int(splits["grid"]["n_cells"])
    placeable = np.asarray(splits["sensor_cells"], dtype=np.int64)
    mask = np.zeros(n * n, dtype=bool)
    mask[placeable] = True
    seed = int(splits["static"]["samples"][split][i]["sensor_seed"])
    passes = int(splits.get("relocation_passes", 5))

    # Densest first, so the eye reads the ladder as losing information left to right.
    layouts = []
    for dens in sorted(splits["sensor_densities_pct"], reverse=True):
        n_sens = int(round(n * n * dens / 100.0))
        cells, _ = draw_sensors(placeable, mask, n, n_sens, seed, passes)
        layouts.append((float(dens), cells))

    plot_density_ladder(
        d["rss"], layouts, street, splits["grid"], jammers=d["jammers"],
        vmin=vmin, vmax=vmax,
        suptitle=(f"Sensor-density ladder - detector sample {i} "
                  f"({split} split, K={k} jammers, identical positions in every panel)"),
        filename=os.path.join(out_dir, f"density_ladder_{split}_k{k:02d}.png"),
    )


def dataset_stats(split_dir, splits, out_dir, split):
    """K balance, density balance, and sensors per receptive field."""
    meta = np.load(os.path.join(split_dir, "meta.npz"))
    kj, dp, ns = meta["num_jammers"], meta["sensor_density_pct"], meta["num_sensors"]

    uk, ck = np.unique(kj, return_counts=True)
    ud, cd = np.unique(dp, return_counts=True)
    k_hist = {int(a): int(b) for a, b in zip(uk, ck)}
    dens_counts = {float(a): int(b) for a, b in zip(ud, cd)}
    n_by_d = {}
    for a in ud:
        vals = np.unique(ns[dp == a])
        if len(vals) != 1:
            print(f"  WARNING: density {a}% has non-unique num_sensors {vals}")
        n_by_d[float(a)] = int(vals[0])

    plot_dataset_stats(
        k_hist, dens_counts, n_by_d,
        suptitle=(f"Detector library balance - multi_static_jammers/{split}, "
                  f"{len(kj):,} samples"),
        filename=os.path.join(out_dir, f"stats_{split}.png"),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="./datasets/batch_simulation_nyc",
                   help="Dataset root (default: %(default)s)")
    p.add_argument("--branch", choices=["static", "trajectory", "both"], default="both",
                   help="Which branch to preview (default: %(default)s)")
    p.add_argument("--split", default="val", choices=["train", "val", "test"],
                   help="Split to sample from (default: %(default)s)")
    p.add_argument("--out-dir", default=None,
                   help="Where to write (default: <dataset-dir>/previews)")
    p.add_argument("--mesh-dir", default="./data/NYC3KM_585751_4512036/mesh",
                   help="PLY meshes for building overlay; pass '' to skip")
    p.add_argument("--max-k", type=int, default=10,
                   help="Preview K=0..MAX_K (default: %(default)s)")
    p.add_argument("--map-bounds-b", type=float, default=1500.0,
                   help="Half-width of the map in metres (default: %(default)s)")
    p.add_argument("--vmin", type=float, default=-145.0)
    p.add_argument("--vmax", type=float, default=0.0)
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--frame-frac", type=float, default=0.5,
                   help="Which frame of each scenario to show, as a fraction (default: %(default)s)")
    p.add_argument("--gif", action="store_true",
                   help="Also write one GIF per previewed trajectory scenario")
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--3d", dest="three_d", action="store_true",
                   help="Also write one 3D PNG per K: street plan plus sensor readings")
    p.add_argument("--sheet", dest="sheet", action="store_true", default=None,
                   help="Write the contact sheets (default unless --3d is given alone)")
    p.add_argument("--density-ladder", dest="density_ladder", action="store_true",
                   help="One configuration at every density rung (static branch only)")
    p.add_argument("--ladder-k", type=int, default=5,
                   help="Which K to use for the density ladder (default: %(default)s)")
    p.add_argument("--stats", action="store_true",
                   help="K/density balance and sensors-per-receptive-field summary")
    p.add_argument("--elev", type=float, default=34.0, help="3D elevation angle")
    p.add_argument("--azim", type=float, default=-120.0, help="3D azimuth angle")
    args = p.parse_args()

    # Any explicit view selection means "just that view"; --sheet re-adds the sheets.
    explicit = args.three_d or args.density_ladder or args.stats
    want_sheet = args.sheet if args.sheet is not None else not explicit

    out_dir = args.out_dir or os.path.join(args.dataset_dir, "previews")
    os.makedirs(out_dir, exist_ok=True)

    extent = build_extent(args.map_bounds_b)
    buildings = load_buildings(args.mesh_dir) if want_sheet else []
    ks = list(range(args.max_k + 1))

    splits = load_splits(args.dataset_dir)
    street = (street_mask_from_splits(splits)
              if (args.three_d or args.density_ladder) else None)

    if args.branch in ("static", "both"):
        split_dir = os.path.join(args.dataset_dir, "multi_static_jammers", args.split)
        print(f"static: {split_dir}")
        picks = static_pick(split_dir, ks)
        if want_sheet:
            plot_rss_sheet(
                static_panels(split_dir, picks), extent, buildings=buildings,
                vmin=args.vmin, vmax=args.vmax, ncols=args.ncols,
                suptitle=f"Detector samples (multi_static_jammers/{args.split}) - one per K",
                filename=os.path.join(out_dir, f"static_{args.split}_by_k.png"),
            )
        if args.three_d:
            static_3d(split_dir, picks, splits, street, out_dir, args.split,
                      args.vmin, args.vmax, args.elev, args.azim)
        if args.density_ladder:
            density_ladder(split_dir, splits, street, out_dir, args.split,
                           args.ladder_k, args.vmin, args.vmax)
        if args.stats:
            dataset_stats(split_dir, splits, out_dir, args.split)

    if args.branch in ("trajectory", "both"):
        split_dir = os.path.join(args.dataset_dir, "multi_trajectory_jammers", args.split)
        print(f"trajectory: {split_dir}")
        scenarios = trajectory_scenarios(split_dir, ks)
        if want_sheet:
            plot_rss_sheet(
                trajectory_panels(scenarios, args.frame_frac), extent, buildings=buildings,
                vmin=args.vmin, vmax=args.vmax, ncols=args.ncols,
                suptitle=f"Tracking scenarios (multi_trajectory_jammers/{args.split}) - one per K",
                filename=os.path.join(out_dir, f"trajectory_{args.split}_by_k.png"),
            )
        if args.three_d:
            trajectory_3d(scenarios, splits, street, args.dataset_dir, out_dir,
                          args.split, args.frame_frac, args.vmin, args.vmax,
                          args.elev, args.azim)
        if args.gif:
            trajectory_gifs(scenarios, buildings, out_dir, args.map_bounds_b,
                            args.vmin, args.vmax, args.fps)

    print(f"\nPreviews written to {out_dir}")


if __name__ == "__main__":
    main()
