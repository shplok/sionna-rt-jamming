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

from utils.plotter import create_jammer_animation, plot_rss_sheet
from utils.scene_objects import gather_bboxes


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


def static_panels(split_dir, ks):
    """One detector sample per K, chosen as the first sample with that K."""
    meta = np.load(os.path.join(split_dir, "meta.npz"))
    labels = np.load(os.path.join(split_dir, "labels.npz"))
    rss = np.load(os.path.join(split_dir, "rss.npy"), mmap_mode="r")

    num_jammers = meta["num_jammers"]
    offsets = meta["position_offsets"]
    density = meta["sensor_density_pct"]

    panels = []
    for k in ks:
        hits = np.where(num_jammers == k)[0]
        if len(hits) == 0:
            print(f"  K={k}: no sample in this split, skipped")
            continue
        i = int(hits[0])
        s, e = offsets[i], offsets[i + 1]
        panels.append({
            "rss": np.asarray(rss[i]).astype(np.float32),
            "jammers": np.column_stack([labels["x"][s:e], labels["y"][s:e]]),
            "title": f"K={k}  sample {i}  ({density[i]:.0f}% sensors)",
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
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.dataset_dir, "previews")
    os.makedirs(out_dir, exist_ok=True)

    extent = build_extent(args.map_bounds_b)
    buildings = load_buildings(args.mesh_dir)
    ks = list(range(args.max_k + 1))

    if args.branch in ("static", "both"):
        split_dir = os.path.join(args.dataset_dir, "multi_static_jammers", args.split)
        print(f"static: {split_dir}")
        panels = static_panels(split_dir, ks)
        plot_rss_sheet(
            panels, extent, buildings=buildings, vmin=args.vmin, vmax=args.vmax,
            ncols=args.ncols,
            suptitle=f"Detector samples (multi_static_jammers/{args.split}) - one per K",
            filename=os.path.join(out_dir, f"static_{args.split}_by_k.png"),
        )

    if args.branch in ("trajectory", "both"):
        split_dir = os.path.join(args.dataset_dir, "multi_trajectory_jammers", args.split)
        print(f"trajectory: {split_dir}")
        scenarios = trajectory_scenarios(split_dir, ks)
        panels = trajectory_panels(scenarios, args.frame_frac)
        plot_rss_sheet(
            panels, extent, buildings=buildings, vmin=args.vmin, vmax=args.vmax,
            ncols=args.ncols,
            suptitle=f"Tracking scenarios (multi_trajectory_jammers/{args.split}) - one per K",
            filename=os.path.join(out_dir, f"trajectory_{args.split}_by_k.png"),
        )
        if args.gif:
            trajectory_gifs(scenarios, buildings, out_dir, args.map_bounds_b,
                            args.vmin, args.vmax, args.fps)

    print(f"\nPreviews written to {out_dir}")


if __name__ == "__main__":
    main()
