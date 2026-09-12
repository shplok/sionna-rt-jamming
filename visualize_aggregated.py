#!/usr/bin/env python3
"""
Visualize Aggregated Radio Map Sequence as an Animated GIF (jammer_animation.gif).

Uses the exact same animation style as main_no_interactive.py / utils.plotter.create_jammer_animation.

Usage:
    # 1. Generate jammer_animation.gif for combo 0:
    python visualize_aggregated.py --combo 0

    # 2. Generate jammer_animation.gif for a specific combination name:
    python visualize_aggregated.py --combo combo_0001_k05

    # 3. Custom output filename or framerate:
    python visualize_aggregated.py --combo 2 --output ./my_combo2.gif --fps 10

    # 4. List all available combinations:
    python visualize_aggregated.py --list
"""

import os
import sys
import glob
import json
import argparse
import numpy as np

# Ensure headless execution
import matplotlib
matplotlib.use("Agg")

from core.engine import MotionEngine
from utils.scene_objects import gather_bboxes
from utils.plotter import create_jammer_animation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate jammer_animation.gif for a selected multi-jammer combination."
    )
    parser.add_argument(
        "--combo",
        type=str,
        default="0",
        help="Combination identifier: index (e.g. 0, 1), name (e.g. combo_0001_k05), or full directory path. Default: 0.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./datasets/simulation_results_nyc",
        help="Folder containing combination subdirectories (default: ./datasets/simulation_results_nyc).",
    )
    parser.add_argument(
        "--single-jammers-dir",
        type=str,
        default="./datasets/nyc_single_jammers",
        help="Folder containing original single jammer trajectory .npy files (default: ./datasets/nyc_single_jammers).",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="./data/NYC3KM_585751_4512036/mesh",
        help="Path to building mesh directory for footprints overlay.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output GIF path. Defaults to <combo_dir>/jammer_animation.gif.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second for the animation (default: 5).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=-145.0,
        help="Minimum RSS in dBW for colorbar scale (default: -145.0, just below noise floor).",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=0.0,
        help="Maximum RSS in dBW for colorbar scale (default: 0.0 dBW).",
    )
    parser.add_argument(
        "--map-bounds-b",
        type=float,
        default=1500.0,
        help="Half-width of coverage area [-b, b] in meters (default: 1500.0m for 3KM scene).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available combinations and exit.",
    )
    return parser.parse_args()


def find_combo_dir(results_dir, combo_arg):
    if os.path.isdir(combo_arg):
        return os.path.abspath(combo_arg)

    direct = os.path.join(results_dir, combo_arg)
    if os.path.isdir(direct):
        return direct

    all_combos = sorted(glob.glob(os.path.join(results_dir, "combo_*")))
    if not all_combos:
        raise FileNotFoundError(f"No combinations found in {results_dir}")

    try:
        idx = int(combo_arg)
        if 0 <= idx < len(all_combos):
            return all_combos[idx]
    except ValueError:
        pass

    for cdir in all_combos:
        if combo_arg in os.path.basename(cdir):
            return cdir

    raise FileNotFoundError(
        f"Could not find combination '{combo_arg}' in {results_dir}. Available: {[os.path.basename(d) for d in all_combos]}."
    )


def main():
    args = parse_args()

    # Mode: list
    if args.list:
        all_combos = sorted(glob.glob(os.path.join(args.results_dir, "combo_*")))
        print(f"\nAvailable combinations in {args.results_dir} ({len(all_combos)} total):")
        for i, cdir in enumerate(all_combos):
            sfile = os.path.join(cdir, "combination_summary.json")
            if os.path.exists(sfile):
                with open(sfile, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                k = meta.get("num_jammers", "?")
                steps = meta.get("total_steps", "?")
                min_dbw = meta.get("min_dbw", "?")
                max_dbw = meta.get("max_dbw", "?")
                print(f"  [{i:02d}] {os.path.basename(cdir)} | K={k} jammers | {steps}s | RSS: [{min_dbw:.1f}, {max_dbw:.1f}] dBW")
            else:
                print(f"  [{i:02d}] {os.path.basename(cdir)}")
        return

    # Find combination directory
    combo_dir = find_combo_dir(args.results_dir, args.combo)
    combo_id = os.path.basename(combo_dir)
    print("=" * 65)
    print(f"Rendering jammer_animation.gif for: {combo_id}")
    print(f"Directory: {combo_dir}")
    print("=" * 65)

    # Load aggregated RSS
    rss_path = os.path.join(combo_dir, "rss_aggregated.npy")
    if not os.path.exists(rss_path):
        raise FileNotFoundError(f"File not found: {rss_path}")

    rss_data = np.load(rss_path).astype(np.float32)  # (T, H, W)
    total_steps, H, W = rss_data.shape
    print(f"Radio map shape: {rss_data.shape} ({total_steps} seconds, {H}x{W} grid)")

    # Load summary metadata
    summary_path = os.path.join(combo_dir, "combination_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"File not found: {summary_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    # Load obstacles footprints
    obstacles = gather_bboxes(args.mesh_dir, footprints=True, use_cache=True)

    # Map bounds setup matching main_no_interactive.py
    b = args.map_bounds_b
    map_bounds = {"x": [-b, b], "y": [-b, b], "z": [1.5, 1.5]}
    map_center = [0.0, 0.0, 1.5]
    map_size = (2 * b, 2 * b)

    # Reconstruct MotionEngine trajectories for jammer markers
    engine = MotionEngine(scene=None, obstacles=obstacles, bounds=map_bounds)
    for j_idx, jmeta in enumerate(summary.get("jammers", [])):
        tid = jmeta["traj_id"]
        tpath = os.path.join(args.single_jammers_dir, f"{tid}.npy")
        if not os.path.exists(tpath):
            raise FileNotFoundError(f"Trajectory file not found: {tpath}")
        tarr = np.load(tpath)
        jname = f"Jammer_{j_idx + 1}"
        engine._jammer_paths[jname] = tarr
        engine._padding_preferences[jname] = "pad_end"
    engine.finalize_trajectories()

    # Output path
    output_path = args.output
    if not output_path:
        output_path = os.path.join(combo_dir, "jammer_animation.gif")

    # Render animation using the exact function from utils.plotter (same as main_no_interactive.py)
    vmin = args.vmin
    vmax = args.vmax

    create_jammer_animation(
        rss_list=rss_data,
        engine=engine,
        buildings=obstacles,
        map_size=map_size,
        map_center=map_center,
        filename=output_path,
        fps=args.fps,
        vmin=vmin,
        vmax=vmax,
    )
    print(f"\nSuccessfully generated animation: {output_path}")


if __name__ == "__main__":
    main()
