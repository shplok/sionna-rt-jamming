#!/usr/bin/env python3
"""
Sionna RT Jamming - batch dataset generation. CLUSTER ONLY.

Headless, no GUI. Intended to run on the GPU cluster via run_pipeline.sh: the
simulate_bases stage needs CUDA Sionna RT, and the full dataset is ~46 GB, which is
why this is not meant for a laptop. For interactive path planning on your own
machine use main_interactive_local.py instead.

Two branches, both writing under --dataset-dir (./datasets/batch_simulation_<tag>).

TRACKING branch - jammers move along trajectories:
    generate         54 collision-free straight trajectories          [CPU]
    plot             2D trajectory map, headless matplotlib           [CPU]
    simulate_bases   ray trace one map per trajectory frame           [GPU]
    aggregate        combine into the scenarios in splits.json        [CPU]
                     -> single_trajectory_jammers/, multi_trajectory_jammers/

DETECTION branch - jammers are static, one snapshot per sample:
    generate_static  sample N continuous positions on street cells    [CPU]
    simulate_static  ray trace one map per position                   [GPU]
    aggregate_static combine into the samples in splits.json          [CPU]
                     -> single_static_jammers/, multi_static_jammers/

Only the two simulate_* stages need a GPU. Everything else is NumPy and runs on a
CPU node. Run scripts/make_splits.py after both simulate_* stages and before either
aggregate stage. See README.md for the full sequence.

Usage:
    python main_batch_cluster.py --action all --dataset-dir ./datasets/batch_simulation_nyc
"""

import os
import sys
import json
import math
import time
import glob
import argparse
import numpy as np

# Configure Mitsuba variant before importing Sionna RT
def setup_mitsuba_variant(preferred="cuda_ad_mono_polarized"):
    import mitsuba as mi
    variant = os.environ.get("MITSUBA_VARIANT", preferred)
    try:
        mi.set_variant(variant)
        print(f"[Mitsuba] Active variant: {mi.variant()}")
    except Exception as e:
        print(f"[Mitsuba] Warning: Could not set variant '{variant}' ({e}). Falling back to 'llvm_ad_mono_polarized'.")
        mi.set_variant("llvm_ad_mono_polarized")

# -----------------------------------------------------------------------------
# CLI Arguments
# -----------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified non-interactive pipeline for trajectory generation and Sionna RT simulation."
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["all", "generate", "plot", "simulate_bases", "aggregate",
                 "generate_static", "simulate_static", "aggregate_static"],
        default="all",
        help="Pipeline stage. TRACKING branch: generate -> plot -> simulate_bases [GPU] "
             "-> aggregate. DETECTION branch: generate_static -> simulate_static [GPU] "
             "-> aggregate_static. Only the two simulate_* stages need a GPU; everything "
             "else is CPU. 'all' runs the tracking branch only - run the static stages "
             "explicitly.",
    )
    # Scene and Paths
    parser.add_argument(
        "--scene-path",
        type=str,
        default="./data/NYC3KM_585751_4512036/simple_OSM_scene.xml",
        help="Path to Mitsuba XML scene file.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="./data/NYC3KM_585751_4512036/mesh",
        help="Path to building mesh directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./datasets/batch_simulation_nyc",
        help="Dataset root. By convention ./datasets/batch_simulation_<tag>. Every other "
             "path derives from it: single_trajectory_jammers/, single_static_jammers/, "
             "multi_trajectory_jammers/, multi_static_jammers/, splits.json, labels/, "
             "sensors/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override for the trajectory base library "
             "(default: <dataset-dir>/single_trajectory_jammers).",
    )
    parser.add_argument(
        "--sim-output-dir",
        type=str,
        default=None,
        help="Override for the tracking scenarios "
             "(default: <dataset-dir>/multi_trajectory_jammers).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Override for splits.json (default: <dataset-dir>/splits.json).",
    )
    parser.add_argument(
        "--static-dir",
        type=str,
        default=None,
        help="Override for the static base library "
             "(default: <dataset-dir>/single_static_jammers).",
    )
    parser.add_argument(
        "--multi-static-dir",
        type=str,
        default=None,
        help="Override for the detector samples "
             "(default: <dataset-dir>/multi_static_jammers).",
    )
    parser.add_argument(
        "--n-static",
        type=int,
        default=20000,
        help="Static jammer positions to ray trace (default: 20000 ~ 78 min GPU, 7.2 GB). "
             "Positions are continuously jittered inside street cells - never at cell "
             "centres, or the sub-cell regression target would be degenerate.",
    )
    # Multi-Jammer Combination Parameters
    # Trajectory Generation Parameters
    parser.add_argument(
        "--durations",
        type=float,
        nargs="+",
        default=[30.0, 60.0, 90.0],
        help="List of trajectory durations in seconds (default: 30 60 90).",
    )
    parser.add_argument(
        "--velocities",
        type=float,
        nargs="+",
        default=[3.0, 9.0, 15.0],
        help="List of velocities in m/s (default: 3 9 15).",
    )
    parser.add_argument(
        "--count-per-combo",
        type=int,
        default=6,
        help="Trajectories per combination (default: 6 -> 54 total).",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=1.0,
        help="Sampling step in seconds (default: 1.0 = 1 fps).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["straight", "curved"],
        default="straight",
        help="Trajectory mode: 'straight' (default) or 'curved'.",
    )
    parser.add_argument(
        "--max-overlap-ratio",
        type=float,
        default=0.75,
        help="Max allowed mutual overlap between any two trajectories (default: 0.75 = 75%%).",
    )
    parser.add_argument(
        "--proximity-threshold",
        type=float,
        default=20.0,
        help="Distance in meters to consider two segments in the same corridor (default: 20.0m).",
    )
    parser.add_argument(
        "--z-height",
        type=float,
        default=1.5,
        help="Jammer altitude in meters (default: 1.5m).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across trajectory generation and combinations (default: 42).",
    )
    # Simulation Parameters
    parser.add_argument(
        "--freq-hz",
        type=float,
        default=1.57542e9,
        help="Carrier frequency (default GPS L1: 1.57542 GHz).",
    )
    parser.add_argument(
        "--power-dbw",
        type=float,
        default=10.0,
        help="Transmitter power in dBW.",
    )
    parser.add_argument(
        "--map-bounds-b",
        type=float,
        default=1500.0,
        help="Half-width of simulation coverage area [-b, b] (default: 1500m for 3KM x 3KM scene).",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        nargs=2,
        default=[10.0, 10.0],
        help="Radio map grid resolution in meters (default: 10 10 -> 300x300 over a 3 km "
             "scene, matching DeepMTL's cell size).",
    )
    parser.add_argument(
        "--samples-per-tx",
        type=int,
        default=10**7,
        help="Ray samples per transmitter (default: 10^7).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=80,
        help="Max ray reflection bounces (default: 80).",
    )
    parser.add_argument(
        "--meas-noise-var",
        type=float,
        default=1.0,
        help="Variance of additive Gaussian measurement noise in dB, applied to the "
             "aggregated RSS (default: 1.0 -> sigma = 1 dB). With Ptx = 10 dBW, "
             "INR = 10*log10(10 / meas_noise_var); the sweep 10, 10/sqrt(10), 1, 0.1 "
             "gives INR = 0, 5, 10, 20 dB. Use 0 to disable.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64", "fp16", "fp32", "fp64"],
        help="Floating-point precision for saving aggregated radio maps (default: 'float16'; options: float16, float32, float64).",
    )
    args = parser.parse_args()
    return resolve_dataset_paths(args)


def resolve_dataset_paths(args):
    """Derive the dataset sub-paths from --dataset-dir unless explicitly overridden."""
    root = os.path.normpath(args.dataset_dir)
    name = os.path.basename(root)
    if not name.startswith("batch_simulation_"):
        print(f"[warn] dataset dir '{name}' does not follow the batch_simulation_<tag> "
              f"convention; continuing anyway.")
    if args.output_dir is None:
        args.output_dir = os.path.join(root, "single_trajectory_jammers")
    if args.sim_output_dir is None:
        args.sim_output_dir = os.path.join(root, "multi_trajectory_jammers")
    if args.splits is None:
        args.splits = os.path.join(root, "splits.json")
    if args.static_dir is None:
        args.static_dir = os.path.join(root, "single_static_jammers")
    if args.multi_static_dir is None:
        args.multi_static_dir = os.path.join(root, "multi_static_jammers")
    return args


# -----------------------------------------------------------------------------
# Step 1: Trajectory Generation
# -----------------------------------------------------------------------------
def run_generation(args, engine):
    from core.trajectory_generator import TrajectoryGenerator, TrajectorySpec, save_trajectory

    np.random.seed(args.seed)
    total_expected = len(args.durations) * len(args.velocities) * args.count_per_combo
    print("\n" + "=" * 70)
    print(f"[1/3] GENERATING {total_expected} TRAJECTORIES (NYC)")
    print("=" * 70)
    print(f"Durations: {args.durations} s | Velocities: {args.velocities} m/s")
    print(f"Sampling: {1.0 / args.time_step:.1f} fps | Mode: {args.mode.upper()}")
    print(f"Overlap limit: <= {args.max_overlap_ratio * 100:.0f}% (within {args.proximity_threshold:.1f}m)")

    generator = TrajectoryGenerator(engine)
    os.makedirs(args.output_dir, exist_ok=True)

    # Sort combinations descending by distance so long corridors are reserved first
    combos = []
    for dur in args.durations:
        for vel in args.velocities:
            num_frames = int(round(dur / args.time_step))
            dist = (num_frames - 1) * vel * args.time_step
            combos.append((dist, dur, vel))
    combos.sort(key=lambda x: x[0], reverse=True)

    manifest_entries = []
    global_accepted = []
    generated_count = 0

    for travel_dist, dur, vel in combos:
        print(f"\n--- Combination: {dur}s @ {vel}m/s (travel dist: {travel_dist:.1f}m) ---")
        corridors = generator.find_straight_street_corridors(
            required_distance=travel_dist,
            num_corridors=args.count_per_combo,
            z_height=args.z_height,
            min_separation=25.0,
            max_attempts=40000,
            existing_trajectories=global_accepted,
            max_overlap_ratio=args.max_overlap_ratio,
            proximity_threshold=args.proximity_threshold,
        )

        for i in range(len(corridors)):
            traj_id = f"traj_dur{int(dur)}s_vel{int(vel):02d}mps_{i:02d}"
            start_p, heading_deg = corridors[i]

            spec = TrajectorySpec(
                duration=dur,
                velocity=vel,
                time_step=args.time_step,
                mode=args.mode,
                start_pos=start_p,
                heading_deg=heading_deg,
                z_height=args.z_height,
                inclusive_end=False,
                name=traj_id,
                extra_metadata={
                    "scene": "NYC3KM_585751_4512036",
                    "index_in_combo": i,
                    "max_overlap_ratio_allowed": args.max_overlap_ratio,
                },
            )

            traj_arr, metadata = generator.generate_straight(spec)
            npy_path, json_path = save_trajectory(args.output_dir, traj_id, traj_arr, metadata)
            global_accepted.append(traj_arr)

            manifest_entries.append({
                "id": traj_id,
                "duration_s": dur,
                "velocity_mps": vel,
                "num_frames": int(traj_arr.shape[0]),
                "travel_distance_m": float(metadata["total_distance_meters"]),
                "start_pos": metadata["start_position"],
                "end_pos": metadata["end_position"],
                "heading_deg": float(metadata["heading_degrees"]),
                "is_collision_free": bool(metadata["is_collision_free"]),
                "npy_file": os.path.basename(npy_path),
                "json_file": os.path.basename(json_path),
            })
            generated_count += 1
            print(f"  [{generated_count:02d}/{total_expected}] {traj_id}: shape={traj_arr.shape}, dist={metadata['total_distance_meters']:.1f}m -> OK (overlap <= 75%)")

    # Save manifest
    manifest_path = os.path.join(args.output_dir, "trajectories_manifest.json")
    manifest = {
        "dataset_name": os.path.basename(args.output_dir),
        "scene": "NYC3KM_585751_4512036",
        "total_trajectories": generated_count,
        "mode": args.mode,
        "durations": args.durations,
        "velocities": args.velocities,
        "count_per_combination": args.count_per_combo,
        "time_step_s": args.time_step,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trajectories": manifest_entries,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest saved: {manifest_path}")
    return manifest_entries


# -----------------------------------------------------------------------------
# Step 2: 2D Visualization
# -----------------------------------------------------------------------------
def run_plot(args, obstacles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from matplotlib.collections import PatchCollection

    print("\n" + "=" * 70)
    print("[2/3] GENERATING 2D TRAJECTORY MAP")
    print("=" * 70)

    npy_files = sorted(glob.glob(os.path.join(args.output_dir, "*.npy")))
    if not npy_files:
        print(f"No trajectory .npy files found in {args.output_dir}. Skipping plot.")
        return

    output_img = os.path.join(args.output_dir, "trajectories_map.png")
    fig, ax = plt.subplots(figsize=(14, 14), dpi=150, facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    # Draw obstacles
    patches = []
    for obs in obstacles:
        fp = obs.get("footprint")
        if fp:
            patches.append(Polygon(fp, closed=True))
        else:
            mn, mx = obs["min"], obs["max"]
            patches.append(Rectangle((mn[0], mn[1]), mx[0] - mn[0], mx[1] - mn[1]))

    pc = PatchCollection(patches, facecolor="#383838", edgecolor="#444444", linewidth=0.2, alpha=0.8, zorder=1)
    ax.add_collection(pc)

    # Color map
    try:
        cmap = matplotlib.colormaps.get_cmap("tab10")
    except Exception:
        cmap = plt.get_cmap("tab10")

    combos = [
        (30, 3), (30, 9), (30, 15),
        (60, 3), (60, 9), (60, 15),
        (90, 3), (90, 9), (90, 15),
    ]
    combo_colors = {c: cmap(i % 10) for i, c in enumerate(combos)}

    legend_labels = set()
    for fpath in npy_files:
        traj = np.load(fpath)
        parts = os.path.basename(fpath).split("_")
        dur = int(parts[1].replace("dur", "").replace("s", ""))
        vel = int(parts[2].replace("vel", "").replace("mps", ""))
        color = combo_colors.get((dur, vel), "#00ffcc")

        label = f"{dur}s @ {vel}m/s" if (dur, vel) not in legend_labels else None
        if label:
            legend_labels.add((dur, vel))

        ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.5, alpha=0.85, zorder=3, label=label)
        ax.scatter(traj[0, 0], traj[0, 1], color="#00ff66", s=15, zorder=4)
        ax.scatter(traj[-1, 0], traj[-1, 1], color="#ff3333", s=15, marker="s", zorder=4)

    ax.set_title("NYC 3KM - Generated Single Jammer Trajectories (Collision-Free, Overlap <= 75%)", color="#ffffff", fontsize=13, pad=12)
    ax.set_xlabel("X (meters)", color="#cccccc")
    ax.set_ylabel("Y (meters)", color="#cccccc")
    ax.tick_params(colors="#cccccc")
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, color="#333333", linestyle="--", alpha=0.5)
    ax.set_aspect("equal")
    b_plot = args.map_bounds_b
    ax.set_xlim(-b_plot, b_plot)
    ax.set_ylim(-b_plot, b_plot)

    leg = ax.legend(loc="upper right", facecolor="#2a2a2a", edgecolor="#555555", fontsize=9)
    for t in leg.get_texts():
        t.set_color("#ffffff")

    plt.savefig(output_img, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Trajectory map successfully saved to: {output_img}")


# -----------------------------------------------------------------------------
# Step 3: Base RadioMap Simulation for 54 Single Jammers (GPU)
# -----------------------------------------------------------------------------
def run_base_simulations(args, obstacles):
    setup_mitsuba_variant()

    from sionna.rt import load_scene, RadioMapSolver, Transmitter
    from utils.scene_objects import create_scene_objects
    from core.engine import MotionEngine

    print("\n" + "=" * 70)
    print("[3/4] SIMULATING BASE RADIO MAPS FOR 54 SINGLE JAMMERS (GPU)")
    print("=" * 70)
    print(f"Max depth: {args.max_depth} | Samples/tx: {args.samples_per_tx} | Cell size: {args.cell_size}")
    print(f"Diffraction: True | Edge Diffraction: True | Refraction: True")

    radio_maps_dir = os.path.join(args.output_dir, "radio_maps")
    os.makedirs(radio_maps_dir, exist_ok=True)

    # 1. Load scene
    scene = load_scene(args.scene_path)
    scene.frequency = args.freq_hz

    b = args.map_bounds_b
    map_bounds = {"x": [-b, b], "y": [-b, b], "z": [1.5, 1.5]}
    cell_size = tuple(args.cell_size)
    map_center, map_size = create_scene_objects(scene, map_bounds=map_bounds, z_height=1.5)
    engine = MotionEngine(scene=scene, obstacles=obstacles, bounds=map_bounds)
    rm_solver = RadioMapSolver()
    NOISE_FLOOR_WATTS = 8e-15
    tx_power_dbm = args.power_dbw + 30.0

    traj_files = sorted(glob.glob(os.path.join(args.output_dir, "traj_*.npy")))
    if not traj_files:
        raise FileNotFoundError(f"No single jammer trajectories found in {args.output_dir}")

    manifest_entries = []
    print(f"Found {len(traj_files)} base trajectories to simulate in {args.output_dir}.")

    for idx, tpath in enumerate(traj_files):
        traj_id = os.path.splitext(os.path.basename(tpath))[0]
        watts_file = os.path.join(radio_maps_dir, f"rm_watts_{traj_id}.npy")
        dbw_file = os.path.join(radio_maps_dir, f"rm_dbw_{traj_id}.npy")
        meta_file = os.path.join(radio_maps_dir, f"rm_{traj_id}.json")

        if os.path.exists(watts_file) and os.path.exists(dbw_file) and os.path.exists(meta_file):
            print(f"[{idx + 1}/{len(traj_files)}] Already simulated: {traj_id} (skipping)")
            with open(meta_file, "r", encoding="utf-8") as jf:
                manifest_entries.append(json.load(jf))
            continue

        path_arr = np.load(tpath)
        num_steps = len(path_arr)
        print(f"\n[{idx + 1}/{len(traj_files)}] Simulating Base Jammer: {traj_id} ({num_steps} steps)...")

        # Clear any existing transmitter
        for tname in list(scene.transmitters.keys()):
            scene.remove(tname)
        engine._jammer_paths.clear()

        # Add single transmitter
        tx_name = f"Jammer_{traj_id}"
        engine._jammer_paths[tx_name] = path_arr
        tx = Transmitter(name=tx_name, position=path_arr[0], power_dbm=tx_power_dbm)
        scene.add(tx)

        watts_history = []
        t_sim_start = time.time()

        for step in range(num_steps):
            engine.update_scene_transmitters(step)
            print(f"  Step {step + 1}/{num_steps}...", end="\r", flush=True)

            rm = rm_solver(
                scene,
                max_depth=args.max_depth,
                samples_per_tx=args.samples_per_tx,
                cell_size=cell_size,
                center=map_center,
                size=[map_size[0], map_size[1]],
                orientation=[0, 0, 0],
                diffraction=True,
                edge_diffraction=True,
                refraction=True,
            )

            lin_watts = rm.rss.numpy()
            if len(lin_watts.shape) == 3:
                watts_frame = lin_watts[0]
            else:
                watts_frame = lin_watts

            watts_history.append(watts_frame)

        sim_duration = time.time() - t_sim_start
        print(f"\n  Completed {num_steps} steps in {sim_duration:.1f}s ({num_steps / max(sim_duration, 1e-4):.2f} fps).")

        # Convert to arrays (native Watts and dBW with noise floor)
        watts_arr = np.array(watts_history, dtype=np.float32)  # (T, H, W)
        dbw_arr = (10.0 * np.log10(watts_arr + NOISE_FLOOR_WATTS)).astype(np.float32)

        # Save to disk
        np.save(watts_file, watts_arr)
        np.save(dbw_file, dbw_arr)

        entry_meta = {
            "traj_id": traj_id,
            "num_steps": num_steps,
            "watts_file": os.path.basename(watts_file),
            "dbw_file": os.path.basename(dbw_file),
            "power_dbw": args.power_dbw,
            "min_dbw": float(dbw_arr.min()),
            "max_dbw": float(dbw_arr.max()),
            "noise_floor_watts": NOISE_FLOOR_WATTS,
            "sim_time_seconds": float(round(sim_duration, 2)),
        }
        with open(meta_file, "w", encoding="utf-8") as jf:
            json.dump(entry_meta, jf, indent=2)

        manifest_entries.append(entry_meta)

    # Clear transmitter at end
    for tname in list(scene.transmitters.keys()):
        scene.remove(tname)

    manifest_path = os.path.join(radio_maps_dir, "radio_maps_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_base_radio_maps": len(manifest_entries),
            "radio_maps": manifest_entries,
        }, f, indent=2)

    print(f"\nAll base radio maps saved in: {radio_maps_dir}")
    return manifest_entries


# -----------------------------------------------------------------------------
# Step 4: Aggregation driven by splits.json
# -----------------------------------------------------------------------------
def _load_bases(radio_maps_dir, entries, traj_ids):
    """Load and sanitize the watts maps for a set of trajectory ids."""
    by_id = {e["traj_id"]: e for e in entries}
    out = {}
    for tid in sorted(traj_ids):
        if tid not in by_id:
            raise KeyError(f"No base radio map for '{tid}'. Run --action simulate_bases first.")
        w = np.load(os.path.join(radio_maps_dir, by_id[tid]["watts_file"]))
        # zero-distance antenna singularities show up as +inf
        if not np.all(np.isfinite(w)):
            fm = np.isfinite(w)
            w = np.nan_to_num(w, posinf=float(np.max(w[fm])) if np.any(fm) else 10.0,
                              neginf=0.0, nan=0.0)
        out[tid] = w
    return out


def run_aggregation_from_splits(args):
    """Aggregate exactly the scenarios listed in splits.json.

    Walking the stored scenario list (rather than drawing fresh random combinations) is
    what preserves the train/val/test trajectory separation and the stratified K
    distribution, and what keeps scenario ids lined up with the labels produced by
    scripts/make_labels.py.

    Base maps are loaded per split rather than all at once, so only that split's pool is
    resident (34 maps for train, 10 for val/test).
    """
    print("\n" + "=" * 70)
    print("AGGREGATING SCENARIOS FROM splits.json")
    print("=" * 70)

    if not os.path.exists(args.splits):
        raise FileNotFoundError(
            f"{args.splits} not found. Run scripts/make_splits.py first."
        )
    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)

    radio_maps_dir = os.path.join(args.output_dir, "radio_maps")
    manifest_path = os.path.join(radio_maps_dir, "radio_maps_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"{manifest_path} not found. Run --action simulate_bases first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        entries = json.load(f)["radio_maps"]

    n_cells = splits["grid"]["n_cells"]
    NOISE_FLOOR_WATTS = 8e-15
    precision_map = {"float16": np.float16, "fp16": np.float16,
                     "float32": np.float32, "fp32": np.float32,
                     "float64": np.float64, "fp64": np.float64}
    target_dtype = precision_map.get(args.precision.lower(), np.float16)

    all_steps = [e["num_steps"] for e in entries]
    written = 0

    for split in ("train", "val", "test"):
        scenarios = splits["scenarios"].get(split, [])
        if not scenarios:
            continue
        pool = splits["trajectory_pools"][split]
        print(f"\n[{split}] {len(scenarios)} scenarios, loading {len(pool)} base maps...")
        maps = _load_bases(radio_maps_dir, entries, pool)
        H, W = next(iter(maps.values())).shape[1:]
        if H != n_cells or W != n_cells:
            raise ValueError(
                f"Base maps are {H}x{W} but splits.json expects {n_cells}x{n_cells}. "
                f"Re-run --action simulate_bases with --cell-size "
                f"{splits['grid']['cell_size_m']:g} {splits['grid']['cell_size_m']:g}, "
                f"deleting {radio_maps_dir} first."
            )

        split_dir = os.path.join(args.sim_output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        t0 = time.time()

        for i, scen in enumerate(scenarios):
            jids = scen["jammers"]
            K = len(jids)
            noise_seed = int(scen.get("noise_seed", scen.get("sensor_seed", args.seed)))
            rng = np.random.default_rng(noise_seed)

            if K:
                max_steps = max(len(maps[t]) for t in jids)
            else:
                max_steps = int(np.random.default_rng(noise_seed).choice(all_steps))

            agg = np.zeros((max_steps, H, W), dtype=np.float32)
            jammers_meta = []
            for j, tid in enumerate(jids):
                w = maps[tid]
                pad = max_steps - len(w)
                padded = np.vstack([w, np.tile(w[-1:], (pad, 1, 1))]) if pad > 0 else w
                agg += padded[:max_steps]
                jammers_meta.append({"jammer_index": j, "traj_id": tid,
                                     "original_steps": int(len(w)),
                                     "padded_steps": int(max(pad, 0))})

            dbw = 10.0 * np.log10(agg + NOISE_FLOOR_WATTS)
            if not np.all(np.isfinite(dbw)):
                fin = dbw[np.isfinite(dbw)]
                dbw = np.nan_to_num(dbw, posinf=float(fin.max()) if fin.size else 40.0,
                                    neginf=-141.0, nan=-141.0)
            if args.meas_noise_var > 0.0:
                dbw = dbw + rng.normal(0.0, math.sqrt(args.meas_noise_var),
                                       size=dbw.shape).astype(np.float32)
            out = dbw.astype(target_dtype)

            sc_dir = os.path.join(split_dir, scen["scenario_id"])
            os.makedirs(sc_dir, exist_ok=True)
            np.save(os.path.join(sc_dir, "rss_aggregated.npy"), out)
            with open(os.path.join(sc_dir, "scenario_summary.json"), "w", encoding="utf-8") as sf:
                json.dump({
                    "scenario_id": scen["scenario_id"],
                    "split": split,
                    "num_jammers": K,
                    "total_steps": int(max_steps),
                    "dtype": str(out.dtype),
                    "min_dbw": float(out.min()),
                    "max_dbw": float(out.max()),
                    "noise_floor_watts": NOISE_FLOOR_WATTS,
                    "meas_noise_var_db": args.meas_noise_var,
                    "noise_seed": noise_seed,
                    "sensor_density_pct": scen.get("sensor_density_pct"),
                    "num_sensors": scen.get("num_sensors"),
                    "sensor_seed": scen.get("sensor_seed"),
                    "jammers": jammers_meta,
                }, sf, indent=2)
            written += 1
            if (i + 1) % 50 == 0 or i == len(scenarios) - 1:
                print(f"  {i + 1}/{len(scenarios)} ({(i + 1) / max(time.time() - t0, 1e-6):.1f}/s)",
                      end="\r", flush=True)
        print(f"\n  [{split}] done in {time.time() - t0:.1f}s -> {split_dir}")
        del maps

    print(f"\nWrote {written} scenarios under {args.sim_output_dir}")


# -----------------------------------------------------------------------------
# Static branch (detection): positions -> ray trace -> combine
# -----------------------------------------------------------------------------
def street_cell_mask(mesh_dir, n_cells, b, cell_size):
    """True where a cell centre falls inside a building footprint."""
    from utils.scene_objects import gather_bboxes
    obstacles = gather_bboxes(mesh_dir, footprints=False, use_cache=True)
    mask = np.zeros((n_cells, n_cells), dtype=bool)
    centres = -b + (np.arange(n_cells) + 0.5) * cell_size
    for ob in obstacles:
        lo, hi = ob["min"], ob["max"]
        c0 = np.searchsorted(centres, lo[0], "left")
        c1 = np.searchsorted(centres, hi[0], "right")
        r0 = np.searchsorted(centres, lo[1], "left")
        r1 = np.searchsorted(centres, hi[1], "right")
        if c1 > c0 and r1 > r0:
            mask[r0:r1, c0:c1] = True
    return ~mask          # True where placeable


def run_static_generation(args):
    """Sample N continuous jammer positions on street cells.  [CPU]"""
    print("\n" + "=" * 70)
    print(f"[STATIC 1/3] SAMPLING {args.n_static} JAMMER POSITIONS  [CPU]")
    print("=" * 70)

    b = args.map_bounds_b
    cell = float(args.cell_size[0])
    n_cells = int(round(2 * b / cell))
    free = street_cell_mask(args.mesh_dir, n_cells, b, cell)
    placeable = np.flatnonzero(free.ravel())
    print(f"Grid {n_cells}x{n_cells} @ {cell:g} m | {len(placeable)} street cells "
          f"({100*len(placeable)/(n_cells*n_cells):.1f}%)")

    rng = np.random.default_rng(args.seed)
    # sample cells with replacement, then jitter continuously inside each one. The
    # jitter is what keeps the sub-cell (dx, dy) regression target non-degenerate -
    # one position per cell centre would make it constant.
    cells = rng.choice(placeable, size=args.n_static, replace=True)
    rows, cols = np.divmod(cells, n_cells)
    x = -b + (cols + rng.random(args.n_static)) * cell
    y = -b + (rows + rng.random(args.n_static)) * cell
    pos = np.column_stack([x, y, np.full(args.n_static, args.z_height)])

    os.makedirs(args.static_dir, exist_ok=True)
    np.save(os.path.join(args.static_dir, "positions.npy"), pos)
    with open(os.path.join(args.static_dir, "positions.json"), "w", encoding="utf-8") as f:
        json.dump({"n_positions": int(args.n_static), "seed": args.seed,
                   "n_cells": n_cells, "cell_size_m": cell, "map_bounds_b_m": b,
                   "z_height_m": args.z_height,
                   "placeable_cells": int(len(placeable)),
                   "placement": "street cells only, continuous jitter within each cell"}, f, indent=2)
    print(f"Wrote {args.n_static} positions -> {args.static_dir}/positions.npy")
    print(f"  distinct cells used: {len(np.unique(cells))}")
    return pos


def run_static_simulation(args):
    """Ray trace one radio map per static position into a single memmap array.  [GPU]"""
    setup_mitsuba_variant()
    from sionna.rt import load_scene, RadioMapSolver, Transmitter
    from utils.scene_objects import create_scene_objects

    pos_path = os.path.join(args.static_dir, "positions.npy")
    if not os.path.exists(pos_path):
        raise FileNotFoundError(f"{pos_path} not found. Run --action generate_static first.")
    pos = np.load(pos_path)
    n = len(pos)

    b = args.map_bounds_b
    cell = tuple(args.cell_size)
    n_cells = int(round(2 * b / cell[0]))

    print("\n" + "=" * 70)
    print(f"[STATIC 2/3] RAY TRACING {n} STATIC RADIO MAPS  [GPU]")
    print("=" * 70)
    print(f"Grid {n_cells}x{n_cells} | max_depth={args.max_depth} | "
          f"samples/tx={args.samples_per_tx}")

    scene = load_scene(args.scene_path)
    scene.frequency = args.freq_hz
    map_bounds = {"x": [-b, b], "y": [-b, b], "z": [args.z_height, args.z_height]}
    map_center, map_size = create_scene_objects(scene, map_bounds=map_bounds,
                                                z_height=args.z_height)
    rm_solver = RadioMapSolver()
    tx_power_dbm = args.power_dbw + 30.0

    # One (N, H, W) file rather than N files: 20k separate .npy would burn 20k inodes
    # and make random access for the combination step awkward. This is memmap-able.
    watts_path = os.path.join(args.static_dir, "watts.npy")
    done_path = os.path.join(args.static_dir, "watts.progress.json")
    start = 0
    if os.path.exists(watts_path) and os.path.exists(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            prog = json.load(f)
        if prog.get("n") == n and prog.get("n_cells") == n_cells:
            start = int(prog.get("completed", 0))
            print(f"Resuming at position {start}/{n}")
        else:
            print("Existing watts.npy has a different shape; starting over.")
    if start == 0:
        arr = np.lib.format.open_memmap(watts_path, mode="w+", dtype=np.float32,
                                        shape=(n, n_cells, n_cells))
    else:
        arr = np.lib.format.open_memmap(watts_path, mode="r+")

    t0 = time.time()
    for i in range(start, n):
        for tname in list(scene.transmitters.keys()):
            scene.remove(tname)
        scene.add(Transmitter(name="J", position=pos[i], power_dbm=tx_power_dbm))
        rm = rm_solver(scene, max_depth=args.max_depth, samples_per_tx=args.samples_per_tx,
                       cell_size=cell, center=map_center,
                       size=[map_size[0], map_size[1]], orientation=[0, 0, 0],
                       diffraction=True, edge_diffraction=True, refraction=True)
        w = rm.rss.numpy()
        arr[i] = w[0] if w.ndim == 3 else w
        if (i + 1) % 50 == 0 or i == n - 1:
            arr.flush()
            with open(done_path, "w", encoding="utf-8") as f:
                json.dump({"n": n, "n_cells": n_cells, "completed": i + 1}, f)
            el = time.time() - t0
            rate = (i + 1 - start) / max(el, 1e-6)
            print(f"  {i+1}/{n}  {rate:.2f}/s  eta {(n-i-1)/max(rate,1e-6)/60:.1f} min",
                  end="\r", flush=True)
    arr.flush()
    for tname in list(scene.transmitters.keys()):
        scene.remove(tname)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {watts_path}")


def run_static_aggregation(args):
    """Combine static base maps into detector samples listed in splits.json.  [CPU]"""
    print("\n" + "=" * 70)
    print("[STATIC 3/3] BUILDING DETECTOR SAMPLES  [CPU]")
    print("=" * 70)

    if not os.path.exists(args.splits):
        raise FileNotFoundError(f"{args.splits} not found. Run scripts/make_splits.py first.")
    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if "static" not in splits:
        raise KeyError("splits.json has no 'static' section. Re-run scripts/make_splits.py "
                       "after generating single_static_jammers/positions.npy.")

    watts_path = os.path.join(args.static_dir, "watts.npy")
    if not os.path.exists(watts_path):
        raise FileNotFoundError(f"{watts_path} not found. Run --action simulate_static first.")
    base = np.load(watts_path, mmap_mode="r")
    n_pos, H, W = base.shape
    n_cells = splits["grid"]["n_cells"]
    if H != n_cells:
        raise ValueError(f"Static maps are {H}x{W} but splits.json expects "
                         f"{n_cells}x{n_cells}. Regenerate one of them.")

    pos = np.load(os.path.join(args.static_dir, "positions.npy"))
    NOISE_FLOOR_WATTS = 8e-15
    prec = {"float16": np.float16, "fp16": np.float16, "float32": np.float32,
            "fp32": np.float32, "float64": np.float64, "fp64": np.float64}
    dtype = prec.get(args.precision.lower(), np.float16)
    cell = splits["grid"]["cell_size_m"]
    x0, y0 = splits["grid"]["origin_m"]

    for split, samples in splits["static"]["samples"].items():
        out_dir = os.path.join(args.multi_static_dir, split)
        os.makedirs(out_dir, exist_ok=True)
        n = len(samples)
        rss = np.lib.format.open_memmap(os.path.join(out_dir, "rss.npy"), mode="w+",
                                        dtype=dtype, shape=(n, H, W))
        lab = {k: [] for k in ("sample_idx", "jammer_idx", "x", "y", "col", "row", "dx", "dy")}
        ks, dens, nsens, sseed, nseed = [], [], [], [], []
        flat_ids, id_off = [], [0]

        t0 = time.time()
        for i, sm in enumerate(samples):
            ids = sm["position_ids"]
            acc = np.zeros((H, W), dtype=np.float32)
            for pid in ids:
                w = np.asarray(base[pid], dtype=np.float32)
                if not np.all(np.isfinite(w)):
                    fm = np.isfinite(w)
                    w = np.nan_to_num(w, posinf=float(w[fm].max()) if fm.any() else 10.0,
                                      neginf=0.0, nan=0.0)
                acc += w
            dbw = 10.0 * np.log10(acc + NOISE_FLOOR_WATTS)
            if not np.all(np.isfinite(dbw)):
                fin = dbw[np.isfinite(dbw)]
                dbw = np.nan_to_num(dbw, posinf=float(fin.max()) if fin.size else 40.0,
                                    neginf=-141.0, nan=-141.0)
            if args.meas_noise_var > 0.0:
                rng = np.random.default_rng(sm["noise_seed"])
                dbw = dbw + rng.normal(0.0, math.sqrt(args.meas_noise_var),
                                       size=dbw.shape).astype(np.float32)
            rss[i] = dbw.astype(dtype)

            for j, pid in enumerate(ids):
                x, y = pos[pid][0], pos[pid][1]
                cf, rf = (x - x0) / cell, (y - y0) / cell
                c, r = int(np.floor(cf)), int(np.floor(rf))
                lab["sample_idx"].append(i); lab["jammer_idx"].append(j)
                lab["x"].append(x); lab["y"].append(y)
                lab["col"].append(c); lab["row"].append(r)
                lab["dx"].append((cf - c) * cell); lab["dy"].append((rf - r) * cell)
            ks.append(len(ids)); dens.append(sm["sensor_density_pct"])
            nsens.append(sm["num_sensors"]); sseed.append(sm["sensor_seed"])
            nseed.append(sm["noise_seed"])
            flat_ids.extend(ids); id_off.append(len(flat_ids))
            if (i + 1) % 200 == 0 or i == n - 1:
                print(f"  {split}: {i+1}/{n} ({(i+1)/max(time.time()-t0,1e-6):.1f}/s)",
                      end="\r", flush=True)
        rss.flush()

        dt = {"sample_idx": np.int32, "jammer_idx": np.int8, "col": np.int16, "row": np.int16}
        np.savez_compressed(os.path.join(out_dir, "labels.npz"),
                            **{k: np.asarray(v, dtype=dt.get(k, np.float64))
                               for k, v in lab.items()})
        np.savez_compressed(os.path.join(out_dir, "meta.npz"),
                            num_jammers=np.asarray(ks, np.int8),
                            position_ids=np.asarray(flat_ids, np.int32),
                            position_offsets=np.asarray(id_off, np.int64),
                            sensor_density_pct=np.asarray(dens, np.float32),
                            num_sensors=np.asarray(nsens, np.int32),
                            sensor_seed=np.asarray(sseed, np.int64),
                            noise_seed=np.asarray(nseed, np.int64))
        print(f"\n  {split}: {n} samples -> {out_dir} "
              f"({os.path.getsize(os.path.join(out_dir,'rss.npy'))/1e9:.1f} GB)")

    print(f"\nDetector samples written under {args.multi_static_dir}")


# -----------------------------------------------------------------------------
# Main Dispatcher
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 70)
    print("Sionna RT Jamming - Unified Non-Interactive Pipeline")
    print("=" * 70)
    print(f"Action requested: {args.action.upper()}")

    # Building meshes are only needed for trajectory generation, plotting, ray tracing
    # and ray tracing. 'aggregate' is pure NumPy over precomputed maps,
    # so skip the load (and the trimesh dependency) entirely for it.
    # Building meshes are needed for trajectory generation, plotting, ray tracing and
    # static placement, but not for the two pure-NumPy combination stages.
    obstacles, engine = None, None
    if args.action not in ("aggregate", "aggregate_static", "simulate_static",
                           "generate_static"):
        t0 = time.time()
        from utils.scene_objects import gather_bboxes
        from core.engine import MotionEngine

        print("Loading building obstacles...")
        obstacles = gather_bboxes(args.mesh_dir, footprints=True, use_cache=True)
        print(f"Loaded {len(obstacles)} obstacles in {time.time() - t0:.2f}s.")

        gen_limit = max(50.0, args.map_bounds_b - 50.0)
        bounds = {"x": [-gen_limit, gen_limit], "y": [-gen_limit, gen_limit],
                  "z": [args.z_height, args.z_height]}
        engine = MotionEngine(scene=None, obstacles=obstacles, bounds=bounds)

    if args.action in ["all", "generate"]:
        run_generation(args, engine)

    if args.action in ["all", "plot"]:
        run_plot(args, obstacles)

    if args.action in ["all", "simulate_bases"]:
        run_base_simulations(args, obstacles)

    if args.action in ["all", "aggregate"]:
        run_aggregation_from_splits(args)

    if args.action == "generate_static":
        run_static_generation(args)

    if args.action == "simulate_static":
        run_static_simulation(args)

    if args.action == "aggregate_static":
        run_static_aggregation(args)

    print("\n" + "=" * 70)
    print(f"Action '{args.action}' completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
