#!/usr/bin/env python3
"""
Sionna RT Jamming - Unified Non-Interactive Entry Point.

Combines all terminal/SLURM operations into a single script:
1. Trajectory generation (54 collision-free straight trajectories, overlap <= 75%).
2. 2D Trajectory map visualization (headless matplotlib).
3. GPU-accelerated Sionna RT RadioMap simulation.

Usage:
    python main_no_interactive.py --action all       # Run generation + plot + simulation
    python main_no_interactive.py --action generate  # Trajectory generation only
    python main_no_interactive.py --action plot      # 2D visualization only
    python main_no_interactive.py --action simulate  # RadioMap simulation only
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
        choices=["all", "generate", "plot", "simulate_bases", "aggregate", "simulate"],
        default="all",
        help="Action: 'all' (default), 'generate', 'plot', 'simulate_bases' (GPU ray trace 54 bases), 'aggregate' (instant combinations), 'simulate'.",
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
        "--output-dir",
        type=str,
        default="./datasets/nyc_single_jammers",
        help="Folder containing generated single jammer trajectories (default: ./datasets/nyc_single_jammers).",
    )
    parser.add_argument(
        "--sim-output-dir",
        type=str,
        default="./datasets/simulation_results_nyc",
        help="Folder to store radio map simulation arrays and summary GIF.",
    )
    # Multi-Jammer Combination Parameters
    parser.add_argument(
        "--num-combinations",
        type=int,
        default=1000,
        help="Number of multi-jammer combinations to aggregate (default: 1000).",
    )
    parser.add_argument(
        "--min-jammers",
        type=int,
        default=2,
        help="Minimum number of simultaneous jammers per combination (default: 2).",
    )
    parser.add_argument(
        "--max-jammers",
        type=int,
        default=10,
        help="Maximum number of simultaneous jammers per combination (default: 10).",
    )
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
        "--sim-trajectories",
        type=str,
        nargs="+",
        default=None,
        help="Specific trajectory .npy files to simulate. If None, picks the first two generated.",
    )
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
        default=[8.0, 8.0],
        help="Radio map grid resolution in meters (default: 8 8).",
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
        "--skip-gif",
        action="store_true",
        help="Options: omitted/default (generate GIF animations for combinations) or flag provided (skip GIF animations to save simulation time and disk space).",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float16",
        choices=["float16", "float32", "float64", "fp16", "fp32", "fp64"],
        help="Floating-point precision for saving aggregated radio maps (default: 'float16'; options: float16, float32, float64).",
    )
    return parser.parse_args()


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
# Step 4: Instant Multi-Jammer Combination Aggregation (NumPy / Physics Superposition)
# -----------------------------------------------------------------------------
def run_aggregation(args, obstacles):
    from utils.plotter import create_jammer_animation

    print("\n" + "=" * 70)
    print(f"[4/4] AGGREGATING {args.num_combinations} MULTI-JAMMER COMBINATIONS ({args.min_jammers}-{args.max_jammers} JAMMERS)")
    print("=" * 70)

    radio_maps_dir = os.path.join(args.output_dir, "radio_maps")
    manifest_path = os.path.join(radio_maps_dir, "radio_maps_manifest.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Base radio maps manifest not found in {radio_maps_dir}. "
            "Please run with '--action simulate_bases' first."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    base_entries = manifest["radio_maps"]

    if len(base_entries) < args.min_jammers:
        raise ValueError(f"Not enough base radio maps ({len(base_entries)}) for min_jammers ({args.min_jammers})")

    os.makedirs(args.sim_output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    NOISE_FLOOR_WATTS = 8e-15

    # Pre-load base radio maps (in Watts) into memory for instantaneous aggregation
    print(f"Loading {len(base_entries)} base radio maps into RAM...")
    loaded_maps = {}
    for entry in base_entries:
        w_path = os.path.join(radio_maps_dir, entry["watts_file"])
        w_arr = np.load(w_path)
        # Sanitize zero-distance antenna singularities (+inf / nan)
        if not np.all(np.isfinite(w_arr)):
            finite_mask = np.isfinite(w_arr)
            max_finite = float(np.max(w_arr[finite_mask])) if np.any(finite_mask) else 10.0
            w_arr = np.nan_to_num(w_arr, posinf=max_finite, neginf=0.0, nan=0.0)
        loaded_maps[entry["traj_id"]] = w_arr

    print(f"Pre-loaded {len(loaded_maps)} base radio maps. Generating {args.num_combinations} combinations...")
    t_agg_start = time.time()
    combinations_manifest = []

    b = args.map_bounds_b
    map_bounds = {"x": [-b, b], "y": [-b, b], "z": [1.5, 1.5]}
    map_center = [0.0, 0.0, 1.5]
    map_size = (2 * b, 2 * b)

    for c in range(args.num_combinations):
        K = int(rng.integers(args.min_jammers, args.max_jammers + 1))
        K = min(K, len(base_entries))

        chosen_indices = rng.choice(len(base_entries), size=K, replace=False)
        chosen_entries = [base_entries[i] for i in chosen_indices]

        # Max steps across chosen jammers in this combination
        max_steps = max(e["num_steps"] for e in chosen_entries)
        H, W = loaded_maps[chosen_entries[0]["traj_id"]].shape[1:]

        combo_id = f"combo_{c:04d}_k{K:02d}"
        combo_dir = os.path.join(args.sim_output_dir, combo_id)
        os.makedirs(combo_dir, exist_ok=True)

        # Incoherent linear addition in Watts: P_agg = sum(P_k)
        agg_watts = np.zeros((max_steps, H, W), dtype=np.float32)

        jammers_meta = []
        for j_idx, entry in enumerate(chosen_entries):
            tid = entry["traj_id"]
            watts_data = loaded_maps[tid]
            cur_steps = len(watts_data)
            pad_amount = max_steps - cur_steps

            if pad_amount > 0:
                # pad_end: stationary at destination
                pad = np.tile(watts_data[-1:], (pad_amount, 1, 1))
                padded_watts = np.vstack([watts_data, pad])
            else:
                padded_watts = watts_data

            agg_watts += padded_watts[:max_steps]

            jname = f"Jammer{j_idx + 1}"
            jammers_meta.append({
                "jammer_index": j_idx,
                "jammer_name": jname,
                "traj_id": tid,
                "original_steps": cur_steps,
                "padded_steps": pad_amount,
            })

        # Apply exact noise floor equation from main.py: agg_dbw = 10 * log10(P_agg_watts + 8e-15)
        agg_dbw_f32 = 10.0 * np.log10(agg_watts + NOISE_FLOOR_WATTS)
        if not np.all(np.isfinite(agg_dbw_f32)):
            finite_dbw = agg_dbw_f32[np.isfinite(agg_dbw_f32)]
            max_dbw_finite = float(np.max(finite_dbw)) if len(finite_dbw) > 0 else 40.0
            agg_dbw_f32 = np.nan_to_num(agg_dbw_f32, posinf=max_dbw_finite, neginf=-141.0, nan=-141.0)
        
        precision_map = {
            "float16": np.float16,
            "fp16": np.float16,
            "float32": np.float32,
            "fp32": np.float32,
            "float64": np.float64,
            "fp64": np.float64,
        }
        target_dtype = precision_map.get(args.precision.lower(), np.float16)
        agg_dbw = agg_dbw_f32.astype(target_dtype)

        np.save(os.path.join(combo_dir, "rss_aggregated.npy"), agg_dbw)

        summary = {
            "combination_id": combo_id,
            "combo_index": c,
            "num_jammers": K,
            "total_steps": max_steps,
            "dtype": str(agg_dbw.dtype),
            "min_dbw": float(agg_dbw.min()),
            "max_dbw": float(agg_dbw.max()),
            "noise_floor_watts": NOISE_FLOOR_WATTS,
            "jammers": jammers_meta,
        }
        with open(os.path.join(combo_dir, "combination_summary.json"), "w", encoding="utf-8") as sf:
            json.dump(summary, sf, indent=2)

        # Generate sample GIF for combination if skip_gif is not set
        if not args.skip_gif:
            gif_path = os.path.join(combo_dir, "jammer_animation.gif")
            from core.engine import MotionEngine
            engine = MotionEngine(scene=None, obstacles=obstacles, bounds=map_bounds)
            for j_idx, jmeta in enumerate(jammers_meta):
                tarr = np.load(os.path.join(args.output_dir, f"{jmeta['traj_id']}.npy"))
                engine._jammer_paths[f"Jammer_{j_idx+1}"] = tarr
                engine._padding_preferences[f"Jammer_{j_idx+1}"] = "pad_end"
            engine.finalize_trajectories()

            create_jammer_animation(
                rss_list=agg_dbw.astype(np.float32),
                engine=engine,
                buildings=obstacles,
                map_size=map_size,
                map_center=map_center,
                filename=gif_path,
            )
            print(f"  [{c+1}/{args.num_combinations}] {combo_id}: animation saved to {gif_path}")

        combinations_manifest.append({
            "combo_id": combo_id,
            "index": c,
            "num_jammers": K,
            "total_steps": max_steps,
            "dir": combo_dir,
            "jammer_ids": [m["traj_id"] for m in jammers_meta],
        })

        if (c + 1) % 100 == 0 or c == args.num_combinations - 1:
            print(f"  Aggregated {c + 1}/{args.num_combinations} combinations...", end="\r", flush=True)

    agg_total_time = time.time() - t_agg_start
    print(f"\nGenerated {args.num_combinations} combinations in {agg_total_time:.2f}s ({args.num_combinations / max(agg_total_time, 1e-4):.1f} combos/s)!")

    with open(os.path.join(args.sim_output_dir, "combinations_manifest.json"), "w", encoding="utf-8") as cf:
        json.dump({
            "total_combinations": args.num_combinations,
            "min_jammers": args.min_jammers,
            "max_jammers": args.max_jammers,
            "combinations": combinations_manifest,
        }, cf, indent=2)

    print(f"All aggregated combinations saved in: {args.sim_output_dir}")


# -----------------------------------------------------------------------------
# Main Dispatcher
# -----------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 70)
    print("Sionna RT Jamming - Unified Non-Interactive Pipeline")
    print("=" * 70)
    print(f"Action requested: {args.action.upper()}")

    # Load obstacles (with disk cache)
    t0 = time.time()
    from utils.scene_objects import gather_bboxes
    from core.engine import MotionEngine

    print("Loading building obstacles...")
    obstacles = gather_bboxes(args.mesh_dir, footprints=True, use_cache=True)
    print(f"Loaded {len(obstacles)} obstacles in {time.time() - t0:.2f}s.")

    gen_limit = max(50.0, args.map_bounds_b - 50.0)
    bounds = {"x": [-gen_limit, gen_limit], "y": [-gen_limit, gen_limit], "z": [args.z_height, args.z_height]}
    engine = MotionEngine(scene=None, obstacles=obstacles, bounds=bounds)

    if args.action in ["all", "generate"]:
        run_generation(args, engine)

    if args.action in ["all", "plot"]:
        run_plot(args, obstacles)

    if args.action in ["all", "simulate_bases", "simulate"]:
        run_base_simulations(args, obstacles)

    if args.action in ["all", "aggregate", "simulate"]:
        run_aggregation(args, obstacles)

    print("\n" + "=" * 70)
    print(f"Action '{args.action}' completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
