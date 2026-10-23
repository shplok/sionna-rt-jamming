#!/usr/bin/env python3
"""
Unified Visualization Suite for Multi-Jammer Simulations (Continuous, Curved & Intermittent).

Features:
1. Dynamic Animated GIF:
   - Full time-lapse simulation showing dynamic RSS heatmap, moving jammers,
     and transmission states ([ON], [OFF], [ON (Stationary)]).
2. Static 6-Panel Snapshot Overview:
   - Trajectory overview on 3D building layout.
   - 4 key time snapshot heatmaps (e.g. t=1s, 30s, 60s, 90s) with vehicle positions.
   - Temporal dynamics: City coverage area jammed (> -90 dBW) and peak power over time.
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.animation import FuncAnimation

from utils.scene_objects import gather_bboxes


def _load_obstacles(mesh_dir: str = "./data/NYC3KM_585751_4512036/mesh") -> list:
    """Helper to load building obstacles from cache or mesh directory."""
    return gather_bboxes(mesh_dir, footprints=True, use_cache=True)


def _load_simulation_data(combo_dir: str, trajectories_dir: str):
    """Helper to load combination summary, RSS array, and padded jammer trajectories."""
    summary_path = os.path.join(combo_dir, "combination_summary.json")
    rss_path = os.path.join(combo_dir, "rss_aggregated.npy")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary: {summary_path}")
    if not os.path.exists(rss_path):
        raise FileNotFoundError(f"Missing RSS array: {rss_path}")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    rss_data = np.load(rss_path).astype(np.float32)
    total_steps, H, W = rss_data.shape
    combo_id = summary.get("combination_id", os.path.basename(combo_dir))
    jammers = summary.get("jammers", [])
    is_intermittent = summary.get("mode") == "intermittent"

    paths_dict = {}
    active_masks = {}

    for j_idx, jmeta in enumerate(jammers):
        tid = jmeta["traj_id"]
        jname = jmeta.get("jammer_name", f"Jammer_{j_idx+1}")
        tpath = os.path.join(trajectories_dir, f"{tid}.npy")
        if not os.path.exists(tpath):
            raise FileNotFoundError(f"Trajectory file not found: {tpath}")
        raw_t = np.load(tpath)

        # Pad stationary at final destination if trajectory shorter than total_steps
        pad_len = total_steps - len(raw_t)
        if pad_len > 0:
            pad = np.tile(raw_t[-1:], (pad_len, 1))
            t_padded = np.vstack([raw_t, pad])
        else:
            t_padded = raw_t[:total_steps]

        paths_dict[jname] = t_padded

        # Intermittent activity mask
        if is_intermittent and "active_intervals" in jmeta:
            mask = np.zeros(total_steps, dtype=bool)
            for s, e in jmeta["active_intervals"]:
                mask[s:e] = True
            active_masks[jname] = mask
        else:
            active_masks[jname] = np.ones(total_steps, dtype=bool)

    return summary, rss_data, paths_dict, active_masks


# =============================================================================
# 1. Dynamic Animated GIF Generation
# =============================================================================

def create_simulation_gif(
    combo_dir: str,
    trajectories_dir: str,
    obstacles: list = None,
    mesh_dir: str = "./data/NYC3KM_585751_4512036/mesh",
    output_filename: str = None,
    map_bounds_b: float = 1500.0,
    fps: int = 6,
    vmin: float = -140.0,
    vmax: float = 10.0,
):
    """Renders a complete animated GIF for the simulation."""
    if obstacles is None:
        obstacles = _load_obstacles(mesh_dir)

    summary, rss_data, paths_dict, active_masks = _load_simulation_data(combo_dir, trajectories_dir)
    total_steps = rss_data.shape[0]
    combo_id = summary.get("combination_id", os.path.basename(combo_dir))
    num_jammers = len(paths_dict)
    is_intermittent = summary.get("mode") == "intermittent"

    if output_filename is None:
        output_filename = os.path.join(combo_dir, "jammer_animation.gif")

    print(f"\n--- Generating GIF for {combo_id} ({num_jammers} jammers, {total_steps} frames) ---")
    print(f"Destination: {output_filename}")

    b = map_bounds_b
    extent = [-b, b, -b, b]

    fig, ax = plt.subplots(figsize=(9, 9), dpi=130, facecolor="#141414")
    ax.set_facecolor("#181818")

    # Building obstacles
    building_patches = []
    for obs in obstacles:
        fp = obs.get("footprint")
        if fp:
            building_patches.append(patches.Polygon(fp, closed=True))
        else:
            mn, mx = obs["min"], obs["max"]
            building_patches.append(patches.Rectangle((mn[0], mn[1]), mx[0]-mn[0], mx[1]-mn[1]))

    pc = PatchCollection(building_patches, facecolor="#303030", edgecolor="#404040", linewidth=0.2, alpha=0.9, zorder=2)
    ax.add_collection(pc)

    # Initial RSS frame
    im = ax.imshow(
        rss_data[0],
        extent=extent,
        origin="lower",
        cmap="inferno",
        vmin=vmin,
        vmax=vmax,
        alpha=0.88,
        zorder=1,
        interpolation="bilinear",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Received Power RSS (dBW)", color="#ffffff", fontsize=10)
    cbar.ax.tick_params(colors="#ffffff")

    # Static trajectory paths
    cmap_jam = plt.get_cmap("tab10")
    color_map = {jname: cmap_jam(i % 10) for i, jname in enumerate(paths_dict.keys())}

    for jname, path in paths_dict.items():
        c = color_map[jname]
        ax.plot(path[:, 0], path[:, 1], color=c, linewidth=1.5, alpha=0.4, linestyle="--", zorder=3)
        ax.scatter(path[0, 0], path[0, 1], color="#00ff66", s=30, zorder=4)
        ax.scatter(path[-1, 0], path[-1, 1], color="#ff3333", s=30, marker="s", zorder=4)

    # Dynamic elements
    trace_lines = {}
    jammer_markers = {}
    jammer_texts = {}

    for jname in paths_dict.keys():
        c = color_map[jname]
        tline, = ax.plot([], [], color=c, linewidth=2.4, alpha=0.85, zorder=5)
        trace_lines[jname] = tline

        marker, = ax.plot([], [], 'o', color=c, markeredgecolor="#ffffff", markeredgewidth=1.2, markersize=8, zorder=7)
        jammer_markers[jname] = marker

        text = ax.text(0, 0, "", color="#ffffff", fontsize=8, weight="bold", zorder=8)
        jammer_texts[jname] = text

    title_text = ax.set_title("", color="#ffffff", fontsize=11, pad=10)
    ax.set_xlabel("X (meters)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("Y (meters)", color="#aaaaaa", fontsize=9)
    ax.set_xlim(-b, b)
    ax.set_ylim(-b, b)
    ax.set_aspect("equal")
    ax.tick_params(colors="#888888")
    ax.grid(True, color="#252525", linestyle=":", alpha=0.5)

    def update(frame):
        im.set_data(rss_data[frame])

        active_count = 0
        for jname, path in paths_dict.items():
            cur_p = path[frame]
            is_active = active_masks[jname][frame]
            c = color_map[jname]

            trace_lines[jname].set_data(path[:frame+1, 0], path[:frame+1, 1])

            # Detect if jammer reached its endpoint and is stationary
            is_stationary = (frame > 0 and np.allclose(cur_p, path[frame - 1]))
            suffix = " (Stationary)" if is_stationary else ""

            if is_active:
                active_count += 1
                jammer_markers[jname].set_data([cur_p[0]], [cur_p[1]])
                jammer_markers[jname].set_color(c)
                jammer_markers[jname].set_markersize(8)
                jammer_markers[jname].set_alpha(1.0)
                jammer_texts[jname].set_position((cur_p[0] + 25, cur_p[1] + 25))
                jammer_texts[jname].set_text(f"{jname} [ON{suffix}]")
                jammer_texts[jname].set_color("#ffffff")
            else:
                jammer_markers[jname].set_data([cur_p[0]], [cur_p[1]])
                jammer_markers[jname].set_color("#555555")
                jammer_markers[jname].set_markersize(4)
                jammer_markers[jname].set_alpha(0.5)
                jammer_texts[jname].set_position((cur_p[0] + 20, cur_p[1] + 20))
                jammer_texts[jname].set_text(f"{jname} [OFF{suffix}]")
                jammer_texts[jname].set_color("#888888")

        mode_str = "INTERMITTENT" if is_intermittent else "CONTINUOUS"
        title_text.set_text(f"{combo_id} ({mode_str}) | Step {frame+1}/{total_steps}s | Active Jammers: {active_count}/{num_jammers}")

        return [im, title_text] + list(trace_lines.values()) + list(jammer_markers.values()) + list(jammer_texts.values())

    ani = FuncAnimation(fig, update, frames=total_steps, interval=1000 // fps, blit=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
    ani.save(output_filename, writer="pillow", fps=fps)
    plt.close(fig)

    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print(f"Saved animation successfully: {output_filename} ({file_size_mb:.2f} MB)")
    return output_filename


# =============================================================================
# 2. Static 6-Panel Snapshot Overview
# =============================================================================

def create_simulation_snapshots(
    combo_dir: str,
    trajectories_dir: str,
    obstacles: list = None,
    mesh_dir: str = "./data/NYC3KM_585751_4512036/mesh",
    output_image: str = None,
    map_bounds_b: float = 1500.0,
    vmin: float = -140.0,
    vmax: float = 10.0,
):
    """Renders a comprehensive 6-panel overview: Trajectories + 4 Snapshots + Temporal Coverage."""
    if obstacles is None:
        obstacles = _load_obstacles(mesh_dir)

    summary, rss_data, paths_dict, active_masks = _load_simulation_data(combo_dir, trajectories_dir)
    total_steps = rss_data.shape[0]
    combo_id = summary.get("combination_id", os.path.basename(combo_dir))
    num_jammers = len(paths_dict)

    if output_image is None:
        output_image = os.path.join(combo_dir, "simulation_snapshot_overview.png")

    print(f"\n--- Generating 6-Panel Snapshot for {combo_id} ({num_jammers} jammers) ---")
    print(f"Destination: {output_image}")

    building_patches = []
    for obs in obstacles:
        fp = obs.get("footprint")
        if fp:
            building_patches.append(Polygon(fp, closed=True))
        else:
            mn, mx = obs["min"], obs["max"]
            building_patches.append(Rectangle((mn[0], mn[1]), mx[0] - mn[0], mx[1] - mn[1]))

    fig, axes = plt.subplots(2, 3, figsize=(22, 14), dpi=160, facecolor="#141414")
    axes = axes.flatten()

    cmap_jam = plt.get_cmap("tab10")
    color_map = {jname: cmap_jam(i % 10) for i, jname in enumerate(paths_dict.keys())}
    b = map_bounds_b
    extent = [-b, b, -b, b]

    # --- Panel 1: Full Trajectories & Urban Layout ---
    ax1 = axes[0]
    ax1.set_facecolor("#181818")
    pc1 = PatchCollection(building_patches, facecolor="#333333", edgecolor="#444444", linewidth=0.2, alpha=0.85, zorder=1)
    ax1.add_collection(pc1)

    for jname, coords in paths_dict.items():
        color = color_map[jname]
        ax1.plot(coords[:, 0], coords[:, 1], color=color, linewidth=2.5, alpha=0.9, zorder=3, label=jname)
        ax1.scatter(coords[0, 0], coords[0, 1], color="#00ff66", s=50, edgecolors="#ffffff", linewidths=0.8, zorder=4)
        ax1.scatter(coords[-1, 0], coords[-1, 1], color="#ff3333", s=50, marker="s", edgecolors="#ffffff", linewidths=0.8, zorder=4)

    ax1.set_title(f"A) {combo_id}: {num_jammers} Jammer Trajectories", color="#ffffff", fontsize=12, pad=10)
    ax1.set_xlim(-b, b)
    ax1.set_ylim(-b, b)
    ax1.set_aspect("equal")
    ax1.grid(True, color="#2e2e2e", linestyle="--", alpha=0.6)
    leg = ax1.legend(loc="upper right", facecolor="#222222", edgecolor="#555555", fontsize=8)
    for t in leg.get_texts():
        t.set_color("#ffffff")
    ax1.tick_params(colors="#888888")

    # Time steps for Panels 2, 3, 4, 5
    snapshot_steps = [
        0,
        int(total_steps * 0.33),
        int(total_steps * 0.66),
        total_steps - 1,
    ]
    step_titles = [f"t = {s+1}s / {total_steps}s" for s in snapshot_steps]

    for p_idx, (step_i, stitle) in enumerate(zip(snapshot_steps, step_titles)):
        ax = axes[p_idx + 1]
        ax.set_facecolor("#181818")

        pc = PatchCollection(building_patches, facecolor="#282828", edgecolor="#3a3a3a", linewidth=0.15, alpha=0.9, zorder=2)
        ax.add_collection(pc)

        frame = rss_data[step_i]
        im = ax.imshow(
            frame,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            alpha=0.88,
            zorder=1,
            interpolation="bilinear",
        )

        for jname, coords in paths_dict.items():
            color = color_map[jname]
            is_active = active_masks[jname][step_i]
            if step_i > 0:
                ax.plot(coords[:step_i+1, 0], coords[:step_i+1, 1], color=color, linewidth=2.0, alpha=0.7, zorder=3)

            cur_p = coords[step_i]
            status_tag = "ON" if is_active else "OFF"
            pt_color = color if is_active else "#555555"
            ax.scatter(cur_p[0], cur_p[1], color=pt_color, s=70, edgecolors="#ffffff", linewidths=1.2, zorder=5)
            ax.annotate(
                f"{jname} [{status_tag}]",
                (cur_p[0], cur_p[1]),
                textcoords="offset points",
                xytext=(6, 6),
                color="#ffffff" if is_active else "#888888",
                fontsize=8,
                weight="bold",
                zorder=6,
            )

        ax.set_title(f"B{p_idx+1}) Jamming Power Field ({stitle})", color="#ffffff", fontsize=12, pad=10)
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)
        ax.set_aspect("equal")
        ax.grid(True, color="#2e2e2e", linestyle=":", alpha=0.4)
        ax.tick_params(colors="#888888")

    # --- Panel 6: Temporal Jamming Metrics ---
    ax6 = axes[5]
    ax6.set_facecolor("#181818")

    time_axis = np.arange(1, total_steps + 1)
    jammed_threshold = -90.0
    coverage_pct = np.mean(rss_data > jammed_threshold, axis=(1, 2)) * 100.0
    max_dbw = np.max(rss_data, axis=(1, 2))

    color_cov = "#00d4ff"
    color_max = "#ff9900"

    ax6.plot(time_axis, coverage_pct, color=color_cov, linewidth=2.4, label="City Area Jammed (> -90 dBW)")
    ax6.set_xlabel("Simulation Time (seconds)", color="#ffffff", fontsize=11)
    ax6.set_ylabel("Coverage (% of City Area)", color=color_cov, fontsize=11)
    ax6.tick_params(axis="y", labelcolor=color_cov)
    ax6.tick_params(axis="x", labelcolor="#ffffff")
    ax6.set_ylim(0, max(25.0, float(np.max(coverage_pct)) * 1.3))
    ax6.grid(True, color="#2e2e2e", linestyle="--", alpha=0.6)

    ax6_twin = ax6.twinx()
    ax6_twin.plot(time_axis, max_dbw, color=color_max, linewidth=1.8, linestyle="--", label="Peak Power (dBW)")
    ax6_twin.set_ylabel("Peak Received Power (dBW)", color=color_max, fontsize=11)
    ax6_twin.tick_params(axis="y", labelcolor=color_max)

    for s in snapshot_steps:
        ax6.axvline(s + 1, color="#666666", linestyle=":", linewidth=1.2, alpha=0.8)

    ax6.set_title("C) Multi-Jammer Temporal Dynamics & City Coverage", color="#ffffff", fontsize=12, pad=10)

    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    leg6 = ax6.legend(lines1 + lines2, labels1 + labels2, loc="upper right", facecolor="#222222", edgecolor="#555555", fontsize=9)
    for t in leg6.get_texts():
        t.set_color("#ffffff")

    plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.12, hspace=0.25, wspace=0.22)
    cbar_ax = fig.add_axes([0.20, 0.045, 0.40, 0.022])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Received Jamming Power RSS (dBW)", color="#ffffff", fontsize=11, labelpad=6)
    cbar.ax.tick_params(colors="#ffffff")

    fig.suptitle(
        f"NYC 3KM - Multi-Jammer Simulation: {combo_id} ({num_jammers} Jammers, {total_steps}s Duration)",
        color="#ffffff",
        fontsize=15,
        y=0.97,
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_image)), exist_ok=True)
    plt.savefig(output_image, bbox_inches="tight", dpi=160)
    plt.close(fig)

    print(f"Saved snapshot successfully: {output_image}")
    return output_image


# =============================================================================
# 3. CLI Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Visualization for Multi-Jammer Simulations (GIF & Snapshots).")
    parser.add_argument("--combo-dir", type=str, required=True, help="Path to combo folder containing combination_summary.json and rss_aggregated.npy.")
    parser.add_argument("--trajectories-dir", type=str, required=True, help="Path to single jammer trajectories directory.")
    parser.add_argument("--mesh-dir", type=str, default="./data/NYC3KM_585751_4512036/mesh", help="Mesh directory for building footprints.")
    parser.add_argument("--gif", action="store_true", help="Generate dynamic animated GIF.")
    parser.add_argument("--snapshot", action="store_true", help="Generate 6-panel static overview image.")
    parser.add_argument("--output-gif", type=str, default=None, help="Custom output path for animated GIF.")
    parser.add_argument("--output-snapshot", type=str, default=None, help="Custom output path for static snapshot.")
    parser.add_argument("--fps", type=int, default=6, help="FPS for animated GIF (default: 6).")
    args = parser.parse_args()

    # If neither flag is provided, generate GIF by default
    do_gif = args.gif or (not args.gif and not args.snapshot)
    do_snapshot = args.snapshot

    obstacles = _load_obstacles(args.mesh_dir)

    if do_gif:
        create_simulation_gif(
            combo_dir=args.combo_dir,
            trajectories_dir=args.trajectories_dir,
            obstacles=obstacles,
            output_filename=args.output_gif,
            fps=args.fps,
        )

    if do_snapshot:
        create_simulation_snapshots(
            combo_dir=args.combo_dir,
            trajectories_dir=args.trajectories_dir,
            obstacles=obstacles,
            output_image=args.output_snapshot,
        )


if __name__ == "__main__":
    main()
