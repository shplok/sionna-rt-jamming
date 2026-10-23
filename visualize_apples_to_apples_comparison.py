"""
visualize_apples_to_apples_comparison.py
Generates a side-by-side, 1-to-1 comparison of identical jammers
simulated by main_no_interactive.py (raw single-seed) vs main_batch_cluster.py
(with 3-seed median + spatial despiking + power cap), including a high-resolution
zoom-in on the exact cells where diffraction spikes occur.
"""

import os
import glob
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def main():
    parser = argparse.ArgumentParser(description="Compare no_interactive vs batch_cluster radio maps.")
    parser.add_argument(
        "--test-dir",
        type=str,
        default="/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/test_validation_slurm/multi_jammer",
        help="Base test directory containing no_interactive and batch subdirectories."
    )
    parser.add_argument(
        "--batch-dir",
        type=str,
        default="batch_cleaned",
        help="Subdirectory name for batch results (e.g. 'batch_cleaned' or 'batch_comparison')"
    )
    parser.add_argument(
        "--jammers",
        type=int,
        nargs="+",
        default=[0, 5],
        help="Indices of jammers to visualize in rows (e.g. 0 5)"
    )
    args = parser.parse_args()

    artifact_dir = "/pomplun/share_home/l.gonzalezgudino001/.gemini/antigravity-ide/brain/68b5f52a-7341-494d-a6b9-df393eb41cba"
    os.makedirs(artifact_dir, exist_ok=True)

    no_int_base = os.path.join(args.test_dir, "no_interactive", "trajectories")
    batch_base = os.path.join(args.test_dir, args.batch_dir, "trajectories")

    # Read cleaning statistics for all jammers
    cleaning_stats = {}
    meta_files = sorted(glob.glob(os.path.join(batch_base, "radio_maps", "rm_traj_*.json")))
    for mf in meta_files:
        try:
            with open(mf, "r", encoding="utf-8") as jf:
                md = json.load(jf)
                tid = md.get("traj_id", os.path.splitext(os.path.basename(mf))[0].replace("rm_", ""))
                cleaning_stats[tid] = md.get("cleaned_cells_total", {}).get("despike", 0)
        except Exception:
            pass

    print("\n=== Multi-Jammer Despiking Summary Across Dataset ===")
    for tid, count in cleaning_stats.items():
        print(f"  {tid}: {count} spatial spikes cleaned")

    selected_indices = args.jammers
    n_rows = len(selected_indices)

    fig = plt.figure(figsize=(24, 6.5 * n_rows))
    # We use GridSpec: 3 main columns + 1 column split into 2 rows for the zoom-in
    gs = fig.add_gridspec(n_rows, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.32, hspace=0.35)

    noise_floor = 8e-15
    vmin, vmax = -140.0, -40.0
    diff_bound = 15.0

    is_cleaned = "cleaned" in args.batch_dir
    batch_title_prefix = "main_batch_cluster (Despiked)" if is_cleaned else "main_batch_cluster (Raw)"

    for row_idx, j_idx in enumerate(selected_indices):
        tid = f"traj_dur5s_vel03mps_{j_idx:02d}"
        p_raw = os.path.join(no_int_base, "radio_maps", f"rm_watts_{tid}.npy")
        p_cln = os.path.join(batch_base, "radio_maps", f"rm_watts_{tid}.npy")
        p_traj = os.path.join(no_int_base, f"{tid}.npy")

        if not (os.path.exists(p_raw) and os.path.exists(p_cln)):
            print(f"Warning: Missing files for {tid}, skipping.")
            continue

        traj_arr = np.load(p_traj)
        pos0 = traj_arr[0]

        w_raw = np.load(p_raw)[0]
        w_cln = np.load(p_cln)[0]

        dbw_raw = 10.0 * np.log10(w_raw + noise_floor)
        dbw_cln = 10.0 * np.log10(w_cln + noise_floor)
        diff_db = dbw_cln - dbw_raw

        spikes_count = cleaning_stats.get(tid, int(np.sum(np.abs(diff_db) > 20.0)))

        # Col 1: Raw
        ax_raw = fig.add_subplot(gs[row_idx, 0])
        im_raw = ax_raw.imshow(dbw_raw, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_raw.set_title(f"main_no_interactive (Raw Single-Seed)\nJammer #{j_idx} | Start: ({pos0[0]:.1f}, {pos0[1]:.1f}) m", fontsize=11, fontweight="bold")
        ax_raw.set_xlabel("X (cells)")
        ax_raw.set_ylabel("Y (cells)")
        fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04, label="RSS (dBW)")

        # Col 2: Despiked
        ax_cln = fig.add_subplot(gs[row_idx, 1])
        im_cln = ax_cln.imshow(dbw_cln, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_cln.set_title(f"{batch_title_prefix}\nJammer #{j_idx} | Despiked: {spikes_count} spikes", fontsize=11, fontweight="bold")
        ax_cln.set_xlabel("X (cells)")
        ax_cln.set_ylabel("Y (cells)")
        fig.colorbar(im_cln, ax=ax_cln, fraction=0.046, pad=0.04, label="RSS (dBW)")

        # Col 3: Difference Map
        ax_diff = fig.add_subplot(gs[row_idx, 2])
        im_diff = ax_diff.imshow(diff_db, origin="lower", cmap="coolwarm", vmin=-diff_bound, vmax=diff_bound)
        max_d = np.max(np.abs(diff_db))
        mean_d = np.mean(np.abs(diff_db))
        ax_diff.set_title(f"Δ Difference (Batch - Raw) [dB]\nMax |Δ|: {max_d:.1f} dB | Mean: {mean_d:.2f} dB", fontsize=11, fontweight="bold")
        ax_diff.set_xlabel("X (cells)")
        ax_diff.set_ylabel("Y (cells)")
        fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04, label="Δ RSS (dB)")

        # Locate maximum spike where Raw had anomalous power over Cleaned
        # Find minimum diff_db (where raw was much larger than cleaned)
        min_idx = np.argmin(diff_db)
        r_sp, c_sp = np.unravel_index(min_idx, diff_db.shape)
        drop_val = diff_db[r_sp, c_sp]

        # Draw a highlight box on Col 1 and Col 2 around this spike
        half_w = 12
        r_min, r_max = max(0, r_sp - half_w), min(diff_db.shape[0], r_sp + half_w)
        c_min, c_max = max(0, c_sp - half_w), min(diff_db.shape[1], c_sp + half_w)

        rect1 = Rectangle((c_min, r_min), c_max - c_min, r_max - r_min, fill=False, edgecolor="red", linewidth=1.5, linestyle="--")
        rect2 = Rectangle((c_min, r_min), c_max - c_min, r_max - r_min, fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--")
        ax_raw.add_patch(rect1)
        ax_cln.add_patch(rect2)

        # Col 4: Zoom-in Subplots (Top: Raw zoom, Bottom: Cleaned zoom)
        sub_gs = gs[row_idx, 3].subgridspec(2, 1, hspace=0.35)
        ax_zoom_raw = fig.add_subplot(sub_gs[0, 0])
        ax_zoom_cln = fig.add_subplot(sub_gs[1, 0])

        patch_raw = dbw_raw[r_min:r_max, c_min:c_max]
        patch_cln = dbw_cln[r_min:r_max, c_min:c_max]

        im_zr = ax_zoom_raw.imshow(patch_raw, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_zoom_raw.set_title(f"Zoom-In RAW: Spike at ({r_sp}, {c_sp})\nRaw: {dbw_raw[r_sp, c_sp]:.1f} dBW (Spurious Hotspot)", fontsize=9, fontweight="bold", color="darkred")
        ax_zoom_raw.tick_params(labelsize=8)

        im_zc = ax_zoom_cln.imshow(patch_cln, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax_zoom_cln.set_title(f"Zoom-In DESPIKED: Cell ({r_sp}, {c_sp})\nCleaned: {dbw_cln[r_sp, c_sp]:.1f} dBW ({drop_val:.1f} dB drop)", fontsize=9, fontweight="bold", color="darkblue")
        ax_zoom_cln.tick_params(labelsize=8)

    summary_str = " | ".join([f"J{k}: {v} spikes" for k, v in sorted(cleaning_stats.items()) if "traj" in k])
    fig.suptitle(
        f"Multi-Jammer Despiking Validation: Raw Single-Seed vs 3-Seed Median + Despiking\n[Cleaning Counts: {summary_str}]",
        fontsize=14,
        fontweight="bold",
        y=0.99
    )

    out_artifact = os.path.join(artifact_dir, "radio_maps_apples_to_apples_despiked.png")
    out_scratch = os.path.join(args.test_dir, "radio_maps_apples_to_apples_despiked.png")
    out_cwd = os.path.join(os.getcwd(), "radio_maps_apples_to_apples_despiked.png")

    fig.savefig(out_artifact, dpi=150, bbox_inches="tight")
    fig.savefig(out_scratch, dpi=150, bbox_inches="tight")
    fig.savefig(out_cwd, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nMulti-jammer comparison successfully saved to:\n  {out_cwd}\n  {out_scratch}\n  {out_artifact}")

if __name__ == "__main__":
    main()


