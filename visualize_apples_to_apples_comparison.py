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

    available_tids = sorted([
        os.path.splitext(os.path.basename(f))[0].replace("rm_watts_", "")
        for f in glob.glob(os.path.join(no_int_base, "radio_maps", "rm_watts_traj_*.npy"))
    ])

    print(f"\nFound {len(available_tids)} jammers to aggregate in multi-jammer test:")
    all_positions = []
    # Load first map to get shape
    first_map = np.load(os.path.join(no_int_base, "radio_maps", f"rm_watts_{available_tids[0]}.npy"))[0]
    w_raw_multi = np.zeros_like(first_map, dtype=np.float64)
    w_cln_multi = np.zeros_like(first_map, dtype=np.float64)

    for tid in available_tids:
        raw_m = np.load(os.path.join(no_int_base, "radio_maps", f"rm_watts_{tid}.npy"))[0]
        cln_m = np.load(os.path.join(batch_base, "radio_maps", f"rm_watts_{tid}.npy"))[0]
        w_raw_multi += raw_m
        w_cln_multi += cln_m
        p_traj = os.path.join(no_int_base, f"{tid}.npy")
        if os.path.exists(p_traj):
            tarr = np.load(p_traj)
            all_positions.append((tid, tarr[0]))
        sp_c = cleaning_stats.get(tid, 0)
        print(f"  - {tid}: initial pos = ({tarr[0,0]:.1f}, {tarr[0,1]:.1f}) m | spikes cleaned = {sp_c}")

    # Map bounds to convert meters to cell coordinates
    # Grid: map_bounds = 1500m, size = 3000m, H, W = first_map.shape
    H, W = first_map.shape
    def m_to_cell(x, y):
        col = (x + 1500.0) / 3000.0 * W
        row = (y + 1500.0) / 3000.0 * H
        return col, row

    # Row 0 will be the COMBINED MULTI-JAMMER SCENARIO (All 6 jammers simultaneously)
    # Row 1 will be JAMMER #5 (deep urban canyon with 756 spikes)
    n_rows = 2
    fig = plt.figure(figsize=(24, 13))
    gs = fig.add_gridspec(n_rows, 4, width_ratios=[1, 1, 1, 0.9], wspace=0.32, hspace=0.38)

    noise_floor = 8e-15
    vmin, vmax = -140.0, -35.0
    diff_bound = 15.0

    is_cleaned = "cleaned" in args.batch_dir
    batch_title_prefix = "main_batch_cluster (Despiked)" if is_cleaned else "main_batch_cluster (Raw)"

    # -------------------------------------------------------------
    # ROW 0: MULTI-JAMMER COMBINED MAP (ALL 6 JAMMERS TRANSMITTING)
    # -------------------------------------------------------------
    dbw_raw_multi = 10.0 * np.log10(w_raw_multi + noise_floor)
    dbw_cln_multi = 10.0 * np.log10(w_cln_multi + noise_floor)
    diff_db_multi = dbw_cln_multi - dbw_raw_multi

    # Col 1: Raw Multi-Jammer
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(dbw_raw_multi, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax0.set_title(f"main_no_interactive (Raw Single-Seed)\nMULTI-JAMMER ({len(available_tids)} Active Jammers Simultaneously)", fontsize=11, fontweight="bold")
    ax0.set_xlabel("X (cells)")
    ax0.set_ylabel("Y (cells)")
    for tid, p in all_positions:
        c, r = m_to_cell(p[0], p[1])
        idx_str = tid.split("_")[-1]
        ax0.plot(c, r, "r*", markersize=8)
        ax0.text(c + 4, r + 4, f"J{idx_str}", color="white", fontsize=8, fontweight="bold")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="RSS (dBW)")

    # Col 2: Despiked Multi-Jammer
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(dbw_cln_multi, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    total_spikes = sum(cleaning_stats.values())
    ax1.set_title(f"{batch_title_prefix}\nMULTI-JAMMER ({len(available_tids)} Jammers | Total Spikes Cleaned: {total_spikes})", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (cells)")
    ax1.set_ylabel("Y (cells)")
    for tid, p in all_positions:
        c, r = m_to_cell(p[0], p[1])
        idx_str = tid.split("_")[-1]
        ax1.plot(c, r, "c*", markersize=8)
        ax1.text(c + 4, r + 4, f"J{idx_str}", color="cyan", fontsize=8, fontweight="bold")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="RSS (dBW)")

    # Col 3: Difference Map
    ax2 = fig.add_subplot(gs[0, 2])
    im2 = ax2.imshow(diff_db_multi, origin="lower", cmap="coolwarm", vmin=-diff_bound, vmax=diff_bound)
    ax2.set_title(f"Δ Difference Multi-Jammer [dB]\nMax |Δ|: {np.max(np.abs(diff_db_multi)):.1f} dB | Mean: {np.mean(np.abs(diff_db_multi)):.2f} dB", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (cells)")
    ax2.set_ylabel("Y (cells)")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Δ RSS (dB)")

    # Col 4: Zoom-in on worst multi-jammer spike
    min_idx_m = np.argmin(diff_db_multi)
    r_sp_m, c_sp_m = np.unravel_index(min_idx_m, diff_db_multi.shape)
    drop_m = diff_db_multi[r_sp_m, c_sp_m]
    half_w = 12
    r_min_m, r_max_m = max(0, r_sp_m - half_w), min(H, r_sp_m + half_w)
    c_min_m, c_max_m = max(0, c_sp_m - half_w), min(W, c_sp_m + half_w)

    ax0.add_patch(Rectangle((c_min_m, r_min_m), c_max_m - c_min_m, r_max_m - r_min_m, fill=False, edgecolor="red", linewidth=1.5, linestyle="--"))
    ax1.add_patch(Rectangle((c_min_m, r_min_m), c_max_m - c_min_m, r_max_m - r_min_m, fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--"))

    sub_gs0 = gs[0, 3].subgridspec(2, 1, hspace=0.35)
    ax_z0_r = fig.add_subplot(sub_gs0[0, 0])
    ax_z0_c = fig.add_subplot(sub_gs0[1, 0])

    im_zr0 = ax_z0_r.imshow(dbw_raw_multi[r_min_m:r_max_m, c_min_m:c_max_m], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_z0_r.set_title(f"Multi-Jammer RAW Zoom at ({r_sp_m}, {c_sp_m})\nRaw: {dbw_raw_multi[r_sp_m, c_sp_m]:.1f} dBW (Spurious Hotspot)", fontsize=9, fontweight="bold", color="darkred")
    ax_z0_r.tick_params(labelsize=8)

    im_zc0 = ax_z0_c.imshow(dbw_cln_multi[r_min_m:r_max_m, c_min_m:c_max_m], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_z0_c.set_title(f"Multi-Jammer DESPIKED Zoom\nCleaned: {dbw_cln_multi[r_sp_m, c_sp_m]:.1f} dBW ({drop_m:.1f} dB drop)", fontsize=9, fontweight="bold", color="darkblue")
    ax_z0_c.tick_params(labelsize=8)

    # -------------------------------------------------------------
    # ROW 1: SINGLE JAMMER #5 DETAIL (DEEP CANYON - 756 SPIKES)
    # -------------------------------------------------------------
    tid_j5 = "traj_dur5s_vel03mps_05"
    p_raw_j5 = os.path.join(no_int_base, "radio_maps", f"rm_watts_{tid_j5}.npy")
    p_cln_j5 = os.path.join(batch_base, "radio_maps", f"rm_watts_{tid_j5}.npy")
    p_traj_j5 = os.path.join(no_int_base, f"{tid_j5}.npy")

    w_raw_j5 = np.load(p_raw_j5)[0]
    w_cln_j5 = np.load(p_cln_j5)[0]
    pos_j5 = np.load(p_traj_j5)[0]

    dbw_raw_j5 = 10.0 * np.log10(w_raw_j5 + noise_floor)
    dbw_cln_j5 = 10.0 * np.log10(w_cln_j5 + noise_floor)
    diff_db_j5 = dbw_cln_j5 - dbw_raw_j5

    # Col 1: Raw J5
    ax_r1 = fig.add_subplot(gs[1, 0])
    im_r1 = ax_r1.imshow(dbw_raw_j5, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_r1.set_title(f"main_no_interactive (Raw Single-Seed)\nSingle Jammer #5 | Start: ({pos_j5[0]:.1f}, {pos_j5[1]:.1f}) m", fontsize=11, fontweight="bold")
    ax_r1.set_xlabel("X (cells)")
    ax_r1.set_ylabel("Y (cells)")
    c5, r5 = m_to_cell(pos_j5[0], pos_j5[1])
    ax_r1.plot(c5, r5, "r*", markersize=9)
    fig.colorbar(im_r1, ax=ax_r1, fraction=0.046, pad=0.04, label="RSS (dBW)")

    # Col 2: Despiked J5
    ax_c1 = fig.add_subplot(gs[1, 1])
    im_c1 = ax_c1.imshow(dbw_cln_j5, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_c1.set_title(f"{batch_title_prefix}\nSingle Jammer #5 | Deep Canyon: 756 Spikes Cleaned", fontsize=11, fontweight="bold")
    ax_c1.set_xlabel("X (cells)")
    ax_c1.set_ylabel("Y (cells)")
    ax_c1.plot(c5, r5, "c*", markersize=9)
    fig.colorbar(im_c1, ax=ax_c1, fraction=0.046, pad=0.04, label="RSS (dBW)")

    # Col 3: Difference Map J5
    ax_d1 = fig.add_subplot(gs[1, 2])
    im_d1 = ax_d1.imshow(diff_db_j5, origin="lower", cmap="coolwarm", vmin=-diff_bound, vmax=diff_bound)
    ax_d1.set_title(f"Δ Difference Jammer #5 [dB]\nMax |Δ|: {np.max(np.abs(diff_db_j5)):.1f} dB | Mean: {np.mean(np.abs(diff_db_j5)):.2f} dB", fontsize=11, fontweight="bold")
    ax_d1.set_xlabel("X (cells)")
    ax_d1.set_ylabel("Y (cells)")
    fig.colorbar(im_d1, ax=ax_d1, fraction=0.046, pad=0.04, label="Δ RSS (dB)")

    # Col 4: Zoom-in on J5 worst spike
    min_idx_5 = np.argmin(diff_db_j5)
    r_sp_5, c_sp_5 = np.unravel_index(min_idx_5, diff_db_j5.shape)
    drop_5 = diff_db_j5[r_sp_5, c_sp_5]
    r_min_5, r_max_5 = max(0, r_sp_5 - half_w), min(H, r_sp_5 + half_w)
    c_min_5, c_max_5 = max(0, c_sp_5 - half_w), min(W, c_sp_5 + half_w)

    ax_r1.add_patch(Rectangle((c_min_5, r_min_5), c_max_5 - c_min_5, r_max_5 - r_min_5, fill=False, edgecolor="red", linewidth=1.5, linestyle="--"))
    ax_c1.add_patch(Rectangle((c_min_5, r_min_5), c_max_5 - c_min_5, r_max_5 - r_min_5, fill=False, edgecolor="cyan", linewidth=1.5, linestyle="--"))

    sub_gs1 = gs[1, 3].subgridspec(2, 1, hspace=0.35)
    ax_z1_r = fig.add_subplot(sub_gs1[0, 0])
    ax_z1_c = fig.add_subplot(sub_gs1[1, 0])

    im_zr1 = ax_z1_r.imshow(dbw_raw_j5[r_min_5:r_max_5, c_min_5:c_max_5], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_z1_r.set_title(f"Jammer #5 RAW Zoom at ({r_sp_5}, {c_sp_5})\nRaw: {dbw_raw_j5[r_sp_5, c_sp_5]:.1f} dBW (Spurious Hotspot)", fontsize=9, fontweight="bold", color="darkred")
    ax_z1_r.tick_params(labelsize=8)

    im_zc1 = ax_z1_c.imshow(dbw_cln_j5[r_min_5:r_max_5, c_min_5:c_max_5], origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax_z1_c.set_title(f"Jammer #5 DESPIKED Zoom\nCleaned: {dbw_cln_j5[r_sp_5, c_sp_5]:.1f} dBW ({drop_5:.1f} dB drop)", fontsize=9, fontweight="bold", color="darkblue")
    ax_z1_c.tick_params(labelsize=8)

    summary_str = " | ".join([f"J{k.split('_')[-1]}: {v} sp" for k, v in sorted(cleaning_stats.items()) if "traj" in k])
    fig.suptitle(
        f"Multi-Jammer Despiking Validation: Raw Single-Seed vs 3-Seed Median + Despiking\n[Cleaning Counts: {summary_str} | Total Spikes Cleaned: {total_spikes}]",
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


