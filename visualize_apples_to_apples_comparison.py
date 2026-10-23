"""
visualize_apples_to_apples_comparison.py
Generates a side-by-side, 1-to-1 comparison of the exact same 2 jammers
simulated by main_no_interactive.py and main_batch_cluster.py with identical
initial positions, trajectories, and Sionna RT ray-tracing parameters.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    test_dir = "/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/test_validation_slurm"
    artifact_dir = "/pomplun/share_home/l.gonzalezgudino001/.gemini/antigravity-ide/brain/68b5f52a-7341-494d-a6b9-df393eb41cba"
    os.makedirs(artifact_dir, exist_ok=True)

    # Paths to no_interactive radio maps
    p_no_int_0 = os.path.join(test_dir, "no_interactive", "trajectories", "radio_maps", "rm_watts_traj_dur5s_vel03mps_00.npy")
    p_no_int_1 = os.path.join(test_dir, "no_interactive", "trajectories", "radio_maps", "rm_watts_traj_dur5s_vel03mps_01.npy")

    # Paths to batch_cluster radio maps (simulated on the exact same trajectories)
    p_batch_0 = os.path.join(test_dir, "batch_comparison", "trajectories", "radio_maps", "rm_watts_traj_dur5s_vel03mps_00.npy")
    p_batch_1 = os.path.join(test_dir, "batch_comparison", "trajectories", "radio_maps", "rm_watts_traj_dur5s_vel03mps_01.npy")

    # Trajectory files to extract exact starting positions
    traj0_arr = np.load(os.path.join(test_dir, "no_interactive", "trajectories", "traj_dur5s_vel03mps_00.npy"))
    traj1_arr = np.load(os.path.join(test_dir, "no_interactive", "trajectories", "traj_dur5s_vel03mps_01.npy"))

    pos0 = traj0_arr[0]
    pos1 = traj1_arr[0]

    w_no_int_0 = np.load(p_no_int_0)
    w_no_int_1 = np.load(p_no_int_1)

    w_batch_0 = np.load(p_batch_0)
    w_batch_1 = np.load(p_batch_1)

    noise_floor = 8e-15
    # Convert Frame 0 to dBW
    dbw_no_int_0 = 10.0 * np.log10(w_no_int_0[0] + noise_floor)
    dbw_batch_0 = 10.0 * np.log10(w_batch_0[0] + noise_floor)

    dbw_no_int_1 = 10.0 * np.log10(w_no_int_1[0] + noise_floor)
    dbw_batch_1 = 10.0 * np.log10(w_batch_1[0] + noise_floor)

    # Difference in Watts and dB
    diff_watts_0 = np.abs(w_batch_0[0] - w_no_int_0[0])
    diff_watts_1 = np.abs(w_batch_1[0] - w_no_int_1[0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(wspace=0.28, hspace=0.32)

    vmin, vmax = -140.0, -40.0

    # Row 1: Jammer 0
    # Col 1: no_interactive
    im0 = axes[0, 0].imshow(dbw_no_int_0, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f"main_no_interactive: Jammer #0\nStart Pos: ({pos0[0]:.1f}, {pos0[1]:.1f})", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("X (cells)")
    axes[0, 0].set_ylabel("Y (cells)")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04, label="RSS (dBW)")

    # Col 2: batch_cluster
    im1 = axes[0, 1].imshow(dbw_batch_0, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f"main_batch_cluster: Jammer #0\nStart Pos: ({pos0[0]:.1f}, {pos0[1]:.1f})", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("X (cells)")
    axes[0, 1].set_ylabel("Y (cells)")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04, label="RSS (dBW)")

    # Col 3: Difference
    im_diff0 = axes[0, 2].imshow(diff_watts_0, origin="lower", cmap="inferno")
    axes[0, 2].set_title(f"Absolute Difference |Batch - NoInt|\nMax: {np.max(diff_watts_0):.1e} W (Mean: {np.mean(diff_watts_0):.1e} W)", fontsize=11, fontweight="bold")
    axes[0, 2].set_xlabel("X (cells)")
    axes[0, 2].set_ylabel("Y (cells)")
    fig.colorbar(im_diff0, ax=axes[0, 2], fraction=0.046, pad=0.04, label="|Diff| (Watts)")

    # Row 2: Jammer 1
    # Col 1: no_interactive
    im2 = axes[1, 0].imshow(dbw_no_int_1, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f"main_no_interactive: Jammer #1\nStart Pos: ({pos1[0]:.1f}, {pos1[1]:.1f})", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("X (cells)")
    axes[1, 0].set_ylabel("Y (cells)")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04, label="RSS (dBW)")

    # Col 2: batch_cluster
    im3 = axes[1, 1].imshow(dbw_batch_1, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f"main_batch_cluster: Jammer #1\nStart Pos: ({pos1[0]:.1f}, {pos1[1]:.1f})", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("X (cells)")
    axes[1, 1].set_ylabel("Y (cells)")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04, label="RSS (dBW)")

    # Col 3: Difference
    im_diff1 = axes[1, 2].imshow(diff_watts_1, origin="lower", cmap="inferno")
    axes[1, 2].set_title(f"Absolute Difference |Batch - NoInt|\nMax: {np.max(diff_watts_1):.1e} W (Mean: {np.mean(diff_watts_1):.1e} W)", fontsize=11, fontweight="bold")
    axes[1, 2].set_xlabel("X (cells)")
    axes[1, 2].set_ylabel("Y (cells)")
    fig.colorbar(im_diff1, ax=axes[1, 2], fraction=0.046, pad=0.04, label="|Diff| (Watts)")

    fig.suptitle("1-to-1 Comparison: Identical Jammers & Initial Positions\n[main_no_interactive vs main_batch_cluster (Both on CUDA OptiX)]", fontsize=14, fontweight="bold", y=0.98)

    out_artifact = os.path.join(artifact_dir, "radio_maps_apples_to_apples.png")
    out_scratch = os.path.join(test_dir, "radio_maps_apples_to_apples.png")

    fig.savefig(out_artifact, dpi=150, bbox_inches="tight")
    fig.savefig(out_scratch, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"1-to-1 comparison successfully saved to:\n  {out_artifact}\n  {out_scratch}")

if __name__ == "__main__":
    main()
