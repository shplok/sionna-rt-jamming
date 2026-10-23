import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from matplotlib.animation import FuncAnimation

def visualize_scene_collisions(obstacles, paths=None, title="Obstacle Validation"):
    """
    3D Plot of building obstacles and jammer paths to verify geometry.
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')

    # 1. Draw Obstacles
    for obs in obstacles:
        min_pt = obs['min']
        max_pt = obs['max']

        x = [min_pt[0], max_pt[0]]
        y = [min_pt[1], max_pt[1]]
        z = [min_pt[2], max_pt[2]]
        
        # Vertices for a rectangular prism
        verts = [
            [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[0], y[1], z[0]]], # Bottom
            [[x[0], y[0], z[1]], [x[1], y[0], z[1]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]], # Top
            [[x[0], y[0], z[0]], [x[0], y[1], z[0]], [x[0], y[1], z[1]], [x[0], y[0], z[1]]], # Left
            [[x[1], y[0], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[1], y[0], z[1]]], # Right
            [[x[0], y[0], z[0]], [x[1], y[0], z[0]], [x[1], y[0], z[1]], [x[0], y[0], z[1]]], # Front
            [[x[0], y[1], z[0]], [x[1], y[1], z[0]], [x[1], y[1], z[1]], [x[0], y[1], z[1]]], # Back
        ]
    
        poly = Poly3DCollection(verts, alpha=0.1, linewidths=1, edgecolors='gray', facecolors='cyan')
        ax.add_collection3d(poly)

    # 2. Draw Paths
    if paths:
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for i, (jammer_id, path) in enumerate(paths.items()):
            c = colors[i % len(colors)]
            # Path Line
            ax.plot(path[:,0], path[:,1], path[:,2], color=c, linewidth=2, label=jammer_id)
            # Start/End Markers
            ax.scatter(path[0,0], path[0,1], path[0,2], color=c, marker='^', s=100) # Start
            ax.scatter(path[-1,0], path[-1,1], path[-1,2], color=c, marker='x', s=100) # End

    # 3. Setup Plot Limits
    all_mins = np.array([o['min'] for o in obstacles])
    all_maxs = np.array([o['max'] for o in obstacles])
    
    if len(all_mins) > 0:
        world_min = all_mins.min(axis=0)
        world_max = all_maxs.max(axis=0)
        
        max_range = (world_max - world_min).max() / 2.0
        mid_x = (world_max[0] + world_min[0]) * 0.5
        mid_y = (world_max[1] + world_min[1]) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(0, max_range*1.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    plt.legend()
    plt.show(block=False)

def draw_buildings(ax, buildings):
    """Overlay building footprints (or bounding boxes) as grey patches."""
    for b in buildings:

        if "footprint" in b and not b['footprint'] is None and len(b["footprint"]) > 2:
            patch = patches.Polygon(
                b["footprint"],
                closed=True,
                linewidth=1,
                edgecolor='black',
                facecolor='gray',
                alpha=0.3,
                zorder=2
            )
        else:
            min_pos = b['min']
            max_pos = b['max']
            width = max_pos[0] - min_pos[0]
            height = max_pos[1] - min_pos[1]

            patch = patches.Rectangle(
                (min_pos[0], min_pos[1]), width, height,
                linewidth=1, edgecolor='black', facecolor='gray', alpha=0.3, zorder=2
            )
        ax.add_patch(patch)


def create_jammer_animation(rss_list, paths_dict, buildings, map_size, map_center, vmin=-160, vmax=0, filename="jammer_animation.gif", fps=5, cbar_label='RSS (dBm)'):
    """
    Creates a 2D GIF showing the dynamic Radio Map + Moving Jammers.

    paths_dict maps jammer id -> (T, >=2) array of positions in metres.
    """
    print("Generating Animation...")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate map extent for imshow
    extent = [
        map_center[0] - map_size[0]/2, map_center[0] + map_size[0]/2,
        map_center[1] - map_size[1]/2, map_center[1] + map_size[1]/2
    ]

    # 1. Draw Static Buildings (Gray Rectangles)
    draw_buildings(ax, buildings)

    # 2. Setup Initial RSS Image
    # We use the first frame to initialize the plot
    first_frame = rss_list[0]
    im = ax.imshow(first_frame, extent=extent, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, zorder=1) # type: ignore
    
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    # 3. Setup Jammer Markers
    jammer_markers = {}
    
    for jid, path in paths_dict.items():
        # Plot initial position
        marker, = ax.plot(path[0, 0], path[0, 1], 'o', 
                          color='white', markeredgecolor='black', markersize=6, 
                          label=jid, zorder=5)
        jammer_markers[jid] = marker
    
    ax.set_title("Jammer Simulation")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    # ax.legend(loc='upper right')

    # 4. Update Function for Animation
    def update(frame):
        # Update Heatmap
        im.set_data(rss_list[frame])
        
        # Update Jammers
        for jid, marker in jammer_markers.items():
            path = paths_dict[jid]
            # Safety check if path is shorter than simulation
            idx = min(frame, len(path) - 1)
            marker.set_data([path[idx, 0]], [path[idx, 1]])
            
        ax.set_title(f"Step {frame} | Active Jammers: {len(jammer_markers)}")
        return [im] + list(jammer_markers.values())

    # 5. Render
    ani = FuncAnimation(fig, update, frames=len(rss_list), interval=1000/fps, blit=True)
    
    # Save
    try:
        writer = 'pillow' if filename.endswith('.gif') else 'ffmpeg'
        ani.save(filename, writer=writer, fps=fps)
        print(f"Animation saved to: {filename}")
    except Exception as e:
        print(f"Failed to save animation: {e}. Try installing ffmpeg or using .gif extension.")

    plt.close(fig)


def plot_rss_panel(ax, rss, extent, buildings=None, jammers=None, vmin=-145, vmax=0, title=None):
    """
    Draw a single RSS frame in the same style as create_jammer_animation.

    rss      (H, W) array indexed [row, col]; row maps to y, col to x.
    jammers  optional (K, 2) array of ground-truth x, y in metres.
    Returns the image handle so a shared colorbar can be attached.
    """
    im = ax.imshow(rss, extent=extent, origin='lower', cmap='viridis',
                   vmin=vmin, vmax=vmax, zorder=1)

    if buildings:
        draw_buildings(ax, buildings)

    if jammers is not None and len(jammers):
        j = np.asarray(jammers, dtype=float).reshape(-1, 2)
        ax.plot(j[:, 0], j[:, 1], 'o', color='white', markeredgecolor='black',
                markersize=6, linestyle='none', zorder=5)

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    if title:
        ax.set_title(title, fontsize=9)
    return im


def plot_rss_3d(rss, sensor_idx, street_mask, grid, jammers=None, vmin=-145, vmax=0,
                title=None, cbar_label='RSS (dBW)', filename="rss_3d.png",
                elev=34, azim=-120, dpi=110):
    """
    3D view of one scenario: street plan on the floor, sensor readings above it.

    Only the cells in sensor_idx are drawn, so this shows what the detector actually
    sees rather than the dense field. Height and colour both encode RSS in dBW.

    rss          (H, W) array indexed [row, col].
    sensor_idx   flat cell indices (row * n_cells + col) of this scenario's sensors.
    street_mask  (H, W) bool, True where a cell is street (placeable).
    grid         dict with n_cells, cell_size_m, origin_m - straight from splits.json.
    jammers      optional (K, 2) ground-truth x, y in metres, drawn as vertical stems.
    """
    n = int(grid["n_cells"])
    cell = float(grid["cell_size_m"])
    x0, y0 = (float(v) for v in grid["origin_m"])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Floor: the street plan, so the readings are legible against the city layout.
    xs = x0 + (np.arange(n) + 0.5) * cell
    ys = y0 + (np.arange(n) + 0.5) * cell
    X, Y = np.meshgrid(xs, ys)
    ax.contourf(X, Y, street_mask.astype(float), levels=[-0.5, 0.5, 1.5],
                colors=['#b0b0b0', '#fafafa'], zdir='z', offset=vmin, alpha=0.9)

    rows, cols = np.divmod(np.asarray(sensor_idx, dtype=np.int64), n)
    sx = x0 + (cols + 0.5) * cell
    sy = y0 + (rows + 0.5) * cell
    sz = np.asarray(rss, dtype=np.float32)[rows, cols]

    sc = ax.scatter(sx, sy, sz, c=sz, cmap='viridis', vmin=vmin, vmax=vmax,
                    s=4, depthshade=False)

    if jammers is not None and len(jammers):
        j = np.asarray(jammers, dtype=float).reshape(-1, 2)
        for jx, jy in j:
            ax.plot([jx, jx], [jy, jy], [vmin, vmax], color='crimson',
                    linewidth=1.2, alpha=0.9, zorder=10)
        ax.scatter(j[:, 0], j[:, 1], np.full(len(j), vmax), color='white',
                   edgecolor='crimson', s=55, marker='o', depthshade=False, zorder=11)

    ax.set_xlim(x0, x0 + n * cell)
    ax.set_ylim(y0, y0 + n * cell)
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel(cbar_label)
    ax.view_init(elev=elev, azim=azim)
    if title:
        ax.set_title(title, fontsize=11)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.08)
    cbar.set_label(cbar_label)

    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    print(f"3D view saved to: {filename}")
    plt.close(fig)


def plot_rss_sheet(panels, extent, buildings=None, vmin=-145, vmax=0, ncols=4,
                   suptitle=None, cbar_label='RSS (dBW)', filename="rss_sheet.png", dpi=110):
    """
    Render a contact sheet of RSS frames, one panel per entry.

    panels is a list of dicts with keys 'rss', optional 'jammers', optional 'title'.
    Every panel shares one colour scale and one colorbar, so brightness is comparable.
    """
    n = len(panels)
    if n == 0:
        raise ValueError("plot_rss_sheet got no panels")

    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.4 * nrows),
                             squeeze=False)
    flat = axes.ravel()

    im = None
    for ax, p in zip(flat, panels):
        im = plot_rss_panel(ax, p["rss"], extent, buildings=buildings,
                            jammers=p.get("jammers"), vmin=vmin, vmax=vmax,
                            title=p.get("title"))
        ax.tick_params(labelsize=7)

    for ax in flat[n:]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)

    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    print(f"Sheet saved to: {filename}")
    plt.close(fig)

def plot_density_ladder(rss, layouts, street_mask, grid, jammers=None,
                        vmin=-145, vmax=0, cbar_label='RSS (dBW)',
                        suptitle=None, filename="density_ladder.png", dpi=130):
    """
    One jammer configuration, seen at every rung of the sensor-density ladder.

    The leftmost panel is the dense ray-traced field -- the ground truth, which no
    model ever sees. The remaining panels are the same field sampled at the sensor
    layouts actually drawn for that density, which IS the model input. Putting them
    side by side is the only way to read what the ladder costs.

    rss          (H, W) dBW, indexed [row, col].
    layouts      list of (density_pct, flat_cell_indices) ordered densest first.
    street_mask  (H, W) bool, True where a cell is street (placeable).
    grid         dict with n_cells, cell_size_m, origin_m - straight from splits.json.
    jammers      optional (K, 2) ground-truth x, y in metres.
    """
    n = int(grid["n_cells"])
    cell = float(grid["cell_size_m"])
    x0, y0 = (float(v) for v in grid["origin_m"])
    extent = [x0, x0 + n * cell, y0, y0 + n * cell]

    fig, axes = plt.subplots(1, len(layouts) + 1,
                             figsize=(3.5 * (len(layouts) + 1), 3.9), squeeze=False)
    flat = axes.ravel()

    im = flat[0].imshow(rss, extent=extent, origin='lower', cmap='viridis',
                        vmin=vmin, vmax=vmax)
    flat[0].set_title(f"dense ray-traced field\n{n*n:,} cells (never an input)",
                      fontsize=9)

    for ax, (dens, cells) in zip(flat[1:], layouts):
        # Street plan underneath, so sparse readings are legible against the city.
        ax.imshow(street_mask.astype(float), extent=extent, origin='lower',
                  cmap='Greys', vmin=0, vmax=6, zorder=0)
        rows, cols = np.divmod(np.asarray(cells, dtype=np.int64), n)
        sx = x0 + (cols + 0.5) * cell
        sy = y0 + (rows + 0.5) * cell
        # Marker area tracks 1/n so the densest rung does not saturate into a blob
        # and the sparsest stays visible.
        s = float(np.clip(6000.0 / max(len(cells), 1), 0.8, 5.0))
        ax.scatter(sx, sy, c=rss[rows, cols], cmap='viridis', vmin=vmin, vmax=vmax,
                   s=s, linewidths=0, zorder=2)
        per_rf = 289.0 * dens / 100.0
        ax.set_title(f"{dens:g}% sensors  ({len(cells):,})\n"
                     f"{per_rf:.1f} per 17x17 receptive field", fontsize=9)

    for ax in flat:
        if jammers is not None and len(jammers):
            j = np.asarray(jammers, dtype=float).reshape(-1, 2)
            ax.plot(j[:, 0], j[:, 1], 'o', color='white', markeredgecolor='black',
                    markersize=6, linestyle='none', zorder=5)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.tick_params(labelsize=7)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.015)
    cbar.set_label(cbar_label)
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    print(f"Density ladder saved to: {filename}")
    plt.close(fig)


def plot_dataset_stats(k_hist, dens_counts, n_sensors_by_dens, rf_cells=289,
                       suptitle=None, filename="dataset_stats.png", dpi=130):
    """
    The three balance/coverage facts a reader needs before trusting any result:
    K is uniform, the density rungs are equally represented, and how much evidence
    a receptive field actually contains at each rung.

    k_hist             dict K -> count
    dens_counts        dict density_pct -> count
    n_sensors_by_dens  dict density_pct -> sensors per sample
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))

    ks = sorted(k_hist)
    axes[0].bar(ks, [k_hist[k] for k in ks], color="#2a78d6", width=0.72)
    axes[0].set_xlabel("jammers per sample, K")
    axes[0].set_ylabel("samples")
    axes[0].set_title("K is stratified uniform over 0-10", fontsize=10)
    axes[0].set_xticks(ks)
    axes[0].set_ylim(0, max(k_hist.values()) * 1.18)

    ds = sorted(dens_counts)
    axes[1].bar(range(len(ds)), [dens_counts[d] for d in ds], color="#1baf7a", width=0.6)
    axes[1].set_xticks(range(len(ds)))
    axes[1].set_xticklabels([f"{d:g}%\n{n_sensors_by_dens[d]:,} sens." for d in ds],
                            fontsize=8)
    axes[1].set_xlabel("sensor density")
    axes[1].set_ylabel("samples")
    axes[1].set_title("density rungs are equally represented", fontsize=10)
    axes[1].set_ylim(0, max(dens_counts.values()) * 1.18)

    per_rf = [rf_cells * d / 100.0 for d in ds]
    bars = axes[2].bar(range(len(ds)), per_rf, color="#eb6834", width=0.6)
    axes[2].axhline(1.0, color="#52514e", linestyle="--", linewidth=1)
    axes[2].text(len(ds) - 0.45, 1.25, "1 sensor", fontsize=7, color="#52514e",
                 ha="right")
    axes[2].set_xticks(range(len(ds)))
    axes[2].set_xticklabels([f"{d:g}%" for d in ds])
    axes[2].set_xlabel("sensor density")
    axes[2].set_ylabel(f"sensors per {int(rf_cells**0.5)}x{int(rf_cells**0.5)} window")
    axes[2].set_title("evidence inside one receptive field", fontsize=10)
    axes[2].set_yscale("log")
    for b, v in zip(bars, per_rf):
        axes[2].text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:.1f}",
                     ha="center", fontsize=8)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    print(f"Dataset stats saved to: {filename}")
    plt.close(fig)
