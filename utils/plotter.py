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