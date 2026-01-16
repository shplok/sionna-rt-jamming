import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle
import os
import glob
import random
import sys
from utils.scene_objects import gather_bboxes


# Use TkAgg for embedding in Tkinter
matplotlib.use("TkAgg")

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_FOLDER = "./datasets/dataset_test"
MESHES_PATH = r"/home/luisg-ubuntu/sionna_rt_jamming/data/downtown_chicago_luis/meshes"
NUM_PATHS_TO_PLOT = 50 

# ==========================================
# THEME COLORS
# ==========================================
COLORS = {
    "bg": "#1e1e1e",
    "panel": "#2d2d2d",
    "fg": "#e0e0e0",
    "accent": "#0078d7",
    "plot_line": "#4cc2ff",
    "plot_bg": "#1e1e1e",
    "grid": "#444444",
    "obstacle": "#444444", 
    "obstacle_edge": "#555555"
}

class ResultViewerApp:
    def __init__(self, root, folder, meshes_path, k_paths):
        self.root = root
        self.root.title(f"Simulation Viewer - {k_paths} Paths")
        self.root.geometry("1100x900")
        self.root.configure(bg=COLORS["bg"])

        # 1. Handle Exit Protocol
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # 2. Data State
        self.folder = folder
        self.k_paths = k_paths
        self.paths = []
        
        print(f"Loading buildings from: {meshes_path}")
        self.obstacles = gather_bboxes(meshes_path)
        print(f"Loaded {len(self.obstacles)} buildings.")

        self._drag_data = {"x": None, "y": None, "pressed": False}

        # 3. Setup UI
        self._setup_styles()
        self._load_data()
        self._setup_canvas()
        
        # 4. Initial Plot
        self.plot_all()

    def on_close(self):
        """Properly kills the app when window is closed."""
        self.root.quit()
        self.root.destroy()
        sys.exit()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background=COLORS["bg"], foreground=COLORS["fg"])
        style.configure("TFrame", background=COLORS["panel"])
        style.configure("TButton", background=COLORS["accent"], foreground="white", borderwidth=0, font=("Segoe UI", 10, "bold"))
        style.map("TButton", background=[('active', '#198ce6')])

    def _load_data(self):
        pattern = os.path.join(self.folder, "*.npy")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No .npy files found in {self.folder}")
            return

        selected_files = random.sample(files, min(self.k_paths, len(files)))
        print(f"Loading {len(selected_files)} paths...")
        
        self.paths = []
        for f in selected_files:
            try:
                arr = np.load(f)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    self.paths.append(arr)
            except Exception as e:
                print(f"Error loading {f}: {e}")

    def _setup_canvas(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        header = ttk.Frame(main_frame)
        header.pack(fill=tk.X, pady=(0, 10))
        
        info_text = f"Paths: {len(self.paths)} | Buildings: {len(self.obstacles)}"
        lbl = ttk.Label(header, text=info_text, background=COLORS["panel"], foreground=COLORS["fg"])
        lbl.pack(side=tk.LEFT, padx=10)
        
        instr = ttk.Label(header, text=" [Left Click+Drag: Pan] [Scroll: Zoom]", 
                          font=("Segoe UI", 9, "italic"), foreground="#888888", background=COLORS["panel"])
        instr.pack(side=tk.LEFT, padx=20)
        
        btn = ttk.Button(header, text="Reload Random Sample", command=self._reload)
        btn.pack(side=tk.RIGHT)

        plt.style.use('dark_background')
        # Optimized: disable tight_layout auto-adjustments during interaction
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.fig.patch.set_facecolor(COLORS["bg"])
        self.ax.set_facecolor(COLORS["bg"])
        self.ax.tick_params(colors=COLORS["fg"], labelcolor=COLORS["fg"])
        for spine in self.ax.spines.values():
            spine.set_edgecolor(COLORS["fg"])

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bind Interaction Events
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("motion_notify_event", self._on_drag)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

    def _reload(self):
        self._load_data()
        self.plot_all()

    def plot_all(self):
        self.ax.clear()
        self.ax.grid(True, color=COLORS["grid"], alpha=0.3)
        self.ax.set_aspect('equal')

        # --- A. PLOT BUILDINGS (Optimized) ---
        # Using list comprehension is slightly faster than a for loop with append
        rects = [
            Rectangle(
                (obs['min'][0], obs['min'][1]), 
                obs['max'][0] - obs['min'][0], 
                obs['max'][1] - obs['min'][1]
            ) for obs in self.obstacles
        ]
            
        if rects:
            pc = PatchCollection(
                rects, 
                facecolor=COLORS["obstacle"], 
                edgecolor=COLORS["obstacle_edge"], 
                alpha=0.8,
                zorder=1
            )
            self.ax.add_collection(pc)

        # --- B. PLOT PATHS (HIGHLY OPTIMIZED) ---        
        # We prepare a list of (N, 2) arrays. LineCollection handles this efficiently.
        lines = [p[:, :2] for p in self.paths]
        
        if lines:
            lc = LineCollection(
                lines, 
                colors=COLORS["plot_line"], 
                linewidths=1.5, # Slightly thicker for visibility
                alpha=0.7, 
                zorder=2
            )
            self.ax.add_collection(lc)

            # Start/End points
            starts = np.array([p[0, :2] for p in self.paths])
            ends = np.array([p[-1, :2] for p in self.paths])
            self.ax.scatter(starts[:,0], starts[:,1], c='#00ff00', s=20, zorder=3, alpha=0.9, label="Start")
            self.ax.scatter(ends[:,0], ends[:,1], c='#ff0055', s=20, zorder=3, alpha=0.9, label="End")

        # --- C. AUTO SCALE ---
        all_x, all_y = [], []
        if self.paths:
            # Quick bounds check without flattening everything
            for p in self.paths:
                all_x.append(np.min(p[:, 0]))
                all_x.append(np.max(p[:, 0]))
                all_y.append(np.min(p[:, 1]))
                all_y.append(np.max(p[:, 1]))
            
            pad = 100
            self.ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
            self.ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
        else:
            self.ax.set_xlim(-500, 500)
            self.ax.set_ylim(-500, 500)

        self.canvas.draw()

    # ==========================================
    # INTERACTION LOGIC
    # ==========================================
    
    def _on_press(self, event):
        if event.button == 1 and event.inaxes == self.ax:
            self._drag_data = {"x": event.xdata, "y": event.ydata, "pressed": True}

    def _on_drag(self, event):
        if self._drag_data["pressed"] and event.inaxes == self.ax:
            # Calculate delta
            dx = event.xdata - self._drag_data["x"]
            dy = event.ydata - self._drag_data["y"]
            
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Update limits
            self.ax.set_xlim(xlim - dx)
            self.ax.set_ylim(ylim - dy)
            
            self.canvas.draw_idle()

    def _on_release(self, event):
        self._drag_data["pressed"] = False

    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        
        base_scale = 1.3
        scale_factor = 1 / base_scale if event.button == 'up' else base_scale
        
        cur_xlim, cur_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        
        new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
        new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
        
        relx = (cur_xlim[1] - xdata) / (cur_xlim[1] - cur_xlim[0])
        rely = (cur_ylim[1] - ydata) / (cur_ylim[1] - cur_ylim[0])
        
        self.ax.set_xlim((xdata - new_width * (1 - relx), xdata + new_width * relx))
        self.ax.set_ylim((ydata - new_height * (1 - rely), ydata + new_height * rely))
        
        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = ResultViewerApp(root, RESULTS_FOLDER, MESHES_PATH, NUM_PATHS_TO_PLOT)
    root.mainloop()