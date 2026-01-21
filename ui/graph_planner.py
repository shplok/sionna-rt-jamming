import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import LineCollection, PatchCollection
from scipy.spatial import KDTree
import collections
from typing import Any
from core.utils import check_line_of_sight
from config import GraphNavConfig
from ui.theme import ModernTheme

matplotlib.use("TkAgg")

class GraphPlannerGUI:
    """
    GUI for configuring the Probabilistic Roadmap (PRM) strategy.
    """
    def __init__(self, root: tk.Tk, engine: Any, config: GraphNavConfig):
        self.root = root
        self.engine = engine
        self.config = config

        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)
        
        self.theme = ModernTheme(self.root)
        self._apply_matplotlib_theme()

        # Data State
        self.nodes = []
        self.edges = []
        self.adjacency = {}

        # Parameters Variables
        self.var_samples = tk.IntVar(value=config.num_samples)
        self.var_radius = tk.DoubleVar(value=config.max_connection_radius)
        self.var_min_dist = tk.DoubleVar(value=config.min_connection_radius) 
        self.var_smoothing = tk.BooleanVar(value=config.enable_smoothing)

        self._setup_ui()
        self._setup_map_canvas()
        self._draw_environment()

    def get_config_updates(self) -> GraphNavConfig:
        self.config.num_samples = self.var_samples.get()
        self.config.max_connection_radius = self.var_radius.get()
        self.config.min_connection_radius = self.var_min_dist.get()
        self.config.enable_smoothing = self.var_smoothing.get()
        return self.config

    def _apply_matplotlib_theme(self):
        plt.style.use('dark_background')
        params = {
            'figure.facecolor': self.theme.bg_color,
            'axes.facecolor': self.theme.bg_color,
            'axes.edgecolor': self.theme.fg_color,
            'axes.labelcolor': self.theme.fg_color,
            'xtick.color': self.theme.fg_color,
            'ytick.color': self.theme.fg_color,
            'text.color': self.theme.fg_color,
            'grid.color': "#555555"
        }
        matplotlib.rcParams.update(params)

    def _setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky="nsew")
        
        # Map Area
        self.map_frame = ttk.Frame(container)
        self.map_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls Area
        ctrl_frame = ttk.Frame(container, padding="15 10", style="Panel.TFrame")
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Left: Sliders
        params_box = ttk.Frame(ctrl_frame, style="Panel.TFrame")
        params_box.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._create_slider(params_box, "Samples:", self.var_samples, 500, 5000, 50)
        self._create_slider(params_box, "Radius (m):", self.var_radius, 1.0, 500.0, 1)
        self._create_slider(params_box, "Min Dist (m):", self.var_min_dist, 1.0, 500.0, 1)

        # Center: Actions
        action_box = ttk.Frame(ctrl_frame, style="Panel.TFrame")
        action_box.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(action_box, text="Build / Preview Graph", command=self._generate_preview_graph).pack(fill=tk.X, pady=2)
        ttk.Checkbutton(action_box, text="Enable Smoothing", variable=self.var_smoothing, style="Card.TCheckbutton").pack(fill=tk.X)

        # Right: Finish
        right_box = ttk.Frame(ctrl_frame, style="Panel.TFrame")
        right_box.pack(side=tk.RIGHT)
        
        # No target instructions needed anymore
        ttk.Button(right_box, text="CONFIRM & GENERATE", command=self.finish, style="Action.TButton").pack(pady=5)

    def _create_slider(self, parent, label, var, min_val, max_val, step=None):
        f = ttk.Frame(parent, style="Panel.TFrame")
        f.pack(fill=tk.X, pady=2)
        ttk.Label(f, text=label, width=12, style="Panel.TLabel").pack(side=tk.LEFT)
        
        def snap_to_step(val):
            v = float(val)
            rounded = round(v / step) * step if step else v
            if isinstance(var, tk.IntVar):
                var.set(int(rounded))
            else:
                var.set(rounded)

        s = ttk.Scale(
            f, from_=min_val, to=max_val, variable=var, 
            orient=tk.HORIZONTAL, style="Horizontal.TScale",
            command=snap_to_step
        )
        s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(f, textvariable=var, width=5, style="Panel.TLabel").pack(side=tk.RIGHT)

    def _setup_map_canvas(self):
        self.fig = plt.figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.05)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self.map_frame)
        toolbar.config(background=self.theme.bg_color)
        for button in toolbar.winfo_children():
            button.config(background=self.theme.bg_color) # type: ignore
        toolbar.update()

    def _draw_environment(self):
        self.ax.clear()
        self.ax.set_title(f"Preview: {len(self.nodes)} Nodes | {len(self.edges)} Edges", color="white")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_aspect('equal')

        patches = []
        for obs in self.engine.obstacles:
            if "footprint" in obs and obs["footprint"]:
                p = Polygon(obs["footprint"], closed=True)
                patches.append(p)
            else:
                # FALLBACK TO BBOX
                mn, mx = obs['min'], obs['max']
                r = Rectangle((mn[0], mn[1]), mx[0]-mn[0], mx[1]-mn[1])
                patches.append(r)
            
        if patches:
            pc = PatchCollection(patches, facecolor='#444444', alpha=0.7, edgecolor=None)
            self.ax.add_collection(pc)
            
        if self.nodes:
            nodes_arr = np.array(self.nodes)
            self.ax.scatter(nodes_arr[:,0], nodes_arr[:,1], s=5, c='#4cc2ff', alpha=0.6, zorder=5)

        if self.edges:
            lc = LineCollection(self.edges, colors='#4cc2ff', linewidths=0.5, alpha=0.3, zorder=4)
            self.ax.add_collection(lc)

        b = self.engine.bounds
        if b:
            self.ax.set_xlim(b['x'][0], b['x'][1])
            self.ax.set_ylim(b['y'][0], b['y'][1])
        else:
            self.ax.set_xlim(-500, 500)
            self.ax.set_ylim(-500, 500)

        self.canvas.draw_idle()

    def _generate_preview_graph(self):
        self.root.config(cursor="watch")
        self.root.update()
        
        num = self.var_samples.get()
        radius = self.var_radius.get()
        min_dist = self.var_min_dist.get()
        
        # --- 1. Sampling ---
        bounds = self.engine.bounds
        x_min, x_max = bounds['x']
        y_min, y_max = bounds['y']

        self.nodes = []
        attempts = 0
        
        while len(self.nodes) < num and attempts < 100:
            # Generate random batch
            candidates = np.column_stack((
                np.random.uniform(x_min, x_max, num),
                np.random.uniform(y_min, y_max, num),
                np.full(num, 1.5) # Fixed Z height!!
            ))
            
            # Filter for Environment Validity (Obstacles)
            valid_env_candidates = [p for p in candidates if self.engine.is_position_valid(p)]
            
            # If this is the very first point, add it blindly to start the tree
            if not self.nodes and valid_env_candidates:
                self.nodes.append(valid_env_candidates.pop(0))

            # If we have existing nodes, we must check distance
            if self.nodes and valid_env_candidates:
                
                # A. Create Tree of established nodes
                tree_existing = KDTree(self.nodes)
                
                # B. Temp list for nodes added IN THIS BATCH
                newly_added = []
                
                for p in valid_env_candidates:
                    if len(self.nodes) >= num: break
                    
                    # Check 1: Distance to established nodes (Fast KDTree)
                    dist, _ = tree_existing.query(p)
                    if dist < min_dist:
                        continue
                        
                    # Check 2: Distance to nodes just added in this batch (Linear Scan)
                    # This prevents two close points from sneaking in together
                    if newly_added:
                        # Vectorized check against the small list of new points
                        dists_new = np.linalg.norm(np.array(newly_added) - p, axis=1)
                        if np.any(dists_new < min_dist):
                            continue
                    
                    # If passed both, accept it
                    self.nodes.append(p)
                    newly_added.append(p)
                    
            attempts += 1
        
        # --- 2. Connecting (Adjacency + Visuals) ---
        self.edges = []
        self.adjacency = collections.defaultdict(list) 

        if len(self.nodes) > 1:
            tree = KDTree(self.nodes)
            k_neighbors = 15
            dists, indices = tree.query(self.nodes, k=k_neighbors + 1, distance_upper_bound=radius)
            
            for i, (nbr_dists, nbr_indices) in enumerate(zip(dists, indices)): #type: ignore
                p1 = self.nodes[i]
                for d, j in zip(nbr_dists, nbr_indices):
                    if np.isinf(d) or i == j or j >= len(self.nodes): continue
                    
                    if i < j: # Process pair once
                        p2 = self.nodes[j]
                        if check_line_of_sight(self.engine, p1, p2, step_size=4.0):
                            # Visuals
                            self.edges.append([(p1[0], p1[1]), (p2[0], p2[1])])
                            # Logic
                            self.adjacency[i].append(int(j))
                            self.adjacency[j].append(int(i))

        self._draw_environment()
        self.root.config(cursor="")

    def on_close_window(self):
        """Handle the user clicking the 'X' button."""
        if messagebox.askyesno("Quit", "Quit without saving graph?"):
            self.root.quit()
            self.root.destroy()

    def finish(self):

        if len(self.nodes) < 2 or not self.adjacency:
            messagebox.showerror("Error", "Graph is too small or empty.\nPlease click 'Build / Preview Graph' to generate at least 2 connected nodes.")
            return

        if self.nodes and self.adjacency:
            self.config.precomputed_nodes = self.nodes
            self.config.precomputed_adjacency = self.adjacency
            plt.close(self.fig)  # Close the matplotlib figure so it doesn't linger
            self.root.quit()
            self.root.destroy()
            self.root.update()