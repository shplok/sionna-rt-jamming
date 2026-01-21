import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk # type: ignore
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
from typing import List, Any

from core.utils import calculate_smooth_path, check_line_of_sight
from config import WaypointConfig
from ui.theme import ModernTheme

matplotlib.use("TkAgg")

class WaypointPlannerGUI:
    def __init__(self, root: tk.Tk, engine: Any, config: WaypointConfig):
        self.root = root
        self.engine = engine
        self.config = config
        
        self.theme = ModernTheme(self.root)
        self._apply_matplotlib_theme()

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)

        self.waypoints: List[np.ndarray] = []

        # State
        self.start_pos = config.starting_position
        self.current_pos = self.start_pos
        self.is_hover_valid = True

        # Smoothing state
        self.enable_smoothing = tk.BooleanVar(value=self.config.enable_smoothing)
        
        # Navigation State (Drag & Zoom)
        self._drag_data = {"x": None, "y": None, "pressed": False, "button": None}
        
        self._setup_ui()
        self._setup_map_canvas()

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

    def get_waypoints(self) -> List[np.ndarray]:
        return self.waypoints

    def _setup_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Main Layout
        container = ttk.Frame(self.root)
        container.grid(row=0, column=0, sticky="nsew")
        
        # Top: Map
        self.map_frame = ttk.Frame(container)
        self.map_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bottom: Controls
        ctrl_frame = ttk.Frame(container, padding="15 10", style="Panel.TFrame")
        ctrl_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Instructions
        instr = "Controls: [Left Click] Add Point | [Right Click] Undo | [Wheel] Zoom | [Middle Click] Pan"
        ttk.Label(ctrl_frame, text=instr, style="Panel.TLabel").pack(side=tk.LEFT)

        right_box = ttk.Frame(ctrl_frame, style="Panel.TFrame")
        right_box.pack(side=tk.RIGHT)

        center_frame = ttk.Frame(ctrl_frame, style="Panel.TFrame")
        center_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        self.chk_smooth = ttk.Checkbutton(
            center_frame, 
            text="Enable Smoothing",
            variable=self.enable_smoothing,
            style="Card.TCheckbutton",
            command=self._update_plot
        )
        self.chk_smooth.pack(anchor="center")
        
        btn_finish = ttk.Button(ctrl_frame, text="FINISH & SAVE", command=self.finish, style="Action.TButton")
        btn_finish.pack(side=tk.RIGHT)

    def _setup_map_canvas(self):
        self.fig = plt.figure(figsize=(9, 9), dpi=100)
        self.ax = self.fig.add_subplot(111)

        self.fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.06)
        
        # Dark Theme for Plot
        plt.style.use('dark_background')
        self.fig.patch.set_facecolor(self.theme.bg_color)
        self.ax.set_facecolor(self.theme.bg_color)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.map_frame)
        toolbar.config(background=self.theme.bg_color)
        for button in toolbar.winfo_children():
            button.config(background=self.theme.bg_color)  # type: ignore
        toolbar.update()
        
        # Connect Events
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_move_and_drag)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        
        # Initial Draw
        self._draw_environment()
        
        # Dynamic Artists
        self.path_line, = self.ax.plot([], [], '-', color='#4cc2ff', lw=2, markersize=4)
        self.control_points, = self.ax.plot([], [], 'o', color='#4cc2ff', markersize=4, alpha=0.6)

        self.preview_line, = self.ax.plot([], [], '--', color='#00ff00', lw=1)
        self.error_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, color='red', fontweight='bold', va='top')

    def _draw_environment(self):
        self.ax.clear()
        self.ax.set_title(f"Waypoint Planner (Vel: {self.config.velocity} m/s)", color="white")
        self.ax.grid(True, alpha=0.2)
        self.ax.set_aspect('equal')
        
        all_x, all_y = [], []
        rects = [] # List to hold patch objects

        # 1. Gather and draw obstacles
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
            
        # Draw Start
        self.ax.plot(self.start_pos[0], self.start_pos[1], 'D', color='#00cc00', markersize=8, label="Start") # type: ignore
        all_x.append(self.start_pos[0]) # type: ignore
        all_y.append(self.start_pos[1]) # type: ignore
        
        # Auto-scale limits based on content + padding
        if all_x:
            pad = 50
            self.ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
            self.ax.set_ylim(min(all_y)-pad, max(all_y)+pad)
        else:
            self.ax.set_xlim(-200, 200)
            self.ax.set_ylim(-200, 200)

        self.path_line, = self.ax.plot([], [], '-', color='#4cc2ff', lw=2, markersize=4)
        self.control_points, = self.ax.plot([], [], 'o', color='#4cc2ff', markersize=4, alpha=0.6)
        self.preview_line, = self.ax.plot([], [], '--', color='#00ff00', lw=1)
        self.error_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes, color='red', fontweight='bold', va='top')

    # ==========================================
    # Mouse Interactions (Logic + Navigation)
    # ==========================================

    def on_close_window(self):
        if self.waypoints:
            if messagebox.askyesno("Quit", "Path data exists. Quit without saving?"):
                self.root.quit()
                self.root.destroy()
        else:
            self.root.quit()
            self.root.destroy()

    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        scale = 1/1.2 if event.button == 'up' else 1.2
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        x, y = event.xdata, event.ydata
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        rx = (xlim[1] - x) / (xlim[1] - xlim[0])
        ry = (ylim[1] - y) / (ylim[1] - ylim[0])
        self.ax.set_xlim((x - new_w * (1-rx), x + new_w * rx))
        self.ax.set_ylim((y - new_h * (1-ry), y + new_h * ry))
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax: return
        self._drag_data = {"x": event.xdata, "y": event.ydata, "pressed": True, "button": event.button}

        if event.button == 3: # Right Click
            if self.waypoints:
                self.waypoints.pop()
                self._update_plot()
            return

    def _on_release(self, event):
        if not self._drag_data["pressed"]: return
        
        is_drag = False
        if event.xdata and event.ydata:
             dist = np.hypot(event.xdata - self._drag_data["x"], event.ydata - self._drag_data["y"])
             if dist > (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]) * 0.01:
                 is_drag = True

        if event.button == 1 and not is_drag:
            if self.is_hover_valid and event.inaxes == self.ax:
                p = np.array([event.xdata, event.ydata, self.start_pos[2]]) # type: ignore
                self.waypoints.append(p)
                self._update_plot()

        self._drag_data["pressed"] = False
        self._drag_data["button"] = None

    def _on_move_and_drag(self, event):
        if event.inaxes != self.ax: return

        if self._drag_data["pressed"] and self._drag_data["button"] == 2:
            dx = event.xdata - self._drag_data["x"]
            dy = event.ydata - self._drag_data["y"]
            self.ax.set_xlim(self.ax.get_xlim() - dx)
            self.ax.set_ylim(self.ax.get_ylim() - dy)
            self.canvas.draw_idle()
            return
        
        z_height = self.start_pos[2] # type: ignore

        target_2d = np.array([event.xdata, event.ydata])
        target_3d = np.array([event.xdata, event.ydata, z_height])
        
        origin_3d = self.waypoints[-1] if self.waypoints else self.start_pos
        origin_2d = origin_3d[:2] # type: ignore

        if not check_line_of_sight(self.engine, origin_3d, target_3d): # type: ignore
            self.preview_line.set_color('red')
            self.error_text.set_text("INVALID PATH: OBSTACLE")
            self.is_hover_valid = False
        else:
            self.preview_line.set_color('#00ff00')
            self.error_text.set_text("")
            self.is_hover_valid = True
            
        self.preview_line.set_data([origin_2d[0], target_2d[0]], [origin_2d[1], target_2d[1]])
        self.canvas.draw_idle()

    def _update_plot(self):
        # Gather all points
        all_pts = [self.start_pos] + self.waypoints
        arr = np.array(all_pts)
        
        self.control_points.set_data(arr[:, 0], arr[:, 1])

        # Track collision state
        spline_collision = False

        if self.enable_smoothing.get() and len(arr) >= 3:
            smooth_arr = calculate_smooth_path(arr, resolution_per_meter=10)
            
            # CHECK VALIDITY OF SPLINE
            for p in smooth_arr:
                if not self.engine.is_position_valid(p):
                    spline_collision = True
                    break

            self.path_line.set_data(smooth_arr[:, 0], smooth_arr[:, 1])
            self.path_line.set_linestyle('-') 
        else:
            self.path_line.set_data(arr[:, 0], arr[:, 1])
            self.path_line.set_linestyle('--')

        # --- VISUAL FEEDBACK ---
        if spline_collision:
            self.path_line.set_color('#ff4444') # Red for danger
            self.path_line.set_linewidth(2.5)
            self.error_text.set_text("⚠️ SMOOTHING COLLISION")
        else:
            self.path_line.set_color('#4cc2ff') # Standard Blue
            self.path_line.set_linewidth(2)
            # Only clear error if the mouse hover isn't also causing an error
            if self.is_hover_valid: 
                self.error_text.set_text("")

        self.canvas.draw_idle()

    def finish(self):
        if not self.waypoints:
            if not messagebox.askyesno("Warning", "No waypoints defined. Use only start position?"):
                return
        
        self.config.enable_smoothing = self.enable_smoothing.get()
        
        self.root.quit()
        self.root.destroy()