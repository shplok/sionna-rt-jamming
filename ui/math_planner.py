import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict, Any, Optional

# Matplotlib integration
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # type: ignore
from matplotlib.patches import Rectangle, Arrow
from matplotlib.collections import PatchCollection

from config import MathModelingConfig, MathSegment
from ui.theme import ModernTheme 

matplotlib.use("TkAgg")
matplotlib.rcParams['axes.unicode_minus'] = False

class MathPlannerGUI:
    """
    Interactive GUI for designing a kinematic path segment by segment.
    Output: A list of MathSegment objects (via self.get_segments()).
    """
    
    def __init__(self, root: tk.Tk, engine: Any, config: MathModelingConfig):
        self.root = root
        self.engine = engine
        self.config = config
        self.dt = config.time_step
        
        # Apply Styling
        self.theme = ModernTheme(self.root)
        self._apply_matplotlib_theme()

        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)
        
        # --- Kinematic State (Current "Head" of the path) ---
        self.start_pos = config.starting_position.copy()
        
        # Current state trackers
        self.curr_pos = self.start_pos.copy()
        self.curr_heading = 0.0 # Will be set by UI
        self.curr_vel = 0.0
        
        # Data Storage
        self.saved_segments: List[MathSegment] = []  # The output
        self.visual_path_history: List[np.ndarray] = [] # For drawing "Committed" lines
        
        # Interaction state
        self._drag_data = {"x": None, "y": None, "pressed": False}

        # --- Build UI ---
        self._setup_layout()
        self._setup_map_canvas()
        self._setup_controls_sidebar()
        
        # Initialize
        self.update_preview()

    def get_segments(self) -> List[MathSegment]:
        """Public accessor for the Controller to retrieve the work done."""
        return self.saved_segments

    # ==========================================
    # UI Setup & Styling
    # ==========================================

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

    def _setup_layout(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.map_frame = ttk.Frame(self.root, padding="0")
        self.map_frame.grid(row=0, column=0, sticky="nsew")

        self.side_frame = ttk.Frame(self.root, padding="15", style="Panel.TFrame")
        self.side_frame.grid(row=0, column=1, sticky="nsew")

    def _setup_map_canvas(self):
        self.fig = plt.figure(figsize=(9, 9), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.map_frame)
        toolbar.config(background=self.theme.bg_color)
        for button in toolbar.winfo_children():
            button.config(background=self.theme.bg_color)  # type: ignore
        toolbar.update()

        # Events
        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag)

        self._draw_static_environment()
        
        # Dynamic Actors
        self.preview_line, = self.ax.plot([], [], '--', lw=2, label='Preview', color='#00ff00')
        self.committed_line, = self.ax.plot([], [], '-', lw=2, label='Path', color='#4cc2ff')
        self.current_marker, = self.ax.plot([], [], 'o', color='#4cc2ff', markersize=6, zorder=10)
        self.heading_arrow = None 
        
        self.collision_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, 
            color='#ff4444', fontsize=12, fontweight='bold', va='top'
        )
        self.ax.legend(loc='lower right', facecolor=self.theme.panel_color, edgecolor=self.theme.fg_color)

    def _draw_static_environment(self):
        self.ax.set_title("Mission Planner", color=self.theme.fg_color)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal', adjustable='datalim')

        rects = []
        all_x, all_y = [], []

        if self.engine.obstacles:
            for obs in self.engine.obstacles:
                min_pt, max_pt = obs['min'], obs['max']
                               
                rects.append(Rectangle((min_pt[0], min_pt[1]), max_pt[0]-min_pt[0], max_pt[1]-min_pt[1]))

                all_x.extend([min_pt[0], max_pt[0]])
                all_y.extend([min_pt[1], max_pt[1]])
            
        if rects:
            pc = PatchCollection(rects, facecolor='#444444', alpha=0.6, edgecolor='#555555')
            self.ax.add_collection(pc)
        
        if all_x:
            pad = 50
            self.ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
            self.ax.set_ylim(min(all_y)-pad, max(all_y)+pad)
        else:
            self.ax.set_xlim(-200, 200)
            self.ax.set_ylim(-200, 200)
            
        self.ax.plot(self.start_pos[0], self.start_pos[1], 'o', color='#00cc00', markersize=8, label='Start')

    # ==========================================
    # Controls & Widgets
    # ==========================================

    def _setup_controls_sidebar(self):
        ttk.Label(self.side_frame, text="Segment Controls", style="Panel.TLabel", font=("Helvetica", 16, "bold")).pack(pady=(0, 15))

        # Mode Tabs
        self.notebook = ttk.Notebook(self.side_frame)
        self.notebook.pack(fill=tk.X, pady=5)
        
        self.tab_vel = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.tab_acc = ttk.Frame(self.notebook, style="Panel.TFrame")
        self.tab_turn = ttk.Frame(self.notebook, style="Panel.TFrame")
        
        self.notebook.add(self.tab_vel, text="Const Velocity")
        self.notebook.add(self.tab_acc, text="Const Accel")
        self.notebook.add(self.tab_turn, text="Coord Turn")
        
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

        # Variables
        self.var_duration = tk.DoubleVar(value=1.0)
        self.var_velocity = tk.DoubleVar(value=0.0)
        self.var_accel = tk.DoubleVar(value=0.0)
        self.var_turn_rate = tk.DoubleVar(value=0.0)
        self.var_start_heading = tk.DoubleVar(value=0.0) # Initial heading

        # Sliders
        self.controls_container = ttk.LabelFrame(self.side_frame, text="Parameters", style="Panel.TLabelframe", padding="10")
        self.controls_container.pack(fill=tk.X, pady=15)
        
        self.slider_widgets = {}
        self.slider_widgets['start_heading'] = self._create_compound_slider("Initial Heading (°)", self.var_start_heading, -180, 180)
        self.slider_widgets['duration'] = self._create_compound_slider("Duration (s)", self.var_duration, 0.0, 60.0, is_duration=True)
        self.slider_widgets['velocity'] = self._create_compound_slider("Velocity (m/s)", self.var_velocity, -25.0, 25.0)
        self.slider_widgets['accel'] = self._create_compound_slider("Accel (m/s²)", self.var_accel, -25.0, 25.0)
        self.slider_widgets['turn'] = self._create_compound_slider("Turn Rate (°/s)", self.var_turn_rate, -90.0, 90.0)

        # Buttons
        btn_frame = ttk.Frame(self.side_frame, style="Panel.TFrame")
        btn_frame.pack(fill=tk.X, pady=15)
        ttk.Button(btn_frame, text="Add Segment", command=self.add_segment).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_frame, text="Undo", command=self.undo_segment).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # List
        ttk.Label(self.side_frame, text="History:", style="Panel.TLabel", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(10, 5))
        self.tree = ttk.Treeview(self.side_frame, columns=("id", "mode", "dur"), show="headings", height=8)
        self.tree.heading("id", text="#"); self.tree.column("id", width=30)
        self.tree.heading("mode", text="Mode"); self.tree.column("mode", width=80)
        self.tree.heading("dur", text="Time (s)"); self.tree.column("dur", width=60)
        self.tree.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.side_frame, text="FINISH & SAVE", command=self.finish).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

    def _create_compound_slider(self, label_text, variable, min_val, max_val, is_duration=False):
        frame = ttk.Frame(self.controls_container, style="Panel.TFrame")
        
        # Step logic for discrete time steps
        step = 0.5
        if is_duration and not self.config.variable_duration:
            step = self.dt
            if min_val < self.dt: min_val = self.dt

        def on_change(*args):
            val = variable.get()
            # Snapping logic for duration
            if is_duration and not self.config.variable_duration:
                steps = round(val / self.dt)
                if steps < 1: steps = 1
                snapped = steps * self.dt
                if abs(val - snapped) > 1e-5:
                    variable.set(snapped)
                    return
            self.update_preview()

        top = ttk.Frame(frame, style="Panel.TFrame")
        top.pack(fill=tk.X)
        ttk.Label(top, text=label_text, style="Panel.TLabel").pack(side=tk.LEFT)
        
        spin = tk.Spinbox(
            top, from_=min_val, to=max_val, increment=step, format="%.2f",
            textvariable=variable, width=8, font=self.theme.main_font,
            bg=self.theme.entry_bg, fg=self.theme.fg_color, 
            buttonbackground=self.theme.panel_color, relief="flat",
            command=on_change
        )
        spin.pack(side=tk.RIGHT, ipady=6)
        spin.bind('<Return>', lambda e: on_change())
        
        scale = ttk.Scale(
            frame, from_=min_val, to=max_val, variable=variable, 
            orient=tk.HORIZONTAL, style="Horizontal.TScale",
            command=lambda v: on_change()
        )
        scale.pack(fill=tk.X, pady=(2, 10))
        return frame

    # ==========================================
    # Logic & Math (Lightweight Visualization)
    # ==========================================

    def on_tab_changed(self, event):
        # Manage visibility of sliders
        for w in self.slider_widgets.values(): w.pack_forget()
        
        self.slider_widgets['duration'].pack(fill=tk.X)
        
        # Only show start heading if it's the very first segment
        if len(self.saved_segments) == 0:
            self.slider_widgets['start_heading'].pack(fill=tk.X)

        idx = self.notebook.index(self.notebook.select())
        if idx == 0: self.slider_widgets['velocity'].pack(fill=tk.X)     # Vel
        elif idx == 1: self.slider_widgets['accel'].pack(fill=tk.X)      # Accel
        elif idx == 2:                                                   # Turn
            self.slider_widgets['velocity'].pack(fill=tk.X)
            self.slider_widgets['turn'].pack(fill=tk.X)
        
        self.update_preview()

    def _calculate_preview(self):
        """
        Calculates the visual path for the current potential segment.
        Returns: (path_array_for_plotting, next_pos, next_vel, next_heading, mode_string)
        """
        # 1. Update initial heading if this is the first segment
        if len(self.saved_segments) == 0:
            self.curr_heading = np.deg2rad(self.var_start_heading.get())

        tab_idx = self.notebook.index(self.notebook.select())
        duration = self.var_duration.get()
        
        # Time array for plotting
        t = np.linspace(0, duration, int(max(10, duration*5))) 
        
        x0, y0 = self.curr_pos[0], self.curr_pos[1]
        z = self.curr_pos[2]
        theta = self.curr_heading
        v_start = self.curr_vel
        
        mode = "Unknown"
        final_v = v_start
        final_theta = theta
        
        # --- Physics ---
        if tab_idx == 0: # Const Vel
            v = self.var_velocity.get()
            x = x0 + v * np.cos(theta) * t
            y = y0 + v * np.sin(theta) * t
            final_v = v
            mode = "Const Vel"
            
        elif tab_idx == 1: # Const Accel
            a = self.var_accel.get()
            dist = v_start * t + 0.5 * a * t**2
            x = x0 + dist * np.cos(theta)
            y = y0 + dist * np.sin(theta)
            final_v = v_start + a * duration
            mode = "Const Accel"
            
        elif tab_idx == 2: # Turn
            v = self.var_velocity.get()
            omega = np.deg2rad(self.var_turn_rate.get())
            if abs(omega) < 1e-4:
                x = x0 + v * np.cos(theta) * t
                y = y0 + v * np.sin(theta) * t
            else:
                r = v / omega
                x = x0 + r * (np.sin(omega * t + theta) - np.sin(theta))
                y = y0 - r * (np.cos(omega * t + theta) - np.cos(theta))
                final_theta = theta + omega * duration
            final_v = v
            mode = "Turn"

        else:
            raise ValueError("Invalid tab index for mode selection.")

        # Final calculated position for the next segment
        next_pos = np.array([x[-1], y[-1], z])
        path_plot = np.stack((x, y, np.full_like(x, z)), axis=1)
        
        return path_plot, next_pos, final_v, final_theta, mode

    def update_preview(self):
        path, next_pos, _, next_heading, _ = self._calculate_preview()
        
        self.preview_line.set_data(path[:,0], path[:,1])
        
        # Collision Check
        collision = False
        for point in path:
            if not self.engine.is_position_valid(point):
                collision = True
                break
        
        if collision:
            self.preview_line.set_color('#ff4444')
            self.collision_text.set_text("⚠️ COLLISION PREDICTED")
        else:
            self.preview_line.set_color('#00ff00')
            self.collision_text.set_text("")
            
        # Draw Arrow
        if self.heading_arrow: self.heading_arrow.remove()
        arrow_len = 20
        dx = arrow_len * np.cos(next_heading)
        dy = arrow_len * np.sin(next_heading)
        # Arrow starts at the end of the preview
        self.heading_arrow = Arrow(next_pos[0], next_pos[1], dx, dy, width=5, color='orange')
        self.ax.add_patch(self.heading_arrow)
        
        self.canvas.draw_idle()

    def add_segment(self):
        # 1. Get calculations
        path_plot, next_pos, next_vel, next_heading, mode = self._calculate_preview()
        
        # 2. Create the Data Object (The strict output)
        # Note: We store the START parameters of the segment
        seg_params = {}
        if mode == "Const Vel": seg_params = {'velocity': self.var_velocity.get()}
        elif mode == "Const Accel": seg_params = {'accel': self.var_accel.get()}
        elif mode == "Turn": seg_params = {'velocity': self.var_velocity.get(), 'turn_rate': self.var_turn_rate.get()}
        
        new_segment = MathSegment(
            mode=mode,
            duration=self.var_duration.get(),
            start_pos=self.curr_pos.copy(),
            start_heading=self.curr_heading,
            start_vel=self.curr_vel,
            params=seg_params
        )
        
        # 3. Save
        self.saved_segments.append(new_segment)
        self.visual_path_history.append(path_plot)
        
        # 4. Advance State
        self.curr_pos = next_pos
        self.curr_vel = next_vel
        self.curr_heading = next_heading
        
        # 5. UI Updates
        self.tree.insert("", tk.END, values=(len(self.saved_segments), mode, f"{new_segment.duration:.1f}s"))
        self._refresh_committed_line()
        self.on_tab_changed(None) # Hides start heading slider if needed

    def undo_segment(self):
        if not self.saved_segments: return
        
        removed = self.saved_segments.pop()
        self.visual_path_history.pop()
        
        # Revert state to the start of the removed segment
        self.curr_pos = removed.start_pos
        self.curr_heading = removed.start_heading
        self.curr_vel = removed.start_vel
        
        self.tree.delete(self.tree.get_children()[-1])
        self._refresh_committed_line()
        self.on_tab_changed(None)

    def _refresh_committed_line(self):
        if self.visual_path_history:
            full = np.vstack(self.visual_path_history)
            self.committed_line.set_data(full[:,0], full[:,1])
            self.current_marker.set_data([self.curr_pos[0]], [self.curr_pos[1]])
        else:
            self.committed_line.set_data([], [])
            self.current_marker.set_data([self.start_pos[0]], [self.start_pos[1]])
        self.update_preview()

    # ==========================================
    # Lifecycle
    # ==========================================

    def on_close_window(self):
        if self.saved_segments:
            if messagebox.askyesno("Quit", "Path data exists. Quit without saving?"):
                self.root.quit()
                self.root.destroy()
        else:
            self.root.quit()
            self.root.destroy()

    def finish(self):
        if not self.saved_segments:
            if not messagebox.askyesno("Warning", "No path created. Return empty path?"):
                return
        self.root.quit()
        self.root.destroy()
    
    # ... Matplotlib Scroll/Pan handlers ...
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
        if event.inaxes != self.ax or self.fig.canvas.toolbar.mode != "": return # type: ignore
        self._drag_data = {"x": event.xdata, "y": event.ydata, "pressed": True}

    def _on_release(self, event):
        self._drag_data["pressed"] = False

    def _on_drag(self, event):
        if not self._drag_data["pressed"] or event.inaxes != self.ax: return
        dx, dy = event.xdata - self._drag_data["x"], event.ydata - self._drag_data["y"]
        self.ax.set_xlim(self.ax.get_xlim() - dx)
        self.ax.set_ylim(self.ax.get_ylim() - dy)
        self.canvas.draw_idle()