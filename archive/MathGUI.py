import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

# Matplotlib integration
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle, Arrow

# Import shared theme
from ui_theme import ModernTheme

matplotlib.use("TkAgg")
matplotlib.rcParams['axes.unicode_minus'] = False

class MathStrategyGUI:
    def __init__(self, root, obstacles, config):
        self.root = root
        self.obstacles = obstacles
        self.config = config
        self.dt = config.time_step
        
        # Apply Theme
        self.theme = ModernTheme(self.root)
        self._apply_matplotlib_theme()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close_window)
        
        # --- Kinematic State ---
        self.start_pos = config.starting_position.copy()
        self.current_pos = self.start_pos.copy()
        self.current_heading = np.deg2rad(config.initial_heading)
        self.current_velocity = 0.0
        
        self.final_path_segments = []
        self.segments_data = []
        self.state_vectors = []
        
        self._drag_data = {"x": None, "y": None, "pressed": False}

        # --- GUI Layout ---
        self._setup_layout()
        self._setup_map_canvas()
        self._setup_controls_sidebar()
        
        self.update_preview()

    def _apply_matplotlib_theme(self):
        """Configures Matplotlib to match the ModernTheme (Dark Mode)."""
        plt.style.use('dark_background')
        # We also manually override specific colors to match our exact palette
        matplotlib.rcParams['figure.facecolor'] = self.theme.bg_color
        matplotlib.rcParams['axes.facecolor'] = self.theme.bg_color
        matplotlib.rcParams['axes.edgecolor'] = self.theme.fg_color
        matplotlib.rcParams['axes.labelcolor'] = self.theme.fg_color
        matplotlib.rcParams['xtick.color'] = self.theme.fg_color
        matplotlib.rcParams['ytick.color'] = self.theme.fg_color
        matplotlib.rcParams['text.color'] = self.theme.fg_color
        matplotlib.rcParams['grid.color'] = "#555555"

    def on_close_window(self):
        if self.final_path_segments:
            if messagebox.askyesno("Quit", "Path data exists. Quit without saving?"):
                self.root.quit()
                self.root.destroy()
        else:
            if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
                self.root.quit()
                self.root.destroy()

    def finish(self):
        if not self.final_path_segments:
            if not messagebox.askyesno("Warning", "No path created. Return empty path?"):
                return
        self.root.quit()
        self.root.destroy()

    def _setup_layout(self):
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        self.map_frame = ttk.Frame(self.root, padding="0")
        self.map_frame.grid(row=0, column=0, sticky="nsew")

        self.side_frame = ttk.Frame(self.root, padding="15", style="Panel.TFrame")
        self.side_frame.grid(row=0, column=1, sticky="nsew")

    def _setup_map_canvas(self):
        # Create Figure with theme background
        self.fig = plt.figure(figsize=(8, 8), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.map_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Helper to style the standard toolbar
        toolbar = NavigationToolbar2Tk(self.canvas, self.map_frame)
        toolbar.config(background=self.theme.bg_color)
        for button in toolbar.winfo_children():
            button.config(background=self.theme.bg_color)
        toolbar.update()

        self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('button_release_event', self._on_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_drag)

        self._draw_static_environment()
        
        self.preview_line, = self.ax.plot([], [], '--', lw=2, label='Preview', color='#00ff00')
        self.committed_line, = self.ax.plot([], [], '-', lw=2, label='Committed', color='#4cc2ff')
        self.current_marker, = self.ax.plot([], [], 'o', color='#4cc2ff', markersize=6, zorder=10)
        self.heading_arrow = None 
        
        self.collision_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes, 
            color='#ff4444', fontsize=12, fontweight='bold', va='top',
            bbox=dict(facecolor=self.theme.bg_color, alpha=0.9, edgecolor='none')
        )
        
        legend = self.ax.legend(loc='lower right', facecolor=self.theme.panel_color, edgecolor=self.theme.fg_color)
        plt.setp(legend.get_texts(), color=self.theme.fg_color)

    def _draw_static_environment(self):
        self.ax.set_title("Path Planner", color=self.theme.fg_color)
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal', adjustable='datalim')
        
        all_x, all_y = [], []
        if self.obstacles:
            for obs in self.obstacles:
                min_pt, max_pt = obs['min'], obs['max']
                rect = Rectangle(
                    (min_pt[0], min_pt[1]), max_pt[0]-min_pt[0], max_pt[1]-min_pt[1],
                    edgecolor='#555555', facecolor='#444444', alpha=0.6
                )
                self.ax.add_patch(rect)
                all_x.extend([min_pt[0], max_pt[0]])
                all_y.extend([min_pt[1], max_pt[1]])
        
        if all_x:
            pad = 50
            self.ax.set_xlim(min(all_x)-pad, max(all_x)+pad)
            self.ax.set_ylim(min(all_y)-pad, max(all_y)+pad)
        else:
            self.ax.set_xlim(-200, 200)
            self.ax.set_ylim(-200, 200)
            
        self.ax.plot(self.start_pos[0], self.start_pos[1], 'o', color='#00cc00', markersize=8, label='Start')

    # Interaction Handlers (Scroll/Pan)
    def _on_scroll(self, event):
        if event.inaxes != self.ax: return
        scale = 1/1.2 if event.button == 'up' else 1.2
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        x, y = event.xdata, event.ydata
        
        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        
        rx = (xlim[1] - x) / (xlim[1] - xlim[0])
        ry = (ylim[1] - y) / (ylim[1] - ylim[0])
        
        self.ax.set_xlim([x - new_w * (1-rx), x + new_w * rx])
        self.ax.set_ylim([y - new_h * (1-ry), y + new_h * ry])
        self.canvas.draw_idle()

    def _on_press(self, event):
        if event.inaxes != self.ax or self.fig.canvas.toolbar.mode != "": return
        self._drag_data = {"x": event.xdata, "y": event.ydata, "pressed": True}

    def _on_release(self, event):
        self._drag_data["pressed"] = False

    def _on_drag(self, event):
        if not self._drag_data["pressed"] or event.inaxes != self.ax: return
        dx, dy = event.xdata - self._drag_data["x"], event.ydata - self._drag_data["y"]
        self.ax.set_xlim(self.ax.get_xlim() - dx)
        self.ax.set_ylim(self.ax.get_ylim() - dy)
        self.canvas.draw_idle()

    # --- Controls ---
    def _setup_controls_sidebar(self):
        lbl_title = ttk.Label(self.side_frame, text="Motion Controls", style="Panel.TLabel", font=("Helvetica", 16, "bold"))
        lbl_title.pack(pady=(0, 15))

        # Notebook
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
        self.var_duration = tk.DoubleVar(value=5.0)
        self.var_velocity = tk.DoubleVar(value=5.0)
        self.var_accel = tk.DoubleVar(value=0.0)
        self.var_turn_rate = tk.DoubleVar(value=15.0)
        self.var_start_heading = tk.DoubleVar(value=self.config.initial_heading)

        # Slider Container
        self.controls_container = ttk.LabelFrame(self.side_frame, text="Parameters", style="Panel.TLabelframe", padding="10")
        self.controls_container.pack(fill=tk.X, pady=15)
        
        self.slider_widgets = {}
        
        self.slider_widgets['start_heading'] = self._create_compound_slider(
            "Initial Heading (°)", self.var_start_heading, 0.0, 360.0
        )
        
        # --- DURATION LOGIC HERE ---
        # We pass specific resolution info to handle the variable_duration check inside
        self.slider_widgets['duration'] = self._create_compound_slider(
            "Duration (s)", self.var_duration, 0.1, 60.0, is_duration=True
        )
        
        self.slider_widgets['velocity'] = self._create_compound_slider(
            "Velocity (m/s)", self.var_velocity, 0.0, 50.0
        )
        self.slider_widgets['accel'] = self._create_compound_slider(
            "Accel (m/s²)", self.var_accel, -20.0, 20.0
        )
        self.slider_widgets['turn'] = self._create_compound_slider(
            "Turn Rate (°/s)", self.var_turn_rate, -90.0, 90.0
        )

        # Buttons
        btn_frame = ttk.Frame(self.side_frame, style="Panel.TFrame")
        btn_frame.pack(fill=tk.X, pady=15)
        
        ttk.Button(btn_frame, text="Add Segment", command=self.add_segment).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_frame, text="Undo", command=self.undo_segment).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Treeview
        lbl_hist = ttk.Label(self.side_frame, text="Path History:", style="Panel.TLabel", font=("Helvetica", 10, "bold"))
        lbl_hist.pack(anchor="w", pady=(10, 5))

        self.tree = ttk.Treeview(self.side_frame, columns=("id", "type", "end"), show="headings", height=8)
        self.tree.heading("id", text="#")
        self.tree.column("id", width=30)
        self.tree.heading("type", text="Type")
        self.tree.column("type", width=80)
        self.tree.heading("end", text="End Pos")
        self.tree.column("end", width=100)
        self.tree.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(self.side_frame, text="FINISH & SAVE", command=self.finish).pack(side=tk.BOTTOM, fill=tk.X, pady=20)

    def _create_compound_slider(self, label_text, variable, min_val, max_val, is_duration=False):
        """
        Creates a compound widget (Label + Spinbox + Scale).
        Handles snapping logic for Duration if variable_duration is False.
        """
        frame = ttk.Frame(self.controls_container, style="Panel.TFrame")
        
        # Determine Step Size
        step_size = 0.1
        if is_duration and not self.config.variable_duration:
            step_size = self.dt
            # Enforce min_val to be at least one time step
            if min_val < self.dt: min_val = self.dt

        # Logic to snap value
        def on_change(*args):
            val = variable.get()
            
            # If strictly discrete duration
            if is_duration and not self.config.variable_duration:
                # Round to nearest multiple of dt
                steps = round(val / self.dt)
                if steps < 1: steps = 1
                snapped_val = steps * self.dt
                
                # Only update if significantly different to avoid loop
                if abs(val - snapped_val) > 1e-5:
                    variable.set(snapped_val)
                    return # variable.set triggers callback again
            
            self.update_preview()

        # Top Row: Label + Spinbox
        top = ttk.Frame(frame, style="Panel.TFrame")
        top.pack(fill=tk.X)
        ttk.Label(top, text=label_text, style="Panel.TLabel").pack(side=tk.LEFT)
        
        spin = tk.Spinbox(
            top, from_=min_val, to=max_val, increment=step_size, format="%.2f",
            textvariable=variable, width=8, font=self.theme.main_font,
            bg=self.theme.entry_bg, fg=self.theme.fg_color, 
            buttonbackground=self.theme.panel_color, relief="flat",
            command=on_change # Triggered on button click
        )
        spin.pack(side=tk.RIGHT, ipady=6)
        spin.bind('<Return>', lambda e: on_change())
        spin.bind('<FocusOut>', lambda e: on_change())

        # Scale
        # Note: ttk.Scale is continuous. We use the 'command' to snap it if needed.
        scale = ttk.Scale(
            frame, from_=min_val, to=max_val, variable=variable, 
            orient=tk.HORIZONTAL, style="Horizontal.TScale",
            command=lambda v: on_change()
        )
        scale.pack(fill=tk.X, pady=(2, 10))
        
        return frame

    # --- Logic ---
    def on_tab_changed(self, event):
        for w in self.slider_widgets.values(): w.pack_forget()
        self.slider_widgets['duration'].pack(fill=tk.X)
        if len(self.final_path_segments) == 0:
            self.slider_widgets['start_heading'].pack(fill=tk.X)
        
        idx = self.notebook.index(self.notebook.select())
        if idx == 0: self.slider_widgets['velocity'].pack(fill=tk.X)
        elif idx == 1: self.slider_widgets['accel'].pack(fill=tk.X)
        elif idx == 2: 
            self.slider_widgets['velocity'].pack(fill=tk.X)
            self.slider_widgets['turn'].pack(fill=tk.X)
        self.update_preview()

    def get_exact_endpoint(self, start_pos, start_heading, start_vel, mode, duration):
        """
        Calculates the theoretical kinematic endpoint at t=duration.
        This ensures the next segment starts at the perfect physical location,
        decoupled from the visual sampling resolution.
        """
        x0, y0, z0 = start_pos
        theta = start_heading
        
        # Default fallback
        new_x, new_y = x0, y0
        new_v = start_vel
        new_theta = theta

        if mode == "Const Vel":
            v = self.var_velocity.get()
            # Exact calculation
            new_x = x0 + v * np.cos(theta) * duration
            new_y = y0 + v * np.sin(theta) * duration
            new_v = v
            new_theta = theta
            
        elif mode == "Const Accel":
            v0 = start_vel
            a = self.var_accel.get()
            # s = ut + 0.5at^2
            dist = v0 * duration + 0.5 * a * (duration ** 2)
            new_x = x0 + dist * np.cos(theta)
            new_y = y0 + dist * np.sin(theta)
            new_v = v0 + a * duration
            new_theta = theta
            
        elif mode == "Turn":
            v = self.var_velocity.get()
            omega_deg = self.var_turn_rate.get()
            omega = np.deg2rad(omega_deg)
            
            if abs(omega) < 1e-4: # Straight line fallback
                new_x = x0 + v * np.cos(theta) * duration
                new_y = y0 + v * np.sin(theta) * duration
                new_theta = theta
            else:
                r = v / omega
                new_theta = theta + omega * duration
                new_x = x0 + r * (np.sin(new_theta) - np.sin(theta))
                new_y = y0 - r * (np.cos(new_theta) - np.cos(theta))
            
            new_v = v

        return np.array([new_x, new_y, z0]), new_v, new_theta

    def calculate_current_segment(self):
        if len(self.final_path_segments) == 0:
            self.current_heading = np.deg2rad(self.var_start_heading.get())

        tab_idx = self.notebook.index(self.notebook.select())
        duration = self.var_duration.get()
        
        # If strict timing, ensure we have integer steps
        if not self.config.variable_duration:
            steps = int(round(duration / self.dt))
            t = np.linspace(0, steps * self.dt, steps + 1)
        else:
            t = np.arange(0, duration, self.dt)
            # Ensure at least start and end
            if len(t) < 2: t = np.array([0.0, duration])

        x0, y0 = self.current_pos[0], self.current_pos[1]
        z = self.current_pos[2]
        theta = self.current_heading
        
        mode = "Unknown"
        final_v = self.current_velocity
        final_theta = theta

        if tab_idx == 0: # Vel
            v = self.var_velocity.get()
            x = x0 + v * np.cos(theta) * t
            y = y0 + v * np.sin(theta) * t
            final_v = v
            mode = "Const Vel"
        elif tab_idx == 1: # Accel
            v0 = self.current_velocity
            a = self.var_accel.get()
            dist = v0 * t + 0.5 * a * t**2
            x = x0 + dist * np.cos(theta)
            y = y0 + dist * np.sin(theta)
            final_v = v0 + a * t[-1]
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
                final_theta = theta + omega * t[-1]
            final_v = v
            mode = "Turn"

        path = np.stack((x, y, np.full_like(x, z)), axis=1)
        return path, final_v, final_theta, mode

    def check_collision(self, path):
        if not self.obstacles: return False
        for obs in self.obstacles:
            mn, mx = obs['min'], obs['max']
            if np.any((path[:,0] >= mn[0]) & (path[:,0] <= mx[0]) & (path[:,1] >= mn[1]) & (path[:,1] <= mx[1])):
                return True
        return False

    def update_preview(self):
        path, _, heading, _ = self.calculate_current_segment()
        self.preview_line.set_data(path[:,0], path[:,1])
        
        if self.check_collision(path):
            self.preview_line.set_color('#ff4444')
            self.collision_text.set_text("⚠️ COLLISION!")
        else:
            self.preview_line.set_color('#00ff00')
            self.collision_text.set_text("")

        if self.heading_arrow: self.heading_arrow.remove()
        dx = 20 * np.cos(self.current_heading)
        dy = 20 * np.sin(self.current_heading)
        self.heading_arrow = Arrow(self.current_pos[0], self.current_pos[1], dx, dy, width=5, color='orange')
        self.ax.add_patch(self.heading_arrow)
        self.current_marker.set_data([self.current_pos[0]], [self.current_pos[1]])
        self.canvas.draw_idle()

    def add_segment(self):
        # 1. Generate Visual Preview (keep existing sampling for the plot)
        path, v, theta, mode = self.calculate_current_segment()
        
        if self.check_collision(path):
            if not messagebox.askyesno("Confirm", "Segment collides! Add anyway?"): return
        
        duration = self.var_duration.get()

        # 2. Store Metadata (Crucial for reconstruction)
        # We explicitly store the slider values to avoid any rounding errors from the visual path
        segment_data = {
            "start_pos": self.current_pos.copy(), 
            "start_heading": self.current_heading, 
            "start_vel": self.current_velocity,
            "mode": mode,
            "duration": duration,
            "params": {
                "velocity": self.var_velocity.get(),
                "accel": self.var_accel.get(),
                "turn_rate": self.var_turn_rate.get()
            }
        }
        self.segments_data.append(segment_data)
        
        # 3. Update State using EXACT Math (Decoupled from 'path' array)
        exact_pos, exact_vel, exact_theta = self.get_exact_endpoint(
            self.current_pos, self.current_heading, self.current_velocity, mode, duration
        )
        
        # 4. Save to lists for functionality
        self.final_path_segments.append(path) # Used only for visual "Undo" now
        
        # Simplified State Vector for the Treeview/Logs
        vx = exact_vel * np.cos(exact_theta)
        vy = exact_vel * np.sin(exact_theta)
        ax_val = self.var_accel.get() if mode == "Const Accel" else 0
        w_val = np.deg2rad(self.var_turn_rate.get()) if mode == "Turn" else 0
        
        self.state_vectors.append(np.array([
            exact_pos[0], exact_pos[1], vx, vy, 
            ax_val*np.cos(exact_theta), ax_val*np.sin(exact_theta), w_val
        ]))
        
        # 5. Advance State
        self.current_pos = exact_pos
        self.current_velocity = exact_vel
        self.current_heading = exact_theta
        
        # UI Updates
        self.tree.insert("", tk.END, values=(
            len(self.segments_data), mode, f"({exact_pos[0]:.1f}, {exact_pos[1]:.1f})"
        ))
        self._refresh_committed_line()
        self.on_tab_changed(None)

    def undo_segment(self):
        if not self.segments_data: return
        data = self.segments_data.pop()
        self.final_path_segments.pop()
        self.state_vectors.pop()
        self.current_pos = data["start_pos"]
        self.current_heading = data["start_heading"]
        self.current_velocity = data["start_vel"]
        self.tree.delete(self.tree.get_children()[-1])
        self._refresh_committed_line()
        self.on_tab_changed(None)

    def _refresh_committed_line(self):
        if self.final_path_segments:
            full = np.vstack(self.final_path_segments)
            self.committed_line.set_data(full[:,0], full[:,1])
        else:
            self.committed_line.set_data([], [])
        self.update_preview()