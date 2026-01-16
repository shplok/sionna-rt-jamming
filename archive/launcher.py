import tkinter as tk
from tkinter import ttk, messagebox
from ui_theme import ModernTheme

class LauncherApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sionna Motion Engine")
        
        # Apply Styling
        self.theme = ModernTheme(self.root)
        
        # --- State Variables ---
        self.selected_strategy = tk.StringVar(value="Math Modeling")
        self.time_step_var = tk.DoubleVar(value=0.1)
        self.use_variable_duration = tk.BooleanVar(value=False)
        
        # Result storage
        self.final_config = None

        self._setup_ui()
        self._center_window()

    def _center_window(self):
        """Centers the window on the screen."""
        self.root.update_idletasks()
        w = 600
        h = 800
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.root.geometry(f'{w}x{h}+{int(x)}+{int(y)}')
        self.root.resizable(False, False)

    def _setup_ui(self):
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)

        # Header
        lbl_title = ttk.Label(main_container, text="Simulation Configuration (Placeholder)", style="Header.TLabel")
        lbl_title.pack(fill=tk.X, pady=(0, 20))

        # --- Section 1: Strategy Selection ---
        strat_frame = ttk.LabelFrame(main_container, text=" Select Motion Strategy ", style="Panel.TLabelframe", padding="15")
        strat_frame.pack(fill=tk.X, pady=(0, 20))

        strategies = [
            ("Math Modeling Planner", "Math Modeling"),
            ("Random Walk", "RandomWalk"),
            ("Waypoint Follower (WIP)", "Waypoint")
        ]

        for text, value in strategies:
            rb = ttk.Radiobutton(
                strat_frame, 
                text=text, 
                value=value, 
                variable=self.selected_strategy,
                command=self._on_strategy_change
            )
            rb.pack(fill=tk.X, pady=4)

        # --- Section 2: Strategy Specific Options ---
        # We create a container that can show/hide based on selection
        self.options_container = ttk.LabelFrame(main_container, text=" Strategy Settings ", style="Panel.TLabelframe", padding="15")
        
        # (Content for Math Modeling Mode)
        self.chk_var_duration = ttk.Checkbutton(
            self.options_container,
            text="Enable Variable Segment Duration",
            variable=self.use_variable_duration,
            style="TCheckbutton",
            command=self._update_description
        )
        self.chk_var_duration.pack(anchor="w")

        self.desc_label = ttk.Label(
            self.options_container, 
            text="Segments will have fixed time steps aligned with global DT.",
            style="Desc.TLabel",
            wraplength=400
        )
        self.desc_label.pack(anchor="w", pady=(5, 0))

        # Initially show this because default is Math Modeling
        self.options_container.pack(fill=tk.X, pady=(0, 20))

        # --- Section 3: Global Parameters ---
        global_frame = ttk.LabelFrame(main_container, text=" Global Simulation Parameters ", style="Panel.TLabelframe", padding="15")
        global_frame.pack(fill=tk.X, pady=(0, 30))

        row_dt = ttk.Frame(global_frame, style="Panel.TFrame")
        row_dt.pack(fill=tk.X)
        
        ttk.Label(row_dt, text="Simulation Time Step (dt):", style="Panel.TLabel").pack(side=tk.LEFT)
        
        # Styled Spinbox
        tk.Spinbox(
            row_dt, 
            from_=1, to=100.0, increment=0.25, 
            textvariable=self.time_step_var, 
            width=6,
            font=self.theme.main_font, 
            bg=self.theme.entry_bg, 
            fg=self.theme.fg_color, 
            buttonbackground=self.theme.panel_color, 
            relief="flat"
        ).pack(side=tk.RIGHT, ipady=9)

        # --- Launch Button ---
        btn_launch = ttk.Button(main_container, text="INITIALIZE SIMULATION", command=self.finish)
        btn_launch.pack(side=tk.BOTTOM, fill=tk.X)

    def _on_strategy_change(self):
        """Show/Hide specific options based on strategy."""
        val = self.selected_strategy.get()
        if val == "Math Modeling":
            self.options_container.pack(fill=tk.X, pady=(0, 20), after=self.root.winfo_children()[0].winfo_children()[1])
        else:
            self.options_container.pack_forget()

    def _update_description(self):
        if self.use_variable_duration.get():
            self.desc_label.config(text="Advanced: Each path segment can have a custom duration.")
        else:
            self.desc_label.config(text="Standard: Segment duration is strictly multiple of Time Step.")

    def finish(self):
        """Validate and Save."""
        try:
            dt = self.time_step_var.get()
            if dt <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Time step must be a positive number.")
            return

        self.final_config = {
            "strategy_type": self.selected_strategy.get(),
            "global_dt": dt,
            "math_gui_params": {
                "variable_duration": self.use_variable_duration.get()
            }
        }
        self.root.quit()
        self.root.destroy()

    def run(self):
        """Public method to start the app and return the config."""
        self.root.mainloop()
        return self.final_config

if __name__ == "__main__":
    # Test run
    app = LauncherApp()
    print(app.run())