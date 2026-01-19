import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional, Dict, Any
from ui.theme import ModernTheme

class LauncherApp:
    def __init__(self, jammer_name: str = "Unknown", fixed_dt: Optional[float] = None, mode: str = "individual"):
        self.root = tk.Tk()
        self.root.title("Sionna Motion Engine")
        
        self.theme = ModernTheme(self.root)
        self._center_window()

        self.jammer_name = jammer_name
        self.fixed_dt = fixed_dt
        self.mode = mode
        
        default_strat = "Math Modeling" if self.mode == "individual" else "GraphNav"
        self.selected_strategy = tk.StringVar(value=default_strat)
        
        self.time_step_var = tk.DoubleVar(value=self.fixed_dt if self.fixed_dt is not None else 1)
        self.velocity_var = tk.DoubleVar(value=10.0)
        self.use_variable_duration = tk.BooleanVar(value=False)
        self.num_simulations_var = tk.IntVar(value=10)
        self.min_path_distance_var = tk.DoubleVar(value=250.0)

        self.padding_mode = tk.StringVar(value="END")
        self.user_selection: Optional[Dict[str, Any]] = None

        self._setup_ui()

    def _center_window(self):
        self.root.update_idletasks()
        w, h = 600, 800 
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.root.geometry(f'{w}x{h}+{int(x)}+{int(y)}')
        self.root.resizable(False, False)

    def _setup_ui(self):
        main = ttk.Frame(self.root, padding="30 40 30 30")
        main.pack(fill=tk.BOTH, expand=True)

        ttk.Label(main, text=f"Configuring: {self.jammer_name}", style="Title.TLabel", anchor="center").pack(fill=tk.X, pady=(0, 35))

        # --- Motion Strategy Card ---
        strat_frame = ttk.LabelFrame(main, text=" Motion Strategy ", style="Card.TLabelframe", padding="15")
        strat_frame.pack(fill=tk.X, pady=(0, 25))

        strat_inner = ttk.Frame(strat_frame, style="Card.TFrame")
        strat_inner.pack(fill=tk.X)

        if self.mode == "individual":
            strategies = [
                ("Math Modeling Planner", "Math Modeling"),
                ("Waypoint Planner", "Waypoint"),
                ("Random Walk Planner", "RandomWalk"),
            ]
        else:
            strategies = [("Graph Navigation", "GraphNav")]

        for text, value in strategies:
            ttk.Radiobutton(
                strat_inner, text=text, value=value, 
                variable=self.selected_strategy, 
                style="Selector.TRadiobutton",
                command=self._on_strategy_change
            ).pack(fill=tk.X, pady=4)

        # --- 2. Parameters Card ---
        self.param_frame = ttk.LabelFrame(main, text=" Parameters ", style="Card.TLabelframe", padding="15")
        self.param_frame.pack(fill=tk.X, pady=(0, 25))
        
        self.param_inner = ttk.Frame(self.param_frame, style="Card.TFrame")
        self.param_inner.pack(fill=tk.X)

        #A. Create Dynamic Containers (Do not pack them yet)
        self.opt_container = ttk.Frame(self.param_inner, style="Card.TFrame")
        ttk.Checkbutton(self.opt_container, text="Enable Variable Segment Duration", 
                        variable=self.use_variable_duration, style="Card.TCheckbutton",
                        command=self._update_description).pack(anchor="w")
        self.desc_label = ttk.Label(self.opt_container, text="", style="Desc.TLabel", wraplength=400)
        self.desc_label.pack(anchor="w", pady=(6, 0), padx=30)

        self.vel_container, _, _ = self._create_spin_row("Const. Velocity (m/s):", self.velocity_var, 0, 100, 1.0)
        self.dist_container, _, _ = self._create_spin_row("Min Path Dist (m):", self.min_path_distance_var, 1, 1000, 10.0)
        self.sim_container, _, _ = self._create_spin_row("Number of Simulations:", self.num_simulations_var, 1, 10000, 1)

        # B. Create Static Bottom Rows (These act as Anchors for ordering)
        
        # Anchor 1: Padding Mode
        self.sync_frame = ttk.Frame(self.param_inner, style="Card.TFrame")
        self.sync_frame.pack(fill=tk.X, pady=(0, 20))
        ttk.Label(self.sync_frame, text="Path Padding Mode:", style="Card.TLabel").pack(side=tk.LEFT, anchor="center")
        ttk.Combobox(
            self.sync_frame, textvariable=self.padding_mode, 
            values=["END", "START"], state="readonly", width=10, style="TCombobox"
        ).pack(side=tk.RIGHT)

        # Anchor 2: Time Step (Always visible at bottom)
        # We capture the frame, spinbox, and label to apply logic later
        self.row_dt, self.spin_dt, self.lbl_dt = self._create_spin_row("Time Step (dt):", self.time_step_var, 0, 100.0, 0.5)
        self.row_dt.pack(fill=tk.X, pady=(0, 5))

        if self.fixed_dt is not None:
            self.spin_dt.config(state='disabled', fg="#888888")
            self.lbl_dt.config(text=f"Time Step (Locked at {self.fixed_dt} s):")

        # --- Finish Button ---
        ttk.Frame(main).pack(expand=True)
        btn_text = "INITIALIZE SIMULATION" if self.mode == "individual" else "GENERATE DATASET"
        ttk.Button(main, text=btn_text, style="Action.TButton", command=self.finish).pack(side=tk.BOTTOM, fill=tk.X)

        self._on_strategy_change()
        self._update_description()

    def _create_spin_row(self, label_text, var, mn, mx, inc):
        """
        Creates a row with a Label and a Spinbox.
        Returns:
            container: The Frame containing the whole row (for packing).
            spin: The actual Spinbox widget (for configuration like disabling).
            lbl: The Label widget (if text needs updating).
        """
        container = ttk.Frame(self.param_inner, style="Card.TFrame")
        
        lbl = ttk.Label(container, text=label_text, style="Card.TLabel")
        lbl.pack(side=tk.LEFT, anchor="center")
        
        wrapper = tk.Frame(container, bg=self.theme.entry_bg, padx=1, pady=1)
        wrapper.pack(side=tk.RIGHT)
        
        spin = tk.Spinbox(wrapper, from_=mn, to=mx, increment=inc, textvariable=var, width=5,
                          bg=self.theme.entry_bg, fg="#ffffff", relief="flat",
                          font=self.theme.main_font, buttonbackground=self.theme.panel_color)
        spin.pack()
        
        return container, spin, lbl

    def _on_strategy_change(self):
        val = self.selected_strategy.get()
        
        # 1. Hide all dynamic containers
        self.opt_container.pack_forget()
        self.vel_container.pack_forget()
        self.sim_container.pack_forget()
        self.dist_container.pack_forget()

        # 3. Pack Dynamic Widgets relative to Anchor
        if val == "Math Modeling":
            self.opt_container.pack(fill=tk.X, pady=(0, 20), before=self.sync_frame)
            
        elif val == "Waypoint":
            self.vel_container.pack(fill=tk.X, pady=(0, 20), before=self.sync_frame)
            
        elif val == "GraphNav":
            # Order: Velocity -> Distance -> Sims
            self.vel_container.pack(fill=tk.X, pady=(0, 20), before=self.sync_frame)
            self.dist_container.pack(fill=tk.X, pady=(0, 20), before=self.sync_frame)
            self.sim_container.pack(fill=tk.X, pady=(0, 20), before=self.sync_frame)


    def _update_description(self):
        if self.use_variable_duration.get():
            self.desc_label.config(text="Current: Each path segment can have a custom duration.")
        else:
            self.desc_label.config(text="Current: Segment duration is strictly multiple of Time Step.")

    def finish(self):
        try:
            dt = self.time_step_var.get()
            if dt <= 0: raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid Time Step")
            return

        self.user_selection = {
            "strategy_type": self.selected_strategy.get(),
            "global_dt": dt,
            "variable_duration": self.use_variable_duration.get(),
            "velocity": self.velocity_var.get(),
            "num_simulations": self.num_simulations_var.get(),
            "min_path_dist": self.min_path_distance_var.get(),
            "padding_mode": "pad_end" if self.padding_mode.get() == "END" else "pad_start"
        }
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.user_selection