import tkinter as tk
from tkinter import ttk
from ui.theme import ModernTheme

class MenuApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sionna Motion Engine")
        
        self.theme = ModernTheme(self.root)
        self._center_window()
        
        self.selected_mode = None
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
        style = ttk.Style()
        style.configure(
            "BigDesc.TLabel", 
            font=("gothic", 14),         # Bigger font size
            foreground="#b0b0b0",      # Light gray text
            background="#1e1e1e",      # Matches the window background (seamless)
            justify="center"
        )

        # Main container for vertical centering
        container = ttk.Frame(self.root)
        container.pack(expand=True, fill=tk.BOTH, padx=40)

        # Centered content frame
        main_frame = ttk.Frame(container)
        main_frame.pack(expand=True, anchor="center")

        # --- Main Title ---
        # Changed text to "Select Operation Mode" and removed the subtitle
        ttk.Label(
            main_frame, 
            text="Select Operation Mode", 
            style="Title.TLabel", 
            anchor="center"
        ).pack(fill=tk.X, pady=(0, 50)) # Increased bottom padding for separation

        # --- Individual Mode Group ---
        individual_frame = ttk.Frame(main_frame)
        individual_frame.pack(fill=tk.X, pady=(0, 30)) # Gap between groups

        btn_individual = ttk.Button(
            individual_frame, 
            text="INDIVIDUAL MODE", 
            style="Action.TButton", 
            command=self._select_individual
        )
        btn_individual.pack(fill=tk.X, pady=(0, 10), ipady=10)
        
        ttk.Label(
            individual_frame, 
            text="Configure specific paths for individual jammers interactively.", 
            style="BigDesc.TLabel",  # Using the new bigger style
            wraplength=400,
            anchor="center"
        ).pack(fill=tk.X)

        # --- Batch Mode Group ---
        batch_frame = ttk.Frame(main_frame)
        batch_frame.pack(fill=tk.X, pady=(0, 0))

        btn_batch = ttk.Button(
            batch_frame, 
            text="BATCH MODE", 
            style="Action.TButton", 
            command=self._select_batch
        )
        btn_batch.pack(fill=tk.X, pady=(0, 10), ipady=10)

        ttk.Label(
            batch_frame, 
            text="Generate N random paths automatically using Graph Navigation.", 
            style="BigDesc.TLabel",  # Using the new bigger style
            wraplength=400,
            anchor="center"
        ).pack(fill=tk.X)

    def _select_individual(self):
        self.selected_mode = "individual"
        self.root.quit()
        self.root.destroy()

    def _select_batch(self):
        self.selected_mode = "batch"
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.selected_mode