import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

class ModernTheme:
    """
    Style theme class.
    """
    def __init__(self, root):
        self.root = root
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Color Palette
        self.bg_color = "#2b2b2b"          # Main window background
        self.fg_color = "#ffffff"          # Main text color
        self.accent_color = "#4cc2ff"      # Light Blue (Active elements)
        self.accent_hover = "#2a9fd6"      # Hover state
        self.panel_color = "#3a3a3a"       # Secondary background (Panels/Tabs)
        self.entry_bg = "#454545"          # Darker grey for Input fields/Treeviews
        self.text_dim = "#aaaaaa"          # Dimmed text

        self._configure_fonts()
        self._configure_root()
        self._configure_styles()

    def _configure_fonts(self):
        """
        Directly creates Font objects using "Gothic".
        """
        self.main_font = tkfont.Font(family="gothic", size=11)
        self.header_font = tkfont.Font(family="gothic", size=16)
        self.button_font = tkfont.Font(family="gothic", size=14)
        self.italic_font = tkfont.Font(family="gothic", size=11, slant="italic")

    def _configure_root(self):
        self.root.configure(bg=self.bg_color)
        # Set default font for standard Tk widgets
        self.root.option_add("*Background", self.bg_color)
        self.root.option_add("*Foreground", self.fg_color)
        self.root.option_add("*TCombobox*Listbox*Background", self.entry_bg)
        self.root.option_add("*TCombobox*Listbox*Foreground", self.fg_color)
        
        # FIX FOR TREEVIEW WHITE BACKGROUND
        self.root.option_add("*Treeview.fieldBackground", self.entry_bg)
        self.root.option_add("*Font", self.main_font)

    def _configure_styles(self):
        # 1. Global Ttk Font Configuration
        self.style.configure(".", 
                             background=self.bg_color, 
                             foreground=self.fg_color,
                             font=self.main_font)

        # Frames
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("Panel.TFrame", background=self.panel_color, relief="flat")
        self.style.configure("Panel.TLabelframe", 
                             background=self.bg_color,
                             foreground=self.fg_color, 
                             relief="flat")
        self.style.configure("Panel.TLabelframe.Label", 
                             background=self.bg_color,
                             foreground=self.accent_color,
                             font=self.header_font)
        
        # Labels
        self.style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        self.style.configure("Header.TLabel", font=self.header_font, padding=(0, 10, 0, 20))
        self.style.configure("Panel.TLabel", background=self.panel_color, foreground=self.fg_color)
        self.style.configure("Desc.TLabel", font=self.italic_font, foreground=self.text_dim, background=self.bg_color)

        # --- (Notebook) ---
        self.style.configure("TNotebook", background=self.panel_color, borderwidth=0)
        self.style.configure("TNotebook.Tab", 
                             background=self.bg_color,     # Unselected Tab Background (Darker)
                             foreground=self.text_dim,     # Unselected Tab Text (Grey)
                             padding=(15, 8),
                             borderwidth=0,
                             font=self.main_font)
        
        # Define how colors change when selected
        self.style.map("TNotebook.Tab",
                       background=[('selected', self.accent_color)], # Active Tab Background (Blue)
                       foreground=[('selected', self.bg_color)])     # Active Tab Text (Dark)

        # --- TREEVIEW (History) ---
        self.style.configure("Treeview", 
                             background=self.entry_bg,      # Row background
                             foreground=self.fg_color,      # Text color
                             fieldbackground=self.entry_bg, # Empty area background
                             borderwidth=0,
                             font=self.main_font,)
        
        self.style.map("Treeview", 
                       background=[('selected', self.accent_color)], 
                       foreground=[('selected', self.bg_color)])

        self.style.configure("Treeview.Heading", 
                             background=self.panel_color, 
                             foreground=self.fg_color,
                             font=self.main_font,
                             relief="flat")
        
        # Hover effect for headings
        self.style.map("Treeview.Heading", background=[('active', self.accent_hover)])
        
        # Radiobuttons (Cards)
        self.style.configure("TRadiobutton", 
                             background=self.panel_color, 
                             foreground=self.fg_color,
                             indicatoron=0, 
                             padding=15,
                             width=30,
                             anchor="center",
                             relief="flat")
        
        # Map colors: Dark text when selected (on blue), White text when not
        self.style.map("TRadiobutton",
                       background=[('selected', self.accent_color), ('active', self.accent_hover)],
                       foreground=[('selected', self.bg_color), ('active', self.bg_color)])

        # Checkbuttons
        self.style.configure("TCheckbutton", 
                             background=self.panel_color, 
                             foreground=self.fg_color,
                             padding=10)
        
        self.style.map("TCheckbutton",
                       background=[('active', self.panel_color)],
                       foreground=[('active', self.accent_color)])

        # Buttons
        self.style.configure("TButton", 
                             background=self.accent_color, 
                             foreground=self.bg_color,
                             font=self.button_font,
                             borderwidth=0, 
                             focuscolor="none", 
                             padding=(30, 15))
              
        self.style.map("TButton", 
                       background=[('active', self.accent_hover), ('pressed', self.accent_hover)],
                       foreground=[('active', self.bg_color), ('pressed', self.bg_color)])
        
        # Scales (Sliders)
        self.style.configure("Horizontal.TScale", 
                             background=self.panel_color,
                             troughcolor=self.entry_bg, # Make the slider track dark
                             sliderthickness=15)
        
        # Labelframes
        self.style.configure("TLabelframe", background=self.panel_color, foreground=self.fg_color)
        self.style.configure("TLabelframe.Label", 
                             background=self.panel_color, 
                             foreground=self.accent_color, 
                             font=self.header_font)