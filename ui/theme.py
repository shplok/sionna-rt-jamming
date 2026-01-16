import tkinter as tk
from tkinter import ttk
import tkinter.font as tkfont

class ModernTheme:
    def __init__(self, root):
        self.root = root
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # --- Color Palette ---
        self.bg_color = "#1e1e1e"
        self.panel_color = "#2d2d2d"
        self.fg_color = "#e0e0e0"
        self.accent_color = "#0078d7"
        self.accent_hover = "#198ce6"
        self.entry_bg = "#383838"
        self.text_dim = "#9e9e9e"

        self._configure_fonts()
        self._configure_root()
        self._configure_styles()

    def _configure_fonts(self):
        family = "gothic"
        
        self.main_font = tkfont.Font(name="AppMainFont", family=family, size=12)
        self.header_font = tkfont.Font(name="AppHeaderFont", family=family, size=12, weight="bold")
        self.title_font = tkfont.Font(name="AppTitleFont", family=family, size=24, weight="bold") 
        self.button_font = tkfont.Font(name="AppButtonFont", family=family, size=12, weight="bold")
        self.italic_font = tkfont.Font(name="AppItalicFont", family=family, size=11, slant="italic")

    def _configure_root(self):
        self.root.configure(bg=self.bg_color)
        self.root.option_add("*Background", self.bg_color)
        self.root.option_add("*Foreground", self.fg_color)
        # self.root.option_add("*Font", self.main_font)

    def _configure_styles(self):
        # Configure the default 'root' style for all ttk widgets
        self.style.configure(".", 
                             background=self.bg_color, 
                             foreground=self.fg_color, 
                             font="AppMainFont")

        self.style.configure("Title.TLabel", 
                             font="AppTitleFont", 
                             foreground=self.fg_color, 
                             background=self.bg_color)

        self.style.configure("Card.TLabelframe", 
                             background=self.panel_color, 
                             relief="flat", 
                             borderwidth=0,
                             labelmargins=(15,20,0,5))
        
        self.style.configure("Card.TLabelframe.Label", 
                             font="AppHeaderFont",
                             foreground=self.accent_color,
                             background=self.panel_color)

        self.style.configure("Card.TFrame", background=self.panel_color)
        self.style.configure("Card.TLabel", background=self.panel_color, foreground=self.fg_color)
        
        self.style.configure("Card.TCheckbutton", 
                             background=self.panel_color, 
                             foreground=self.fg_color, 
                             font="AppMainFont",
                             focuscolor=self.panel_color)
        
        self.style.map("Card.TCheckbutton",
               background=[('active', self.panel_color)],
               foreground=[('active', self.fg_color)])

        self.style.configure("Selector.TRadiobutton", 
                             background=self.entry_bg,
                             foreground=self.fg_color,
                             indicatoron=0, 
                             padding=(20, 15),
                             font="AppMainFont",
                             anchor="w",
                             relief="flat")
        
        self.style.map("Selector.TRadiobutton",
                       background=[('selected', self.accent_color), ('active', self.accent_hover)],
                       foreground=[('selected', "#ffffff"), ('active', "#ffffff")])

        self.style.configure("Action.TButton", 
                             background=self.accent_color, 
                             foreground="#ffffff",
                             font="AppButtonFont",
                             borderwidth=0, 
                             padding=(0, 15))
              
        self.style.map("Action.TButton", 
                       background=[('active', self.accent_hover), ('pressed', self.accent_hover)])

        self.style.configure("Desc.TLabel", font="AppItalicFont", foreground=self.text_dim, background=self.panel_color)

        self.style.configure("TNotebook", background=self.bg_color, borderwidth=0)
        self.style.configure("TNotebook.Tab", 
                             background=self.panel_color, 
                             foreground=self.fg_color,
                             padding=[15, 5],
                             font="AppMainFont",
                             borderwidth=0)
        
        self.style.map("TNotebook.Tab",
                       background=[("selected", self.accent_color), ("active", self.accent_hover)],
                       foreground=[("selected", "#ffffff"), ("active", "#ffffff")])
        
        self.style.configure("Treeview", 
                             background=self.panel_color, 
                             foreground=self.fg_color, 
                             fieldbackground=self.panel_color,
                             font="AppMainFont",
                             rowheight=25,
                             borderwidth=0)
        
        self.style.configure("Treeview.Heading", 
                             background=self.entry_bg, 
                             foreground=self.fg_color, 
                             font="AppHeaderFont",
                             relief="flat")
        
        self.style.map("Treeview", 
                       background=[('selected', self.accent_color)],
                       foreground=[('selected', "#ffffff")])
        
        self.style.map('TCombobox', fieldbackground=[('readonly', self.entry_bg)])
        self.style.map('TCombobox', selectbackground=[('readonly', self.entry_bg)])
        self.style.map('TCombobox', selectforeground=[('readonly', self.fg_color)])
        self.style.configure('TCombobox', 
                        background=self.panel_color, 
                        foreground=self.fg_color,
                        arrowcolor=self.fg_color)