import numpy as np
import tkinter as tk
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, Any
from abc import ABC, abstractmethod

# Imports from your other modules
from motion_engine import MotionEngine
from MathGUI import MathStrategyGUI
from launcher import LauncherApp
from ui_theme import ModernTheme

# ==========================================
# 1. Base Interfaces
# ==========================================

class MotionStrategy(ABC):
    """
    Abstract base class for all path generation logic.
    """
    @abstractmethod
    def generate(self, engine: MotionEngine, config: Any) -> Tuple[np.ndarray, Dict]:
        """
        Generates a path based on the provided configuration.
        
        Args:
            engine: The MotionEngine instance (provides collision checks, bounds).
            config: A specific configuration data class (e.g., RandomWalkConfig).
            
        Returns:
            (path, metadata): A tuple containing the (N,3) position array and a metadata dict.
        """
        pass

# ==========================================
# 2. Random Walk Strategy
# ==========================================

@dataclass
class RandomWalkConfig:
    starting_position: np.ndarray
    num_steps: int = 100
    step_size: float = 1.0
    time_step: float = 0.1
    max_retries: int = 10
    random_seed: Optional[int] = None

    def __post_init__(self):
        if self.starting_position.shape != (3,):
            raise ValueError("starting_position must be a 1D array of size 3 (x, y, z)")

class RandomWalkStrategy(MotionStrategy):
    """
    Generates a stochastic path that avoids obstacles by retrying invalid steps.
    """
    def generate(self, engine: MotionEngine, config: RandomWalkConfig) -> Tuple[np.ndarray, Dict]:
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        path = np.zeros((config.num_steps, 3))
        path[0] = config.starting_position
        
        # We assume 2D motion on the Z-plane of the start position
        z_height = config.starting_position[2]

        for step in range(1, config.num_steps):
            previous_position = path[step - 1]
            valid_step_found = False
            
            for _ in range(config.max_retries):
                # Generate random direction in XY plane
                direction = np.random.randn(2)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                
                # Create candidate (maintaining original Z height)
                offset = np.array([direction[0], direction[1], 0.0]) * config.step_size
                candidate_position = previous_position + offset
                candidate_position[2] = z_height 
                
                if engine.is_position_valid(candidate_position):
                    path[step] = candidate_position
                    valid_step_found = True
                    break
            
            # If stuck, stay in place
            if not valid_step_found:
                path[step] = previous_position

        # Metadata calculations
        total_distance = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        metadata = {
            "total_distance": total_distance,
            "avg_velocity": config.step_size / config.time_step,
            "duration": config.num_steps * config.time_step
        }
        
        return path, metadata

# ==========================================
# 3. Math Modeling Strategy
# ==========================================

@dataclass
class MathStrategyConfig:
    starting_position: np.ndarray
    initial_heading: float = 0.0     # Degrees
    time_step: float = 0.1
    variable_duration: bool = False  # If True, segments can have arbitrary lengths

    def __post_init__(self):
        if self.starting_position.shape != (3,):
            raise ValueError("starting_position must be a 1D array of size 3 (x, y, z)")

class MathStrategy(MotionStrategy):
    """
    Launches a GUI for manual path planning using kinematic segments.
    """
    def generate(self, engine: MotionEngine, config: MathStrategyConfig) -> Tuple[np.ndarray, Dict]:
        root = tk.Tk()
        root.title("Sionna Path Planner")
        
        _ = ModernTheme(root)
        
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry(f"{int(screen_width*0.8)}x{int(screen_height*0.8)}")

        app = MathStrategyGUI(root, engine.obstacles, config)
        root.mainloop()

        # --- Process Results after Window Close ---
        if not app.final_path_segments:
            print("Math Modeling Mode: No path generated. Returning stationary point.")
            # Return a short stationary path
            return np.tile(config.starting_position, (2, 1)), {"status": "empty"}

        # 1. Define the Global Timeline
        # Sum of all durations from the metadata
        total_duration = sum(seg['duration'] for seg in app.segments_data)
        
        # Create uniform time steps: 0, 0.1, 0.2 ... Total
        t_global = np.arange(0, total_duration, config.time_step)
        
        # We want to ensure the final time is included
        if t_global[-1] < total_duration - 1e-9:
            t_global = np.append(t_global, total_duration)

        full_path = []
        
        # 2. Sample along the timeline
        current_seg_idx = 0
        current_seg_start_time = 0.0
        
        for t in t_global:
            # Advance segment if t has moved past the current segment's duration
            while (current_seg_idx < len(app.segments_data) - 1 and 
                   t >= current_seg_start_time + app.segments_data[current_seg_idx]['duration'] - 1e-9):
                current_seg_start_time += app.segments_data[current_seg_idx]['duration']
                current_seg_idx += 1
            
            seg = app.segments_data[current_seg_idx]
            
            # Local time relative to the segment start
            t_local = t - current_seg_start_time
            
            # Clamp to prevent tiny overshoots due to float math
            if t_local > seg['duration']: t_local = seg['duration']
            if t_local < 0: t_local = 0
            
            pos = self._calculate_position_at_t(seg, t_local)
            full_path.append(pos)
            
        path_array = np.array(full_path)
        
        metadata = {
            "total_distance": np.sum(np.linalg.norm(np.diff(path_array, axis=0), axis=1)),
            "total_duration": total_duration,
            "num_segments": len(app.segments_data),
            "segment_details": app.segments_data
        }

        return path_array, metadata
    
    def _calculate_position_at_t(self, seg, t):
        """Re-implements the physics equations for precise sampling."""
        x0, y0, z0 = seg['start_pos']
        theta = seg['start_heading']
        mode = seg['mode']
        
        # Default
        x, y = x0, y0

        if mode == "Const Vel":
            v = seg['params']['velocity']
            x = x0 + v * np.cos(theta) * t
            y = y0 + v * np.sin(theta) * t
            
        elif mode == "Const Accel":
            v0 = seg['start_vel']
            a = seg['params']['accel']
            dist = v0 * t + 0.5 * a * (t**2)
            x = x0 + dist * np.cos(theta)
            y = y0 + dist * np.sin(theta)
            
        elif mode == "Turn":
            v = seg['params']['velocity']
            omega_deg = seg['params']['turn_rate']
            omega = np.deg2rad(omega_deg)
            
            if abs(omega) < 1e-4:
                x = x0 + v * np.cos(theta) * t
                y = y0 + v * np.sin(theta) * t
            else:
                r = v / omega
                # Note: t is local time starting at 0 for this segment
                x = x0 + r * (np.sin(omega * t + theta) - np.sin(theta))
                y = y0 - r * (np.cos(omega * t + theta) - np.cos(theta))

        return np.array([x, y, z0])

# ==========================================
# 4. The Mission Wizard
# ==========================================

def run_mission_wizard(engine: MotionEngine, jammer_id: str, default_start_pos: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Main entry point for user interaction.
    1. Launches LauncherApp to select strategy.
    2. Configures the selected strategy.
    3. Runs the Engine to generate the path.
    """
    print("--- Starting Mission Wizard ---")
    
    # 1. Run Launcher
    launcher = LauncherApp()
    launcher_config = launcher.run() # Blocks until 'Launch' is clicked

    if not launcher_config:
        print("Mission Wizard cancelled by user.")
        return None, {}

    strategy_type = launcher_config["strategy_type"]
    global_dt = launcher_config["global_dt"]

    # 2. Configure Strategy based on selection
    if strategy_type == "RandomWalk":
        strategy = RandomWalkStrategy()
        config = RandomWalkConfig(
            starting_position=default_start_pos,
            num_steps=500,  # Default, could be exposed in Launcher later
            step_size=1.0, # Default, could be exposed in Launcher later
            time_step=global_dt
        )

    elif strategy_type == "Math Modeling":
        strategy = MathStrategy()
        math_gui_params = launcher_config.get("math_gui_params", {})
        
        config = MathStrategyConfig(
            starting_position=default_start_pos,
            initial_heading=0.0,
            time_step=global_dt,
            variable_duration=math_gui_params.get("variable_duration", False)
        )
    
    elif strategy_type == "Waypoint":
        # Placeholder for future expansion
        print("Waypoint strategy not implemented yet.")
        return None, {}

    else:
        print(f"Unknown strategy: {strategy_type}")
        return None, {}

    # 3. Generate Path via Engine
    # This automatically stores it in the engine and returns it here
    path, metadata = engine.generate_path(jammer_id, strategy, config)
    
    return path, metadata