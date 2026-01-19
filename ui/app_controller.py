import tkinter as tk
import numpy as np
from typing import Optional, Tuple, Dict

# Imports from core
from core.engine import MotionEngine
from core.strategies import MathStrategy, RandomWalkStrategy, WaypointStrategy, GraphNavStrategy
from config import MathModelingConfig, RandomWalkConfig, WaypointConfig, GraphNavConfig

# Imports from UI
from ui.launcher import LauncherApp
from ui.math_planner import MathPlannerGUI
from ui.waypoint_planner import WaypointPlannerGUI
from ui.graph_planner import GraphPlannerGUI

class MissionController:
    """
    Orchestrates the user flow:
    1. Launch Selection Window.
    2. Launch Specific Planner (if needed).
    3. Construct Configuration Objects.
    4. Execute Strategy via Engine.
    """
    
    def __init__(self, engine: MotionEngine, jammer_id: str, start_pos: Optional[np.ndarray] = None, fixed_dt: Optional[float] = None):
        self.engine = engine
        self.jammer_id = jammer_id
        self.start_pos = start_pos
        self.fixed_dt = fixed_dt

    def run(self, mode: str = "individual") -> Tuple[Optional[np.ndarray], Dict]:
        """Main entry point called by main.py"""

        if self.start_pos is None and mode == "individual":
             raise RuntimeError(f"Cannot run individual mode for {self.jammer_id} without a start position.")
        
        # 1. Launch Selection Window (With ID and Locking capability)
        launcher = LauncherApp(jammer_name=self.jammer_id, fixed_dt=self.fixed_dt, mode=mode)
        selection = launcher.run()
        
        if not selection:
            print(f"Setup cancelled for {self.jammer_id}.")
            return None, {}

        # 2. Extract Common Parameters
        strategy_type = selection["strategy_type"]
        dt = selection["global_dt"]
        pad_mode = selection["padding_mode"]

        # 3. Branch Logic
        path, metadata = None, {}

        if strategy_type == "Math Modeling":
            path, metadata = self._run_math_workflow(dt, selection["variable_duration"])
        
        elif strategy_type == "Waypoint":
            path, metadata = self._run_waypoint_workflow(dt, selection["velocity"])
            
        elif strategy_type == "RandomWalk":
            path, metadata = self._run_random_workflow(dt)
        
        elif strategy_type == "GraphNav":
            raise RuntimeError("GraphNav strategy must be run in Batch Mode via batch_run().")
            # path, metadata = self._run_graph_workflow(dt, selection["num_simulations"], selection["min_path_dist"], selection['velocity'])
        
        else:
            raise ValueError(f"Unknown strategy type selected: {strategy_type}")

        # 4. If successful, notify Engine about the Padding Preference
        if path is not None and mode == "individual":
            self.engine.set_padding_mode(self.jammer_id, pad_mode)
            metadata['dt_used'] = dt

        return path, metadata
    
    def batch_run(self) -> Tuple[Optional[GraphNavConfig], str]:
        """
        Dedicated workflow for Batch Mode. 
        Does not require a start position.
        Returns the configuration object instead of a path.
        """
        launcher = LauncherApp(jammer_name=self.jammer_id, fixed_dt=None, mode="batch")
        selection = launcher.run()
        
        if not selection:
            raise RuntimeError("No graph data was created during batch setup.")

        # Build Config
        config = GraphNavConfig(
            strategy_type="GraphNav",
            time_step=selection["global_dt"],
            num_simulations=selection["num_simulations"],
            min_path_distance=selection["min_path_dist"],
            velocity=selection["velocity"],
        )

        # Run Graph GUI
        root = tk.Tk()
        root.title("Graph Navigation Planner")
        app = GraphPlannerGUI(root, self.engine, config)
        root.mainloop()
        
        final_config = app.get_config_updates()
        
        # Validate that a graph was actually built
        if not final_config.precomputed_nodes:
            raise RuntimeError("No graph data was created during batch setup.")

        return final_config, selection['padding_mode']

    def _run_math_workflow(self, dt: float, var_duration: bool):
        gui_config = MathModelingConfig(
            strategy_type="Math",
            time_step=dt,
            starting_position=self.start_pos,
            variable_duration=var_duration
        )

        root = tk.Tk()
        root.title("Path Planner")
        
        app = MathPlannerGUI(root, self.engine, gui_config)
        root.mainloop() # Blocks until user finishes planning

        segments = app.get_segments()
        
        if not segments:
            print("No path segments created.")
            return None, {}
        
        final_config = MathModelingConfig(
            strategy_type="Math",
            time_step=dt,
            starting_position=self.start_pos,
            variable_duration=var_duration,
            segments=segments
        )
        
        strategy = MathStrategy()
        return self.engine.generate_path(self.jammer_id, strategy, final_config)
    
    def _run_waypoint_workflow(self, dt: float, velocity: float):
        config = WaypointConfig(
            strategy_type="Waypoint",
            time_step=dt,
            starting_position=self.start_pos,
            velocity=velocity,
            enable_smoothing=True
        )
        
        root = tk.Tk()
        root.title("Waypoint Planner")
        
        app = WaypointPlannerGUI(root, self.engine, config)
        root.mainloop()
        
        # C. Retrieve and Execute
        config.waypoints = app.get_waypoints()
        
        strategy = WaypointStrategy()
        return self.engine.generate_path(self.jammer_id, strategy, config)
    
    # def _run_graph_workflow(self, dt: float, num_sims: int, min_path_dist: float, velocity: float):
    #     # 1. Setup Config
    #     config = GraphNavConfig(
    #         strategy_type="GraphNav",
    #         time_step=dt,
    #         num_simulations=num_sims,
    #         min_path_distance=min_path_dist,
    #         velocity=velocity,
    #     )
        
    #     # 2. Run GUI to build graph
    #     root = tk.Tk()
    #     root.title("Graph Navigation Planner")
    #     app = GraphPlannerGUI(root, self.engine, config)
    #     root.mainloop()
        
    #     # 3. Retrieve finalized config (with graph data)
    #     final_config = app.get_config_updates()
    #     final_config.num_simulations = num_sims # Ensure this comes from launcher
        
    #     strategy = GraphNavStrategy()
        
    #     # We return the sample path + the config in metadata so Main can use it
    #     try:
    #         sample_path, sample_meta = self.engine.generate_path(self.jammer_id, strategy, final_config)
    #         sample_meta["graph_config"] = final_config
    #         return sample_path, sample_meta
    #     except Exception as e:
    #         print(f"Graph generation failed or cancelled: {e}")
    #         return None, {}
    
    def _run_random_workflow(self, dt: float):
        # A. Build Config (No GUI needed for Random Walk)
        config = RandomWalkConfig(
            strategy_type="Random",
            time_step=dt,
            starting_position=self.start_pos,
            num_steps=500, # Could be parameterized in Launcher if desired
            step_size=1.0
        )
        
        # B. Execute
        strategy = RandomWalkStrategy()
        return self.engine.generate_path(self.jammer_id, strategy, config)