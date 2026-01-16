import numpy as np
from typing import Dict, List, Tuple, Any

class MotionEngine:
    """
    Core logic handler for scene management, collision checking, and trajectory storage.
    
    This class acts as the interface between the physical constraints of the scene
    (obstacles, boundaries) and the motion strategies that generate paths.
    """

    def __init__(self, scene, obstacles: List[Dict], bounds: Dict[str, List[float]]):
        """
        Initialize the Motion Engine.

        Args:
            scene: The Sionna scene object (used for updating transmitter positions).
            obstacles: A list of dictionaries defining obstacle bounding boxes 
                       (e.g., [{'min': [x1, y1], 'max': [x2, y2]}, ...]).
            bounds: A dictionary defining the map boundaries 
                    (e.g., {'x': [-500, 500], 'y': [-500, 500]}).
        """
        self.scene = scene
        self.obstacles = obstacles if obstacles is not None else []
        self.bounds = bounds
        
        # Storage for generated paths: {jammer_id: numpy_array_of_positions}
        self.jammer_paths: Dict[str, np.ndarray] = {}
        
        # Storage for metadata (e.g., velocity, distance): {jammer_id: dict}
        self.jammer_metadata: Dict[str, Any] = {}
        
        # Preferences for how to pad paths (start vs end) for synchronization
        self.padding_preferences: Dict[str, str] = {}

    def is_position_valid(self, position: np.ndarray) -> bool:
        """
        Checks if a given 3D position is valid (within bounds and not inside an obstacle).
        
        Args:
            position: A numpy array [x, y, z].
            
        Returns:
            True if the position is valid, False otherwise.
        """
        # 1. Check Scene Boundaries
        if self.bounds:
            for i, axis in enumerate(['x', 'y', 'z']):
                if axis in self.bounds:
                    lower, upper = self.bounds[axis]
                    # Check if coordinate is strictly outside the allowed range
                    if not (lower <= position[i] <= upper):
                        return False

        # 2. Check Obstacles (Bounding Boxes)
        for obstacle in self.obstacles:
            if np.all(position >= obstacle['min']) and np.all(position <= obstacle['max']):
                return False
                
        return True

    def generate_path(self, jammer_id: str, strategy, config, padding_mode: str = 'pad_end') -> Tuple[np.ndarray, Dict]:
        """
        Executes a motion strategy to generate a path for a specific jammer.

        Args:
            jammer_id: Unique identifier for the transmitter/jammer.
            strategy: An instance of a MotionStrategy class (must implement .generate()).
            config: A configuration object/dataclass specific to the chosen strategy.
            padding_mode: 'pad_end' (stationary at finish) or 'pad_start' (stationary at start)
                          used when synchronizing multiple paths of different lengths.

        Returns:
            A tuple (path, metadata).
        """
        print(f"[{jammer_id}] Generating path using {strategy.__class__.__name__}...")
        
        # The strategy uses the engine (self) to validate steps against obstacles
        path, metadata = strategy.generate(self, config)
        
        # Store results
        self.jammer_paths[jammer_id] = path
        self.jammer_metadata[jammer_id] = metadata
        self.padding_preferences[jammer_id] = padding_mode
        
        return path, metadata

    def get_max_path_length(self) -> int:
        """Returns the number of steps in the longest currently stored path."""
        if not self.jammer_paths:
            return 0
        return max(len(path) for path in self.jammer_paths.values())

    def finalize_trajectories(self):
        """
        Synchronizes all stored paths by padding them to match the length of the 
        longest path. This ensures that during a simulation loop, all arrays 
        have the same size.
        """
        if not self.jammer_paths:
            return

        max_steps = self.get_max_path_length()
        
        for jammer_id, path in self.jammer_paths.items():
            current_len = len(path)
            
            if current_len < max_steps:
                pad_amount = max_steps - current_len
                mode = self.padding_preferences.get(jammer_id, 'pad_end')
                
                if mode == 'pad_end':
                    # Repeat the last position
                    last_pos = path[-1]
                    padding = np.tile(last_pos, (pad_amount, 1))
                    self.jammer_paths[jammer_id] = np.vstack([path, padding])
                    
                elif mode == 'pad_start':
                    # Repeat the first position
                    first_pos = path[0]
                    padding = np.tile(first_pos, (pad_amount, 1))
                    self.jammer_paths[jammer_id] = np.vstack([padding, path])
                
                print(f"[{jammer_id}] Padded path length from {current_len} to {max_steps} (Mode: {mode}).")

    def update_scene_transmitters(self, step_index: int):
        """
        Updates the actual Sionna scene objects for a specific time step.
        
        Args:
            step_index: The current simulation frame index.
        """
        for jammer_id, path in self.jammer_paths.items():
            if step_index < len(path):
                try:
                    # Access the transmitter in the Sionna scene
                    tx = self.scene.get(jammer_id)
                    tx.position = path[step_index]
                except KeyError:
                    print(f"Warning: Transmitter '{jammer_id}' not found in Sionna scene.")
                except Exception as e:
                    print(f"Error updating '{jammer_id}': {e}")

    def get_all_positions_at_step(self, step_index: int) -> Dict[str, np.ndarray]:
        """
        Retrieves a dictionary of positions for all jammers at a given time step.
        Useful for visualization or debugging without touching the scene directly.
        """
        return {
            jid: path[step_index] 
            for jid, path in self.jammer_paths.items() 
            if step_index < len(path)
        }