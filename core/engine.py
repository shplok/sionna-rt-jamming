import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from config import BaseMotionConfig
from core.strategies import MotionStrategy

class MotionEngine:
    """
    Core logic handler for scene management, collision checking, and trajectory storage.
    Acts as the single source of truth for the physical world constraints.
    """

    def __init__(self, scene: Any, obstacles: List[Dict[str, Any]], bounds: Dict[str, List[float]]):
        self.scene = scene
        self.bounds = bounds
        
        self.obstacles = obstacles if obstacles is not None else []
        
        if self.obstacles:
            # Create (N, 3) arrays for min and max coordinates
            self._obs_min = np.array([o['min'] for o in self.obstacles])
            self._obs_max = np.array([o['max'] for o in self.obstacles])
        else:
            self._obs_min = np.empty((0, 3))
            self._obs_max = np.empty((0, 3))

        # Internal storage
        self._jammer_paths: Dict[str, np.ndarray] = {}
        self._jammer_metadata: Dict[str, Any] = {}
        self._padding_preferences: Dict[str, str] = {}

    def is_position_valid(self, position: np.ndarray) -> bool:
        """
        Checks if a given 3D position is valid (within bounds and not inside an obstacle).
        Optimized to avoid Python loops.
        """
        # 1. Check Scene Boundaries
        if self.bounds:
            if 'x' in self.bounds:
                if not (self.bounds['x'][0] <= position[0] <= self.bounds['x'][1]): return False
            if 'y' in self.bounds:
                if not (self.bounds['y'][0] <= position[1] <= self.bounds['y'][1]): return False
            if 'z' in self.bounds:
                if not (self.bounds['z'][0] <= position[2] <= self.bounds['z'][1]): return False

        # 2. Check Obstacles (Vectorized AABB Collision Check)
        if self._obs_min.shape[0] == 0:
            return True
        
        # A point is inside a box if: point >= min AND point <= max (for all x,y,z)
        inside_mask = np.all(
            (position >= self._obs_min) & (position <= self._obs_max), 
            axis=1
        )
        
        # If inside_mask is True for any building, it's a collision
        if np.any(inside_mask):
            return False
                
        return True

    def generate_path(self, 
                      jammer_id: str, 
                      strategy: MotionStrategy, 
                      config: BaseMotionConfig, 
                      padding_mode: str = 'pad_end') -> Tuple[np.ndarray, Dict]:
        """
        Executes a motion strategy to generate a path for a specific jammer.
        """
        # Validate that the config object matches the expected base class
        if not isinstance(config, BaseMotionConfig):
            raise TypeError("Config must inherit from BaseMotionConfig")

        # Execute strategy (The strategy will callback engine.is_position_valid if needed)
        path, metadata = strategy.generate(self, config)
        
        # Store results internally
        self._jammer_paths[jammer_id] = path
        self._jammer_metadata[jammer_id] = metadata
        self._padding_preferences[jammer_id] = padding_mode
        
        return path, metadata
    
    def set_padding_mode(self, jammer_id: str, mode: str):
        """Updates the padding preference for an existing jammer."""
        self._padding_preferences[jammer_id] = mode

    def get_all_paths(self) -> Dict[str, np.ndarray]:
        """Returns the dictionary of all generated paths."""
        return self._jammer_paths

    def get_all_metadata(self) -> Dict[str, Any]:
        """Returns the dictionary of all metadata."""
        return self._jammer_metadata

    def get_path(self, jammer_id: str) -> Optional[np.ndarray]:
        """Accessor to retrieve a stored path."""
        return self._jammer_paths.get(jammer_id)

    def get_max_path_length(self) -> int:
        """Returns the number of steps in the longest currently stored path."""
        if not self._jammer_paths:
            return 0
        return max(len(path) for path in self._jammer_paths.values())

    def finalize_trajectories(self):
        """
        Synchronizes all stored paths by padding them to match the length of the 
        longest path. Ensures consistent array sizes for the simulation loop.
        """
        if not self._jammer_paths:
            return

        max_steps = self.get_max_path_length()
        
        for jammer_id, path in self._jammer_paths.items():
            current_len = len(path)
            
            if current_len < max_steps:
                pad_amount = max_steps - current_len
                mode = self._padding_preferences.get(jammer_id, 'pad_end')
                
                if mode == 'pad_end':
                    # Repeat the last position (stationary at end)
                    last_pos = path[-1]
                    padding = np.tile(last_pos, (pad_amount, 1))
                    self._jammer_paths[jammer_id] = np.vstack([path, padding])
                    
                elif mode == 'pad_start':
                    # Repeat the first position (delayed start)
                    first_pos = path[0]
                    padding = np.tile(first_pos, (pad_amount, 1))
                    self._jammer_paths[jammer_id] = np.vstack([padding, path])

    def update_scene_transmitters(self, step_index: int):
        """
        Updates the actual Sionna scene objects for a specific time step.
        """
        for jammer_id, path in self._jammer_paths.items():
            if step_index < len(path):
                try:
                    # Update Sionna object directly
                    tx = self.scene.get(jammer_id)
                    tx.position = path[step_index]
                except Exception as e:
                    print(f"Error updating transmitter '{jammer_id}' at step {step_index}: {e}")