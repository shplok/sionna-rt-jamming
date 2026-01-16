import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ==========================================
# Base Configuration
# ==========================================

@dataclass
class BaseMotionConfig:
    """
    Base configuration class for all motion strategies.
    Holds parameters common to any path generation task.
    """
    strategy_type: str
    time_step: float
    starting_position: Optional[np.ndarray] = None
    output_path: Optional[str] = None

# ==========================================
# 1. Random Walk Configuration
# ==========================================

@dataclass
class RandomWalkConfig(BaseMotionConfig):
    """
    Configuration specific to the Stochastic/Random Walk strategy.
    """
    num_steps: int = 500
    step_size: float = 1.0
    max_retries: int = 10
    random_seed: Optional[int] = None

# ==========================================
# 2. Math Modeling Configuration
# ==========================================

@dataclass
class MathSegment:
    """
    Represents a single kinematic segment (chunk) of a path.
    Used to transfer data from the GUI to the Strategy logic.
    """
    mode: str                  # e.g., "Const Vel", "Const Accel", "Turn"
    duration: float            # Duration in seconds
    start_pos: np.ndarray      # Position at the start of this segment
    start_heading: float       # Heading (radians) at start
    start_vel: float           # Velocity (m/s) at start
    
    # Specific parameters for the mode (e.g., {'accel': 2.0} or {'turn_rate': 15.0})
    params: Dict[str, float] = field(default_factory=dict)

@dataclass
class MathModelingConfig(BaseMotionConfig):
    """
    Configuration for the deterministic Math/Kinematic strategy.
    Contains the list of segments designed by the user.
    """
    variable_duration: bool = False
    segments: List[MathSegment] = field(default_factory=list)

# ==========================================
# 3. Waypoint Configuration
# ==========================================
@dataclass
class WaypointConfig(BaseMotionConfig):
    """
    Configuration for a Waypoint-based strategy.
    """
    waypoints: List[np.ndarray] = field(default_factory=list)
    velocity: float = 5.0
    enable_smoothing: bool = True


# ==========================================
# 4. Procedural Generation Configuration
# ==========================================
@dataclass
class GraphNavConfig(BaseMotionConfig):
    """
    Configuration for Graph Navigation Strategy.
    """
    # Graph Building Parameters
    num_samples: int = 1500          # How many random nodes to scatter in the city
    max_connection_radius: float = 200.0 # Max distance to connect two nodes
    min_connection_radius: float = 20.0 # Min distance to place a node from existing nodes
    enable_smoothing: bool = True
    
    # Launcher Parameters
    num_simulations: int = 100        # How many simulations to run
    min_path_distance: float = 100.0  # Minimum Euclidean distance between Start and End nodes
    velocity: float = 1.0              # Constant velocity for the jammer

    precomputed_nodes: Optional[List[np.ndarray]] = None
    precomputed_adjacency: Optional[Dict[int, List[int]]] = None