import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, List
from config import BaseMotionConfig, RandomWalkConfig, MathModelingConfig, WaypointConfig, GraphNavConfig, MathSegment
from core.utils import calculate_smooth_path, a_star_search
from scipy.spatial import KDTree
import collections
import os

class MotionStrategy(ABC):
    """
    Abstract base class for all path generation logic.
    """
    @abstractmethod
    def generate(self, engine: Any, config: BaseMotionConfig) -> Tuple[np.ndarray, Dict]:
        """
        Generates a path based on the provided configuration.
        
        Args:
            engine: The MotionEngine instance (provides bounds/obstacle checks).
            config: A specific configuration data class.
            
        Returns:
            (path, metadata): (N,3) position array and metadata dict.
        """
        pass

# ==========================================
# 1. Random Walk Strategy
# ==========================================

class RandomWalkStrategy(MotionStrategy):
    """
    Generates a stochastic path that strictly avoids obstacles defined in the engine.
    """
    def generate(self, engine: Any, config: RandomWalkConfig) -> Tuple[np.ndarray, Dict]: # type: ignore
        # 1. Setup
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        path = np.zeros((config.num_steps, 3))
        path[0] = config.starting_position
        
        # We assume movement is primarily on the XY plane, preserving initial Z
        z_height = config.starting_position[2] # type: ignore

        # 2. Iterative Step Generation
        for step in range(1, config.num_steps):
            previous_position = path[step - 1]
            valid_step_found = False
            
            for _ in range(config.max_retries):
                # Generate random direction in XY plane
                direction = np.random.randn(2)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction /= norm
                
                # Create candidate
                offset = np.array([direction[0], direction[1], 0.0]) * config.step_size
                candidate_position = previous_position + offset
                candidate_position[2] = z_height 
                
                # Check against engine's world truth (obstacles/bounds)
                if engine.is_position_valid(candidate_position):
                    path[step] = candidate_position
                    valid_step_found = True
                    break
            
            # Fallback: if stuck, stay in place
            if not valid_step_found:
                path[step] = previous_position

        # 3. Calculate Metadata
        # (Vectorized calculation of distance)
        diffs = np.diff(path, axis=0)
        total_distance = np.sum(np.linalg.norm(diffs, axis=1))
        
        metadata = {
            "strategy": "RandomWalk",
            "total_distance": total_distance,
            "avg_velocity": config.step_size / config.time_step,
            "duration": config.num_steps * config.time_step
        }
        
        return path, metadata

# ==========================================
# 2. Math Modeling Strategy
# ==========================================

class MathStrategy(MotionStrategy):
    """
    Deterministic Kinematic Strategy.
    
    It does NOT launch a GUI. It expects the segments to be already defined
    in the `config.segments` list. It simply interpolates/samples them 
    into a discrete path array.
    """
    def generate(self, engine: Any, config: MathModelingConfig) -> Tuple[np.ndarray, Dict]: # type: ignore
        
        # Edge case: No segments provided
        if not config.segments:
            raise RuntimeError("MathStrategy Error: No segments provided in configuration.")

        # 1. Define the Global Timeline
        total_duration = sum(seg.duration for seg in config.segments)
        
        # Create uniform time steps: 0, dt, 2*dt ... Total
        # using config.time_step from the base class
        t_global = np.arange(0, total_duration, config.time_step)
        
        # Ensure final time is included to complete the path
        if len(t_global) == 0 or t_global[-1] < total_duration - 1e-9:
            t_global = np.append(t_global, total_duration)

        full_path = []
        
        # 2. Sample along the timeline
        current_seg_idx = 0
        current_seg_start_time = 0.0
        
        for t in t_global:
            # Advance segment if t has moved past the current segment's duration
            # We use a while loop to handle cases where dt is larger than a tiny segment
            while (current_seg_idx < len(config.segments) - 1 and 
                   t >= current_seg_start_time + config.segments[current_seg_idx].duration - 1e-9):
                current_seg_start_time += config.segments[current_seg_idx].duration
                current_seg_idx += 1
            
            seg = config.segments[current_seg_idx]
            
            # Local time relative to the specific segment start
            t_local = t - current_seg_start_time
            
            # Clamp to prevent tiny float math overshoots
            if t_local > seg.duration: t_local = seg.duration
            if t_local < 0: t_local = 0
            
            # Calculate Physics
            pos = self._calculate_position_at_t(seg, t_local)
            full_path.append(pos)
            
        path_array = np.array(full_path)
        
        # 3. Metadata
        diffs = np.diff(path_array, axis=0)
        total_dist = np.sum(np.linalg.norm(diffs, axis=1))

        metadata = {
            "strategy": "MathModeling",
            "total_distance": total_dist,
            "total_duration": total_duration,
            "num_segments": len(config.segments)
        }

        return path_array, metadata
    
    def _calculate_position_at_t(self, seg: MathSegment, t: float) -> np.ndarray:
        """
        Pure physics calculation for a single point in time within a segment.
        """
        x0, y0, z0 = seg.start_pos
        theta = seg.start_heading
        mode = seg.mode
        
        # Default initialization
        x, y = x0, y0

        if mode == "Const Vel":
            v = seg.params.get('velocity', 0.0)
            x = x0 + v * np.cos(theta) * t
            y = y0 + v * np.sin(theta) * t
            
        elif mode == "Const Accel":
            v0 = seg.start_vel
            a = seg.params.get('accel', 0.0)
            dist = v0 * t + 0.5 * a * (t**2)
            x = x0 + dist * np.cos(theta)
            y = y0 + dist * np.sin(theta)
            
        elif mode == "Turn":
            v = seg.params.get('velocity', 0.0)
            omega_deg = seg.params.get('turn_rate', 0.0)
            omega = np.deg2rad(omega_deg)
            
            if abs(omega) < 1e-4:
                # Treat as straight line if turn rate is near zero
                x = x0 + v * np.cos(theta) * t
                y = y0 + v * np.sin(theta) * t
            else:
                r = v / omega
                # Standard circular motion formulas
                x = x0 + r * (np.sin(omega * t + theta) - np.sin(theta))
                y = y0 - r * (np.cos(omega * t + theta) - np.cos(theta))

        return np.array([x, y, z0])
    
# ==========================================
# 3. Waypoint Strategy
# ==========================================

class WaypointStrategy(MotionStrategy):
    def generate(self, engine: Any, config: WaypointConfig) -> Tuple[np.ndarray, Dict]: # type: ignore
        # 1. Prepare Control Points
        all_points = [config.starting_position]
        for wp in config.waypoints:
            if np.linalg.norm(wp - all_points[-1]) > 1e-4:
                all_points.append(wp)

        points_array = np.array(all_points)
        
        if len(points_array) < 2:
            raise RuntimeError("WaypointStrategy Error: At least one waypoint is required.")

        # 2. Generate Geometry & COLLISION CHECK
        # Default to linear
        geometric_path = points_array
        is_smoothed = False

        if config.enable_smoothing and len(points_array) >= 3:
            # Try to generate smooth path
            candidate_path = calculate_smooth_path(points_array, resolution_per_meter=10.0)
            
            is_valid = True
            for point in candidate_path:
                if not engine.is_position_valid(point):
                    is_valid = False
                    break
            
            if is_valid:
                geometric_path = candidate_path
                is_smoothed = True
            else:
                print("Warning: Smoothed path collides with obstacle. Reverting to Linear Path.")

        # 3. Time Parameterization (Global Distance)
        dists = np.linalg.norm(np.diff(geometric_path, axis=0), axis=1)
        cumulative_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_length = cumulative_dist[-1]

        total_duration = total_length / config.velocity
        t_eval = np.arange(0, total_duration, config.time_step)
        if t_eval.size == 0 or t_eval[-1] < total_duration:
            t_eval = np.append(t_eval, total_duration)

        target_distances = t_eval * config.velocity
        
        final_x = np.interp(target_distances, cumulative_dist, geometric_path[:, 0])
        final_y = np.interp(target_distances, cumulative_dist, geometric_path[:, 1])
        final_z = np.interp(target_distances, cumulative_dist, geometric_path[:, 2])
        
        path_array = np.column_stack((final_x, final_y, final_z))

        # 4. Metadata
        metadata = {
            "strategy": "Waypoint",
            "mode": "Smoothed" if is_smoothed else "Linear (Fallback)",
            "total_distance": total_length,
            "velocity": config.velocity,
            "duration": total_duration
        }
        
        return path_array, metadata
    
# ==========================================
# 4. Graph Navigation Strategy (PRM)
# ==========================================

class GraphNavStrategy(MotionStrategy):
    """
    Probabilistic Roadmap (PRM) Strategy.
    Generates a dataset of paths by repeatedly querying the graph.
    """

    def __init__(self):
        # Move cache to instance variable to prevent data leaking between runs
        self._cache = {
            "nodes": [],
            "adjacency": collections.defaultdict(list),
            "built": False
        }

    def generate(self, engine: Any, config: GraphNavConfig) -> Tuple[np.ndarray, Dict]: # type: ignore
        
        # 1. Load Graph (Lazy Loading)
        if not self._cache["built"]:
            if config.precomputed_nodes and config.precomputed_adjacency:
                self._cache["nodes"] = config.precomputed_nodes
                self._cache["adjacency"] = config.precomputed_adjacency
                self._cache["built"] = True
            else:
                raise RuntimeError("GraphNav Error: Graph not built. Please run the GUI to build the graph first.")

        # 2. Search Loop        
        valid_path = None
        attempts = 0
        max_attempts = 100 
        
        num_nodes = len(self._cache["nodes"])
        if num_nodes < 2:
             raise RuntimeError("Graph is too small (less than 2 nodes).")

        while attempts < max_attempts:
            attempts += 1

            # A. Pick Random Start & End
            start_node_idx = np.random.randint(0, num_nodes)
            end_node_idx = np.random.randint(0, num_nodes)
            
            if start_node_idx == end_node_idx: 
                continue

            # B. Pathfinding (A* Search)
            raw_waypoints = a_star_search(
                self._cache["nodes"], 
                self._cache["adjacency"], 
                int(start_node_idx), 
                int(end_node_idx)
            )
            
            if not raw_waypoints: 
                continue # Nodes are not connected in the graph
            
            path_array = np.array(raw_waypoints)

            # C. Filter: Minimum Distance
            diffs = np.diff(path_array, axis=0)
            total_dist = np.sum(np.linalg.norm(diffs, axis=1))

            if total_dist < config.min_path_distance: 
                continue
            
            # D. Smoothing & Validation
            if config.enable_smoothing:
                smoothed = calculate_smooth_path(path_array, resolution_per_meter=10.0)
                
                # Verify safety of smoothed path (it might cut corners into obstacles)
                is_safe = True
                for point in smoothed:
                    if not engine.is_position_valid(point):
                        is_safe = False
                        break
                
                if is_safe:
                    path_array = smoothed
                else:
                    # If smoothing fails, we could fallback to raw_waypoints, 
                    # but usually we want high quality paths, so we skip.
                    continue 

            valid_path = path_array
            break 

        if valid_path is None:
            raise RuntimeError("GraphNav Error: Failed to find a valid path after maximum attempts.")
        
        # ---------------------------------------------------------
        # 3. Time Parameterization
        # ---------------------------------------------------------
        # We now transform the geometric path into a timed trajectory
        # based on config.velocity and config.time_step
        
        # Calculate cumulative distance along the path
        dists = np.linalg.norm(np.diff(valid_path, axis=0), axis=1)
        cumulative_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_length = cumulative_dist[-1]

        # Calculate total duration required            
        total_duration = total_length / config.velocity
        
        # Generate time steps
        t_eval = np.arange(0, total_duration, config.time_step)
        if t_eval.size == 0 or t_eval[-1] < total_duration:
            t_eval = np.append(t_eval, total_duration)

        # Map time steps to distance along the path
        target_distances = t_eval * config.velocity
        
        # Interpolate positions (Resample)
        final_x = np.interp(target_distances, cumulative_dist, valid_path[:, 0])
        final_y = np.interp(target_distances, cumulative_dist, valid_path[:, 1])
        final_z = np.interp(target_distances, cumulative_dist, valid_path[:, 2])
        
        final_trajectory = np.column_stack((final_x, final_y, final_z))

        # 4. Metadata
        metadata = {
            "strategy": "GraphNav",
            "graph_nodes": num_nodes,
            "attempts": attempts,
            "total_distance": float(total_length),
            "duration": float(total_duration),
            "velocity": float(config.velocity)
        }
        
        return final_trajectory, metadata