import numpy as np
from scipy.interpolate import CubicSpline
from typing import List, Any, Dict
import heapq
import collections

def check_line_of_sight(engine: Any, p1: np.ndarray, p2: np.ndarray, step_size: float = 2.0) -> bool:
    """
    Checks if a straight line between p1 and p2 is valid (collision-free).
    Samples the segment at discrete intervals.
    """
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-3:
        return True

    direction = (p2 - p1) / dist
    steps = int(dist / step_size)
    
    # Sample points along the line
    for i in range(1, steps + 1):
        test_point = p1 + direction * (i * step_size)
        # We assume the engine acts as the source of truth for validity
        if not engine.is_position_valid(test_point):
            return False
            
    return True

def calculate_smooth_path(points: np.ndarray, resolution_per_meter: float = 2.0) -> np.ndarray:
    """
    Applies Cubic Spline interpolation to smooth a path of control points.
    Handles 3D paths correctly by interpolating X, Y, and Z.
    """
    if len(points) < 3:
        return points  # Not enough points for cubic spline
        
    # 1. Calculate cumulative distance (chord length parameterization)
    # Updated: Now uses all dimensions (axis=1) instead of just X/Y to support 3D distance
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    cumulative_dist = np.insert(np.cumsum(dists), 0, 0.0)
    
    # 2. Handle duplicates (Spline requires strictly increasing x)
    # If consecutive points are identical, add a tiny epsilon
    if np.any(np.diff(cumulative_dist) < 1e-5):
        cumulative_dist += np.arange(len(cumulative_dist)) * 1e-5

    try:
        # 3. Create Splines for X, Y, and Z
        cs_x = CubicSpline(cumulative_dist, points[:, 0], bc_type='natural')
        cs_y = CubicSpline(cumulative_dist, points[:, 1], bc_type='natural')
        cs_z = CubicSpline(cumulative_dist, points[:, 2], bc_type='natural')
        
        # 4. Resample
        total_len = cumulative_dist[-1]
        num_samples = int(total_len * resolution_per_meter)
        num_samples = max(num_samples, len(points) * 2) # Ensure decent resolution
        
        ts = np.linspace(0, total_len, num_samples)
        
        xs = cs_x(ts)
        ys = cs_y(ts)
        zs = cs_z(ts)
        
        return np.column_stack((xs, ys, zs))
        
    except Exception as e:
        print(f"Spline smoothing failed: {e}. Returning linear path.")
        return points
    
def a_star_search(nodes: List[np.ndarray], adjacency: Dict[int, List[int]], start_idx: int, end_idx: int) -> List[np.ndarray]:
    """
    Standard A* pathfinding algorithm on a pre-built graph.
    
    Args:
        nodes: List of 3D points (the graph vertices).
        adjacency: Dictionary mapping node_index -> list of neighbor_indices.
        start_idx: Index of start node.
        end_idx: Index of target node.
        
    Returns:
        List of points [Node_start, ..., Node_end] or empty list if no path.
    """
    # Priority Queue: (f_score, node_index)
    open_set = []
    heapq.heappush(open_set, (0, start_idx))
    
    came_from = {}
    
    g_score = collections.defaultdict(lambda: float('inf'))
    g_score[start_idx] = 0
    
    f_score = collections.defaultdict(lambda: float('inf'))
    # Heuristic: Euclidean distance to target
    f_score[start_idx] = float(np.linalg.norm(nodes[start_idx] - nodes[end_idx]))
    
    visited = set()

    while open_set:
        current = heapq.heappop(open_set)[1]
        
        if current == end_idx:
            return _reconstruct_path(nodes, came_from, current)
        
        if current in visited:
            continue
        visited.add(current)
        
        for neighbor in adjacency[current]:
            # Distance between current and neighbor (edge weight)
            dist = np.linalg.norm(nodes[current] - nodes[neighbor])
            tentative_g = g_score[current] + dist
            
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = float(tentative_g)
                f_score[neighbor] = float(tentative_g) + float(np.linalg.norm(nodes[neighbor] - nodes[end_idx]))
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
    return []

def _reconstruct_path(nodes: List[np.ndarray], came_from: Dict[int, int], current: int) -> List[np.ndarray]:
    path_indices = [current]
    while current in came_from:
        current = came_from[current]
        path_indices.append(current)
    path_indices.reverse()
    return [nodes[i] for i in path_indices]