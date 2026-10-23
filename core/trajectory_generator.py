"""
Modular Trajectory Generator for Sionna RT Jamming.

Provides:
- Straight-line trajectory generation (constant velocity vector, zero acceleration).
- Curved trajectory generation (cubic spline, constant speed).
- Obstacle-aware street corridor discovery in urban scenes (e.g. NYC).
- Customizable start/end positions and headings.
- Clean serialization of NumPy arrays and JSON metadata.
"""

import os
import json
import math
import pickle
import heapq
import collections
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from core.engine import MotionEngine
from core.utils import check_line_of_sight, calculate_smooth_path
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


class TrajectorySpec:
    """Configuration for a single trajectory."""

    def __init__(
        self,
        duration: float,                  # Total duration in seconds (e.g. 30, 60, 90)
        velocity: float,                  # Speed in m/s (e.g. 3, 9, 15)
        time_step: float = 1.0,           # Frame interval in seconds (1 fps)
        mode: str = "straight",           # "straight" or "curved"
        start_pos: Optional[np.ndarray] = None,
        end_pos: Optional[np.ndarray] = None,
        heading_deg: Optional[float] = None,  # Angle in degrees for straight line (0 = +X, 90 = +Y)
        z_height: float = 1.5,
        inclusive_end: bool = False,      # False -> duration/dt frames (30s -> 30 frames); True -> 31 frames
        name: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.duration = float(duration)
        self.velocity = float(velocity)
        self.time_step = float(time_step)
        self.mode = mode.lower()
        self.start_pos = np.asarray(start_pos, dtype=float) if start_pos is not None else None
        self.end_pos = np.asarray(end_pos, dtype=float) if end_pos is not None else None
        self.heading_deg = heading_deg
        self.z_height = float(z_height)
        self.inclusive_end = inclusive_end
        self.name = name or f"traj_dur{int(duration)}s_vel{int(velocity)}mps"
        self.extra_metadata = extra_metadata or {}

        # Compute number of frames
        self.num_frames = int(round(self.duration / self.time_step))
        if self.inclusive_end:
            self.num_frames += 1

        # Distance covered across frames
        self.travel_distance = (self.num_frames - 1) * self.velocity * self.time_step


def compute_trajectory_overlap(
    traj1: np.ndarray,
    traj2: np.ndarray,
    proximity_threshold: float = 20.0,
) -> float:
    """
    Computes the mutual overlap ratio between two trajectories (straight or curved).
    Two trajectories are considered overlapping at points where they are within
    `proximity_threshold` meters of each other (e.g. traveling along the same street canyon).

    The overlap ratio is defined as:
        max(fraction of traj1 close to traj2, fraction of traj2 close to traj1)

    Returns:
        float in [0.0, 1.0]: 0.0 means completely separate, 1.0 means fully overlapping.
    """
    pts1 = traj1[:, :2]
    pts2 = traj2[:, :2]

    # If inputs are just 2 endpoints (e.g. candidate segment [p0, p_end]), sample densely
    if len(pts1) < 20:
        n_samples1 = max(len(pts1), 50)
        diffs1 = np.diff(pts1, axis=0)
        dists1 = np.linalg.norm(diffs1, axis=1)
        cum1 = np.insert(np.cumsum(dists1), 0, 0.0)
        if cum1[-1] > 1e-4:
            s_eval1 = np.linspace(0, cum1[-1], n_samples1)
            pts1 = np.column_stack((
                np.interp(s_eval1, cum1, pts1[:, 0]),
                np.interp(s_eval1, cum1, pts1[:, 1])
            ))

    if len(pts2) < 20:
        n_samples2 = max(len(pts2), 50)
        diffs2 = np.diff(pts2, axis=0)
        dists2 = np.linalg.norm(diffs2, axis=1)
        cum2 = np.insert(np.cumsum(dists2), 0, 0.0)
        if cum2[-1] > 1e-4:
            s_eval2 = np.linspace(0, cum2[-1], n_samples2)
            pts2 = np.column_stack((
                np.interp(s_eval2, cum2, pts2[:, 0]),
                np.interp(s_eval2, cum2, pts2[:, 1])
            ))

    dists_matrix = cdist(pts1, pts2)
    d1 = np.min(dists_matrix, axis=1)
    d2 = np.min(dists_matrix, axis=0)

    ratio1 = float(np.mean(d1 <= proximity_threshold))
    ratio2 = float(np.mean(d2 <= proximity_threshold))

    return max(ratio1, ratio2)


def simplify_collinear(points: np.ndarray, angle_tol_deg: float = 15.0) -> np.ndarray:
    """Removes intermediate collinear points along straight roads while preserving intersection turns."""
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    for i in range(1, len(points) - 1):
        v1 = points[i] - simplified[-1]
        v2 = points[i + 1] - points[i]
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)
        if d1 < 1e-3 or d2 < 1e-3:
            continue
        u1 = v1 / d1
        u2 = v2 / d2
        cos_ang = np.clip(np.dot(u1, u2), -1.0, 1.0)
        ang = math.degrees(math.acos(cos_ang))
        if ang > angle_tol_deg:
            simplified.append(points[i])
    simplified.append(points[-1])
    return np.array(simplified)


def round_corners(points: np.ndarray, corner_radius: float = 10.0, num_arc_pts: int = 10) -> np.ndarray:
    """
    Rounds sharp turns in a piecewise-linear path using quadratic Bezier fillets.
    This guarantees that the trajectory smoothly turns around corners without
    overshooting into buildings (avoiding cubic spline overshoot).
    """
    if len(points) <= 2:
        return points

    refined = [points[0]]
    for i in range(1, len(points) - 1):
        p_prev = points[i - 1]
        p_curr = points[i]
        p_next = points[i + 1]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        d1 = np.linalg.norm(v1)
        d2 = np.linalg.norm(v2)

        if d1 < 1e-3 or d2 < 1e-3:
            continue

        u1 = v1 / d1
        u2 = v2 / d2

        cos_angle = np.clip(np.dot(u1, u2), -1.0, 1.0)
        if cos_angle < -0.96:
            refined.append(p_curr)
            continue

        r = min(corner_radius, d1 * 0.4, d2 * 0.4)
        p_start = p_curr + r * u1
        p_end = p_curr + r * u2

        ts = np.linspace(0.0, 1.0, num_arc_pts)
        for t in ts:
            b_pt = ((1.0 - t) ** 2) * p_start + 2.0 * (1.0 - t) * t * p_curr + (t ** 2) * p_end
            refined.append(b_pt)

    refined.append(points[-1])
    return np.array(refined)


def build_city_prm(
    engine: MotionEngine,
    bounds: Dict[str, Any],
    num_samples: int = 2500,
    radius: float = 180.0,
    min_dist: float = 22.0,
    cache_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Constructs a generic Probabilistic Roadmap (PRM) of collision-free street network
    for ANY city layout (Manhattan, European curved, synthetic, etc.).
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return np.array(data["nodes"]), data["adjacency"]

    x_min, x_max = bounds['x']
    y_min, y_max = bounds['y']
    z_h = bounds['z'][0]

    obs_tree = None
    if engine._obs_min is not None and len(engine._obs_min) > 0:
        obs_centers = 0.5 * (engine._obs_min[:, :2] + engine._obs_max[:, :2])
        obs_tree = KDTree(obs_centers)

    nodes = []
    attempts = 0
    while len(nodes) < num_samples and attempts < 150:
        batch = 600
        xs = np.random.uniform(x_min, x_max, batch)
        ys = np.random.uniform(y_min, y_max, batch)
        zs = np.full(batch, z_h)
        cand = np.column_stack((xs, ys, zs))

        valid = []
        for p in cand:
            if engine.is_position_valid(p):
                if obs_tree is not None:
                    d_bldg, _ = obs_tree.query(p[:2])
                    if d_bldg <= 80.0:
                        valid.append(p)
                else:
                    valid.append(p)

        if not nodes and valid:
            nodes.append(valid.pop(0))

        if nodes and valid:
            tree = KDTree(nodes)
            for p in valid:
                if len(nodes) >= num_samples:
                    break
                d, _ = tree.query(p)
                if d >= min_dist:
                    nodes.append(p)
        attempts += 1

    tree = KDTree(nodes)
    k_neighbors = 16
    dists, indices = tree.query(nodes, k=k_neighbors + 1, distance_upper_bound=radius)

    adjacency = collections.defaultdict(list)
    for i, (nbr_dists, nbr_indices) in enumerate(zip(dists, indices)):
        p1 = nodes[i]
        for d, j in zip(nbr_dists, nbr_indices):
            if np.isinf(d) or i >= j or j >= len(nodes):
                continue
            p2 = nodes[j]
            if check_line_of_sight(engine, p1, p2, step_size=4.0):
                adjacency[i].append(j)
                adjacency[j].append(i)

    visited = set()
    components = []
    for i in range(len(nodes)):
        if i not in visited:
            comp = []
            q = [i]
            visited.add(i)
            for u in q:
                comp.append(u)
                for v in adjacency[u]:
                    if v not in visited:
                        visited.add(v)
                        q.append(v)
            components.append(comp)

    components.sort(key=len, reverse=True)
    if components:
        main_nodes_indices = sorted(components[0])
        idx_map = {old_i: new_i for new_i, old_i in enumerate(main_nodes_indices)}
        filt_nodes = [nodes[i] for i in main_nodes_indices]
        filt_adj = collections.defaultdict(list)
        for old_i in main_nodes_indices:
            new_i = idx_map[old_i]
            for nbr in adjacency[old_i]:
                if nbr in idx_map:
                    filt_adj[new_i].append(idx_map[nbr])
        nodes_arr = np.array(filt_nodes)
        adjacency = filt_adj
    else:
        nodes_arr = np.array(nodes)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"nodes": nodes_arr, "adjacency": adjacency}, f)

    return nodes_arr, adjacency



class TrajectoryGenerator:

    """
    Core engine to produce straight or curved collision-free trajectories.
    """

    def __init__(self, engine: Optional[MotionEngine] = None):
        self.engine = engine
        self._prm_nodes: Optional[np.ndarray] = None
        self._prm_adjacency: Optional[Dict[int, List[int]]] = None

    # =========================================================================
    # 1. Straight Line Generation (No turns, zero acceleration)
    # =========================================================================
    def generate_straight(
        self,
        spec: TrajectorySpec,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates a straight-line trajectory from spec.start_pos along a direction
        or toward spec.end_pos, with exactly constant velocity.
        """
        if spec.start_pos is None:
            raise ValueError("start_pos must be specified to generate a straight trajectory.")

        p0 = np.asarray(spec.start_pos, dtype=float)
        if len(p0) == 2:
            p0 = np.array([p0[0], p0[1], spec.z_height])

        # Determine unit direction vector
        if spec.end_pos is not None:
            p_target = np.asarray(spec.end_pos, dtype=float)
            diff = p_target[:2] - p0[:2]
            dist = np.linalg.norm(diff)
            if dist < 1e-4:
                raise ValueError("start_pos and end_pos are identical.")
            unit_dir = np.array([diff[0] / dist, diff[1] / dist, 0.0])
            heading_rad = math.atan2(unit_dir[1], unit_dir[0])
        elif spec.heading_deg is not None:
            heading_rad = math.radians(spec.heading_deg)
            unit_dir = np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])
        else:
            raise ValueError("Either end_pos or heading_deg must be specified.")

        # Generate points at each discrete timestep k: p(k) = p0 + k * dt * v * unit_dir
        frames = []
        for k in range(spec.num_frames):
            dist_k = k * spec.time_step * spec.velocity
            pos_k = p0 + dist_k * unit_dir
            frames.append(pos_k)

        trajectory = np.array(frames)
        end_pos_actual = trajectory[-1]

        # Verify collisions if engine is available
        is_collision_free = True
        collision_step = min(1.0, spec.velocity * spec.time_step / 2.0)
        if self.engine is not None:
            is_collision_free = self.is_path_collision_free(p0, end_pos_actual, step_size=collision_step)

        metadata = {
            "name": spec.name,
            "mode": "straight",
            "duration_seconds": spec.duration,
            "velocity_mps": spec.velocity,
            "time_step_seconds": spec.time_step,
            "num_frames": spec.num_frames,
            "total_distance_meters": float(np.linalg.norm(end_pos_actual - p0)),
            "start_position": p0.tolist(),
            "end_position": end_pos_actual.tolist(),
            "heading_degrees": math.degrees(heading_rad) % 360.0,
            "unit_direction": unit_dir.tolist(),
            "is_collision_free": is_collision_free,
            "kinematics": {
                "constant_velocity_vector": True,
                "tangential_acceleration": 0.0,
                "centripetal_acceleration": 0.0,
            },
            **spec.extra_metadata,
        }

        return trajectory, metadata

    # =========================================================================
    # 2. Curved / Spline Generation (Constant speed, centripetal acceleration)
    # =========================================================================
    def generate_curved(
        self,
        spec: TrajectorySpec,
        waypoints: List[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generates a smooth curved trajectory passing through waypoints using CubicSpline,
        resampled at constant scalar speed v along the arc length.
        """
        if len(waypoints) < 2:
            raise ValueError("Curved trajectory requires at least 2 control waypoints.")

        pts = np.asarray(waypoints, dtype=float)
        # Apply smoothing
        smooth_curve = calculate_smooth_path(pts, resolution_per_meter=10.0)

        # Compute cumulative distance along the curve
        diffs = np.diff(smooth_curve, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        cum_dist = np.insert(np.cumsum(dists), 0, 0.0)
        total_curve_len = cum_dist[-1]

        # Resample at target distances: k * dt * v
        target_distances = np.arange(spec.num_frames) * (spec.time_step * spec.velocity)

        if target_distances[-1] > total_curve_len:
            raise ValueError(
                f"Waypoints provide path length {total_curve_len:.1f}m, but required distance is {target_distances[-1]:.1f}m."
            )

        final_x = np.interp(target_distances, cum_dist, smooth_curve[:, 0])
        final_y = np.interp(target_distances, cum_dist, smooth_curve[:, 1])
        final_z = np.full_like(final_x, spec.z_height)
        trajectory = np.column_stack((final_x, final_y, final_z))

        # Collision check
        is_collision_free = True
        if self.engine is not None:
            for pt in trajectory:
                if not self.engine.is_position_valid(pt):
                    is_collision_free = False
                    break

        metadata = {
            "name": spec.name,
            "mode": "curved",
            "duration_seconds": spec.duration,
            "velocity_mps": spec.velocity,
            "time_step_seconds": spec.time_step,
            "num_frames": spec.num_frames,
            "total_distance_meters": float(target_distances[-1]),
            "start_position": trajectory[0].tolist(),
            "end_position": trajectory[-1].tolist(),
            "is_collision_free": is_collision_free,
            "kinematics": {
                "constant_speed": True,
                "constant_velocity_vector": False,
                "tangential_acceleration": 0.0,
                "has_centripetal_acceleration": True,
            },
            **spec.extra_metadata,
        }

        return trajectory, metadata

    # =========================================================================
    # 3. Collision and Street Corridor Discovery
    # =========================================================================
    def is_path_collision_free(
        self,
        p_start: np.ndarray,
        p_end: np.ndarray,
        step_size: float = 2.0,
    ) -> bool:
        """Checks if a straight line between p_start and p_end is collision-free using coarse-to-fine testing."""
        if self.engine is None:
            return True

        # Quick check: endpoints
        if not self.engine.is_position_valid(p_start):
            return False
        if not self.engine.is_position_valid(p_end):
            return False

        dist = float(np.linalg.norm(p_end[:2] - p_start[:2]))
        if dist < 1e-3:
            return True

        # Stage 1: Coarse check (fast fail on middle points)
        coarse_fractions = [0.5, 0.25, 0.75]
        for frac in coarse_fractions:
            p_test = p_start + frac * (p_end - p_start)
            if not self.engine.is_position_valid(p_test):
                return False

        # Stage 2: Medium check (every ~10m)
        if dist > 20.0:
            medium_steps = max(4, int(dist / 10.0))
            for i in range(1, medium_steps):
                p_test = p_start + (float(i) / medium_steps) * (p_end - p_start)
                if not self.engine.is_position_valid(p_test):
                    return False

        # Stage 3: Fine check (step_size)
        num_steps = int(dist / step_size)
        if num_steps > 0:
            for i in range(1, num_steps):
                p_test = p_start + (float(i) / num_steps) * (p_end - p_start)
                if not self.engine.is_position_valid(p_test):
                    return False

        return True


    def find_straight_street_corridors(
        self,
        required_distance: float,
        num_corridors: int = 6,
        z_height: float = 1.5,
        candidate_angles_deg: Optional[List[float]] = None,
        min_separation: float = 25.0,
        max_attempts: int = 30000,
        existing_trajectories: Optional[List[np.ndarray]] = None,
        max_overlap_ratio: float = 0.75,
        proximity_threshold: float = 20.0,
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Finds `num_corridors` straight street segments (start_pos, heading_deg)
        of length `required_distance` that:
        1. Are completely collision-free in the scene.
        2. Do not overlap or run very close (> max_overlap_ratio, default 75%)
           to any already accepted trajectory in existing_trajectories or this batch.

        Returns:
            List of (start_pos, heading_deg) tuples.
        """
        if self.engine is None:
            raise RuntimeError("MotionEngine must be provided to find collision-free corridors.")

        bounds = self.engine.bounds
        x_min, x_max = bounds['x'][0] + 50.0, bounds['x'][1] - 50.0
        y_min, y_max = bounds['y'][0] + 50.0, bounds['y'][1] - 50.0

        if candidate_angles_deg is None:
            # Manhattan grid dominant directions:
            # Avenues (~61.5 deg NNE, ~241.5 deg SSW)
            # Streets (~151.0 deg WNW, ~331.0 deg ESE)
            # Plus cardinal and small angular offsets for perfect alignment
            angles = []
            for base in [61.5, 151.0, 241.5, 331.0, 0.0, 90.0, 180.0, 270.0]:
                angles.extend([base - 1.0, base - 0.5, base, base + 0.5, base + 1.0])
        else:
            angles = candidate_angles_deg

        found_corridors: List[Tuple[np.ndarray, float]] = []
        batch_trajectories: List[np.ndarray] = []
        if existing_trajectories is not None:
            batch_trajectories.extend(existing_trajectories)

        attempts = 0

        while len(found_corridors) < num_corridors and attempts < max_attempts:
            attempts += 1

            # 1. Random start point within bounds
            p0 = np.array([
                np.random.uniform(x_min, x_max),
                np.random.uniform(y_min, y_max),
                z_height,
            ])

            # Must start in open space (street)
            if not self.engine.is_position_valid(p0):
                continue

            # 2. Pick a direction
            heading = float(np.random.choice(angles))
            heading_rad = math.radians(heading)
            unit_dir = np.array([math.cos(heading_rad), math.sin(heading_rad), 0.0])

            # 3. Calculate end point
            p_end = p0 + required_distance * unit_dir

            # End point must also be within bounds
            if not (bounds['x'][0] <= p_end[0] <= bounds['x'][1] and bounds['y'][0] <= p_end[1] <= bounds['y'][1]):
                continue

            # 4. Check separation from already found corridors in this batch
            too_close = False
            for prev_p0, prev_heading in found_corridors:
                if np.linalg.norm(p0 - prev_p0) < min_separation and abs(heading - prev_heading) < 1e-2:
                    too_close = True
                    break
            if too_close:
                continue

            # 5. Check Line of Sight (straight line collision against all buildings)
            if not self.is_path_collision_free(p0, p_end, step_size=2.0):
                continue

            # 6. Overlap check against all existing trajectories
            cand_traj = np.array([p0, p_end])
            is_overlapping = False
            for other_traj in batch_trajectories:
                overlap = compute_trajectory_overlap(
                    cand_traj,
                    other_traj,
                    proximity_threshold=proximity_threshold,
                )
                if overlap > max_overlap_ratio:
                    is_overlapping = True
                    break

            if is_overlapping:
                continue

            # Passed all checks!
            found_corridors.append((p0, heading))
            batch_trajectories.append(cand_traj)

        if len(found_corridors) < num_corridors:
            print(
                f"Warning: Found {len(found_corridors)}/{num_corridors} collision-free corridors of length {required_distance:.1f}m (overlap <= {max_overlap_ratio*100:.0f}%) after {attempts} attempts."
            )

        return found_corridors

    def get_clear_distance(
        self,
        p_start: np.ndarray,
        heading_deg: float,
        max_dist: float = 1500.0,
        step: float = 2.0,
    ) -> float:
        """
        Computes the maximum collision-free distance along heading_deg from p_start.
        """
        if self.engine is None:
            return max_dist
        rad = math.radians(heading_deg)
        u = np.array([math.cos(rad), math.sin(rad), 0.0])
        num_steps = int(max_dist / step)
        for s in range(1, num_steps + 1):
            pt = p_start + s * step * u
            if not self.engine.is_position_valid(pt):
                return (s - 1) * step
        return max_dist

    def get_or_build_prm(self, cache_path: Optional[str] = None) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        """
        Loads or builds a city-agnostic Probabilistic Roadmap (PRM) of collision-free street network.
        """
        if self._prm_nodes is not None and self._prm_adjacency is not None:
            return self._prm_nodes, self._prm_adjacency

        if cache_path is None:
            candidates = [
                "./data/NYC3KM_585751_4512036/mesh/city_street_prm.pkl",
                "./mesh/city_street_prm.pkl",
            ]
            for c in candidates:
                if os.path.exists(c):
                    cache_path = c
                    break
            if cache_path is None:
                cache_path = "./data/NYC3KM_585751_4512036/mesh/city_street_prm.pkl"

        bounds = self.engine.bounds
        nodes, adj = build_city_prm(self.engine, bounds, cache_path=cache_path)
        self._prm_nodes = nodes
        self._prm_adjacency = adj
        return self._prm_nodes, self._prm_adjacency

    def find_curved_trajectories(
        self,
        duration: float,
        velocity: float,
        num_trajectories: int = 6,
        time_step: float = 1.0,
        z_height: float = 1.5,
        min_separation: float = 25.0,
        max_attempts: int = 2000,
        existing_trajectories: Optional[List[np.ndarray]] = None,
        max_overlap_ratio: float = 0.75,
        proximity_threshold: float = 20.0,
        cache_path: Optional[str] = None,
    ) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Finds `num_trajectories` collision-free curved trajectories that navigate
        through urban street intersections (start street != end street) using a
        city-agnostic shortest-path algorithm:
        - Works for ANY city layout (Manhattan, European curved, synthetic, etc.).
        - Samples two random points in navigable street space and calculates shortest path (Dijkstra/A*).
        - Natural direction changes at intersections.
        - Smooth rounded cornering arcs inside street intersections.
        - Constant scalar speed v (zero tangential acceleration, centripetal acceleration during turns).
        - Arc-length parameterization ensuring exact timing across frames.
        - Mutual overlap <= max_overlap_ratio (default 75%) against all existing trajectories.

        Returns:
            List of (trajectory_array, metadata_dict) tuples.
        """
        if self.engine is None:
            raise RuntimeError("MotionEngine must be provided to generate curved trajectories.")

        nodes, adjacency = self.get_or_build_prm(cache_path=cache_path)
        num_nodes = len(nodes)
        if num_nodes < 2:
            raise RuntimeError("Not enough valid street nodes found in city PRM.")

        num_frames = int(round(duration / time_step))
        req_dist = (num_frames - 1) * velocity * time_step

        found_trajectories: List[Tuple[np.ndarray, Dict[str, Any]]] = []
        batch_trajectories: List[np.ndarray] = []
        if existing_trajectories is not None:
            batch_trajectories.extend(existing_trajectories)

        attempts = 0
        while len(found_trajectories) < num_trajectories and attempts < max_attempts:
            attempts += 1

            start_idx = np.random.randint(0, num_nodes)
            p0 = nodes[start_idx]

            # Start point separation check from trajectories found in this batch
            if any(np.linalg.norm(p0[:2] - prev_traj[0, :2]) < min_separation for prev_traj, _ in found_trajectories):
                continue

            # Dijkstra shortest path search to candidate targets at >= req_dist
            pq = [(0.0, start_idx)]
            dist = {start_idx: 0.0}
            parent = {start_idx: None}
            candidates = []

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue
                if d >= req_dist:
                    candidates.append(u)
                    if len(candidates) >= 15:
                        break
                    continue
                for v in adjacency[u]:
                    w = np.linalg.norm(nodes[u] - nodes[v])
                    new_d = d + w
                    if v not in dist or new_d < dist[v]:
                        dist[v] = new_d
                        parent[v] = u
                        heapq.heappush(pq, (new_d, v))

            if not candidates:
                continue

            target_idx = candidates[np.random.choice(len(candidates))]
            path = []
            curr = target_idx
            while curr is not None:
                path.append(curr)
                curr = parent[curr]
            path.reverse()
            path_pts = nodes[path]

            # Simplify collinear points along streets, keeping turns
            simple_pts = simplify_collinear(path_pts, angle_tol_deg=15.0)

            # For paths with travel distance > 150m, require at least 1 turn
            if req_dist > 150.0 and len(simple_pts) < 3:
                continue

            # Round corner turns smoothly with intersection fillet arcs
            rounded = round_corners(simple_pts, corner_radius=10.0)

            diffs = np.diff(rounded, axis=0)
            cum_dist = np.insert(np.cumsum(np.linalg.norm(diffs, axis=1)), 0, 0.0)
            if cum_dist[-1] < req_dist:
                continue

            # Arc-length parameterization at constant speed v * t
            target_distances = np.arange(num_frames) * (time_step * velocity)
            final_x = np.interp(target_distances, cum_dist, rounded[:, 0])
            final_y = np.interp(target_distances, cum_dist, rounded[:, 1])
            final_z = np.full(num_frames, z_height)
            traj_arr = np.column_stack((final_x, final_y, final_z))

            # Collision check
            if not all(self.engine.is_position_valid(p) for p in traj_arr):
                continue

            # Mutual overlap check against all existing trajectories
            is_overlapping = False
            for other_traj in batch_trajectories:
                overlap = compute_trajectory_overlap(
                    traj_arr,
                    other_traj,
                    proximity_threshold=proximity_threshold,
                )
                if overlap > max_overlap_ratio:
                    is_overlapping = True
                    break

            if is_overlapping:
                continue

            # Turn metrics & heading
            diffs_traj = np.diff(traj_arr[:, :2], axis=0)
            headings = np.arctan2(diffs_traj[:, 1], diffs_traj[:, 0])
            ang_diffs = np.abs(np.diff(headings))
            ang_diffs = np.minimum(ang_diffs, 2 * np.pi - ang_diffs)
            num_turns = int(np.sum(ang_diffs > np.radians(15.0)))
            initial_heading = float(math.degrees(headings[0])) % 360.0

            metadata = {
                "mode": "curved",
                "curve_type": f"shortest_path_turns_{num_turns}",
                "num_turns": num_turns,
                "duration_seconds": duration,
                "velocity_mps": velocity,
                "time_step_seconds": time_step,
                "num_frames": num_frames,
                "total_distance_meters": float(req_dist),
                "start_position": traj_arr[0].tolist(),
                "end_position": traj_arr[-1].tolist(),
                "heading_degrees": initial_heading,
                "is_collision_free": True,
                "kinematics": {
                    "constant_speed": True,
                    "constant_velocity_vector": False,
                    "tangential_acceleration": 0.0,
                    "has_centripetal_acceleration": True,
                },
            }

            found_trajectories.append((traj_arr, metadata))
            batch_trajectories.append(traj_arr)

        if len(found_trajectories) < num_trajectories:
            print(
                f"Warning: Found {len(found_trajectories)}/{num_trajectories} curved trajectories of length {req_dist:.1f}m (overlap <= {max_overlap_ratio*100:.0f}%) after {attempts} attempts."
            )

        return found_trajectories




# =============================================================================
# Helper: Save trajectory and metadata to dataset
# =============================================================================
def save_trajectory(
    output_dir: str,
    trajectory_id: str,
    trajectory: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    save_json: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Saves a trajectory array (.npy) to output_dir.
    Metadata is aggregated at dataset level in trajectories_manifest.json,
    avoiding redundant individual JSON files.
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f"{trajectory_id}.npy")
    np.save(npy_path, trajectory)

    json_path = None
    if save_json and metadata:
        json_path = os.path.join(output_dir, f"{trajectory_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    return npy_path, json_path

