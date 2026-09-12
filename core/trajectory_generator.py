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
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union

from core.engine import MotionEngine
from core.utils import check_line_of_sight, calculate_smooth_path


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
    Computes the mutual overlap ratio between two trajectories.
    Two segments are considered overlapping at points where they are within
    `proximity_threshold` meters of each other (e.g. traveling along the same street canyon).

    The overlap ratio is defined as:
        max(fraction of traj1 close to traj2, fraction of traj2 close to traj1)

    Returns:
        float in [0.0, 1.0]: 0.0 means completely separate, 1.0 means fully overlapping.
    """
    p1_a, p1_b = traj1[0, :2], traj1[-1, :2]
    p2_a, p2_b = traj2[0, :2], traj2[-1, :2]

    def _point_to_segment_dists(pts_2d: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-6:
            return np.linalg.norm(pts_2d - a, axis=1)
        ap = pts_2d - a
        t = np.clip(np.sum(ap * ab, axis=1) / ab_len2, 0.0, 1.0)
        proj = a + t[:, np.newaxis] * ab
        return np.linalg.norm(pts_2d - proj, axis=1)

    n_samples1 = max(len(traj1), 50)
    pts1 = np.linspace(traj1[0], traj1[-1], n_samples1)

    n_samples2 = max(len(traj2), 50)
    pts2 = np.linspace(traj2[0], traj2[-1], n_samples2)

    d1 = _point_to_segment_dists(pts1[:, :2], p2_a, p2_b)
    ratio1 = float(np.mean(d1 <= proximity_threshold))

    d2 = _point_to_segment_dists(pts2[:, :2], p1_a, p1_b)
    ratio2 = float(np.mean(d2 <= proximity_threshold))

    return max(ratio1, ratio2)


class TrajectoryGenerator:

    """
    Core engine to produce straight or curved collision-free trajectories.
    """

    def __init__(self, engine: Optional[MotionEngine] = None):
        self.engine = engine

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


# =============================================================================
# Helper: Save trajectory and metadata to dataset
# =============================================================================
def save_trajectory(
    output_dir: str,
    trajectory_id: str,
    trajectory: np.ndarray,
    metadata: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Saves a trajectory array (.npy) and its metadata (.json) to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    npy_path = os.path.join(output_dir, f"{trajectory_id}.npy")
    json_path = os.path.join(output_dir, f"{trajectory_id}.json")

    np.save(npy_path, trajectory)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return npy_path, json_path

