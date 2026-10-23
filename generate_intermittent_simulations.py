#!/usr/bin/env python3
"""
Intermittent Jammer Simulation Generator.

Clones a multi-jammer simulation dataset and applies customizable
intermittent transmission schedules (time intervals where jammers turn
off/deactivate/disappear).

Features:
- Reads an existing simulation results directory (e.g. simulation_results_nyc).
- Conveniently defines active/inactive time intervals per jammer:
    * Custom explicit intervals: [(start, end), ...] per jammer or via JSON.
    * Periodic pulsing (on_steps, off_steps, staggered phases).
    * Random bursts (min_on, max_on, min_off, max_off).
    * Round-robin / alternating transmission.
- Filters by maximum number of jammers (--max-jammers) or exact jammers (--exact-jammers).
- Limits output to N simulations (--num-simulations N).
- Preserves the exact dataset structure:
    clone_dir/
      combinations_manifest.json
      combo_XXXX_kYY/
        combination_summary.json (augmented with intermittent intervals)
        rss_aggregated.npy
- Optional visualization: activity timeline plot (--plot-timeline).
"""

import os
import glob
import json
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np

NOISE_FLOOR_WATTS_DEFAULT = 8e-15


# =============================================================================
# 1. Intermittent Schedule Definitions
# =============================================================================

class BaseSchedule:
    """Abstract base class for intermittent transmission schedules."""

    def get_mask(self, jammer_idx: int, total_steps: int) -> np.ndarray:
        """Returns boolean mask of shape (total_steps,) where True = transmitting."""
        raise NotImplementedError

    def get_intervals(self, jammer_idx: int, total_steps: int) -> Dict[str, Any]:
        """Returns active and inactive interval lists [[start, end], ...]."""
        mask = self.get_mask(jammer_idx, total_steps)
        active_intervals = []
        inactive_intervals = []

        in_active = False
        start_step = 0

        for t in range(total_steps):
            if mask[t] and not in_active:
                if t > start_step and t > 0:
                    inactive_intervals.append([start_step, t])
                in_active = True
                start_step = t
            elif not mask[t] and in_active:
                active_intervals.append([start_step, t])
                in_active = False
                start_step = t

        if in_active:
            active_intervals.append([start_step, total_steps])
        elif start_step < total_steps:
            inactive_intervals.append([start_step, total_steps])

        active_count = int(np.sum(mask))
        duty_cycle = float(active_count) / float(total_steps) if total_steps > 0 else 0.0

        return {
            "active_intervals": active_intervals,
            "inactive_intervals": inactive_intervals,
            "active_steps_count": active_count,
            "total_steps": total_steps,
            "duty_cycle": round(duty_cycle, 4),
        }


class IntervalSchedule(BaseSchedule):
    """
    Explicit interval schedule defined per jammer.
    intervals_dict: {jammer_idx: [(start_step, end_step), ...]}
    """

    def __init__(self, intervals_dict: Dict[int, List[Tuple[int, int]]]):
        self.intervals_dict = intervals_dict

    def get_mask(self, jammer_idx: int, total_steps: int) -> np.ndarray:
        mask = np.zeros(total_steps, dtype=bool)
        intervals = self.intervals_dict.get(jammer_idx, [])
        for start, end in intervals:
            s = max(0, min(total_steps, int(start)))
            e = max(0, min(total_steps, int(end)))
            mask[s:e] = True
        return mask


class PeriodicSchedule(BaseSchedule):
    """
    Periodic square-wave pulsing (on for `on_steps`, off for `off_steps`).
    Supports phase staggering across jammers so jammers pulse out-of-phase.
    """

    def __init__(
        self,
        on_steps: int = 15,
        off_steps: int = 15,
        stagger: bool = True,
        initial_offset: int = 0,
    ):
        self.on_steps = max(1, int(on_steps))
        self.off_steps = max(1, int(off_steps))
        self.period = self.on_steps + self.off_steps
        self.stagger = stagger
        self.initial_offset = int(initial_offset)

    def get_mask(self, jammer_idx: int, total_steps: int) -> np.ndarray:
        mask = np.zeros(total_steps, dtype=bool)
        if self.stagger:
            # Shift phase for each jammer across the period
            offset = (self.initial_offset + jammer_idx * (self.period // 3)) % self.period
        else:
            offset = self.initial_offset % self.period

        for t in range(total_steps):
            phase = (t + offset) % self.period
            if phase < self.on_steps:
                mask[t] = True
        return mask


class RandomBurstSchedule(BaseSchedule):
    """
    Random on/off bursts using uniform or Poisson-like intervals.
    """

    def __init__(
        self,
        min_on: int = 10,
        max_on: int = 25,
        min_off: int = 10,
        max_off: int = 25,
        seed: Optional[int] = None,
    ):
        self.min_on = min_on
        self.max_on = max_on
        self.min_off = min_off
        self.max_off = max_off
        self.seed = seed

    def get_mask(self, jammer_idx: int, total_steps: int) -> np.ndarray:
        rng = np.random.default_rng(None if self.seed is None else self.seed + jammer_idx * 101)
        mask = np.zeros(total_steps, dtype=bool)
        t = 0
        is_on = rng.choice([True, False])
        while t < total_steps:
            if is_on:
                dur = int(rng.integers(self.min_on, self.max_on + 1))
                end = min(total_steps, t + dur)
                mask[t:end] = True
                t = end
                is_on = False
            else:
                dur = int(rng.integers(self.min_off, self.max_off + 1))
                end = min(total_steps, t + dur)
                mask[t:end] = False
                t = end
                is_on = True
        return mask


class StaggeredRoundRobinSchedule(BaseSchedule):
    """
    Round-robin schedule: exactly one jammer (or rotating subsets) is active at any time.
    """

    def __init__(self, slot_duration: int = 20, num_jammers: int = 3):
        self.slot_duration = slot_duration
        self.num_jammers = max(1, num_jammers)

    def get_mask(self, jammer_idx: int, total_steps: int) -> np.ndarray:
        mask = np.zeros(total_steps, dtype=bool)
        for t in range(total_steps):
            active_jammer = (t // self.slot_duration) % self.num_jammers
            if jammer_idx % self.num_jammers == active_jammer:
                mask[t] = True
        return mask


# =============================================================================
# 2. Main Cloner and Intermittent Generator
# =============================================================================

class IntermittentSimulator:
    """
    Loads base radio maps and aggregates new intermittent combinations.
    """

    def __init__(
        self,
        input_sim_dir: str,
        output_dir: str,
        radio_maps_dir: Optional[str] = None,
        noise_floor_watts: float = NOISE_FLOOR_WATTS_DEFAULT,
        precision: str = "float16",
    ):
        self.input_sim_dir = os.path.abspath(input_sim_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.noise_floor_watts = float(noise_floor_watts)
        self.precision = precision.lower()
        self.dtype = np.float16 if self.precision == "float16" else np.float32

        # Auto-detect radio maps dir if not supplied
        if radio_maps_dir is not None:
            self.radio_maps_dir = os.path.abspath(radio_maps_dir)
        else:
            candidates = [
                os.path.join(os.path.dirname(self.input_sim_dir), "nyc_curved_jammers", "radio_maps"),
                os.path.join(self.input_sim_dir, "radio_maps"),
                "./datasets/nyc_single_jammers/radio_maps",
                "./nyc_curved_jammers/radio_maps",
            ]
            found = None
            for c in candidates:
                if os.path.exists(c) and glob.glob(os.path.join(c, "rm_watts_*.npy")):
                    found = os.path.abspath(c)
                    break
            if found is None:
                raise FileNotFoundError(
                    f"Could not automatically locate base radio maps directory with rm_watts_*.npy files. "
                    f"Please specify --radio-maps-dir."
                )
            self.radio_maps_dir = found

        self._loaded_maps: Dict[str, np.ndarray] = {}

    def get_base_map(self, traj_id: str) -> np.ndarray:
        """Loads and caches base Watts radio map in RAM."""
        if traj_id in self._loaded_maps:
            return self._loaded_maps[traj_id]

        w_path = os.path.join(self.radio_maps_dir, f"rm_watts_{traj_id}.npy")
        if not os.path.exists(w_path):
            raise FileNotFoundError(f"Base watts map not found: {w_path}")

        w_arr = np.load(w_path)
        if not np.all(np.isfinite(w_arr)):
            finite_mask = np.isfinite(w_arr)
            max_finite = float(np.max(w_arr[finite_mask])) if np.any(finite_mask) else 10.0
            w_arr = np.nan_to_num(w_arr, posinf=max_finite, neginf=0.0, nan=0.0)

        self._loaded_maps[traj_id] = w_arr
        return w_arr

    def process(
        self,
        schedule: BaseSchedule,
        num_simulations: Optional[int] = None,
        max_jammers: Optional[int] = None,
        exact_jammers: Optional[int] = None,
        combo_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Clones combinations, applies the intermittent schedule, and saves output.
        """
        manifest_path = os.path.join(self.input_sim_dir, "combinations_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found in {self.input_sim_dir}: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            src_manifest = json.load(f)

        combinations = src_manifest.get("combinations", [])
        if not combinations:
            raise ValueError(f"No combinations found in {manifest_path}")

        # Filter combinations by jammer criteria or combo_id
        filtered_combos = []
        for combo in combinations:
            if combo_id is not None and combo.get("combo_id") != combo_id:
                continue
            k = combo.get("num_jammers", len(combo.get("jammer_ids", [])))
            if exact_jammers is not None and k != exact_jammers:
                continue
            if max_jammers is not None and k > max_jammers:
                continue
            filtered_combos.append(combo)

        if not filtered_combos:
            msg = f"No combinations found matching criteria (max_jammers={max_jammers}, exact_jammers={exact_jammers})."
            raise ValueError(msg)

        # Limit to num_simulations
        if num_simulations is not None and num_simulations > 0:
            chosen_combos = filtered_combos[:num_simulations]
        else:
            chosen_combos = filtered_combos

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"\n==================================================================")
        print(f"Generating Intermittent Simulations")
        print(f"Source:         {self.input_sim_dir}")
        print(f"Radio Maps:     {self.radio_maps_dir}")
        print(f"Destination:    {self.output_dir}")
        print(f"Simulations:    {len(chosen_combos)} (from {len(filtered_combos)} matching)")
        print(f"Precision:      {self.precision} (noise floor: {self.noise_floor_watts:.1e} W)")
        print(f"==================================================================")

        out_combinations_manifest = []

        for out_idx, combo_info in enumerate(chosen_combos):
            combo_id = combo_info["combo_id"]
            jammer_ids = combo_info["jammer_ids"]
            K = len(jammer_ids)

            print(f"\n[{out_idx + 1}/{len(chosen_combos)}] Processing {combo_id} (K={K} jammers)...")

            # Load base maps for jammers
            base_maps = [self.get_base_map(tid) for tid in jammer_ids]
            max_steps = max(len(m) for m in base_maps)
            H, W = base_maps[0].shape[1:]

            # Aggregate linear watts with intermittent mask
            agg_watts = np.zeros((max_steps, H, W), dtype=np.float32)
            jammers_meta = []

            for j_idx, (tid, w_data) in enumerate(zip(jammer_ids, base_maps)):
                cur_steps = len(w_data)
                pad_amount = max_steps - cur_steps

                # Pad stationary if shorter than max_steps
                if pad_amount > 0:
                    pad = np.tile(w_data[-1:], (pad_amount, 1, 1))
                    padded_w = np.vstack([w_data, pad])
                else:
                    padded_w = w_data

                # Get intermittent boolean mask
                mask_1d = schedule.get_mask(j_idx, max_steps)
                intervals_info = schedule.get_intervals(j_idx, max_steps)

                # Broadcast mask across spatial dimensions: (T, 1, 1)
                mask_3d = mask_1d[:, None, None].astype(np.float32)

                # Active watts (zero when deactivated)
                active_watts = padded_w[:max_steps] * mask_3d
                agg_watts += active_watts

                jname = f"Jammer{j_idx + 1}"
                jmeta = {
                    "jammer_index": j_idx,
                    "jammer_name": jname,
                    "traj_id": tid,
                    "movement_steps": cur_steps,
                    "stationary_steps": pad_amount,
                    "original_steps": cur_steps,
                    "padded_steps": pad_amount,
                    "is_intermittent": bool(intervals_info["duty_cycle"] < 1.0 or len(intervals_info["active_intervals"]) > 1),
                    **intervals_info,
                }
                jammers_meta.append(jmeta)
                print(f"  {jname} ({tid}): Active {intervals_info['active_steps_count']}/{max_steps} steps "
                      f"(duty={intervals_info['duty_cycle']*100:.1f}%) | intervals={intervals_info['active_intervals']}")

            # Incoherent dBW conversion: 10 * log10(P_watts + noise_floor)
            agg_dbw_f32 = 10.0 * np.log10(agg_watts + self.noise_floor_watts)
            if not np.all(np.isfinite(agg_dbw_f32)):
                finite_dbw = agg_dbw_f32[np.isfinite(agg_dbw_f32)]
                min_f = float(np.min(finite_dbw)) if len(finite_dbw) > 0 else -141.0
                max_f = float(np.max(finite_dbw)) if len(finite_dbw) > 0 else 20.0
                agg_dbw_f32 = np.nan_to_num(agg_dbw_f32, posinf=max_f, neginf=min_f, nan=min_f)

            agg_dbw_final = agg_dbw_f32.astype(self.dtype)

            # Save in clone directory structure
            target_combo_dir = os.path.join(self.output_dir, combo_id)
            os.makedirs(target_combo_dir, exist_ok=True)

            out_npy_path = os.path.join(target_combo_dir, "rss_aggregated.npy")
            out_summary_path = os.path.join(target_combo_dir, "combination_summary.json")

            np.save(out_npy_path, agg_dbw_final)

            summary_dict = {
                "combination_id": combo_id,
                "combo_index": out_idx,
                "num_jammers": K,
                "total_steps": max_steps,
                "dtype": self.precision,
                "min_dbw": float(agg_dbw_final.min()),
                "max_dbw": float(agg_dbw_final.max()),
                "noise_floor_watts": self.noise_floor_watts,
                "mode": "intermittent",
                "jammers": jammers_meta,
            }

            with open(out_summary_path, "w", encoding="utf-8") as jf:
                json.dump(summary_dict, jf, indent=2)

            out_combinations_manifest.append({
                "combo_id": combo_id,
                "index": out_idx,
                "num_jammers": K,
                "total_steps": max_steps,
                "dir": target_combo_dir,
                "jammer_ids": jammer_ids,
                "is_intermittent": True,
            })

            print(f"  -> Saved {out_npy_path} (shape={agg_dbw_final.shape}, {self.precision})")

        # Save clone combinations manifest
        out_manifest_path = os.path.join(self.output_dir, "combinations_manifest.json")
        with open(out_manifest_path, "w", encoding="utf-8") as jf:
            json.dump({
                "total_combinations": len(out_combinations_manifest),
                "is_intermittent": True,
                "source_sim_dir": self.input_sim_dir,
                "combinations": out_combinations_manifest,
            }, jf, indent=2)

        print(f"\n==================================================================")
        print(f"Intermittent dataset successfully created at: {self.output_dir}")
        print(f"Manifest saved to: {out_manifest_path}")
        print(f"==================================================================")

        return out_combinations_manifest


# =============================================================================
# 3. Visualization Helper: Timeline / Activity Plot
# =============================================================================

def plot_intermittent_timeline(summary_json_path: str, output_image_path: str):
    """Generates a clean Gantt-chart timeline showing active/inactive periods of each jammer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(summary_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    combo_id = data["combination_id"]
    total_steps = data["total_steps"]
    jammers = data["jammers"]
    K = len(jammers)

    fig, ax = plt.subplots(figsize=(10, max(3, 1.2 * K)), dpi=150, facecolor="#1a1a1a")
    ax.set_facecolor("#1a1a1a")

    cmap = plt.get_cmap("tab10")

    for idx, jmeta in enumerate(jammers):
        y_pos = K - 1 - idx
        color = cmap(idx % 10)
        label = f"{jmeta['jammer_name']} ({jmeta['traj_id']})"

        # Background inactive bar (dim gray)
        ax.barh(y_pos, total_steps, left=0, height=0.55, color="#2c2c2c", edgecolor="#444444", linewidth=0.5)

        # Active bars
        movement_steps = jmeta.get("movement_steps")
        for start, end in jmeta.get("active_intervals", []):
            if movement_steps and movement_steps < end:
                # Part 1: Moving active interval
                if movement_steps > start:
                    ax.barh(y_pos, movement_steps - start, left=start, height=0.55, color=color, edgecolor="#ffffff", linewidth=0.6, alpha=0.9)
                # Part 2: Stationary active interval (hatched)
                stat_start = max(start, movement_steps)
                ax.barh(y_pos, end - stat_start, left=stat_start, height=0.55, color=color, hatch="///", edgecolor="#ffff99", linewidth=0.8, alpha=0.75)
            else:
                dur = end - start
                ax.barh(y_pos, dur, left=start, height=0.55, color=color, edgecolor="#ffffff", linewidth=0.6, alpha=0.9)

        # Duty cycle text annotation
        duty = jmeta.get("duty_cycle", 0.0) * 100.0
        if movement_steps and movement_steps < total_steps:
            info_str = f"Duty: {duty:.1f}% ({movement_steps}s move + {total_steps - movement_steps}s stationary)"
        else:
            info_str = f"Duty: {duty:.1f}% ({jmeta['active_steps_count']}s)"
        ax.text(total_steps + 1.5, y_pos, info_str,
                va="center", ha="left", color="#cccccc", fontsize=9)

    ax.set_yticks(range(K))
    ax.set_yticklabels([f"{jammers[K - 1 - i]['jammer_name']}" for i in range(K)], color="#ffffff", fontsize=10)
    ax.set_xlabel("Time Step (seconds)", color="#ffffff", fontsize=10, labelpad=8)
    ax.set_xlim(0, total_steps * 1.25)
    ax.set_title(f"Intermittent Transmission Schedule: {combo_id} ({K} Jammers)", color="#ffffff", fontsize=12, pad=12)
    ax.tick_params(colors="#aaaaaa")
    ax.grid(True, axis="x", color="#333333", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_image_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Timeline plot saved to: {output_image_path}")


def run_long_3jammers_preset(
    traj_dir: str = "/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/nyc_long_jammers",
    output_dir: str = "/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/simulation_results_nyc_intermittent_long",
    mesh_dir: str = "./data/NYC3KM_585751_4512036/mesh",
    fps: int = 6,
    generate_gif: bool = True,
    generate_snapshot: bool = True,
):
    """
    Curated scenario preset: 3 spatially separated long curved jammers (1.5x Jammer3 = 1201.5m).
    - Jammer 1 (Zone 1 NE): 45s movement + 45s stationary continuous transmission [0, 90]s.
    - Jammer 2 (Zone 2 SW): Intermittent [0, 30]s and [60, 90]s.
    - Jammer 3 (Zone 3 SE): Silent first quarter [0, 23]s, then active [23, 90]s.
    """
    from visualize_simulation import create_simulation_gif, create_simulation_snapshots, _load_obstacles

    rm_dir = os.path.join(traj_dir, "radio_maps")
    combo_dir = os.path.join(output_dir, "combo_long_3jammers")
    os.makedirs(combo_dir, exist_ok=True)

    jammers_config = [
        {
            "jammer_index": 0,
            "jammer_name": "Jammer1",
            "name": "Jammer1",
            "traj_id": "traj_long1200m_zone1",
            "zone": "Zone 1 (North-East, 600m / 45s move + 45s stationary)",
            "movement_steps": 45,
            "stationary_steps": 45,
            "is_intermittent": False,
            "active_intervals": [[0, 90]],
            "inactive_intervals": [],
            "active_steps_count": 90,
            "duty_cycle": 1.0,
            "motion_description": "Moves for 45s (0-45s), then remains stationary at destination for 45-90s while continuously transmitting [ON].",
        },
        {
            "jammer_index": 1,
            "jammer_name": "Jammer2",
            "name": "Jammer2",
            "traj_id": "traj_long1200m_zone2",
            "zone": "Zone 2 (South-West, 1202m)",
            "movement_steps": 90,
            "stationary_steps": 0,
            "is_intermittent": True,
            "active_intervals": [[0, 30], [60, 90]],
            "inactive_intervals": [[30, 60]],
            "active_steps_count": 60,
            "duty_cycle": 0.6667,
            "motion_description": "Moves for 90s. Active [0, 30]s and [60, 90]s, paused [30, 60]s.",
        },
        {
            "jammer_index": 2,
            "jammer_name": "Jammer3",
            "name": "Jammer3",
            "traj_id": "traj_long1200m_zone3",
            "zone": "Zone 3 (South-East, 1202m)",
            "movement_steps": 90,
            "stationary_steps": 0,
            "is_intermittent": True,
            "active_intervals": [[23, 90]],
            "inactive_intervals": [[0, 23]],
            "active_steps_count": 67,
            "duty_cycle": 0.7444,
            "motion_description": "Moves for 90s. Silent first quarter [0, 23]s, turns ON at 23s through 90s.",
        },
    ]

    print("\n" + "=" * 70)
    print("RUNNING INTERMITTENT SIMULATION PRESET: long_3jammers")
    print("=" * 70)

    watts_maps = []
    for jc in jammers_config:
        wp = os.path.join(rm_dir, f"rm_watts_{jc['traj_id']}.npy")
        if not os.path.exists(wp):
            raise FileNotFoundError(f"Missing base radio map: {wp}")
        w = np.load(wp)
        watts_maps.append(w)

    total_steps = 90
    H, W = watts_maps[0].shape[1:]
    agg_watts = np.zeros((total_steps, H, W), dtype=np.float32)

    for jc, w_data in zip(jammers_config, watts_maps):
        if len(w_data) < total_steps:
            pad_len = total_steps - len(w_data)
            w_full = np.vstack([w_data, np.tile(w_data[-1:], (pad_len, 1, 1))])
        else:
            w_full = w_data[:total_steps]

        mask_1d = np.zeros(total_steps, dtype=bool)
        for s, e in jc["active_intervals"]:
            mask_1d[s:e] = True
        mask_3d = mask_1d[:, None, None].astype(np.float32)
        active_w = w_full * mask_3d
        agg_watts += active_w
        print(f"  {jc['name']} ({jc['traj_id']}): Active {jc['active_steps_count']}/{total_steps}s ({jc['duty_cycle']*100:.1f}%) | {jc['active_intervals']} | Move: {jc.get('movement_steps', total_steps)}s, Stat: {jc.get('stationary_steps', 0)}s")

    noise_floor_watts = 8e-15
    agg_dbw = 10.0 * np.log10(agg_watts + noise_floor_watts)
    min_f = float(np.min(agg_dbw[np.isfinite(agg_dbw)])) if np.any(np.isfinite(agg_dbw)) else -141.0
    max_f = float(np.max(agg_dbw[np.isfinite(agg_dbw)])) if np.any(np.isfinite(agg_dbw)) else 20.0
    agg_dbw_clean = np.nan_to_num(agg_dbw, posinf=max_f, neginf=min_f, nan=min_f).astype(np.float16)

    out_npy = os.path.join(combo_dir, "rss_aggregated.npy")
    out_json = os.path.join(combo_dir, "combination_summary.json")
    np.save(out_npy, agg_dbw_clean)

    summary = {
        "combination_id": "combo_long_3jammers",
        "num_jammers": 3,
        "total_steps": total_steps,
        "dtype": "float16",
        "min_dbw": float(agg_dbw_clean.min()),
        "max_dbw": float(agg_dbw_clean.max()),
        "noise_floor_watts": noise_floor_watts,
        "mode": "intermittent",
        "description": "3 long curved trajectories (1.5x Jammer3 = 1201.5m) in separated urban sectors (NE, SW, SE)",
        "jammers": jammers_config,
    }
    with open(out_json, "w") as jf:
        json.dump(summary, jf, indent=2)

    print(f"\nSaved aggregated radio map: {out_npy}")
    print(f"Saved combination summary:  {out_json}")

    # Timeline plot
    timeline_path = os.path.join(output_dir, "combo_long_3jammers_timeline.png")
    plot_intermittent_timeline(out_json, timeline_path)

    # Visualizations
    obstacles = _load_obstacles(mesh_dir)

    if generate_gif:
        gif_path = os.path.join(combo_dir, "jammer_animation.gif")
        create_simulation_gif(
            combo_dir=combo_dir,
            trajectories_dir=traj_dir,
            obstacles=obstacles,
            output_filename=gif_path,
            fps=fps,
        )

    if generate_snapshot:
        snapshot_path = os.path.join(combo_dir, "simulation_snapshot_overview.png")
        create_simulation_snapshots(
            combo_dir=combo_dir,
            trajectories_dir=traj_dir,
            obstacles=obstacles,
            output_image=snapshot_path,
        )

    print("\n" + "=" * 70)
    print("Preset 'long_3jammers' complete!")
    print(f"Output: {combo_dir}")
    print("=" * 70)


# =============================================================================
# 4. CLI Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone a simulation directory with customizable intermittent (deactivating/disappearing) jammer schedules."
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["none", "long_3jammers"],
        default="none",
        help="Run a curated scenario preset (e.g. 'long_3jammers' for the 3 spatially separated 1.5x jammers).",
    )
    parser.add_argument(
        "--input-sim-dir",
        type=str,
        default="/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/simulation_results_nyc",
        help="Input simulation directory containing generated combinations (default: scratch simulation_results_nyc)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/simulation_results_nyc_intermittent",
        help="Clone output directory for intermittent simulations",
    )
    parser.add_argument(
        "--radio-maps-dir",
        type=str,
        default=None,
        help="Directory containing base radio maps (rm_watts_*.npy). If None, auto-detected.",
    )
    parser.add_argument(
        "-n", "--num-simulations",
        type=int,
        default=1,
        help="Number of simulations to generate (N <= length of original folder). Default: 1",
    )
    parser.add_argument(
        "-k", "--max-jammers",
        type=int,
        default=None,
        help="Maximum number of jammers allowed per combination (e.g. 3).",
    )
    parser.add_argument(
        "--exact-jammers",
        type=int,
        default=None,
        help="Require combinations with exactly this number of jammers (e.g. 3).",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        choices=["periodic", "staggered", "random", "round_robin", "custom"],
        default="staggered",
        help="Intermittent schedule pattern (default: staggered).",
    )
    parser.add_argument(
        "--on-steps",
        type=int,
        default=25,
        help="Number of active steps per period for 'periodic' pattern (default: 25).",
    )
    parser.add_argument(
        "--off-steps",
        type=int,
        default=20,
        help="Number of inactive/deactivated steps per period for 'periodic' pattern (default: 20).",
    )
    parser.add_argument(
        "--custom-json",
        type=str,
        default=None,
        help="JSON string or file path specifying custom intervals: {0: [[0, 30], [60, 90]], ...}",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Storage precision for aggregated radio maps (default: float16).",
    )
    parser.add_argument(
        "--combo-id",
        type=str,
        default=None,
        help="Process a specific combination ID only (e.g. combo_0001_k03).",
    )
    parser.add_argument(
        "--generate-gif",
        action="store_true",
        help="Generate animated GIF for the processed intermittent simulations.",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        default="/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/nyc_curved_jammers",
        help="Directory with single jammer trajectory coordinates .npy.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="./data/NYC3KM_585751_4512036/mesh",
        help="Path to mesh directory for building footprints in visualization.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second for output GIF (default: 6).",
    )
    parser.add_argument(
        "--plot-timeline",
        action="store_true",
        help="Generate a timeline visualization plot of the intermittent schedules.",
    )

    parser.add_argument(
        "--generate-snapshot",
        action="store_true",
        help="Generate 6-panel overview snapshot for intermittent simulations.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Preset shortcut
    if args.preset == "long_3jammers":
        traj_dir = args.trajectories_dir if args.trajectories_dir != "/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/nyc_curved_jammers" else "/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/nyc_long_jammers"
        out_dir = args.output_dir if args.output_dir != "/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/simulation_results_nyc_intermittent" else "/pomplun/scratch01/cs_research/l.gonzalezgudino001/sionna_rt_jamming/simulation_results_nyc_intermittent_long"
        run_long_3jammers_preset(
            traj_dir=traj_dir,
            output_dir=out_dir,
            mesh_dir=args.mesh_dir,
            fps=args.fps,
            generate_gif=True,
            generate_snapshot=args.generate_snapshot,
        )
        return

    # Construct the selected schedule
    if args.pattern == "custom" and args.custom_json:
        if os.path.exists(args.custom_json):
            with open(args.custom_json, "r", encoding="utf-8") as f:
                intervals_data = json.load(f)
        else:
            intervals_data = json.loads(args.custom_json)
        # Convert keys to int
        intervals_dict = {int(k): [tuple(iv) for iv in v] for k, v in intervals_data.items()}
        schedule = IntervalSchedule(intervals_dict)

    elif args.pattern == "periodic":
        schedule = PeriodicSchedule(
            on_steps=args.on_steps,
            off_steps=args.off_steps,
            stagger=False,
        )

    elif args.pattern == "staggered":
        # Periodic with staggered phases across jammers (great default for realistic multi-jammer pulsing)
        schedule = PeriodicSchedule(
            on_steps=args.on_steps,
            off_steps=args.off_steps,
            stagger=True,
        )

    elif args.pattern == "random":
        schedule = RandomBurstSchedule(
            min_on=args.on_steps // 2,
            max_on=args.on_steps * 2,
            min_off=args.off_steps // 2,
            max_off=args.off_steps * 2,
            seed=42,
        )

    elif args.pattern == "round_robin":
        schedule = StaggeredRoundRobinSchedule(
            slot_duration=args.on_steps,
            num_jammers=args.exact_jammers or args.max_jammers or 3,
        )

    else:
        # Default staggered
        schedule = PeriodicSchedule(on_steps=25, off_steps=20, stagger=True)

    simulator = IntermittentSimulator(
        input_sim_dir=args.input_sim_dir,
        output_dir=args.output_dir,
        radio_maps_dir=args.radio_maps_dir,
        precision=args.precision,
    )

    manifest = simulator.process(
        schedule=schedule,
        num_simulations=args.num_simulations,
        max_jammers=args.max_jammers,
        exact_jammers=args.exact_jammers,
        combo_id=args.combo_id,
    )

    if args.plot_timeline and manifest:
        for item in manifest:
            cid = item["combo_id"]
            summary_path = os.path.join(args.output_dir, cid, "combination_summary.json")
            plot_path = os.path.join(args.output_dir, f"{cid}_timeline.png")
            plot_intermittent_timeline(summary_path, plot_path)

    if (args.generate_gif or args.generate_snapshot) and manifest:
        from utils.scene_objects import gather_bboxes
        from visualize_simulation import create_simulation_gif, create_simulation_snapshots
        print(f"\nLoading mesh obstacles for visualization from {args.mesh_dir}...")
        obstacles = gather_bboxes(args.mesh_dir, footprints=True, use_cache=True)
        for item in manifest:
            cid = item["combo_id"]
            combo_dir = item["dir"]

            if args.generate_gif:
                gif_out = os.path.join(combo_dir, "jammer_animation.gif")
                print(f"Generating animation GIF for {cid} -> {gif_out}")
                create_simulation_gif(
                    combo_dir=combo_dir,
                    trajectories_dir=args.trajectories_dir,
                    obstacles=obstacles,
                    output_filename=gif_out,
                    fps=args.fps,
                )

            if args.generate_snapshot:
                snap_out = os.path.join(combo_dir, "simulation_snapshot_overview.png")
                print(f"Generating 6-panel snapshot for {cid} -> {snap_out}")
                create_simulation_snapshots(
                    combo_dir=combo_dir,
                    trajectories_dir=args.trajectories_dir,
                    obstacles=obstacles,
                    output_image=snap_out,
                )


if __name__ == "__main__":
    main()
