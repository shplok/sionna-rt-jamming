"""
paths_config.py
Centralized, multi-user and multi-cluster path configuration for sionna_rt_jamming.

Automatically detects the current execution environment (e.g., Pomplun cluster,
Northeastern Explorer cluster, or local development) and user, assigning appropriate
high-capacity scratch and dataset storage paths.

All paths can be overridden via:
1. Editing the 2 folder names below.
2. Setting environment variables (SIONNA_TRAJ_DIR, SIONNA_SIM_OUTPUT_DIR).
3. CLI argument flags (--output-dir, --sim-output-dir, etc.).
"""

import os
import sys
import getpass
import socket
from pathlib import Path

# Repo Root
REPO_ROOT = Path(__file__).resolve().parent

# ==============================================================================
# User-Configurable Subfolder Names (Easily modify these 2 anytime)
# ==============================================================================
TRAJECTORIES_DIR_NAME = "nyc_curved_jammers"
SIM_RESULTS_DIR_NAME = "simulation_results_nyc"


def detect_site() -> str:
    """
    Detect the compute site/cluster: 'pomplun', 'explorer', or 'local'.
    Respects SIONNA_SITE environment variable if already exported.
    """
    env_site = os.environ.get("SIONNA_SITE")
    if env_site:
        return env_site.strip().lower()

    # Fast filesystem and hostname checks
    if os.path.exists("/pomplun"):
        return "pomplun"
    if os.path.exists("/projects/ipl_lab"):
        return "explorer"

    hostname = socket.gethostname().lower()
    if "explorer" in hostname or "northeastern" in hostname:
        return "explorer"
    if "chimera" in hostname or "pomplun" in hostname:
        return "pomplun"

    return "local"


def get_current_user() -> str:
    """Returns the current operating system username."""
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "unknown_user")


def get_scratch_base_dir() -> str:
    """
    Returns the base scratch directory for high-volume simulation data.
    Can be overridden by exporting SIONNA_SCRATCH_DIR.
    """
    if "SIONNA_SCRATCH_DIR" in os.environ:
        return os.path.abspath(os.environ["SIONNA_SCRATCH_DIR"])

    site = detect_site()
    user = get_current_user()

    if site == "pomplun":
        pomplun_user_scratch = f"/pomplun/scratch01/cs_research/{user}/sionna_rt_jamming"
        if os.path.exists(f"/pomplun/scratch01/cs_research/{user}") or os.path.exists("/pomplun/scratch01"):
            return pomplun_user_scratch

    elif site == "explorer":
        explorer_proj = os.environ.get("SIONNA_PROJ", f"/projects/ipl_lab/{user}")
        return os.path.join(explorer_proj, "sionna_rt_jamming")

    # Fallback for local or unconfigured machines: repo's datasets folder
    return str(REPO_ROOT / "datasets")


def get_dataset_root() -> str:
    """Returns root directory for datasets."""
    if "SIONNA_DATASET_ROOT" in os.environ:
        return os.path.abspath(os.environ["SIONNA_DATASET_ROOT"])
    return str(REPO_ROOT / "datasets")


# ==============================================================================
# The 2 Main Target Output Paths
# ==============================================================================

def get_trajectories_dir(subfolder: str = TRAJECTORIES_DIR_NAME) -> str:
    """
    Returns the output directory for single jammer trajectories.
    Override with env var SIONNA_TRAJ_DIR.
    """
    if "SIONNA_TRAJ_DIR" in os.environ:
        return os.path.abspath(os.environ["SIONNA_TRAJ_DIR"])
    return os.path.join(get_scratch_base_dir(), subfolder)


def get_sim_results_dir(subfolder: str = SIM_RESULTS_DIR_NAME) -> str:
    """
    Returns the output directory for simulation results (radio maps / intermittent runs).
    Override with env var SIONNA_SIM_OUTPUT_DIR.
    """
    if "SIONNA_SIM_OUTPUT_DIR" in os.environ:
        return os.path.abspath(os.environ["SIONNA_SIM_OUTPUT_DIR"])
    return os.path.join(get_scratch_base_dir(), subfolder)


# Backward-compatibility aliases
get_sim_output_dir = get_sim_results_dir
get_intermittent_dir = get_sim_results_dir


def get_scene_defaults(scene_name: str = "NYC3KM_585751_4512036") -> dict:
    """Returns default scene XML and mesh paths within the repository data folder."""
    data_dir = REPO_ROOT / "data" / scene_name
    return {
        "mesh_dir": str(data_dir / "mesh"),
        "scene_path": str(data_dir / "simple_OSM_scene.xml"),
    }


def print_summary():
    """Prints a clean, concise diagnostic of active paths."""
    print("=" * 60)
    print("sionna_rt_jamming - Environment & Path Configuration")
    print("=" * 60)
    print(f"Repository Root:    {REPO_ROOT}")
    print(f"Current User:       {get_current_user()}")
    print(f"Detected Site:      {detect_site()}")
    print(f"Scratch Base Dir:   {get_scratch_base_dir()}")
    print(f"Trajectories Dir:   {get_trajectories_dir()}")
    print(f"Sim Results Dir:    {get_sim_results_dir()}")
    scene = get_scene_defaults()
    print(f"Default Mesh Dir:   {scene['mesh_dir']}")
    print(f"Default Scene XML:  {scene['scene_path']}")
    print("=" * 60)


if __name__ == "__main__":
    print_summary()
