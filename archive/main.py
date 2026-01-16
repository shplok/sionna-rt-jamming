import sys
import numpy as np
import mitsuba as mi

# Set Mitsuba variant before importing Sionna
mi.set_variant("llvm_ad_mono_polarized")

from sionna.rt import load_scene
from scene_objects import create_scene_objects, gather_bboxes
from motion_engine import MotionEngine
from motion_strategies import run_mission_wizard

# --- Configuration ---
SCENE_PATH = r"/home/luisg-ubuntu/sionna_rt_jamming/data/downtown_chicago_luis/ChicagoMarionaClean.xml"
MESHES_PATH = r"/home/luisg-ubuntu/sionna_rt_jamming/data/downtown_chicago_luis/meshes"
FREQ_HZ = 1.57542e9  # GPS L1 Frequency

def main():
    # 1. Load the Physical Scene
    print(f"Loading scene from: {SCENE_PATH}")
    scene = load_scene(SCENE_PATH)
    scene.frequency = FREQ_HZ
    
    # 2. Extract Obstacles (Bounding Boxes)
    print("Gathering building obstacles...")
    buildings = gather_bboxes(MESHES_PATH)

    # 3. Define Transmitters (Initial State)
    transmitter_config = [
        {"name": "Jammer1", "position": np.array([200, -185, 0.0]), "color": [1.0, 0.0, 0.0]},
        {"name": "Jammer2", "position": np.array([-200, 200, 0.0]), "color": [0.0, 0.0, 1.0]}
    ]

    # 4. Setup Map Boundaries & Helper Objects
    # Note: create_scene_objects is a user-provided helper not shown here, 
    # but we assume it returns bounds/center/size.
    bounds, map_center, map_size, cell_size = create_scene_objects(
        scene,
        transmitters_config=transmitter_config,
        map_bounds={'x': [-600, 600], 'y': [-600, 600]},
        cell_size=(20, 20),
        z_height=0.0
    )

    # 5. Initialize the Motion Engine
    # This engine holds the "Truth" of the simulation world
    engine = MotionEngine(scene=scene, obstacles=buildings, bounds=bounds)

    # 6. Launch the Mission Wizard
    # This handles the GUI Launcher -> Strategy Config -> Path Generation flow
    target_jammer = "Jammer1"
    start_pos = transmitter_config[0]['position']
    
    path, metadata = run_mission_wizard(
        engine=engine, 
        jammer_id=target_jammer, 
        default_start_pos=start_pos
    )

    # 7. Final Output / Next Steps
    if path is not None:
        print("\n" + "="*40)
        print(" MISSION GENERATION SUCCESSFUL")
        print("="*40)
        print(f"Target: {target_jammer}")
        print(f"Total Steps: {len(path)}")
        
        if "total_distance" in metadata:
            print(f"Distance: {metadata['total_distance']:.2f} m")

        # print path for verification
        print("Generated Path (x, y, z):")
        for step in path:
            print(f"  {step}")
        
        # Here you would typically start your main simulation loop
        # e.g., run_simulation_loop(engine, path)
        
    else:
        print("\nMission Setup Cancelled.")

if __name__ == "__main__":
    main()