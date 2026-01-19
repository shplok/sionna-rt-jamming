import sys, os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

from sionna.rt import load_scene, RadioMapSolver, Transmitter
from utils.scene_objects import create_scene_objects, gather_bboxes
from utils.plotter import create_jammer_animation

from core.engine import MotionEngine
from ui.app_controller import MissionController
from ui.menu import MenuApp
from core.strategies import GraphNavStrategy 

def main():
    # --- 1. Global Setup ---
    SCENE_PATH = r"/home/luisg-ubuntu/sionna_rt_jamming/data/downtown_chicago_luis/ChicagoMarionaClean.xml"
    MESHES_PATH = r"/home/luisg-ubuntu/sionna_rt_jamming/data/downtown_chicago_luis/meshes"
    OUTPUT_DIR = "./datasets"
    DATASET_NAME = "dataset_test"
    FREQ_HZ = 1.57542e9
    Z_HEIGHT = 1.5

    # --- 2. Load Physical World ---
    scene = load_scene(SCENE_PATH)
    scene.frequency = FREQ_HZ
    buildings = gather_bboxes(MESHES_PATH)

    # --- 3. Define Transmitters Config (for individual mode) ---
    transmitters_config = [
        {"name": "Jammer1", "position": np.array([220, -185, Z_HEIGHT])},
        {"name": "Jammer2", "position": np.array([-180, 200, Z_HEIGHT])}
    ]

    map_bounds = {'x': [-600, 600], 'y': [-600, 600], 'z': [Z_HEIGHT, Z_HEIGHT]}
    cell_size = (10, 10)

    # --- 4. Static Scene Setup ---
    map_center, map_size = create_scene_objects(
        scene, 
        map_bounds=map_bounds,
        z_height=Z_HEIGHT
    )
    
    engine = MotionEngine(scene=scene, obstacles=buildings, bounds=map_bounds)

    # --- 5. User Selection ---
    menu = MenuApp()
    mode = menu.run()
    
    if mode == "individual":
        # Pass the config so we can spawn them inside this function
        run_individual_mode(engine, transmitters_config, scene, map_center, map_size, cell_size, OUTPUT_DIR, DATASET_NAME, buildings)
    elif mode == "batch":
        # Batch mode doesn't need the default config, it makes its own
        run_batch_mode(engine, scene, map_center, map_size, cell_size, OUTPUT_DIR, DATASET_NAME, buildings, Z_HEIGHT)
    else:
        raise RuntimeError("No valid mode selected. Exiting.")

def run_individual_mode(engine, transmitters_config, scene, map_center, map_size, cell_size, output_dir, dataset_name, buildings):
    """Configures specific paths for each jammer, saves them, then runs Sionna simulation."""
    
    print("Spawning individual Transmitters...")
    for tx_info in transmitters_config:
        if tx_info["name"] not in scene.transmitters:
            tx = Transmitter(
                name=tx_info["name"],
                position=tx_info["position"],
                color=tx_info.get("color", [0.5, 0.5, 0.5])
            )
            scene.add(tx)
    # ------------------------------------------

    global_dt = None
    
    # A. Path Configuration Loop
    for tx in transmitters_config:
        j_id = tx['name']
        print(f"--- Configuring {j_id} ---")
        
        controller = MissionController(engine, j_id, tx['position'], fixed_dt=global_dt)
        path, metadata = controller.run(mode="individual")
        
        if path is None:
            print(f"Setup cancelled for {j_id}. Exiting.")
            return

        if global_dt is None: 
            global_dt = metadata['dt_used']

    # B. Synchronization & Save
    print("\nSynchronizing trajectories...")
    engine.finalize_trajectories()
    
    save_dataset(output_dir, dataset_name, paths=engine.get_all_paths(), metadata=engine.get_all_metadata())

    # C. Run Simulation
    run_simulation(engine, scene, map_center, map_size, cell_size, os.path.join(output_dir, dataset_name), buildings=buildings)

def run_batch_mode(engine, scene, map_center, map_size, cell_size, output_dir, dataset_name, buildings, z_height):
    """Generates N paths, ADDS them as new jammers to the scene, and runs one massive simulation."""
    
    # 1. Configure Graph
    controller = MissionController(engine, "BatchGenerator")
    config, padding_mode = controller.batch_run()

    if config is None:
        raise RuntimeError("Batch setup was cancelled. Exiting.")
    
    num_paths = config.num_simulations
    print(f"\nGenerating {num_paths} paths (to be used as simultaneous jammers)...")
    
    strategy = GraphNavStrategy()
    
    # 2. Generate Paths & Register Jammers
    for i in range(num_paths):
        try:
            # Generate geometry
            path, _ = strategy.generate(engine, config)
            
            jammer_name = f"Jammer_{i:03d}"
            
            # A. Add Path to Engine
            engine._jammer_paths[jammer_name] = path
            engine._jammer_metadata[jammer_name] = {"strategy": "GraphNav_Batch"}
            engine.set_padding_mode(jammer_name, padding_mode)
            
            # B. Add Physical Transmitter to Scene
            if jammer_name not in scene.transmitters:
                # print(f"  Spawning {jammer_name} in scene...", end="\r")
                new_tx = Transmitter(name=jammer_name, position=path[0], power_dbm=30)
                scene.add(new_tx)
            
            print(f"  Generated {jammer_name}...", end="\r")
            
        except Exception as e:
            raise RuntimeError(f"Error generating path for jammer {i}: {e}")
    
    print(f"\n\nCreated {len(engine._jammer_paths)} simultaneous jammers.")
    
    # 3. Synchronize Lengths (Padding)
    engine.finalize_trajectories()

    # 4. Save Dataset
    save_dataset(output_dir, dataset_name, paths=engine.get_all_paths(), metadata=engine.get_all_metadata())

    # 4. Run Simulation
    run_simulation(engine, scene, map_center, map_size, cell_size, os.path.join(output_dir, dataset_name), buildings=buildings)

def run_simulation(engine, scene, map_center, map_size, cell_size, output_dir, buildings):
    """Executes the Sionna RadioMapSolver loop for all active jammers in the engine."""
    total_steps = engine.get_max_path_length()
    if total_steps == 0: return

    print(f"\nStarting Sionna simulation for {total_steps} steps...")
    
    # Initialize Solver
    rm_solver = RadioMapSolver()

    # --- Storage Initialization ---
    # 1. List for the Aggregated RSS frames (dBm)
    aggregated_rss_history = []
    
    # 2. Dict for Individual RSS frames (dBm)
    # Mapping: jammer_name -> list of frames
    individual_rss_histories = {name: [] for name in scene.transmitters}

    tx_power_dbm = 30.0
    tx_power_linear = 10**(tx_power_dbm / 10.0) / 1000.0 # Convert dBm to Watts

    for step in range(total_steps):

        # 1. Move Actors
        engine.update_scene_transmitters(step)
        
        print(f"Processing step {step}/{total_steps}...", end="\r")

        # 2. Compute Radio Map
        try:
            rm = rm_solver(
                scene,
                max_depth=50,                       # Maximum ray bounces 
                samples_per_tx=10**6,               # More samples = less noise but more memory
                cell_size=cell_size,                # Resolution of the radio map
                center=map_center,                  # Center of the coverage area
                size=[map_size[0], map_size[1]],    # Total size of the radio map #type: ignore
                orientation=[0, 0, 0],              # Horizontal orientation (Z-up) #type: ignore
                diffraction=True,
                edge_diffraction=False 
            )

            # linear_gain shape is typically (Tx, H, W)
            linear_gain = rm.rss.numpy() 

            # Ensure shape consistency if there is only 1 transmitter
            if len(linear_gain.shape) == 2:
                linear_gain = np.expand_dims(linear_gain, axis=0)
            
            # --- A. Save Aggregated Map ---
            aggregate_rss_watts = np.sum(linear_gain * tx_power_linear, axis=0)
            agg_dbm = 10 * np.log10(np.maximum(aggregate_rss_watts, 1e-20)) + 30 
            aggregated_rss_history.append(agg_dbm)

            # --- B. Save Individual Maps ---
            # Sionna returns results in the order of insertion into `scene.transmitters`
            # We iterate safely by index
            tx_names = list(scene.transmitters.keys())
            
            for i, name in enumerate(tx_names):
                # Extract specific slice for this transmitter
                indiv_watts = linear_gain[i] * tx_power_linear
                indiv_dbm = 10 * np.log10(np.maximum(indiv_watts, 1e-20)) + 30
                
                individual_rss_histories[name].append(indiv_dbm)

        except Exception as e:
            raise RuntimeError(f"Error during RadioMap computation at step {step}: {e}")
            

    # --- Save RSS Data ---    
    # 1. Save Aggregated
    agg_stack = np.array(aggregated_rss_history) # Shape: (T, N, M)
    np.save(os.path.join(output_dir, "rss_aggregated.npy"), agg_stack)

    # 2. Save Individual
    for name, frames in individual_rss_histories.items():
        indiv_stack = np.array(frames) # Shape: (T, N, M)
        clean_name = name.replace(" ", "_") # Safety for filenames
        filename = f"rss_{clean_name}.npy"
        np.save(os.path.join(output_dir, filename), indiv_stack)

    # --- Generate GIF (using aggregated data) ---
    print("Creating summary animation...")
    gif_path = os.path.join(output_dir, "jammer_animation.gif")
    create_jammer_animation(
        rss_list=aggregated_rss_history,
        engine=engine,
        buildings=buildings,
        map_size=map_size,
        map_center=map_center,
        filename=gif_path
    )

def save_dataset(root_dir, dataset_name, paths, metadata=None):
    """Unified logic to clear directory and save numpy arrays."""
    save_path = os.path.join(root_dir, dataset_name)
    
    # Clear existing
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    # Save Items
    for name, data in paths.items():
        # Save Path
        clean_name = name.replace(" ", "_")
        np.save(os.path.join(save_path, f"path_{clean_name}.npy"), data)
        
        # Save Metadata (if available for this item)
        if metadata and name in metadata:
            with open(os.path.join(save_path, f"meta_{clean_name}.txt"), "w") as f:
                for k, v in metadata[name].items():
                    f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()