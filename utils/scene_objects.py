import sionna.rt as rt
import os
import numpy as np
import trimesh
from collections import defaultdict

def create_scene_objects(
    scene,
    map_bounds,
    tx_array_args={"num_rows": 1, "num_cols": 1, "pattern": "iso", "polarization": "V"}, 
    rx_array_args={"num_rows": 1, "num_cols": 1, "pattern": "iso", "polarization": "V"},
    z_height=0.0
):
    """
    Configures static scene elements: Antenna arrays, map boundaries, and center.
    """
    
    # 1. Setup Arrays (Global config)
    scene.tx_array = rt.PlanarArray(**tx_array_args)
    scene.rx_array = rt.PlanarArray(**rx_array_args)

    # 2. Calculated Map Parameters    
    x_min, x_max = map_bounds['x']
    y_min, y_max = map_bounds['y']
    
    map_width = x_max - x_min
    map_height = y_max - y_min
    map_center = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_height]

    return map_center, (map_width, map_height)

def gather_bboxes(mesh_dir):
    obstacles = []
    
    # Get all .ply files
    files = [f for f in os.listdir(mesh_dir) if f.endswith(".ply")]

    # Filter out non-building objects if necessary
    ignored_keywords = ['road', 'sidewalk', 'ground']
    
    print(f"Processing {len(files)} buildings directly from geometry...")

    for file in files:
        if any(k in file for k in ignored_keywords):
            continue

        mesh_path = os.path.join(mesh_dir, file)
        # 1. Load the mesh
        mesh = trimesh.load(mesh_path, force='mesh')
        
        # 2. Get standard bounds (Min/Max) for the engine collisions
        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]

        try:
            # 3. EXTRACT FOOTPRINT FROM GEOMETRY
            # We cut a slice 0.5 meters above the bottom of the building.
            # This avoids issues with uneven ground or bottom faces.
            slice_height = bbox_min[2] + 0.5
            
            # Create a cross-section
            section = mesh.section(plane_origin=[0, 0, slice_height], plane_normal=[0, 0, 1]) #type: ignore
            
            footprint_coords = None

            if section:
                # Convert the 3D slice to a 2D planar polygon
                # 'to_planar' returns (2D_geometry, transformation_matrix)
                planar_section, to_3D = section.to_planar()
                
                # A slice might result in multiple polygons (e.g. inner courtyards).
                # We usually just want the largest outer boundary.
                if len(planar_section.polygons_closed) > 0:
                    # Sort by area and take the largest one to be safe
                    largest_poly = max(planar_section.polygons_closed, key=lambda p: p.area)
                    
                    # Simplify slightly to reduce point count (optional, improves FPS)
                    simplified_poly = largest_poly.simplify(0.1, preserve_topology=False)
                    
                    coords_2d = np.array(list(simplified_poly.exterior.coords))
                        
                    # 1. Pad 2D coords with Z=0 to make them 3D compatible
                    coords_3d_local = np.column_stack((coords_2d, np.zeros(len(coords_2d))))
                    
                    # 2. Apply the transformation matrix to get back to World Space
                    coords_3d_world = trimesh.transform_points(coords_3d_local, to_3D)
                    
                    # 3. Extract just X and Y
                    footprint_coords = coords_3d_world[:, :2].tolist()

        # 4. Fallback if slicing fails (e.g. flat planes, broken meshes)
        except Exception as e:
            print(f"  Warning: Footprint extraction failed for {file} with error: {e}")
            # Use a simple rectangle as fallback so the app doesn't crash
            min_x, min_y = bbox_min[0], bbox_min[1]
            max_x, max_y = bbox_max[0], bbox_max[1]
            footprint_coords = [
                (min_x, min_y), (max_x, min_y), 
                (max_x, max_y), (min_x, max_y), 
                (min_x, min_y)
            ]

        obstacles.append({
            "file": file,
            "min": bbox_min,
            "max": bbox_max,
            "footprint": footprint_coords
        })

    return obstacles
