import os
import numpy as np
import trimesh
from collections import defaultdict

MESH_DIR = r"C:\Users\sawyer\Documents\sionna_rt_jamming\data\downtown_chicago\meshes"
PRINT_LIMIT = 20

building_bboxes = []

for file in os.listdir(MESH_DIR):
    if file.endswith(".ply"):
        mesh_path = os.path.join(MESH_DIR, file)
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
        
        bbox_min = mesh.bounds[0]
        bbox_max = mesh.bounds[1]
        building_bboxes.append({
            "file": file,
            "bbox_min": bbox_min,
            "bbox_max": bbox_max
        })

print(f"Found {len(building_bboxes)} individual meshes.")

building_groups = defaultdict(list)
for b in building_bboxes:
    # Strip material suffixes to get building base name
    base_name = b['file'].replace('-itu_concrete.ply', '').replace('-itu_metal.ply', '')
    building_groups[base_name].append(b)

merged_bboxes = []
for name, meshes in building_groups.items():
    mins = np.array([m['bbox_min'] for m in meshes])
    maxs = np.array([m['bbox_max'] for m in meshes])
    bbox_min = mins.min(axis=0)
    bbox_max = maxs.max(axis=0)
    merged_bboxes.append({
        "name": name,
        "bbox_min": bbox_min,
        "bbox_max": bbox_max
    })

for b in merged_bboxes[:PRINT_LIMIT]:
    print(f"{b['name']} | min: {b['bbox_min']}, max: {b['bbox_max']}")
