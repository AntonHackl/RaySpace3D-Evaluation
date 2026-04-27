import os
import trimesh
import numpy as np
import glob
from tqdm import tqdm

def analyze_bboxes(data_dir):
    glb_files = glob.glob(os.path.join(data_dir, "*.glb"))
    if not glb_files:
        print(f"No GLB files found in {data_dir}")
        return

    print(f"Found {len(glb_files)} GLB files. Analyzing bounding boxes...")

    stats = {
        'width': [],   # X extent
        'height': [],  # Y extent
        'depth': [],   # Z extent
        'max_dim': [], # max(X, Y, Z extent)
        'diagonal': [] # sqrt(X^2 + Y^2 + Z^2)
    }

    for f in tqdm(glb_files):
        try:
            mesh = trimesh.load(f, force='mesh')
            bounds = mesh.bounds
            extents = bounds[1] - bounds[0]
            
            w, h, d = extents
            stats['width'].append(w)
            stats['height'].append(h)
            stats['depth'].append(d)
            stats['max_dim'].append(max(w, h, d))
            stats['diagonal'].append(np.sqrt(np.sum(extents**2)))
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print("\n--- Bounding Box Statistics ---")
    for key, values in stats.items():
        v = np.array(values)
        print(f"\n{key.capitalize()}:")
        print(f"  Min:    {np.min(v):.2f}")
        print(f"  Max:    {np.max(v):.2f}")
        print(f"  Mean:   {np.mean(v):.2f}")
        print(f"  Median: {np.median(v):.2f}")
        print(f"  Std:    {np.std(v):.2f}")

if __name__ == "__main__":
    target_dir = "/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/datasets_scripts/microns_data/microns_region_4gb_glb"
    analyze_bboxes(target_dir)
