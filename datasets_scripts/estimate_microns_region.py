#!/usr/bin/env python3
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Estimate MICrONS bounding box size for a target GB.")
    parser.add_argument("--target-gb", type=float, required=True)
    parser.add_argument("--avg-mesh-mb", type=float, default=112.0, help="Average mesh size in MB.")
    parser.add_argument("--neuron-density-mm3", type=float, default=10000.0, help="Estimated proofread neurons per mm3.")
    args = parser.parse_args()

    # Target bytes
    target_mb = args.target_gb * 1024
    num_neurons = target_mb / args.avg_mesh_mb
    
    # Volume needed in mm3
    volume_mm3 = num_neurons / args.neuron_density_mm3
    side_mm = volume_mm3 ** (1/3)
    side_um = side_mm * 1000
    side_nm = side_um * 1000

    print(f"--- MICrONS Region Estimation for {args.target_gb} GB ---")
    print(f"Target Neurons: {num_neurons:.1f}")
    print(f"Required Volume: {volume_mm3:.6f} mm3")
    print(f"Cube Side Length: {side_um:.1f} um ({side_nm:.0f} nm)")
    
    # Center of Minnie65
    cx, cy, cz = 897688, 708628, 854160
    
    x_min, x_max = cx - side_nm/2, cx + side_nm/2
    y_min, y_max = cy - side_nm/2, cy + side_nm/2
    z_min, z_max = cz - side_nm/2, cz + side_nm/2
    
    print("\nSuggested Bounding Box Arguments:")
    print(f"  --x-min-nm {x_min:.0f} --x-max-nm {x_max:.0f}")
    print(f"  --y-min-nm {y_min:.0f} --y-max-nm {y_max:.0f}")
    print(f"  --z-min-nm {z_min:.0f} --z-max-nm {z_max:.0f}")

if __name__ == "__main__":
    main()
