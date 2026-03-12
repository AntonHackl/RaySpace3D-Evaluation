#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import json
import sys
from pathlib import Path

def visualize_speedup(json_path, output_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    counts = data["counts"]
    tdbase_means = np.array(data["tdbase"]["mean"])
    exact_means = np.array(data["exact"]["mean"])
    
    # Handle different names for estimated approach
    if "estimated" in data:
        estimated_means = np.array(data["estimated"]["mean"])
    elif "direct_estimation" in data:
        estimated_means = np.array(data["direct_estimation"]["mean"])
    else:
        # Fallback if neither is present
        estimated_means = np.ones_like(exact_means) * np.nan
        print("Warning: No estimated approach data found in JSON")
    
    # Calculate speedup
    exact_speedup = tdbase_means / exact_means
    estimated_speedup = tdbase_means / estimated_means
    
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(counts))
    width = 0.35
    
    plt.bar(x - width/2, exact_speedup, width, label='RaySpace Exact Speedup', color='#1f77b4', alpha=0.8)
    plt.bar(x + width/2, estimated_speedup, width, label='RaySpace Estimated Speedup', color='#2ca02c', alpha=0.8)
    
    # Add labels on top of bars
    for i, (ex, est) in enumerate(zip(exact_speedup, estimated_speedup)):
        plt.text(i - width/2, ex + 0.5, f'{ex:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(i + width/2, est + 0.5, f'{est:.1f}x', ha='center', va='bottom', fontsize=10, fontweight='bold', color='green')

    plt.xlabel('Nuclei per Vessel', fontsize=12)
    plt.ylabel('Speedup over TDBase (x)', fontsize=12)
    plt.title('RaySpace3D Speedup relative to TDBase', fontsize=14, fontweight='bold')
    plt.xticks(x, [str(c) for c in counts])
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "mesh_overlap_nu_speedup.png"
    plt.savefig(output_path, dpi=300)
    print(f"Speedup visualization saved to {output_path}")
    
    plt.savefig(str(output_path).replace('.png', '.pdf'))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 visualize_nu_speedup.py <results_json>")
        sys.exit(1)
        
    json_file = sys.argv[1]
    output_directory = Path(__file__).parent / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)
    
    visualize_speedup(json_file, output_directory)
