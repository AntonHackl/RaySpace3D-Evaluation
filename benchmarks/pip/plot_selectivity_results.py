#!/usr/bin/env python3
"""
Visualize selectivity benchmark results showing runtime vs selectivity 
for different point counts and approaches.
"""

import json
import os
import glob
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Define output directory
OUTPUT_DIR = "results/visualizations"
RESULTS_DIR = "results"

# Selectivity values we're looking for
SELECTIVITIES = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8]

# Point counts - only use non-annotated (implicit 100M) data
POINT_COUNTS = ["100M"]  # No annotation in filename


def parse_selectivity_filename(filename):
    """Extract selectivity and point count from filename."""
    # Example: grid_benchmark_sphere_selectivity_0.01_200M_20251211_045047.json
    # Example: grid_benchmark_sphere_selectivity_0.01_20251211_025400.json (100M, no annotation)
    
    parts = filename.replace(".json", "").split("_")
    
    # Find selectivity
    selectivity = None
    point_count = None
    timestamp = None
    
    for i, part in enumerate(parts):
        if part == "selectivity" and i + 1 < len(parts):
            try:
                selectivity = float(parts[i + 1])
            except ValueError:
                pass
        
        # Check for point count annotation (200M, 500M)
        if part in ["200M", "500M"]:
            point_count = part
        
        # Get timestamp (last part before .json)
        if i == len(parts) - 1:
            timestamp = part
    
    # If no point count found, it's 100M (no annotation)
    if point_count is None and selectivity is not None:
        point_count = "100M"
    
    return selectivity, point_count, timestamp


def load_benchmark_data():
    """Load all selectivity benchmark JSONs and organize by selectivity, point count, approach."""
    pattern = os.path.join(RESULTS_DIR, "grid_benchmark_sphere_selectivity_*.json")
    files = glob.glob(pattern)
    
    # Group by (selectivity, point_count) and keep the latest timestamp
    data_by_config = defaultdict(list)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        selectivity, point_count, timestamp = parse_selectivity_filename(filename)
        
        if selectivity is None or point_count is None:
            continue
        
        # Load the JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        key = (selectivity, point_count)
        data_by_config[key].append({
            'timestamp': timestamp,
            'data': data,
            'filepath': filepath
        })
    
    # Keep only the latest for each (selectivity, point_count)
    latest_data = {}
    for key, entries in data_by_config.items():
        # Sort by timestamp descending
        entries.sort(key=lambda x: x['timestamp'], reverse=True)
        latest_data[key] = entries[0]['data']
    
    return latest_data


def extract_runtimes(benchmark_data):
    """Extract runtime data organized by approach, point count, and selectivity."""
    # Structure: {approach: {point_count: {selectivity: runtime_ms}}}
    runtimes = defaultdict(lambda: defaultdict(dict))
    
    for (selectivity, point_count), data in benchmark_data.items():
        results = data.get('results', {})
        
        for approach, runs in results.items():
            if not runs or len(runs) == 0:
                continue
            
            # Average runtime across runs (if multiple)
            avg_runtime = np.mean([run.get('total_query_ms', 0) for run in runs])
            runtimes[approach][point_count][selectivity] = avg_runtime
    
    return runtimes


def plot_selectivity_runtimes(runtimes, output_file):
    """Create plot with selectivity on x-axis (log scale) and runtime on y-axis (log scale)."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors and markers for different approaches
    colors = {'CGAL': 'blue', 'SQL': 'green', 'Raytracer': 'red', 
              'FilterRefine': 'purple', 'CUDA': 'orange'}
    markers = {'CGAL': 'o', 'SQL': 's', 'Raytracer': '^', 'FilterRefine': 'D', 'CUDA': 'v'}
    
    # Plot each approach - only 100M data
    for approach in sorted(runtimes.keys()):
        for point_count in sorted(runtimes[approach].keys()):
            # Only plot 100M data
            if point_count != '100M':
                continue
            selectivities = []
            runtime_values = []
            
            for sel in sorted(runtimes[approach][point_count].keys()):
                selectivities.append(sel)
                runtime_values.append(runtimes[approach][point_count][sel])
            
            if selectivities:
                label = approach
                marker = markers.get(approach, 'o')
                color = colors.get(approach, 'gray')
                
                ax.plot(selectivities, runtime_values, 
                       marker=marker, linestyle='-', linewidth=2, markersize=8,
                       label=label, color=color, alpha=0.7)
    
    # Configure axes
    ax.set_xlabel('Selectivity', fontsize=14, fontweight='bold')
    ax.set_ylabel('Runtime (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Sphere Query Performance: Runtime vs Selectivity\n(100M points, Log-Log Scale)', 
                fontsize=16, fontweight='bold')
    
    # Log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def main():
    """Main function to generate selectivity visualization."""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading selectivity benchmark data...")
    benchmark_data = load_benchmark_data()
    print(f"Loaded {len(benchmark_data)} unique configurations")
    
    print("Extracting runtime data...")
    runtimes = extract_runtimes(benchmark_data)
    
    # Print summary
    print("\nData summary:")
    for approach in sorted(runtimes.keys()):
        print(f"  {approach}:")
        for point_count in sorted(runtimes[approach].keys()):
            n_selectivities = len(runtimes[approach][point_count])
            print(f"    {point_count}: {n_selectivities} selectivities")
    
    print("\nGenerating plot...")
    output_file = os.path.join(OUTPUT_DIR, "selectivity_runtimes.png")
    plot_selectivity_runtimes(runtimes, output_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
