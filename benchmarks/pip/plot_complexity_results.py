#!/usr/bin/env python3
"""
Visualize complexity benchmark results showing runtime vs mesh complexity (faces)
for different approaches.
"""

import json
import os
import glob
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Define output directory
OUTPUT_DIR = "results/visualizations"
RESULTS_DIR = "results"

# Complexity levels and their corresponding face counts (from README)
COMPLEXITY_FACES = {
    1: 48,
    2: 5928,
    3: 21608,
    4: 47088,
    5: 82368,
    6: 127804,
    7: 182754,
    8: 247504,
    9: 322054,
    10: 408320
}


def parse_complexity_filename(filename):
    """Extract stage number and timestamp from filename."""
    # Example: grid_benchmark_sphere_complexity_stage_5_20251211_044554.json
    
    parts = filename.replace(".json", "").split("_")
    
    stage = None
    timestamp = None
    
    for i, part in enumerate(parts):
        if part == "stage" and i + 1 < len(parts):
            try:
                stage = int(parts[i + 1])
            except ValueError:
                pass
        
        # Get timestamp (last part before .json)
        if i == len(parts) - 1:
            timestamp = part
    
    return stage, timestamp


def load_benchmark_data():
    """Load all complexity benchmark JSONs and organize by stage."""
    pattern = os.path.join(RESULTS_DIR, "grid_benchmark_sphere_complexity_stage_*.json")
    files = glob.glob(pattern)
    
    # Group by stage and keep the latest timestamp
    data_by_stage = defaultdict(list)
    
    for filepath in files:
        filename = os.path.basename(filepath)
        stage, timestamp = parse_complexity_filename(filename)
        
        if stage is None:
            continue
        
        # Load the JSON
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        data_by_stage[stage].append({
            'timestamp': timestamp,
            'data': data,
            'filepath': filepath
        })
    
    # Keep only the latest for each stage
    latest_data = {}
    for stage, entries in data_by_stage.items():
        # Sort by timestamp descending
        entries.sort(key=lambda x: x['timestamp'], reverse=True)
        latest_data[stage] = entries[0]['data']
    
    return latest_data


def extract_runtimes(benchmark_data):
    """Extract runtime data organized by approach and stage."""
    # Structure: {approach: {stage: runtime_ms}}
    runtimes = defaultdict(dict)
    
    for stage, data in benchmark_data.items():
        results = data.get('results', {})
        
        for approach, runs in results.items():
            if not runs or len(runs) == 0:
                continue
            
            # Average runtime across runs (if multiple)
            avg_runtime = np.mean([run.get('total_query_ms', 0) for run in runs])
            runtimes[approach][stage] = avg_runtime
    
    return runtimes


def plot_complexity_runtimes(runtimes, output_file):
    """Create plot with face count on x-axis (linear) and runtime on y-axis (log scale)."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Colors and markers for different approaches
    colors = {'CGAL': 'blue', 'SQL': 'green', 'Raytracer': 'red', 
              'FilterRefine': 'purple', 'CUDA': 'orange'}
    markers = {'CGAL': 'o', 'SQL': 's', 'Raytracer': '^', 
               'FilterRefine': 'D', 'CUDA': 'v'}
    
    # Plot each approach
    for approach in sorted(runtimes.keys()):
        stages = []
        face_counts = []
        runtime_values = []
        
        for stage in sorted(runtimes[approach].keys()):
            if stage in COMPLEXITY_FACES:
                stages.append(stage)
                face_counts.append(COMPLEXITY_FACES[stage])
                runtime_values.append(runtimes[approach][stage])
        
        if face_counts:
            marker = markers.get(approach, 'o')
            color = colors.get(approach, 'gray')
            
            ax.plot(face_counts, runtime_values, 
                   marker=marker, linestyle='-', linewidth=2, markersize=8,
                   label=approach, color=color, alpha=0.7)
    
    # Configure axes
    ax.set_xlabel('Number of Faces (Mesh Complexity)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Runtime (ms, log scale)', fontsize=14, fontweight='bold')
    ax.set_title('Sphere Query Performance: Runtime vs Mesh Complexity\n(100M points, 1% selectivity)', 
                fontsize=16, fontweight='bold')
    
    # Log scale for y-axis, linear for x-axis
    ax.set_yscale('log')
    
    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--')
    
    # Legend
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # Format x-axis to show face counts nicely
    ax.ticklabel_format(style='plain', axis='x')
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def main():
    """Main function to generate complexity visualization."""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading complexity benchmark data...")
    benchmark_data = load_benchmark_data()
    print(f"Loaded {len(benchmark_data)} stages")
    
    print("Extracting runtime data...")
    runtimes = extract_runtimes(benchmark_data)
    
    # Print summary
    print("\nData summary:")
    for approach in sorted(runtimes.keys()):
        n_stages = len(runtimes[approach])
        stages = sorted(runtimes[approach].keys())
        print(f"  {approach}: {n_stages} stages ({stages})")
    
    print("\nGenerating plot...")
    output_file = os.path.join(OUTPUT_DIR, "complexity_runtimes.png")
    plot_complexity_runtimes(runtimes, output_file)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
