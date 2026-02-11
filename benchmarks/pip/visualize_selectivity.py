#!/usr/bin/env python3
"""
Visualize selectivity benchmark results showing runtime vs selectivity 
for different point counts (10M, 200M, 500M) and approaches.
Creates separate plots for each point count.
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

# Selectivity values we're looking for
SELECTIVITIES = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8]

# Point count labels
POINT_COUNT_LABELS = {
    "10M": "10 Million",
    "100M": "100 Million",
    "200M": "200 Million",
    "500M": "500 Million"
}

# Approach colors and markers
APPROACH_COLORS = {
    'CGAL': '#1f77b4',
    'SQL': '#2ca02c',
    'Raytracer': '#d62728',
    'FilterRefine': '#9467bd',
    'CUDA': '#ff7f0e'
}

APPROACH_MARKERS = {
    'CGAL': 'o',
    'SQL': 's',
    'Raytracer': '^',
    'FilterRefine': 'D',
    'CUDA': 'v'
}


def parse_selectivity_filename(filename):
    """Extract selectivity, point count, and timestamp from filename."""
    # Example: grid_benchmark_sphere_selectivity_0.01_200M_20251211_045047.json
    # Example: grid_benchmark_sphere_selectivity_0.01_20251211_025400.json (10M, no annotation)
    
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
    
    # If no point count found, it's 10M (no annotation in filename)
    if point_count is None and selectivity is not None:
        point_count = "10M"
    
    return selectivity, point_count, timestamp


def load_benchmark_data():
    """Load all selectivity benchmark JSONs and organize by selectivity and point count."""
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
        selectivity, point_count = key
        print(f"  Selectivity {selectivity:6.4f}, {point_count}: Using timestamp {entries[0]['timestamp']}")
    
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


def plot_selectivity_for_point_count(runtimes, point_count, output_file):
    """Create plot for a specific point count."""
    
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot each approach
    for approach in sorted(runtimes.keys()):
        if point_count not in runtimes[approach]:
            continue
            
        selectivities = []
        runtime_values = []
        
        for sel in sorted(runtimes[approach][point_count].keys()):
            selectivities.append(sel)
            runtime_values.append(runtimes[approach][point_count][sel])
        
        if selectivities:
            marker = APPROACH_MARKERS.get(approach, 'o')
            color = APPROACH_COLORS.get(approach, 'gray')
            
            ax.plot(selectivities, runtime_values, 
                   marker=marker, linestyle='-', linewidth=2.5, markersize=10,
                   label=approach, color=color, alpha=0.8)
    
    # Configure axes
    ax.set_xlabel('Selectivity (log scale)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Runtime (ms, log scale)', fontsize=16, fontweight='bold')
    
    point_label = POINT_COUNT_LABELS.get(point_count, point_count)
    ax.set_title(f'Sphere Query Performance vs Selectivity\n({point_label} Points)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Grid
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.7)
    ax.grid(True, which='minor', alpha=0.15, linestyle=':', linewidth=0.5)
    
    # Legend
    ax.legend(loc='best', fontsize=13, framealpha=0.95, shadow=True)
    
    # Tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved {point_count} plot to {output_file}")
    plt.close()


def print_summary_table(runtimes, point_count):
    """Print a summary table for a specific point count."""
    print(f"\n{'='*80}")
    print(f"SELECTIVITY BENCHMARK SUMMARY - {POINT_COUNT_LABELS.get(point_count, point_count)} Points")
    print(f"{'='*80}")
    
    # Get all selectivities for this point count
    all_selectivities = sorted(set(
        sel for approach_data in runtimes.values()
        if point_count in approach_data
        for sel in approach_data[point_count].keys()
    ))
    
    if not all_selectivities:
        print(f"No data available for {point_count}")
        return
    
    # Print header
    print(f"\n{'Selectivity':<15}", end="")
    approaches = sorted([app for app in runtimes.keys() if point_count in runtimes[app]])
    for approach in approaches:
        print(f"{approach:<15}", end="")
    print()
    print("-" * 80)
    
    # Print data rows
    for sel in all_selectivities:
        print(f"{sel:<15.4f}", end="")
        for approach in approaches:
            runtime = runtimes[approach][point_count].get(sel, None)
            if runtime is not None:
                print(f"{runtime:<15.2f}", end="")
            else:
                print(f"{'N/A':<15}", end="")
        print()
    
    print("="*80)


def main():
    """Main function to generate selectivity visualizations."""
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*80)
    print("SELECTIVITY VISUALIZATION GENERATOR")
    print("="*80)
    
    print("\nLoading selectivity benchmark data...")
    benchmark_data = load_benchmark_data()
    print(f"\n✓ Loaded {len(benchmark_data)} unique configurations")
    
    print("\nExtracting runtime data...")
    runtimes = extract_runtimes(benchmark_data)
    
    # Find all point counts available
    all_point_counts = sorted(set(
        pc for approach_data in runtimes.values()
        for pc in approach_data.keys()
    ))
    
    print(f"\n✓ Found data for point counts: {', '.join(all_point_counts)}")
    
    # Print summary for each approach
    print("\nApproaches found:")
    for approach in sorted(runtimes.keys()):
        point_counts = sorted(runtimes[approach].keys())
        print(f"  • {approach}: {', '.join(point_counts)}")
    
    # Generate plots for each point count
    print("\nGenerating selectivity visualizations...")
    for point_count in all_point_counts:
        print(f"\n  Processing {point_count}...")
        
        # Print summary table
        print_summary_table(runtimes, point_count)
        
        # Generate plot
        output_file = os.path.join(OUTPUT_DIR, f"sphere_selectivity_{point_count}.png")
        plot_selectivity_for_point_count(runtimes, point_count, output_file)
    
    print("\n" + "="*80)
    print("✓ SELECTIVITY VISUALIZATIONS COMPLETE")
    print(f"✓ Generated {len(all_point_counts)} plots")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
