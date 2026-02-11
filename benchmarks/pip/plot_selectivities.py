#!/usr/bin/env python3
"""Extract and plot selectivities from benchmark results.

This script:
1. Finds benchmark JSON files for a specified experiment type (cube, sphere, etc.)
2. Extracts selectivities (inside_count / total_points) for each extent
3. Averages selectivities across all grid positions and approaches
4. Plots as a line chart
5. Saves as PNG

Usage:
    python plot_selectivities.py cube
    python plot_selectivities.py sphere
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def extract_experiment_info(filename):
    """Extract experiment type and extent from filename.
    
        Filename pattern examples:
            - grid_benchmark_<name>_<number>.json
            - grid_benchmark_<name>_<number>_<timestamp>.json
            - grid_benchmark_<name>_<number>_<timestamp>_<subtimestamp>.json
    
    Args:
        filename: Path or filename string
        
    Returns:
        tuple: (experiment_type, extent) or (None, None) if not found
               e.g., ('cube', 1), ('sphere', 5)
    """
    s = str(filename)
    # Match the core pattern with optional trailing timestamp parts and the .json extension
    match = re.match(r'grid_benchmark_([^_]+)_(\d+)(?:_.*)?\.json$', s)
    if match:
        return match.group(1), int(match.group(2))

    # Fallback: try to find name and number anywhere in the filename
    match = re.search(r'grid_benchmark_(\w+)_(\d+)', s)
    if match:
        return match.group(1), int(match.group(2))

    return None, None


def load_benchmark_file(filepath):
    """Load and parse a benchmark JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        dict: Parsed JSON data or None if invalid
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None


def extract_selectivities(data):
    """Extract all selectivities from benchmark data.
    
    Args:
        data: Parsed benchmark JSON data
        
    Returns:
        list: List of selectivity values (inside_count / total_points)
    """
    selectivities = []
    
    if 'results' not in data:
        return selectivities
    
    for approach, results_list in data['results'].items():
        if not isinstance(results_list, list):
            continue
            
        for result in results_list:
            if not isinstance(result, dict):
                continue
                
            inside_count = result.get('inside_count')
            total_points = result.get('total_points')
            
            if inside_count is not None and total_points is not None and total_points > 0:
                selectivity = inside_count / total_points
                selectivities.append(selectivity)
    
    return selectivities


def main():
    """Main function to process benchmarks and create plot."""
    parser = argparse.ArgumentParser(
        description='Plot selectivities for a specific experiment type'
    )
    parser.add_argument(
        'experiment',
        type=str,
        help='Experiment type to plot (e.g., cube, sphere)'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default=None,
        help='Directory containing benchmark results (default: ./results)'
    )
    
    args = parser.parse_args()
    experiment_name = args.experiment.lower()
    
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent / 'results'
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1
    
    experiment_data = defaultdict(list)
    
    pattern = f'grid_benchmark_{experiment_name}_*.json'
    for json_file in results_dir.glob(pattern):
        exp_type, extent = extract_experiment_info(json_file.name)
        if exp_type is None or extent is None:
            continue
        if exp_type != experiment_name:
            continue
        
        data = load_benchmark_file(json_file)
        if data is None:
            continue
        
        selectivities = extract_selectivities(data)
        if selectivities:
            experiment_data[extent].extend(selectivities)
            print(f"Processed {json_file.name}: {len(selectivities)} selectivities")
    
    if not experiment_data:
        print(f"Error: No {experiment_name} benchmark files found or no selectivities extracted")
        return 1
    
    extents = sorted(experiment_data.keys())
    avg_selectivities = []
    
    for extent in extents:
        selectivities = experiment_data[extent]
        avg_selectivity = np.mean(selectivities)
        avg_selectivities.append(avg_selectivity)
        print(f"{experiment_name.capitalize()} extent {extent}: {len(selectivities)} measurements, "
              f"average selectivity = {avg_selectivity:.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(extents, avg_selectivities, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Extent', fontsize=12)
    plt.ylabel('Average Selectivity', fontsize=12)
    plt.title(f'Average Selectivity vs {experiment_name.capitalize()} Extent\n(10 million points dataset)', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = results_dir / f'{experiment_name}_selectivities.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())

