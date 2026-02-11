#!/usr/bin/env python3
"""Evaluate and compare multiple benchmark results.

This script loads timestamped benchmark results and generates
comparison visualizations showing mean and median query times.

Usage:
    python evaluate_benchmarks.py --results-dir ../results [--output-dir figures]
    python evaluate_benchmarks.py --results results/grid_benchmark_*.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import glob

import numpy as np
import matplotlib.pyplot as plt


def load_benchmark_results(result_files: List[str]) -> List[Dict[str, Any]]:
    """Load multiple benchmark result JSON files.
    
    Args:
        result_files: List of paths to result JSON files
        
    Returns:
        List of loaded benchmark data dicts
    """
    results = []
    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                data['_source_file'] = filepath
                results.append(data)
                print(f"Loaded: {filepath}")
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")
    
    return results


def extract_query_times(benchmark_data: Dict[str, Any], approach: str, metric: str = 'query_ms') -> List[float]:
    """Extract query times for a specific approach from benchmark data.
    
    Args:
        benchmark_data: Loaded benchmark JSON data
        approach: Name of the approach (e.g., 'CGAL', 'Raytracer')
        metric: Which timing metric to use ('query_ms' or 'rayspace_total_ms')
        
    Returns:
        List of query times in milliseconds
    """
    results = benchmark_data.get('results', {})
    if approach not in results:
        return []
    
    times = []
    for result in results[approach]:
        if result.get('success') and result.get(metric) is not None:
            times.append(result[metric])
    
    return times


def plot_approach_comparison(benchmarks: List[Dict[str, Any]], output_dir: Path, metric: str = 'query_ms'):
    """Generate bar charts comparing approaches across benchmarks.
    
    Creates visualizations showing mean and median query times.
    
    Args:
        benchmarks: List of loaded benchmark data dicts
        output_dir: Directory to save figures
        metric: Which timing metric to use
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all approaches across all benchmarks
    all_approaches = set()
    for bench in benchmarks:
        all_approaches.update(bench.get('results', {}).keys())
    
    all_approaches = sorted(all_approaches)
    
    if not all_approaches:
        print("No approaches found in benchmark data")
        return
    
    # Determine metric label
    metric_label = "RaySpace Total Time (ms)" if metric == 'rayspace_total_ms' else "Query Time (ms)"
    
    # For each benchmark, collect statistics
    benchmark_labels = []
    approach_means = {app: [] for app in all_approaches}
    approach_medians = {app: [] for app in all_approaches}
    
    for bench in benchmarks:
        # Use timestamp or filename as label
        label = bench.get('timestamp', Path(bench['_source_file']).stem)
        benchmark_labels.append(label)
        
        for approach in all_approaches:
            times = extract_query_times(bench, approach, metric)
            if times:
                approach_means[approach].append(np.mean(times))
                approach_medians[approach].append(np.median(times))
            else:
                approach_means[approach].append(None)
                approach_medians[approach].append(None)
    
    # Plot mean comparison
    fig, ax = plt.subplots(figsize=(max(12, len(all_approaches) * 2), 6))
    
    x = np.arange(len(all_approaches))
    width = 0.8 / len(benchmarks) if benchmarks else 0.8
    
    for i, (bench, label) in enumerate(zip(benchmarks, benchmark_labels)):
        means = [approach_means[app][i] for app in all_approaches]
        positions = x + (i - len(benchmarks)/2 + 0.5) * width
        
        # Filter out None values
        plot_positions = []
        plot_means = []
        for pos, mean in zip(positions, means):
            if mean is not None:
                plot_positions.append(pos)
                plot_means.append(mean)
        
        if plot_means:
            ax.bar(plot_positions, plot_means, width, label=label, alpha=0.8)
    
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title('Mean Query Time Comparison Across Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_approaches, rotation=15, ha='right')
    ax.legend(title='Benchmark Run')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mean_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'mean_comparison.png'}")
    plt.close()
    
    # Plot median comparison
    fig, ax = plt.subplots(figsize=(max(12, len(all_approaches) * 2), 6))
    
    for i, (bench, label) in enumerate(zip(benchmarks, benchmark_labels)):
        medians = [approach_medians[app][i] for app in all_approaches]
        positions = x + (i - len(benchmarks)/2 + 0.5) * width
        
        # Filter out None values
        plot_positions = []
        plot_medians = []
        for pos, median in zip(positions, medians):
            if median is not None:
                plot_positions.append(pos)
                plot_medians.append(median)
        
        if plot_medians:
            ax.bar(plot_positions, plot_medians, width, label=label, alpha=0.8)
    
    ax.set_xlabel('Approach', fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title('Median Query Time Comparison Across Benchmarks', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_approaches, rotation=15, ha='right')
    ax.legend(title='Benchmark Run')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'median_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'median_comparison.png'}")
    plt.close()
    
    # Single benchmark detailed comparison (use most recent)
    if benchmarks:
        latest_bench = benchmarks[-1]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        approaches = []
        means = []
        medians = []
        
        for approach in all_approaches:
            times = extract_query_times(latest_bench, approach, metric)
            if times:
                approaches.append(approach)
                means.append(np.mean(times))
                medians.append(np.median(times))
        
        x_pos = np.arange(len(approaches))
        
        # Mean plot
        bars1 = ax1.bar(x_pos, means, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_xlabel('Approach', fontsize=12)
        ax1.set_ylabel(metric_label, fontsize=12)
        ax1.set_title('Mean Query Times (Latest Benchmark)', fontsize=13, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(approaches, rotation=15, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Median plot
        bars2 = ax2.bar(x_pos, medians, alpha=0.7, color='coral', edgecolor='black')
        ax2.set_xlabel('Approach', fontsize=12)
        ax2.set_ylabel(metric_label, fontsize=12)
        ax2.set_title('Median Query Times (Latest Benchmark)', fontsize=13, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(approaches, rotation=15, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'latest_benchmark_detailed.png', dpi=300, bbox_inches='tight')
        print(f"Saved: {output_dir / 'latest_benchmark_detailed.png'}")
        plt.close()


def print_statistics_table(benchmarks: List[Dict[str, Any]], metric: str = 'query_ms'):
    """Print a formatted table of statistics for each benchmark.
    
    Args:
        benchmarks: List of loaded benchmark data dicts
        metric: Which timing metric to use
    """
    print("\n" + "="*80)
    print("BENCHMARK STATISTICS")
    print("="*80)
    
    for bench in benchmarks:
        timestamp = bench.get('timestamp', 'unknown')
        source = bench.get('_source_file', 'unknown')
        print(f"\nBenchmark: {timestamp}")
        print(f"Source: {source}")
        print("-" * 80)
        
        results = bench.get('results', {})
        for approach in sorted(results.keys()):
            times = extract_query_times(bench, approach, metric)
            
            if times:
                mean_time = np.mean(times)
                median_time = np.median(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)
                count = len(times)
                
                print(f"\n  {approach}:")
                print(f"    Successful queries: {count}")
                print(f"    Mean:               {mean_time:.2f} ms")
                print(f"    Median:             {median_time:.2f} ms")
                print(f"    Std Dev:            {std_time:.2f} ms")
                print(f"    Min:                {min_time:.2f} ms")
                print(f"    Max:                {max_time:.2f} ms")
            else:
                print(f"\n  {approach}: No successful queries")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--results-dir', 
                        help='Directory containing benchmark result JSON files')
    parser.add_argument('--results', nargs='+',
                        help='Specific result files to compare (can use wildcards)')
    parser.add_argument('--output-dir', default='figures',
                        help='Output directory for figures (default: figures)')
    parser.add_argument('--metric', default='rayspace_total_ms',
                        choices=['query_ms', 'rayspace_total_ms'],
                        help='Timing metric to use (default: rayspace_total_ms for RaySpace approaches)')
    
    args = parser.parse_args()
    
    # Collect result files
    result_files = []
    
    if args.results:
        for pattern in args.results:
            result_files.extend(glob.glob(pattern))
    elif args.results_dir:
        results_dir = Path(args.results_dir)
        result_files = list(results_dir.glob('grid_benchmark_*.json'))
        result_files = [str(f) for f in result_files]
    else:
        # Default: look in ../results/
        results_dir = Path(__file__).parent.parent / 'results'
        result_files = list(results_dir.glob('grid_benchmark_*.json'))
        result_files = [str(f) for f in result_files]
    
    if not result_files:
        print("Error: No result files found")
        print("Use --results-dir or --results to specify benchmark results")
        return 1
    
    # Sort by filename (timestamp)
    result_files.sort()
    
    print(f"Found {len(result_files)} benchmark result file(s)")
    
    # Load benchmarks
    benchmarks = load_benchmark_results(result_files)
    
    if not benchmarks:
        print("Error: No valid benchmark data loaded")
        return 1
    
    print(f"\nSuccessfully loaded {len(benchmarks)} benchmark(s)")
    
    # Print statistics
    print_statistics_table(benchmarks, args.metric)
    
    # Generate comparison plots
    output_dir = Path(args.output_dir)
    print(f"\nGenerating comparison figures in {output_dir}/")
    plot_approach_comparison(benchmarks, output_dir, args.metric)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Figures saved to: {output_dir}/")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
