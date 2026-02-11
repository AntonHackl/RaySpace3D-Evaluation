#!/usr/bin/env python3
"""
Visualization script for mesh overlap benchmark results.
Generates bar charts showing mean, min, max, and standard deviation for each adapter.
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_file):
    """Load benchmark results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def visualize_results(json_file, output_dir=None):
    """Generate visualization from benchmark results."""
    data = load_results(json_file)
    
    # Extract metadata
    metadata = data.get("metadata", {})
    dataset = metadata.get("dataset", "unknown")
    num_runs = metadata.get("num_runs", 0)
    timestamp = metadata.get("timestamp", "")
    
    # Extract results
    results = data.get("results", {})
    
    # Filter out failed adapters
    valid_results = {name: res for name, res in results.items() if "error" not in res}
    
    if not valid_results:
        print("No valid results to visualize")
        return
    
    # Prepare data for plotting
    adapters = list(valid_results.keys())
    means = [valid_results[name]["mean"] for name in adapters]
    mins = [valid_results[name]["min"] for name in adapters]
    maxs = [valid_results[name]["max"] for name in adapters]
    stds = [valid_results[name]["std"] for name in adapters]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Bar chart with error bars (showing std dev)
    x_pos = np.arange(len(adapters))
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(adapters)])
    
    ax1.set_xlabel('Adapter', fontsize=12)
    ax1.set_ylabel('Query Time (ms) [Log Scale]', fontsize=12)
    ax1.set_title(f'Mean Query Time with Std Dev\n{dataset} ({num_runs} runs)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(adapters, rotation=15, ha='right')
    ax1.grid(axis='y', which='both', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        # Offset label slightly in log space
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                f'{mean:.2f} ms',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Min/Max range visualization
    width = 0.6
    for i, adapter in enumerate(adapters):
        min_val = mins[i]
        max_val = maxs[i]
        mean_val = means[i]
        
        # Draw range line
        ax2.plot([i, i], [min_val, max_val], 'k-', linewidth=2, alpha=0.5)
        # Draw min marker
        ax2.plot(i, min_val, 'v', markersize=8, color='blue', label='Min' if i == 0 else '')
        # Draw max marker
        ax2.plot(i, max_val, '^', markersize=8, color='red', label='Max' if i == 0 else '')
        # Draw mean marker
        ax2.plot(i, mean_val, 'o', markersize=10, color='green', label='Mean' if i == 0 else '')
    
    ax2.set_xlabel('Adapter', fontsize=12)
    ax2.set_ylabel('Query Time (ms) [Log Scale]', fontsize=12)
    ax2.set_title(f'Min/Max/Mean Comparison\n{dataset} ({num_runs} runs)', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(adapters)))
    ax2.set_xticklabels(adapters, rotation=15, ha='right')
    ax2.grid(axis='y', which='both', alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    # Determine output path
    if output_dir is None:
        json_path = Path(json_file)
        output_dir = json_path.parent.parent / "figures"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    output_name = f"{dataset}_{num_runs}runs_{timestamp}.png"
    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    
    # Also save as PDF for publications
    output_path_pdf = output_dir / f"{dataset}_{num_runs}runs_{timestamp}.pdf"
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"PDF saved to {output_path_pdf}")
    
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print(f"Benchmark Results Summary: {dataset}")
    print("="*60)
    for adapter in adapters:
        res = valid_results[adapter]
        print(f"\n{adapter}:")
        print(f"  Mean:   {res['mean']:.4f} ms")
        print(f"  Min:    {res['min']:.4f} ms")
        print(f"  Max:    {res['max']:.4f} ms")
        print(f"  Std:    {res['std']:.4f} ms")
        print(f"  CV:     {(res['std']/res['mean']*100):.2f}%")  # Coefficient of variation

def main():
    parser = argparse.ArgumentParser(description="Visualize mesh overlap benchmark results")
    parser.add_argument("json_file", type=str, help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", type=str, default=None, 
                        help="Output directory for figures (default: ../figures relative to JSON)")
    
    args = parser.parse_args()
    
    visualize_results(args.json_file, args.output_dir)

if __name__ == "__main__":
    main()
