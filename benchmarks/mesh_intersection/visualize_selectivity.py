#!/usr/bin/env python3
"""
Visualization script for mesh intersection selectivity benchmark results.
Generates:
1. Line chart (runtime vs selectivity) with log-log scale.
2. Bar chart showing runtime breakdown for each selectivity point.
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_file):
    """Load summary results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def visualize_selectivity(json_file, output_dir=None):
    """Generate visualizations from selectivity summary results."""
    data = load_results(json_file)
    
    # Filter out entries with errors
    valid_data = [d for d in data if "two_pass" in d and "error" not in d["two_pass"]]
    if not valid_data:
        print("No valid data to visualize")
        return

    selectivities = [d["selectivity"] for d in valid_data]
    
    # --- 1. Line Chart: Runtime vs Selectivity ---
    plt.figure(figsize=(10, 6))
    
    approaches = []
    if "two_pass" in valid_data[0]: approaches.append("two_pass")
    if "estimated" in valid_data[0]: approaches.append("estimated")
    
    colors = {"two_pass": "#1f77b4", "estimated": "#2ca02c"}
    labels = {"two_pass": "Exact Query (Two-Pass)", "estimated": "Estimated Query"}
    markers = {"two_pass": "o", "estimated": "s"}
    
    for app in approaches:
        runtimes = [d[app]["mean_ms"] for d in valid_data]
        plt.plot(selectivities, runtimes, marker=markers[app], linestyle='-', 
                 color=colors[app], label=labels[app], linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Selectivity (Fraction of Intersecting Pairs)', fontsize=12, fontweight='bold')
    plt.ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    plt.title('Mesh Intersection Performance: Runtime vs Selectivity\n(Log-Log Scale)', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Determine output path
    if output_dir is None:
        output_dir = Path(json_file).parent / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    line_chart_path = output_dir / "selectivity_runtime_line.png"
    plt.savefig(line_chart_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(line_chart_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Line chart saved to {line_chart_path}")
    plt.close()
    
    # --- 2. Runtime Breakdown Bar Chart ---
    # We'll create a stacked bar chart for each selectivity point for both approaches
    
    # Get all unique breakdown keys across all data
    breakdown_keys = set()
    for d in valid_data:
        for app in approaches:
            if app in d and "breakdown" in d[app]:
                breakdown_keys.update(d[app]["breakdown"].keys())
    
    breakdown_keys = sorted(list(breakdown_keys))
    
    # Colors for breakdown components
    # Map common keys to consistent colors
    color_map = {
        "query": "#1f77b4",
        "download results": "#ff7f0e",
        "selectivity estimation": "#2ca02c",
        "gpu deduplication": "#d62728"
    }
    # Add generic colors for any other keys
    standard_colors = plt.cm.tab10.colors
    idx = 0
    for key in breakdown_keys:
        if key not in color_map:
            color_map[key] = standard_colors[idx % len(standard_colors)]
            idx += 1

    # Create one plot for each approach or a grouped plot?
    # Let's do a side-by-side grouped stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 7))
    
    num_sel = len(valid_data)
    width = 0.35  # width of each bar
    x = np.arange(num_sel)
    
    for i, app in enumerate(approaches):
        offset = (i - len(approaches)/2 + 0.5) * width
        bottom = np.zeros(num_sel)
        
        for key in breakdown_keys:
            vals = []
            for d in valid_data:
                vals.append(d[app]["breakdown"].get(key, 0.0))
            
            ax.bar(x + offset, vals, width, bottom=bottom, label=f"{key} ({app})" if i==0 else None, 
                   color=color_map[key], alpha=0.8 if i==0 else 0.5)
            bottom += np.array(vals)

    ax.set_ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Mesh Intersection Runtime Breakdown', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"Sel: {s}" for s in selectivities], rotation=45)
    
    # Custom legend to avoid duplication
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color='black', alpha=0.8, lw=4, label='Exact (Solid)'),
                       Line2D([0], [0], color='black', alpha=0.5, lw=4, label='Estimated (Faded)')]
    for key in breakdown_keys:
        legend_elements.append(plt.Rectangle((0,0),1,1, color=color_map[key], label=key))
    
    ax.legend(handles=legend_elements, loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    breakdown_chart_path = output_dir / "selectivity_breakdown_bars.png"
    plt.savefig(breakdown_chart_path, dpi=300, bbox_inches='tight')
    print(f"Breakdown chart saved to {breakdown_chart_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize mesh intersection selectivity results")
    parser.add_argument("json_file", type=str, help="Path to summary.json file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures")
    
    args = parser.parse_args()
    visualize_selectivity(args.json_file, args.output_dir)

if __name__ == "__main__":
    main()
