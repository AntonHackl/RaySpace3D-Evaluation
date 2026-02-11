
"""
Analysis script for selectivity test results.
Reads the summary.json produced by selectivity_test.py and generates visualization.
"""
import io
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def visualize_selectivity(summary_file, output_path=None):
    with open(summary_file, 'r') as f:
        data = json.load(f)

    # Sort checks if json is not sorted
    data.sort(key=lambda x: x["selectivity"])

    selectivities = [d["selectivity"] for d in data]
    exact_means = []
    exact_stds = []
    est_means = []
    est_stds = []

    # Filter data
    valid_selectivities = []
    for d in data:
        if "error" in d.get("exact", {}) or "error" in d.get("estimated", {}):
            continue
        valid_selectivities.append(d["selectivity"])
        exact_means.append(d["exact"]["mean_ms"])
        exact_stds.append(d["exact"]["std_ms"])
        est_means.append(d["estimated"]["mean_ms"])
        est_stds.append(d["estimated"]["std_ms"])

    if not valid_selectivities:
        print("No valid data points found.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get num_cubes if available
    num_cubes_str = ""
    if data and "num_cubes" in data[0]:
        num_cubes = data[0]["num_cubes"]
        num_cubes_str = f" ({num_cubes} cubes)"

    # X-axis: Selectivity (log scale possibly better if range is large, but 0.0001 to 0.01 is 2 orders)
    # Let's use log scale for X if evenly spaced in log, but user provided [0.0001, 0.0005, 0.001, 0.005, 0.01]
    # These are somewhat log-spaced.
    
    # Plot lines
    ax.errorbar(valid_selectivities, exact_means, yerr=exact_stds, label='Exact Raytracer', 
                marker='o', capsize=5, linestyle='-', color='#1f77b4')
    ax.errorbar(valid_selectivities, est_means, yerr=est_stds, label='Estimated Raytracer', 
                marker='s', capsize=5, linestyle='--', color='#2ca02c')

    ax.set_xscale('log')
    ax.set_yscale('log')
    
    ax.set_xlabel('Selectivity (Log Scale)', fontsize=12)
    ax.set_ylabel('Query Time (ms) [Log Scale]', fontsize=12)
    ax.set_title(f'Mesh Overlap Join Performance vs. Selectivity{num_cubes_str}', fontsize=14, fontweight='bold')
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(fontsize=12)

    # Annotate improvement factor
    for sl, ex, est in zip(valid_selectivities, exact_means, est_means):
        speedup = ex / est
        ax.annotate(f"{speedup:.1f}x", 
                    xy=(sl, est), 
                    xytext=(0, -15), textcoords="offset points",
                    ha='center', fontsize=9, color='#2ca02c')

    # Ensure output directory exists
    if output_path is None:
        summary_path = Path(summary_file)
        # Default: figures directory in mesh_overlap_benchmark
        output_dir = summary_path.parent.parent.parent / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        img_name = f"selectivity_scaling_{int(valid_selectivities[0]*10000)}to{int(valid_selectivities[-1]*100)}.png"
        output_path = output_dir / img_name

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")
    
    # Also save PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path)
    print(f"PDF saved to {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize Selectivity Test Results")
    parser.add_argument("summary_file", nargs='?', default="results/selectivity_test/summary.json",
                        help="Path to summary.json")
    parser.add_argument("--output", help="Path to output image")
    
    args = parser.parse_args()
    
    # Resolve path relative to script location if default
    script_dir = Path(__file__).parent
    input_file = Path(args.summary_file)
    if not input_file.is_absolute():
        input_file = script_dir / input_file
        
    if not input_file.exists():
        print(f"Error: Summary file {input_file} not found.")
        return

    visualize_selectivity(input_file, args.output)

if __name__ == "__main__":
    main()
