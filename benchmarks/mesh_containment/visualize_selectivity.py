#!/usr/bin/env python3
"""
Visualization script for mesh containment selectivity benchmark results.
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
    """Load results from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def visualize_selectivity(json_file, output_dir=None):
    """Generate visualizations from selectivity summary results."""
    data_root = load_results(json_file)
    
    # Check if data_root is a list (summary.json) or dict (detailed run log)
    if isinstance(data_root, list):
        results = data_root
    else:
        results = data_root.get("results", [])
    
    # Filter out entries with errors
    valid_results = [r for r in results if not any("error" in r.get(app, {}) for app in ["raytracer", "cgal"])]
    if not valid_results:
        print("No valid results to visualize")
        return

    selectivities = [r["selectivity"] for r in valid_results]
    
    # --- 1. Line Chart: Runtime vs Selectivity ---
    plt.figure(figsize=(10, 6))
    
    approaches = []
    # Check what approaches we actually have in the data
    available_apps = ["raytracer", "cgal"]
    for app in available_apps:
        if any(app in r for r in valid_results):
            approaches.append(app)
    
    colors = {"raytracer": "#1f77b4", "cgal": "#ff7f0e"}
    labels = {"raytracer": "RaySpace3D (OptiX)", "cgal": "CGAL (AABB Tree)"}
    markers = {"raytracer": "o", "cgal": "s"}
    
    for app in approaches:
        runtimes = [r[app]["mean"] for r in valid_results if app in r]
        sels = [r["selectivity"] for r in valid_results if app in r]
        plt.plot(sels, runtimes, marker=markers[app], linestyle='-', 
                 color=colors[app], label=labels[app], linewidth=2, markersize=8)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Selectivity (Fraction of Overlapping Pairs)', fontsize=12, fontweight='bold')
    plt.ylabel('Runtime (ms)', fontsize=12, fontweight='bold')
    plt.title('Mesh Containment Performance: Runtime vs Selectivity\n(Log-Log Scale)', fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()
    
    # Determine output path
    if output_dir is None:
        # Save to benchmarks/mesh_containment/figures
        output_dir = Path(__file__).resolve().parent / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    line_chart_path = output_dir / "selectivity_runtime_line.png"
    plt.savefig(line_chart_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(line_chart_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Line chart saved to {line_chart_path}")
    plt.close()
    
    # --- 2. Runtime Breakdown Bar Chart (Raytracer only) ---
    if "raytracer" in approaches:
        plt.figure(figsize=(12, 7))
        
        # Get breakdown keys (nested in raytracer results)
        breakdown_keys = []
        # Find first result with breakdown
        for r in valid_results:
            if "raytracer" in r and "breakdown" in r["raytracer"]:
                breakdown_keys = sorted(r["raytracer"]["breakdown"].keys())
                break
        
        if breakdown_keys:
            n_sel = len(valid_results)
            ind = np.arange(n_sel)
            width = 0.6
            
            bottoms = np.zeros(n_sel)
            
            # Map common internal keys to cleaner labels
            label_map = {
                "query": "GPU Query Kernel",
                "warmup": "Warmup Runs",
                "upload mesh a": "Upload Mesh A",
                "upload mesh b": "Upload Mesh B",
                "build a index": "Index A",
                "build b index": "Index B",
                "prepare kernel parameters": "Kernel Params",
                "load mesh a": "Load A (IO)",
                "load mesh b": "Load B (IO)",
                "download results": "Download Results",
            }

            for key in breakdown_keys:
                vals = []
                for r in valid_results:
                    if "raytracer" in r and "breakdown" in r["raytracer"] and key in r["raytracer"]["breakdown"]:
                        # JSON might have mean inside component or just value
                        comp = r["raytracer"]["breakdown"][key]
                        if isinstance(comp, dict):
                            vals.append(comp.get("mean", 0))
                        else:
                            vals.append(comp)
                    else:
                        vals.append(0)
                        
                plt.bar(ind, vals, width, bottom=bottoms, label=label_map.get(key, key))
                bottoms += np.array(vals)
            
            plt.xticks(ind, [f"{s:.4f}" for s in selectivities], rotation=45)
            plt.xlabel('Selectivity', fontweight='bold')
            plt.ylabel('Runtime (ms)', fontweight='bold')
            plt.title('RaySpace3D Containment Breakdown by Selectivity', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            bar_chart_path = output_dir / "raytracer_breakdown.png"
            plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
            print(f"Breakdown chart saved to {bar_chart_path}")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize mesh containment selectivity results")
    parser.add_argument("--input", type=str, required=True, help="Input JSON results file")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save figures")
    args = parser.parse_args()
    
    visualize_selectivity(args.input, args.output_dir)

if __name__ == "__main__":
    main()
