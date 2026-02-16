
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
    tdbase_means = []
    tdbase_stds = []
    cgal_means = []
    cgal_stds = []

    # Filter data
    valid_selectivities = []
    for d in data:
        # We require at least one successful run to plot something for this selectivity point
        # But to keep indices aligned, let's just skip if basic ray tracing failed
        if "error" in d.get("exact", {}) or "error" in d.get("estimated", {}):
            continue
        valid_selectivities.append(d["selectivity"])
        exact_means.append(d["exact"]["mean_ms"])
        exact_stds.append(d["exact"]["std_ms"])
        est_means.append(d["estimated"]["mean_ms"])
        est_stds.append(d["estimated"]["std_ms"])
        
        if "tdbase" in d and "error" not in d["tdbase"]:
            tdbase_means.append(d["tdbase"]["mean_ms"])
            tdbase_stds.append(d["tdbase"]["std_ms"])
        else:
            tdbase_means.append(None)
            tdbase_stds.append(None)
            
        if "cgal" in d and "error" not in d["cgal"]:
            cgal_means.append(d["cgal"]["mean_ms"])
            cgal_stds.append(d["cgal"]["std_ms"])
        else:
            cgal_means.append(None)
            cgal_stds.append(None)

    if not valid_selectivities:
        print("No valid data points found.")
        return

    # Plot
    fig, (ax_main, ax_breakdown) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Get num_cubes if available
    num_cubes_str = ""
    if data and "num_cubes" in data[0]:
        num_cubes = data[0]["num_cubes"]
        num_cubes_str = f" ({num_cubes} cubes)"

    # --- Plot 1: Scaling Line Plot ---
    ax_main.errorbar(valid_selectivities, exact_means, yerr=exact_stds, label='Exact Raytracer', 
                marker='o', capsize=5, linestyle='-', color='#1f77b4')
    ax_main.errorbar(valid_selectivities, est_means, yerr=est_stds, label='Estimated Raytracer', 
                marker='s', capsize=5, linestyle='--', color='#2ca02c')

    if any(x is not None for x in tdbase_means):
        td_sel = [s for s, m in zip(valid_selectivities, tdbase_means) if m is not None]
        td_means = [m for m in tdbase_means if m is not None]
        td_stds = [s for s, m in zip(tdbase_stds, tdbase_means) if m is not None]
        
        ax_main.errorbar(td_sel, td_means, yerr=td_stds, label='TDBase',
                    marker='^', capsize=5, linestyle='-.', color='#d62728')

    if any(x is not None for x in cgal_means):
        cgal_sel = [s for s, m in zip(valid_selectivities, cgal_means) if m is not None]
        cm = [m for m in cgal_means if m is not None]
        cs = [s for s, m in zip(cgal_stds, cgal_means) if m is not None]
        
        ax_main.errorbar(cgal_sel, cm, yerr=cs, label='CGAL',
                    marker='d', capsize=5, linestyle=':', color='#9467bd')

    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_xlabel('Selectivity (Log Scale)', fontsize=12)
    ax_main.set_ylabel('Query Time (ms) [Log Scale]', fontsize=12)
    ax_main.set_title(f'Mesh Overlap Join Performance vs. Selectivity{num_cubes_str}', fontsize=14, fontweight='bold')
    ax_main.grid(True, which="both", ls="-", alpha=0.2)
    ax_main.legend(fontsize=12)

    # Annotate improvement factor
    for sl, ex, est in zip(valid_selectivities, exact_means, est_means):
        speedup = ex / est
        ax_main.annotate(f"{speedup:.1f}x", 
                    xy=(sl, est), 
                    xytext=(0, -15), textcoords="offset points",
                    ha='center', fontsize=9, color='#2ca02c')

    # --- Plot 2: Breakdown Chart ---
    # Prepare data for breakdown
    modes_in_data = ["exact", "estimated", "tdbase"]
    phase_mapping = {
        "selectivity estimation_": "Selectivity Est.",
        "execute hash query_": "Hash Query",
        "query_": "Ray Query",
        "gpu deduplication_": "Deduplication",
        "download results_": "Download"
    }
    ordered_phases_raw = [
        "selectivity estimation_",
        "query_",
        "execute hash query_",
        "gpu deduplication_",
        "download results_"
    ]
    colors = {
        "selectivity estimation_": "#ff9999", # Red-ish
        "query_": "#66b3ff",              # Blue-ish
        "execute hash query_": "#3399ff",   # Darker Blue
        "gpu deduplication_": "#99ff99",    # Green-ish
        "download results_": "#ffcc99"      # Orange-ish
    }

    # Find all active phases in any mode/selectivity
    all_active_phases = set(ordered_phases_raw) # Always include the ones we know
    for d in data:
        if d["selectivity"] not in valid_selectivities: continue
        for mode in modes_in_data:
            if mode in d and "breakdown" in d[mode]:
                all_active_phases.update(d[mode]["breakdown"].keys())
    
    active_phases_ordered = [p for p in ordered_phases_raw if p in all_active_phases]
    # Add any remaining phases
    for p in all_active_phases:
        if p not in active_phases_ordered:
            active_phases_ordered.append(p)

    # X positions for bars: groups by selectivity
    num_selectivities = len(valid_selectivities)
    active_modes = [m for m in modes_in_data if any(m in d and "error" not in d[m] for d in data if d["selectivity"] in valid_selectivities)]
    num_modes = len(active_modes)
    
    legend_handles = []
    legend_labels = []
    for phase in active_phases_ordered:
        label = phase_mapping.get(phase, phase)
        color = colors.get(phase, "#cccccc")
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, ec='white')
        legend_handles.append(patch)
        legend_labels.append(label)

    group_width = 0.8
    mode_width = group_width / num_modes
    
    for i, sel in enumerate(valid_selectivities):
        # find the record for this selectivity
        d = next(item for item in data if item["selectivity"] == sel)
        
        for j, mode in enumerate(active_modes):
            if mode not in d or "error" in d[mode]:
                continue
                
            x_pos = i - group_width/2 + (j + 0.5) * mode_width
            
            breakdown = d[mode].get("breakdown", {})
            if not breakdown and "mean_ms" in d[mode]:
                ax_breakdown.bar(x_pos, d[mode]["mean_ms"], mode_width, color="#cccccc", edgecolor='white', alpha=0.5)
            else:
                bottom = 0
                for phase in active_phases_ordered:
                    val = breakdown.get(phase, 0.0)
                    if val > 0:
                        ax_breakdown.bar(x_pos, val, mode_width, bottom=bottom, 
                                         color=colors.get(phase, None), edgecolor='white')
                        bottom += val

    ax_breakdown.set_xticks(range(num_selectivities))
    ax_breakdown.set_xticklabels([f"{s}" for s in valid_selectivities])
    ax_breakdown.set_xlabel('Selectivity', fontsize=12)
    ax_breakdown.set_ylabel('Query Time (ms)', fontsize=12)
    ax_breakdown.set_title('Query Time Breakdown', fontsize=14, fontweight='bold')
    ax_breakdown.grid(True, axis='y', which='both', ls='-', alpha=0.1)

    # Add legend
    ax_breakdown.legend(legend_handles, legend_labels, 
                       bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Ensure output directory exists
    if output_path is None:
        summary_path = Path(summary_file)
        # Default: figures directory in mesh_overlap_benchmark
        output_dir = summary_path.parent.parent.parent / "figures"
        output_dir.mkdir(parents=True, exist_ok=True)
        img_name = f"selectivity_scaling_{int(valid_selectivities[0]*10000)}to{int(valid_selectivities[-1]*100)}_with_breakdown.png"
        output_path = output_dir / img_name

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Also save PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
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
