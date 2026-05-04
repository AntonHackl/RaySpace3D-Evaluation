
"""
Analysis script for selectivity test results.
Reads the summary.json produced by selectivity_test.py and generates visualization.
"""
import io
import json
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def phase_label(phase_key: str) -> str:
    custom = {
        "selectivity estimation": "Selectivity Estimation",
        "execute hash query": "Hash Query (Total)",
        "query": "Ray Query (Total)",
        "gpu deduplication": "Deduplication",
        "download results": "Download Results",
        "raytrace_hash_mesh1tomesh2": "Raytrace Hash Mesh1→Mesh2",
        "raytrace_hash_mesh2tomesh1": "Raytrace Hash Mesh2→Mesh1",
        "raytrace_mesh1tomesh2_pass1": "Raytrace Mesh1→Mesh2 Pass1",
        "raytrace_mesh2tomesh1_pass1": "Raytrace Mesh2→Mesh1 Pass1",
        "raytrace_mesh1tomesh2_pass2": "Raytrace Mesh1→Mesh2 Pass2",
        "raytrace_mesh2tomesh1_pass2": "Raytrace Mesh2→Mesh1 Pass2",
        "raytrace_overlap_mesh1tomesh2_pass1": "Raytrace Overlap Mesh1→Mesh2 Pass1",
        "raytrace_overlap_mesh2tomesh1_pass1": "Raytrace Overlap Mesh2→Mesh1 Pass1",
        "raytrace_overlap_mesh1tomesh2_pass2": "Raytrace Overlap Mesh1→Mesh2 Pass2",
        "raytrace_overlap_mesh2tomesh1_pass2": "Raytrace Overlap Mesh2→Mesh1 Pass2",
        "raytrace_containment_mesh1tomesh2_pass1": "Raytrace Containment Mesh1→Mesh2 Pass1",
        "raytrace_containment_mesh2tomesh1_pass1": "Raytrace Containment Mesh2→Mesh1 Pass1",
        "raytrace_containment_mesh1tomesh2_pass2": "Raytrace Containment Mesh1→Mesh2 Pass2",
        "raytrace_containment_mesh2tomesh1_pass2": "Raytrace Containment Mesh2→Mesh1 Pass2",
    }
    if phase_key in custom:
        return custom[phase_key]
    return re.sub(r"\s+", " ", phase_key.replace("_", " ")).strip().title()

def visualize_selectivity(summary_file, output_path=None):
    with open(summary_file, 'r') as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        data = data["results"]

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
    touch_means = []
    touch_stds = []
    direct_est_means = []
    direct_est_stds = []
    mem10_means = []
    mem10_stds = []

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
        
        if "estimated" in d and "error" not in d["estimated"]:
            est_means.append(d["estimated"]["mean_ms"])
            est_stds.append(d["estimated"]["std_ms"])
        else:
            est_means.append(None)
            est_stds.append(None)
        
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

        if "touch" in d and "error" not in d["touch"]:
            touch_means.append(d["touch"]["mean_ms"])
            touch_stds.append(d["touch"]["std_ms"])
        else:
            touch_means.append(None)
            touch_stds.append(None)

        if "direct_estimation" in d and "error" not in d["direct_estimation"]:
            direct_est_means.append(d["direct_estimation"]["mean_ms"])
            direct_est_stds.append(d["direct_estimation"]["std_ms"])
        else:
            direct_est_means.append(None)
            direct_est_stds.append(None)

        if "estimated_mem10" in d and "error" not in d["estimated_mem10"]:
            mem10_means.append(d["estimated_mem10"]["mean_ms"])
            mem10_stds.append(d["estimated_mem10"]["std_ms"])
        else:
            mem10_means.append(None)
            mem10_stds.append(None)

    if not valid_selectivities:
        print("No valid data points found.")
        return

    # Setup output directory
    summary_path = Path(summary_file)
    if "runs" in summary_path.parts:
        output_dir = summary_path.parent / "figures"
    else:
        output_dir = summary_path.parent.parent.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get num_cubes if available
    num_cubes_str = ""
    if data and "num_cubes" in data[0]:
        num_cubes = data[0]["num_cubes"]
        num_cubes_str = f" ({num_cubes} cubes)"

    # --- Plot 1: Scaling Line Plot ---
    fig_main, ax_main = plt.subplots(figsize=(10, 8))
    ax_main.errorbar(valid_selectivities, exact_means, yerr=exact_stds, label='Two Pass', 
                marker='o', capsize=5, linestyle='-', color='#1f77b4')
    
    if any(x is not None for x in est_means):
        ax_main.errorbar(valid_selectivities, est_means, yerr=est_stds, label='Estimated Raytracer', 
                    marker='s', capsize=5, linestyle='--', color='#2ca02c')
    
    if any(x is not None for x in direct_est_means):
        ax_main.errorbar(valid_selectivities, direct_est_means, yerr=direct_est_stds, label='Selectivity Estimation', 
                    marker='X', capsize=5, linestyle=':', color='#ff7f0e')

    if any(x is not None for x in mem10_means):
        ax_main.errorbar(valid_selectivities, mem10_means, yerr=mem10_stds, label='10% remaining', 
                    marker='v', capsize=5, linestyle='--', color='#e377c2')

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

    if any(x is not None for x in touch_means):
        touch_sel = [s for s, m in zip(valid_selectivities, touch_means) if m is not None]
        tm = [m for m in touch_means if m is not None]
        ts = [s for s, m in zip(touch_stds, touch_means) if m is not None]
        
        ax_main.errorbar(touch_sel, tm, yerr=ts, label='TOUCH',
                    marker='p', capsize=5, linestyle='-', color='#8c564b')

    ax_main.set_xscale('log')
    ax_main.set_yscale('log')
    ax_main.set_xlabel('Selectivity (Log Scale)', fontsize=12)
    ax_main.set_ylabel('Query Time (ms) [Log Scale]', fontsize=12)
    ax_main.set_title(f'Mesh Overlap Join Performance vs. Selectivity{num_cubes_str}', fontsize=14, fontweight='bold')
    ax_main.grid(True, which="both", ls="-", alpha=0.2)
    ax_main.legend(fontsize=12)

    # Annotate improvement factor
    for sl, ex, est in zip(valid_selectivities, exact_means, est_means):
        if est is not None and ex is not None:
            speedup = ex / est
            ax_main.annotate(f"{speedup:.1f}x", 
                        xy=(sl, est), 
                        xytext=(0, -15), textcoords="offset points",
                        ha='center', fontsize=9, color='#2ca02c')

    scaling_output = output_dir / "selectivity_scaling.png"
    plt.tight_layout()
    plt.savefig(scaling_output, dpi=300, bbox_inches='tight')
    plt.savefig(str(scaling_output).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Scaling plot saved to {scaling_output}")
    plt.close(fig_main)

    # --- Plot 2: Breakdown Chart ---
    fig_breakdown, ax_breakdown = plt.subplots(figsize=(12, 8))
    # Prepare data for breakdown
    modes_in_data = ["exact", "estimated", "direct_estimation", "estimated_mem10", "tdbase", "cgal", "touch"]
    ordered_phases_raw = [
        "selectivity estimation",
        "query",
        "execute hash query",
        "raytrace_hash_mesh1tomesh2",
        "raytrace_hash_mesh2tomesh1",
        "raytrace_mesh1tomesh2_pass1",
        "raytrace_mesh2tomesh1_pass1",
        "raytrace_mesh1tomesh2_pass2",
        "raytrace_mesh2tomesh1_pass2",
        "raytrace_overlap_mesh1tomesh2_pass1",
        "raytrace_overlap_mesh2tomesh1_pass1",
        "raytrace_overlap_mesh1tomesh2_pass2",
        "raytrace_overlap_mesh2tomesh1_pass2",
        "raytrace_containment_mesh1tomesh2_pass1",
        "raytrace_containment_mesh2tomesh1_pass1",
        "raytrace_containment_mesh1tomesh2_pass2",
        "raytrace_containment_mesh2tomesh1_pass2",
        "gpu deduplication",
        "download results",
    ]
    
    mode_short_names = {
        "exact": "2P",
        "estimated": "Est",
        "direct_estimation": "SE",
        "estimated_mem10": "10%",
        "tdbase": "TD",
        "cgal": "CG",
        "touch": "TO"
    }

    # Find all active phases in any mode/selectivity
    all_active_phases = set()
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
    
    palette = plt.get_cmap("tab20")
    colors = {phase: palette(i % 20) for i, phase in enumerate(active_phases_ordered)}

    legend_handles = []
    legend_labels = []
    for phase in active_phases_ordered:
        label = phase_label(phase)
        color = colors[phase]
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
                                         color=colors[phase], edgecolor='white')
                        bottom += val
            
            # Add short label for mode
            short_name = mode_short_names.get(mode, mode[:3])
            ax_breakdown.text(x_pos, -2, short_name, ha='center', va='top', fontsize=8, rotation=45)

    ax_breakdown.set_xticks(range(num_selectivities))
    ax_breakdown.set_xticklabels([f"{s}" for s in valid_selectivities])
    ax_breakdown.set_xlabel('Selectivity (Grouped by Approach: 2P, SE, 10%, ...)', fontsize=12)
    ax_breakdown.set_ylabel('Query Time (ms)', fontsize=12)
    ax_breakdown.set_title('Query Time Breakdown', fontsize=14, fontweight='bold')
    ax_breakdown.grid(True, axis='y', which='both', ls='-', alpha=0.1)

    # Add legend
    ax_breakdown.legend(legend_handles, legend_labels, 
                       bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    breakdown_output = output_dir / "selectivity_breakdown.png"
    plt.tight_layout()
    plt.savefig(breakdown_output, dpi=300, bbox_inches='tight')
    plt.savefig(str(breakdown_output).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Breakdown plot saved to {breakdown_output}")
    plt.close(fig_breakdown)

def main():
    parser = argparse.ArgumentParser(description="Visualize Selectivity Test Results")
    parser.add_argument("summary_file", nargs='?', default="results/selectivity_test/summary.json",
                        help="Path to summary.json")
    parser.add_argument("--output", help="Ignored (now saves to two separate files in the run directory)")
    
    args = parser.parse_args()
    
    # Resolve path relative to script location if default
    script_dir = Path(__file__).parent
    input_file = Path(args.summary_file)
    if not input_file.is_absolute():
        input_file = script_dir / input_file
        
    if not input_file.exists():
        print(f"Error: Summary file {input_file} not found.")
        return

    visualize_selectivity(input_file)

if __name__ == "__main__":
    main()
