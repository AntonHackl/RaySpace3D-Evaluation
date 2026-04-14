#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess 
import json
from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json

# Add current directory to path to import adapters
sys.path.append(str(Path(__file__).parent))
from adapters.raytracer_adapter import RaytracerAdapter
from adapters.cgal_adapter import CGALAdapter
from adapters.touch_adapter import TOUCHAdapter

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
CGAL_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/CGAL"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

TIMEOUT_SECONDS = 120.0

# Cube Counts for Dataset B (Dataset A is fixed at 200k)
CUBE_COUNTS = [200000, 400000, 600000, 1000000]
FIXED_COUNT = "200k_a"

def run_experiment(runs, grid_resolution, run_log_dir):
    print("--- Starting Cube Scalability Experiment ---")
    
    # Ensure directories exist
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Logging runs to: {run_log_dir}")

    # Initialize Adapters
    print("Initializing adapters...")
    exact_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), 
        mode="exact", 
        preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(TIMINGS_DIR),
        grid_resolution=grid_resolution,
        warmup_runs=1
    )
    
    estimated_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), 
        mode="estimated", 
        preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(TIMINGS_DIR),
        grid_resolution=grid_resolution,
        warmup_runs=1
    )

    cgal_adapter = CGALAdapter(
        str(CGAL_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR)
    )
    
    touch_adapter = TOUCHAdapter(
        str(CGAL_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR)
    )
    
    results = {
        "counts": [],
        "exact": {"mean": [], "std": [], "breakdown": []},
        "estimated": {"mean": [], "std": [], "breakdown": []},
        "cgal": {"mean": [], "std": []},
        "touch": {"mean": [], "std": []}
    }

    filename_a = f"cubes_{FIXED_COUNT}.obj"
    f1_path = RAW_DIR / filename_a
    
    if not f1_path.exists():
        print(f"Error: Dataset A ({f1_path}) not found!")
        return None

    for count in CUBE_COUNTS:
        filename_b = f"cubes_{count // 1000}k_b.obj"
        f2_path = RAW_DIR / filename_b
        
        if not f2_path.exists():
            print(f"Error: Dataset B ({f2_path}) not found! Skipping.")
            continue
        
        print(f"\nProcessing: {filename_a} vs {filename_b}")

        # Check/Run Preprocessing (Exact/Estimated share preprocessed files)
        print("Checking preprocessing...")
        # Force preprocessing if not exists or ensure it's up to date
        # Note: RaytracerAdapter.preprocess_from_source checks existence inside
        exact_adapter.preprocess_from_source(str(f1_path), str(f1_path), log_dir=str(run_log_dir))
        exact_adapter.preprocess_from_source(str(f2_path), str(f2_path), log_dir=str(run_log_dir))

        # Run Exact Benchmark
        print(f"Running Exact Mode ({runs} runs)...")
        res_exact = exact_adapter.run_overlap(
            str(f1_path), 
            str(f2_path), 
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS
        )
        if "error" in res_exact:
            print(f"Error in exact run: {res_exact['error']}")
            continue # Assuming if exact fails, we skip this point
            
        # Run Estimated Benchmark
        print(f"Running Estimated Mode ({runs} runs)...")
        res_est = estimated_adapter.run_overlap(
            str(f1_path), 
            str(f2_path), 
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS
        )
        if "error" in res_est:
            print(f"Error in estimated run: {res_est['error']}")
            # We continue even if estimated fails? Let's say yes for robustness
            res_est = {"mean": 0, "std": 0, "breakdown": {}}

        # Run CGAL Benchmark
        print(f"Running CGAL Mode ({runs} runs)...")
        res_cgal = cgal_adapter.run_overlap(
            str(f1_path), 
            str(f2_path), 
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS
        )
        if "error" in res_cgal:
            print(f"Error in CGAL run: {res_cgal['error']}")
            # Allow CGAL to fail (e.g. timeout)
            res_cgal = {"mean": None, "std": None}

        # Run TOUCH Benchmark
        print(f"Running TOUCH Mode ({runs} runs)...")
        res_touch = touch_adapter.run_overlap(
            str(f1_path), 
            str(f2_path), 
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS
        )
        if "error" in res_touch:
            print(f"Error in TOUCH run: {res_touch['error']}")
            res_touch = {"mean": None, "std": None}

        results["counts"].append(count)
        
        results["exact"]["mean"].append(res_exact["mean"])
        results["exact"]["std"].append(res_exact["std"])
        results["exact"]["breakdown"].append(res_exact.get("breakdown", {}))
        
        results["estimated"]["mean"].append(res_est["mean"])
        results["estimated"]["std"].append(res_est["std"])
        results["estimated"]["breakdown"].append(res_est.get("breakdown", {}))
        
        results["cgal"]["mean"].append(res_cgal["mean"])
        results["cgal"]["std"].append(res_cgal["std"])
        
        results["touch"]["mean"].append(res_touch["mean"])
        results["touch"]["std"].append(res_touch["std"])
        
        # Add dataset sizes
        if "num_obj1" not in results:
             results["num_obj1"] = []
             results["num_obj2"] = []
             results["size_bytes1"] = []
             results["size_bytes2"] = []
             results["universe_extents1"] = []
             results["universe_extents2"] = []
        
        results["num_obj1"].append(int(res_exact.get("num_obj1", 0)))
        results["num_obj2"].append(int(res_exact.get("num_obj2", 0)))
        results["size_bytes1"].append(f1_path.stat().st_size if f1_path.exists() else 0)
        results["size_bytes2"].append(f2_path.stat().st_size if f2_path.exists() else 0)
        results["universe_extents1"].append(res_exact.get("universe_extents1", [0.0, 0.0, 0.0]))
        results["universe_extents2"].append(res_exact.get("universe_extents2", [0.0, 0.0, 0.0]))
        
        cgal_str = f"{res_cgal['mean']:.2f}ms" if res_cgal['mean'] else "TIMEOUT/ERR"
        touch_str = f"{res_touch['mean']:.2f}ms" if res_touch['mean'] else "TIMEOUT/ERR"
        print(f"Done {count}: Exact={res_exact['mean']:.2f}ms, Est={res_est['mean']:.2f}ms, CGAL={cgal_str}, TOUCH={touch_str}")

    return results

def plot_results(results, figures_dir):
    print("\nPlotting results...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    counts = results["counts"]
    if not counts:
        print("No results to plot.")
        return

    fig, (ax_main, ax_breakdown) = plt.subplots(1, 2, figsize=(18, 8))

    # --- Plot 1: Line Chart (Scaling) ---
    ax_main.errorbar(counts, results["exact"]["mean"], yerr=results["exact"]["std"], 
                     fmt='-o', label='Exact Raytracer', capsize=5, color='#1f77b4')
    ax_main.errorbar(counts, results["estimated"]["mean"], yerr=results["estimated"]["std"], 
                     fmt='--s', label='Estimated Raytracer', capsize=5, color='#2ca02c')
    
    # Filter valid CGAL points
    cgal_valid_indices = [i for i, m in enumerate(results["cgal"]["mean"]) if m is not None]
    if cgal_valid_indices:
        cgal_counts = [counts[i] for i in cgal_valid_indices]
        cgal_means = [results["cgal"]["mean"][i] for i in cgal_valid_indices]
        cgal_stds = [results["cgal"]["std"][i] for i in cgal_valid_indices]
        ax_main.errorbar(cgal_counts, cgal_means, yerr=cgal_stds, 
                         fmt=':d', label='CGAL', capsize=5, color='#9467bd')

    # Filter valid TOUCH points
    touch_valid_indices = [i for i, m in enumerate(results["touch"]["mean"]) if m is not None]
    if touch_valid_indices:
        touch_counts = [counts[i] for i in touch_valid_indices]
        touch_means = [results["touch"]["mean"][i] for i in touch_valid_indices]
        touch_stds = [results["touch"]["std"][i] for i in touch_valid_indices]
        ax_main.errorbar(touch_counts, touch_means, yerr=touch_stds, 
                         fmt='-^', label='TOUCH', capsize=5, color='#8c564b')

    ax_main.set_xlabel('Number of Cubes in Dataset B (A=200k)', fontsize=12)
    ax_main.set_ylabel('Execution Time (ms) [Log Scale]', fontsize=12)
    ax_main.set_title('Scalability: Mesh Overlap Query Time', fontsize=14, fontweight='bold')
    ax_main.set_yscale('log')
    ax_main.legend(fontsize=12)
    ax_main.grid(True, which="both", ls="-", alpha=0.2)
    ax_main.set_xticks(counts)

    # --- Plot 2: Breakdown Bar Chart (Exact & Estimated ONLY) ---
    # Breakdown visual settings
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
        "selectivity estimation_": "#ff9999", 
        "query_": "#66b3ff",              
        "execute hash query_": "#3399ff",   
        "gpu deduplication_": "#99ff99",    
        "download results_": "#ffcc99"      
    }
    
    # Collect active phases
    all_active_phases = set(ordered_phases_raw)
    for mode in ["exact", "estimated"]:
        for bd in results[mode]["breakdown"]:
            all_active_phases.update(bd.keys())
    
    active_phases_ordered = [p for p in ordered_phases_raw if p in all_active_phases]
    for p in all_active_phases:
        if p not in active_phases_ordered: active_phases_ordered.append(p)

    legend_handles = []
    legend_labels = []
    for phase in active_phases_ordered:
        label = phase_mapping.get(phase, phase)
        color = colors.get(phase, "#cccccc")
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, ec='white')
        legend_handles.append(patch)
        legend_labels.append(label)

    modes_to_plot = ["exact", "estimated"]
    num_modes = len(modes_to_plot)
    group_width = 0.8
    mode_width = group_width / num_modes
    
    # Use indices (0, 1, 2...) for x-axis of bar chart, label with counts
    x_indices = range(len(counts))

    for i, count_idx in enumerate(x_indices):
        for j, mode in enumerate(modes_to_plot):
            x_pos = i - group_width/2 + (j + 0.5) * mode_width
            
            # Get breakdown for this run
            breakdown = results[mode]["breakdown"][i]
            mean_time = results[mode]["mean"][i] # Fallback if no breakdown
            
            if not breakdown:
                ax_breakdown.bar(x_pos, mean_time, mode_width, color="#cccccc", edgecolor='white', alpha=0.5)
            else:
                bottom = 0
                for phase in active_phases_ordered:
                    val = breakdown.get(phase, 0.0)
                    if val > 0:
                        ax_breakdown.bar(x_pos, val, mode_width, bottom=bottom, 
                                         color=colors.get(phase, None), edgecolor='white')
                        bottom += val

    ax_breakdown.set_xticks(x_indices)
    ax_breakdown.set_xticklabels([f"{c//1000}k" for c in counts])
    ax_breakdown.set_xlabel('Dataset Size (Cubes)', fontsize=12)
    ax_breakdown.set_ylabel('Query Time (ms)', fontsize=12)
    ax_breakdown.set_title('Query Time Breakdown', fontsize=14, fontweight='bold')
    ax_breakdown.grid(True, axis='y', which='both', ls='-', alpha=0.1)
    
    # Legend
    ax_breakdown.legend(legend_handles, legend_labels, 
                       bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    output_path = figures_dir / "mesh_overlap_cube_scalability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Also save PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Mesh Overlap Cube Scalability Experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per method")
    parser.add_argument("--grid-resolution", type=int, default=20, help="Grid resolution for RaySpace")
    args = parser.parse_args()
    
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_cube_scalability")
    run_log_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])
    results = run_experiment(args.runs, args.grid_resolution, run_log_dir)
    
    if results and results["counts"]:
        print("\nResults Summary:")
        print(f"{'Count':<10} {'Exact (ms)':<15} {'Estimated (ms)':<15} {'CGAL (ms)':<15} {'TOUCH (ms)':<15}")
        for i, n in enumerate(results["counts"]):
            ex = results['exact']['mean'][i]
            est = results['estimated']['mean'][i]
            cg = results['cgal']['mean'][i]
            to = results['touch']['mean'][i]
            cg_str = f"{cg:.2f}" if cg else "N/A"
            to_str = f"{to:.2f}" if to else "N/A"
            print(f"{n:<10} {ex:<15.2f} {est:<15.2f} {cg_str:<15} {to_str:<15}")
                
        plot_results(results, figures_dir)
        
        # Save summary to canonical run results path
        out_json = Path(run_layout["results_json"])
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean_results[k] = {ki: (vi.tolist() if isinstance(vi, np.ndarray) else vi) for ki, vi in v.items()}
            elif isinstance(v, list):
                clean_results[k] = v
            else:
                clean_results[k] = v
        write_json(
            out_json,
            {
                "metadata": {
                    "timestamp": run_layout["timestamp"],
                    "run_name": run_layout["run_name"],
                    "run_dir": str(run_layout["run_dir"]),
                    "runs": args.runs,
                    "grid_resolution": args.grid_resolution,
                },
                "results": clean_results,
            },
        )
        print(f"Raw results saved to {out_json}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
