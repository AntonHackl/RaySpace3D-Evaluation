#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess 
import json
from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.scenario_utils import (
    canonical_sphere_pair_paths,
    count_vertices,
    ensure_sphere_pair_dataset,
    get_shared_data_dirs,
)

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
SINGLE_OBJ_DIR = DATA_DIR / "single_obj_files"
SPHERE_TEMPLATE_DIR = REPO_ROOT / "benchmarks" / "mesh_overlap" / "data" / "single_obj_files"
SHARED_SCENARIO = "mesh_complexity"

TIMEOUT_SECONDS = 3600.0  # Allow longer timeout for dense meshes

def run_experiment(runs, grid_cell_size, num_objects, selectivity, run_log_dir):
    print("--- Starting Mesh Complexity Experiment ---")

    shared_dirs = get_shared_data_dirs(SHARED_SCENARIO)
    
    exact_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), mode="exact", preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(shared_dirs["timings"]), grid_cell_size=grid_cell_size, warmup_runs=1
    )
    estimated_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), mode="estimated", preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(shared_dirs["timings"]), grid_cell_size=grid_cell_size, warmup_runs=1
    )
    exact_adapter.preprocessed_dir = shared_dirs["preprocessed"]
    estimated_adapter.preprocessed_dir = shared_dirs["preprocessed"]
    cgal_adapter = CGALAdapter(str(CGAL_DIR), preprocessed_dir=str(shared_dirs["preprocessed"]))
    touch_adapter = TOUCHAdapter(str(CGAL_DIR), preprocessed_dir=str(shared_dirs["preprocessed"]))
    
    results = {
        "complexities": [],
        "stages": [],
        "exact": {"mean": [], "std": [], "breakdown": []},
        "estimated": {"mean": [], "std": [], "breakdown": []},
        "cgal": {"mean": [], "std": []},
        "touch": {"mean": [], "std": []}
    }

    # Iterate over exactly 10 stages
    for stage in range(1, 11):
        template_name = f"Sphere_Stage_{stage}.obj"
        template_path = SPHERE_TEMPLATE_DIR / template_name
        
        if not template_path.exists():
            print(f"Warning: {template_path} not found. Skipping Stage {stage}.")
            continue
            
        vertices_count = count_vertices(template_path)
        print(f"\n--- Processing Stage {stage} (Vertices per mesh: {vertices_count}) ---")
        
        file_a, file_b = canonical_sphere_pair_paths(
            shared_dirs["raw"],
            template_name=template_name,
            num_objects=num_objects,
            min_size=1.0,
            max_size=5.0,
            selectivity=selectivity,
            seed=42,
            grid_cell_size=grid_cell_size,
        )
        ensure_sphere_pair_dataset(
            file_a,
            file_b,
            template_obj=template_path,
            num_objects=num_objects,
            min_size=1.0,
            max_size=5.0,
            selectivity=selectivity,
            seed=42,
        )
            
        exact_adapter.preprocess_from_source(str(file_a), str(file_a), log_dir=str(run_log_dir))
        exact_adapter.preprocess_from_source(str(file_b), str(file_b), log_dir=str(run_log_dir))

        # Benchmarks
        print("Running Exact Mode...")
        res_exact = exact_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=TIMEOUT_SECONDS)
        
        print("Running Estimated Mode...")
        res_est = estimated_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=TIMEOUT_SECONDS)
        
        # Optional baseline bounds or skip if too slow
        print("Skipping CGAL Mode...")
        # res_cgal = cgal_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=TIMEOUT_SECONDS)
        res_cgal = {"error": "Skipped"}
        
        print("Skipping TOUCH Mode...")
        # res_touch = touch_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=TIMEOUT_SECONDS)
        res_touch = {"error": "Skipped"}

        # Handle errors gracefully
        for res in [res_exact, res_est, res_cgal, res_touch]:
            if "error" in res:
                res["mean"] = None
                res["std"] = None

        results["stages"].append(stage)
        results["complexities"].append(vertices_count)
        
        results["exact"]["mean"].append(res_exact.get("mean"))
        results["exact"]["std"].append(res_exact.get("std"))
        results["exact"]["breakdown"].append(res_exact.get("breakdown", {}))
        
        results["estimated"]["mean"].append(res_est.get("mean"))
        results["estimated"]["std"].append(res_est.get("std"))
        results["estimated"]["breakdown"].append(res_est.get("breakdown", {}))
        
        results["cgal"]["mean"].append(res_cgal.get("mean"))
        results["cgal"]["std"].append(res_cgal.get("std"))
        
        results["touch"]["mean"].append(res_touch.get("mean"))
        results["touch"]["std"].append(res_touch.get("std"))
        
        print(f"Stage {stage} done. Vertices={vertices_count}, Exact={res_exact.get('mean')}, Est={res_est.get('mean')}")

    return results

def plot_results(results, num_objects, selectivity, figures_dir):
    print("\nPlotting results...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    complexities = results["complexities"]
    if not complexities:
        return

    plt.figure(figsize=(10, 8))

    for key, color, label, fmt in [
        ('exact', '#1f77b4', 'Exact Raytracer', '-o'),
        ('estimated', '#2ca02c', 'Estimated Raytracer', '--s'),
        ('cgal', '#9467bd', 'CGAL', ':d'),
        ('touch', '#8c564b', 'TOUCH', '-^')
    ]:
        valid_indices = [i for i, m in enumerate(results[key]["mean"]) if m is not None]
        if valid_indices:
            x_vals = [complexities[i] for i in valid_indices]
            means = [results[key]["mean"][i] for i in valid_indices]
            stds = [results[key]["std"][i] for i in valid_indices]
            plt.errorbar(x_vals, means, yerr=stds, fmt=fmt, label=label, capsize=5, color=color)

    plt.xlabel('Mesh Complexity (Vertices per Mesh)', fontsize=12)
    plt.ylabel('Execution Time (ms) [Log Scale]', fontsize=12)
    plt.title(f'Mesh Complexity Benchmark\nAmount of Objects: {num_objects}, Selectivity: {selectivity}', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.xscale('log') # Use log scale for complexity as well to spread points evenly
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)

    plt.tight_layout()
    output_path = figures_dir / "mesh_complexity_scalability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path} and .pdf")

def main():
    parser = argparse.ArgumentParser(description="Mesh Complexity Benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per method")
    parser.add_argument("--grid-cell-size", type=float, default=1.0, help="Grid resolution for RaySpace")
    parser.add_argument("--num-objects", type=int, default=50000, help="Number of objects per dataset")
    parser.add_argument("--selectivity", type=float, default=0.0005, help="Fixed selectivity target")
    args = parser.parse_args()
    
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_mesh_complexity")
    run_log_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])
    results = run_experiment(args.runs, args.grid_cell_size, args.num_objects, args.selectivity, run_log_dir)
    
    if results and results["complexities"]:
        plot_results(results, args.num_objects, args.selectivity, figures_dir)

        out_json = Path(run_layout["results_json"])
        payload = {
            "metadata": {
                "timestamp": run_layout["timestamp"],
                "run_name": run_layout["run_name"],
                "run_dir": str(run_layout["run_dir"]),
                "runs": args.runs,
                "grid_cell_size": args.grid_cell_size,
                "num_objects": args.num_objects,
                "selectivity": args.selectivity,
            },
            "results": results,
        }
        write_json(out_json, payload)
        print(f"Raw results saved to {out_json}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
