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

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import PAPER_FIGSIZE, apply_paper_style, make_legend_bold, set_log_timing_axis_limits, style_for
from benchmarks.common.scenario_utils import (
    canonical_sphere_pair_paths,
    count_vertices,
    ensure_sphere_pair_dataset,
    get_shared_data_dirs,
)

# Add current directory to path to import adapters
sys.path.append(str(Path(__file__).parent))
from benchmarks.mesh_overlap.adapters.raytracer_adapter import RaytracerAdapter
from benchmarks.mesh_overlap.adapters.cgal_adapter import CGALAdapter
from benchmarks.mesh_overlap.adapters.touch_adapter import TOUCHAdapter
from benchmarks.mesh_overlap.adapters.tdbase_adapter import TDBaseAdapter

# Configuration
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
CGAL_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/CGAL"
TDBASE_DIR = REPO_ROOT.parent / "tdbase"
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

def run_experiment(runs, grid_cell_size, num_objects, selectivity, run_log_dir, approaches=None, timeout=3600.0):
    if approaches is None:
        approaches = ["exact", "direct_estimation", "cgal", "touch", "tdbase"]

    print("--- Starting Mesh Complexity Experiment ---")

    shared_dirs = get_shared_data_dirs(SHARED_SCENARIO)
    
    exact_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), mode="exact", preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(shared_dirs["timings"]), grid_cell_size=grid_cell_size, warmup_runs=1
    )
    direct_estimation_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), mode="direct_estimation", preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(shared_dirs["timings"]), grid_cell_size=grid_cell_size, warmup_runs=1
    )
    exact_adapter.preprocessed_dir = shared_dirs["preprocessed"]
    direct_estimation_adapter.preprocessed_dir = shared_dirs["preprocessed"]
    cgal_adapter = CGALAdapter(str(CGAL_DIR), preprocessed_dir=str(shared_dirs["preprocessed"]), grid_cell_size=grid_cell_size)
    touch_adapter = TOUCHAdapter(str(CGAL_DIR), preprocessed_dir=str(shared_dirs["preprocessed"]), grid_cell_size=grid_cell_size)
    tdbase_adapter = TDBaseAdapter(str(TDBASE_DIR), preprocessed_dir=str(shared_dirs["preprocessed"]))
    
    results = {
        "complexities": [],
        "stages": [],
        "exact": {"mean": [], "std": [], "breakdown": []},
        "direct_estimation": {"mean": [], "std": [], "breakdown": []},
        "cgal": {"mean": [], "std": []},
        "touch": {"mean": [], "std": []},
        "tdbase": {"mean": [], "std": []},
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
            
        # Check/Run Preprocessing for Raytracer (also used by Face/TOUCH)
        if any(a in approaches for a in ["exact", "direct_estimation", "cgal", "touch"]):
            print("Checking preprocessing (Raytracer)...")
            exact_adapter.preprocess_from_source(str(file_a), str(file_a), log_dir=str(run_log_dir))
            exact_adapter.preprocess_from_source(str(file_b), str(file_b), log_dir=str(run_log_dir))

        if "tdbase" in approaches:
            print("Checking preprocessing (TDBase)...")
            if not tdbase_adapter.check_preprocessed(str(file_a)):
                tdbase_adapter.preprocess_from_source(str(file_a), str(file_a), log_dir=str(run_log_dir))
            if not tdbase_adapter.check_preprocessed(str(file_b)):
                tdbase_adapter.preprocess_from_source(str(file_b), str(file_b), log_dir=str(run_log_dir))

        # Benchmarks
        res_exact = {"error": "Skipped"}
        if "exact" in approaches:
            print("Running Exact Mode...")
            res_exact = exact_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=timeout)
        
        res_direct = {"error": "Skipped"}
        if "direct_estimation" in approaches:
            print("Running Selectivity Estimation Mode...")
            res_direct = direct_estimation_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=TIMEOUT_SECONDS)
        
        res_cgal = {"error": "Skipped"}
        if "cgal" in approaches:
            print("Running Face Mode...")
            res_cgal = cgal_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=TIMEOUT_SECONDS)
        
        res_touch = {"error": "Skipped"}
        if "touch" in approaches:
            print("Running TOUCH Mode...")
            res_touch = touch_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=TIMEOUT_SECONDS)

        res_tdbase = {"error": "Skipped"}
        if "tdbase" in approaches:
            print("Running TDBase...")
            res_tdbase = tdbase_adapter.run_overlap(str(file_a), str(file_b), runs, log_dir=str(run_log_dir), timeout=TIMEOUT_SECONDS)

        # Handle errors gracefully
        for res in [res_exact, res_direct, res_cgal, res_touch, res_tdbase]:
            if "error" in res:
                res["mean"] = None
                res["std"] = None

        results["stages"].append(stage)
        results["complexities"].append(vertices_count)
        
        results["exact"]["mean"].append(res_exact.get("mean"))
        results["exact"]["std"].append(res_exact.get("std"))
        results["exact"]["breakdown"].append(res_exact.get("breakdown", {}))
        
        results["direct_estimation"]["mean"].append(res_direct.get("mean"))
        results["direct_estimation"]["std"].append(res_direct.get("std"))
        results["direct_estimation"]["breakdown"].append(res_direct.get("breakdown", {}))
        
        results["cgal"]["mean"].append(res_cgal.get("mean"))
        results["cgal"]["std"].append(res_cgal.get("std"))
        
        results["touch"]["mean"].append(res_touch.get("mean"))
        results["touch"]["std"].append(res_touch.get("std"))

        results["tdbase"]["mean"].append(res_tdbase.get("mean"))
        results["tdbase"]["std"].append(res_tdbase.get("std"))
        
        print(
            f"Stage {stage} done. Vertices={vertices_count}, Exact={res_exact.get('mean')}, "
            f"Selectivity Estimation={res_direct.get('mean')}, Face={res_cgal.get('mean')}, "
            f"TOUCH={res_touch.get('mean')}, TDBase={res_tdbase.get('mean')}"
        )

    return results

def plot_results(results, num_objects, selectivity, figures_dir):
    print("\nPlotting results...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    complexities = results["complexities"]
    if not complexities:
        return

    apply_paper_style()
    plt.figure(figsize=PAPER_FIGSIZE)
    all_y_vals = []

    for key in ["exact", "direct_estimation", "cgal", "touch", "tdbase"]:
        valid_indices = [i for i, m in enumerate(results[key]["mean"]) if m is not None]
        if valid_indices:
            x_vals = [complexities[i] for i in valid_indices]
            means = [results[key]["mean"][i] for i in valid_indices]
            stds = [results[key]["std"][i] for i in valid_indices]
            all_y_vals.extend(means)
            st = style_for(key)
            plt.plot(
                x_vals,
                means,
                f"-{st['marker']}",
                label=st["label"],
                color=st["color"],
            )

    plt.xlabel('Mesh Complexity (Vertices per Mesh)')
    plt.ylabel('Execution Time (ms) [Log Scale]')
    plt.yscale('log')
    set_log_timing_axis_limits(plt.gca(), all_y_vals)
    plt.xscale('log') # Use log scale for complexity as well to spread points evenly
    make_legend_bold(plt.gca())
    plt.grid(False)
    plt.tight_layout()
    output_path = figures_dir / "mesh_complexity_scalability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path} and .pdf")

def main():
    parser = argparse.ArgumentParser(description="Mesh Complexity Benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per method")
    parser.add_argument("--grid-cell-size", type=float, default=5.0, help="Grid resolution for RaySpace")
    parser.add_argument("--num-objects", type=int, default=50000, help="Number of objects per dataset")
    parser.add_argument("--selectivity", type=float, default=0.0005, help="Fixed selectivity target")
    parser.add_argument("--approaches", type=str, nargs="+", default=["exact", "direct_estimation", "cgal", "touch", "tdbase"], 
                        choices=["exact", "direct_estimation", "cgal", "touch", "tdbase"], help="Approaches to run")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout in seconds per run")
    parser.add_argument("--revisualize", type=str, help="Path to results.json to re-generate plots from")
    args = parser.parse_args()

    if args.revisualize:
        print(f"Re-visualizing results from {args.revisualize}...")
        with open(args.revisualize, "r") as f:
            data = json.load(f)
        results = data["results"]
        figures_dir = Path(args.revisualize).parent / "figures"
        plot_results(results, data.get("metadata", {}).get("num_objects"), data.get("metadata", {}).get("selectivity"), figures_dir)
        return

    
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_mesh_complexity")
    run_log_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])
    results = run_experiment(args.runs, args.grid_cell_size, args.num_objects, args.selectivity, run_log_dir, approaches=args.approaches, timeout=args.timeout)

    
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
                "approaches": args.approaches,
                "timeout": args.timeout,
            },

            "results": results,
        }
        write_json(out_json, payload)
        print(f"Raw results saved to {out_json}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
