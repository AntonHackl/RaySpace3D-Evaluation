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

TIMEOUT_SECONDS = 3600.0  # Allow longer timeout for dense meshes

def count_vertices(obj_path):
    count = 0
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                count += 1
    return count

def generate_datasets(stage_idx, num_objects, selectivity, template_path, file_a, file_b):
    print(f"Generating datasets for Stage {stage_idx} ({num_objects} objects, sel={selectivity})...")
    gen_script = RAYSPACE_DIR / "scripts/cpp_generator/generate_spheres"
    
    cmd = [
        str(gen_script),
        "--template-obj", str(template_path),
        "--num-objs-a", str(num_objects),
        "--num-objs-b", str(num_objects),
        "--min-size", "1.0",
        "--max-size", "5.0",
        "--selectivity", str(selectivity),
        "-oa", str(file_a),
        "-ob", str(file_b),
        "--seed", "42"
    ]
    subprocess.run(cmd, check=True)

def run_experiment(runs, grid_resolution, num_objects, selectivity):
    print("--- Starting Mesh Complexity Experiment ---")
    
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mesh_complexity_{runs}runs_{timestamp}"
    run_log_dir = RUNS_DIR / "logs" / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
    
    exact_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), mode="exact", preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(TIMINGS_DIR), grid_resolution=grid_resolution, warmup_runs=1
    )
    estimated_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), mode="estimated", preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(TIMINGS_DIR), grid_resolution=grid_resolution, warmup_runs=1
    )
    cgal_adapter = CGALAdapter(str(CGAL_DIR), preprocessed_dir=str(PREPROCESSED_DIR))
    touch_adapter = TOUCHAdapter(str(CGAL_DIR), preprocessed_dir=str(PREPROCESSED_DIR))
    
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
        template_path = SINGLE_OBJ_DIR / template_name
        
        if not template_path.exists():
            print(f"Warning: {template_path} not found. Skipping Stage {stage}.")
            continue
            
        vertices_count = count_vertices(template_path)
        print(f"\n--- Processing Stage {stage} (Vertices per mesh: {vertices_count}) ---")
        
        obj_str = f"{num_objects//1000}k" if num_objects >= 1000 else str(num_objects)
        file_a = RAW_DIR / f"sphere_stage_{stage}_{obj_str}_a.obj"
        file_b = RAW_DIR / f"sphere_stage_{stage}_{obj_str}_b.obj"
        
        # Generator
        if not file_a.exists() or not file_b.exists():
            generate_datasets(stage, num_objects, selectivity, template_path, file_a, file_b)
            
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

        # Delete datasets again as requested
        if file_a.exists():
            file_a.unlink()
        if file_b.exists():
            file_b.unlink()

    return results

def plot_results(results, num_objects, selectivity):
    print("\nPlotting results...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
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
    output_path = FIGURES_DIR / "mesh_complexity_scalability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Visualization saved to {output_path} and .pdf")

def main():
    parser = argparse.ArgumentParser(description="Mesh Complexity Benchmark")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per method")
    parser.add_argument("--grid-resolution", type=int, default=20, help="Grid resolution for RaySpace")
    parser.add_argument("--num-objects", type=int, default=50000, help="Number of objects per dataset")
    parser.add_argument("--selectivity", type=float, default=0.0005, help="Fixed selectivity target")
    args = parser.parse_args()
    
    results = run_experiment(args.runs, args.grid_resolution, args.num_objects, args.selectivity)
    
    if results and results["complexities"]:
        plot_results(results, args.num_objects, args.selectivity)
        
        out_json = RUNS_DIR / f"complexity_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Raw results saved to {out_json}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
