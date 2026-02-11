#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
import subprocess 
from adapters.raytracer_adapter import RaytracerAdapter

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

# Cube Counts for Dataset B (Dataset A is fixed at 200k)
CUBE_COUNTS = [200000, 400000, 600000, 1000000]
FIXED_COUNT = "200k_a"

def run_experiment(runs, grid_resolution):
    print("--- Starting Cube Scalability Experiment ---")
    
    # Ensure directories exist
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"cube_scalability_{runs}runs_{timestamp}"
    run_log_dir = RUNS_DIR / "logs" / run_name
    run_log_dir.mkdir(parents=True, exist_ok=True)
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
    
    results = {
        "counts": [],
        "exact_times": [],
        "estimated_times": [],
        "exact_std": [],
        "estimated_std": []
    }

    filename_a = f"cubes_{FIXED_COUNT}.obj"
    f1_path = RAW_DIR / filename_a
    
    if not f1_path.exists():
        print(f"Error: Dataset A ({f1_path}) not found!")
        return

    for count in CUBE_COUNTS:
        filename_b = f"cubes_{count // 1000}k_b.obj"
        f2_path = RAW_DIR / filename_b
        
        if not f2_path.exists():
            print(f"Error: Dataset B ({f2_path}) not found! Skipping.")
            continue
        
        print(f"\nProcessing: {filename_a} vs {filename_b}")

        # Check/Run Preprocessing
        print("Checking preprocessing...")
        for f in [f1_path, f2_path]:
            # Use suffix .pre for checking, but adapter handles naming.
            # We want to make sure we don't keep reprocessing if not needed.
            # The adapter uses .with_suffix('.pre')
            if not exact_adapter.check_preprocessed(str(f)):
                print(f"Preprocessing {f.name}...")
                # Note: We name the output .pre file based on the original .obj name
                exact_adapter.preprocess_from_source(str(f), str(f), log_dir=str(run_log_dir))
            else:
                 print(f"Already preprocessed: {f.name}")

        # Run Exact Benchmark
        print(f"Running Exact Mode ({runs} runs)...")
        res_exact = exact_adapter.run_overlap(
            str(f1_path), 
            str(f2_path), 
            runs,
            log_dir=str(run_log_dir)
        )
        if "error" in res_exact:
            print(f"Error in exact run: {res_exact['error']}")
            continue
            
        # Run Estimated Benchmark
        print(f"Running Estimated Mode ({runs} runs)...")
        res_est = estimated_adapter.run_overlap(
            str(f1_path), 
            str(f2_path), 
            runs,
            log_dir=str(run_log_dir)
        )
        if "error" in res_est:
            print(f"Error in estimated run: {res_est['error']}")
            continue

        results["counts"].append(count)
        results["exact_times"].append(res_exact["mean"])
        results["exact_std"].append(res_exact["std"])
        results["estimated_times"].append(res_est["mean"])
        results["estimated_std"].append(res_est["std"])
        
        print(f"Done {count}: Exact={res_exact['mean']:.2f}ms, Est={res_est['mean']:.2f}ms")

    return results

def plot_results(results):
    print("\nPlotting results...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    counts = results["counts"]
    exact_times = results["exact_times"]
    estimated_times = results["estimated_times"]
    exact_std = results["exact_std"]
    estimated_std = results["estimated_std"]
    
    if not counts:
        print("No results to plot.")
        return

    plt.figure(figsize=(10, 6))
    
    plt.errorbar(counts, exact_times, yerr=exact_std, fmt='-o', label='Exact', capsize=5)
    plt.errorbar(counts, estimated_times, yerr=estimated_std, fmt='-o', label='Estimated', capsize=5)
    
    plt.xlabel('Number of Cubes in Dataset B (Dataset A fixed at 200k)')
    plt.ylabel('Execution Time (ms)')
    plt.title('Scalability: Mesh Overlap Query Time (Cubes)')
    plt.legend()
    plt.grid(True)
    
    # Set x-axis ticks
    plt.xticks(counts)
    
    output_path = FIGURES_DIR / "mesh_overlap_cube_scalability.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Mesh Overlap Cube Scalability Experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per method")
    parser.add_argument("--grid-resolution", type=int, default=10, help="Grid resolution for RaySpace")
    args = parser.parse_args()
    
    results = run_experiment(args.runs, args.grid_resolution)
    
    if results and results["counts"]:
        print("\nResults Summary:")
        print(f"{'Count':<10} {'Exact (ms)':<15} {'Estimated (ms)':<15}")
        for i, n in enumerate(results["counts"]):
            print(f"{n:<10} {results['exact_times'][i]:<15.2f} {results['estimated_times'][i]:<15.2f}")
                
        plot_results(results)
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
