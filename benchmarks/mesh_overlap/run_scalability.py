#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess 
import json
import time

# Add current directory to path to import adapters
sys.path.append(str(Path(__file__).parent))
from adapters.raytracer_adapter import RaytracerAdapter
from adapters.tdbase_adapter import TDBaseAdapter

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent # Assuming standard structure
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
TDBASE_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/tdbase"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

TIMEOUT_SECONDS = 120.0

# Nu Counts for Dataset B (Dataset A is fixed at 150 vessels)
NU_COUNTS = [200, 400, 600, 800, 1000]
NV_COUNT = 150 # Fixed vessel count

def run_experiment(runs, grid_resolution):
    print("--- Starting Mesh Overlap Scalability Experiment ---")
    
    # Ensure directories exist
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"mesh_scalability_{runs}runs_{timestamp}"
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

    tdbase_adapter = TDBaseAdapter(
        str(TDBASE_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR)
    )
    
    results = {
        "counts": [],
        "exact": {"mean": [], "std": [], "breakdown": []},
        "estimated": {"mean": [], "std": [], "breakdown": []},
        "tdbase": {"mean": [], "std": []}
    }

    # Helper to find file
    def find_file(prefix_part):
        # Look for file matching pattern in RAW_DIR
        # expected: tdbase_n_nv150_nu{nu}_[vn]_nv150_nu{nu}_vs100_r30.dt
        # We need flexible matching because simulator output depends on params
        candidates = list(RAW_DIR.glob(f"*{prefix_part}*.dt"))
        if not candidates:
            return None
        # Return the shortest match or just the first?
        # usually simulator outputs specific name.
        # User defined prefix: tdbase_n_nv150_nu{nu}
        # Simulator appends: _n_nv150_nu{nu}_vs100_r30.dt and _v_...
        return candidates[0]

    for nu in NU_COUNTS:
        # Construct expected filenames based on generation script
        # "tdbase_n_nv150_nu200" is the prefix (-o)
        prefix = f"tdbase_n_nv150_nu{nu}"
        
        # Dataset A: Vessel (v)
        # Pattern: prefix + "_v_nv150_nu" + nu + "_vs100_r30.dt"
        file_a_name = f"{prefix}_v_nv150_nu{nu}_vs100_r30.dt"
        f_a_path = RAW_DIR / file_a_name
        
        # Dataset B: Nuclei (n)
        # Pattern: prefix + "_n_nv150_nu" + nu + "_vs100_r30.dt"
        file_b_name = f"{prefix}_n_nv150_nu{nu}_vs100_r30.dt"
        f_b_path = RAW_DIR / file_b_name
        
        if not f_a_path.exists() or not f_b_path.exists():
            print(f"Error: Datasets for nu={nu} not found at {RAW_DIR}.")
            print(f"Expected: {file_a_name} and {file_b_name}")
            # Try to find partial matches if exact name fails
            candidates_a = list(RAW_DIR.glob(f"{prefix}*_v_*.dt"))
            candidates_b = list(RAW_DIR.glob(f"{prefix}*_n_*.dt"))
            if candidates_a and candidates_b:
                f_a_path = candidates_a[0]
                f_b_path = candidates_b[0]
                print(f"Found via glob: {f_a_path.name} and {f_b_path.name}")
            else:
                print("Skipping.")
                continue
        
        print(f"\nProcessing nu={nu}: {f_a_path.name} vs {f_b_path.name}")

        # Check/Run Preprocessing for Raytracer (TDBase handles .dt natively via adapter assuming pass through)
        print("Checking preprocessing...")
        # Note: Raytracer needs .pre files generated from .dt files
        # The adapter's preprocess_from_source(src, dt_name) calls preprocess_dataset
        # We pass the full path to source .dt file.
        # The adapter generates .pre file named based on 2nd arg.
        exact_adapter.preprocess_from_source(str(f_a_path), str(f_a_path), log_dir=str(run_log_dir))
        exact_adapter.preprocess_from_source(str(f_b_path), str(f_b_path), log_dir=str(run_log_dir))

        # Run Exact Benchmark
        print(f"Running Exact Mode ({runs} runs)...")
        res_exact = exact_adapter.run_overlap(
            str(f_a_path), 
            str(f_b_path), 
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS
        )
        if "error" in res_exact:
            print(f"Error in exact run: {res_exact['error']}")
            # Continue?
            res_exact = {"mean": 0, "std": 0, "breakdown": {}}
            
        # Run Estimated Benchmark
        print(f"Running Estimated Mode ({runs} runs)...")
        res_est = estimated_adapter.run_overlap(
            str(f_a_path), 
            str(f_b_path), 
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS
        )
        if "error" in res_est:
            print(f"Error in estimated run: {res_est['error']}")
            res_est = {"mean": 0, "std": 0, "breakdown": {}}

        # Run TDBase Benchmark
        print(f"Running TDBase Mode ({runs} runs)...")
        # Note: TDBaseAdapter expects .obj or .dt paths. If .dt, it should just pass them if logic allows.
        # TDBaseAdapter logic: attempts to find processing in preprocessed_dir if set.
        # We'll rely on it finding the file or using the absolute path.
        res_td = tdbase_adapter.run_overlap(
            str(f_a_path), 
            str(f_b_path), 
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS
        )
        if "error" in res_td:
            print(f"Error in TDBase run: {res_td['error']}")
            res_td = {"mean": None, "std": None}

        results["counts"].append(nu)
        
        results["exact"]["mean"].append(res_exact["mean"])
        results["exact"]["std"].append(res_exact["std"])
        results["exact"]["breakdown"].append(res_exact.get("breakdown", {}))
        
        results["estimated"]["mean"].append(res_est["mean"])
        results["estimated"]["std"].append(res_est["std"])
        results["estimated"]["breakdown"].append(res_est.get("breakdown", {}))
        
        results["tdbase"]["mean"].append(res_td["mean"])
        results["tdbase"]["std"].append(res_td["std"])
        
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
        results["size_bytes1"].append(f_a_path.stat().st_size if f_a_path.exists() else 0)
        results["size_bytes2"].append(f_b_path.stat().st_size if f_b_path.exists() else 0)
        results["universe_extents1"].append(res_exact.get("universe_extents1", [0.0, 0.0, 0.0]))
        results["universe_extents2"].append(res_exact.get("universe_extents2", [0.0, 0.0, 0.0]))
        
        td_str = f"{res_td['mean']:.2f}ms" if res_td['mean'] else "TIMEOUT/ERR"
        print(f"Done nu={nu}: Exact={res_exact['mean']:.2f}ms, Est={res_est['mean']:.2f}ms, TDBase={td_str}")

    return results

def plot_results(results):
    print("\nPlotting results...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    counts = results["counts"]
    if not counts:
        print("No results to plot.")
        return

    fig, ax_main = plt.subplots(figsize=(10, 8))

    # --- Plot: Line Chart (Scaling) ---
    ax_main.errorbar(counts, results["exact"]["mean"], yerr=results["exact"]["std"], 
                     fmt='-o', label='RaySpace Exact', capsize=5, color='#1f77b4')
    ax_main.errorbar(counts, results["estimated"]["mean"], yerr=results["estimated"]["std"], 
                     fmt='--s', label='RaySpace Estimated', capsize=5, color='#2ca02c')
    
    # Filter valid TDBase points
    td_valid_indices = [i for i, m in enumerate(results["tdbase"]["mean"]) if m is not None]
    if td_valid_indices:
        td_counts = [counts[i] for i in td_valid_indices]
        td_means = [results["tdbase"]["mean"][i] for i in td_valid_indices]
        td_stds = [results["tdbase"]["std"][i] for i in td_valid_indices]
        ax_main.errorbar(td_counts, td_means, yerr=td_stds, 
                         fmt=':d', label='TDBase', capsize=5, color='#d62728')

    ax_main.set_xlabel('Number of Nuclei per Vessel (Dataset B)', fontsize=12)
    ax_main.set_ylabel('Execution Time (ms) [Log Scale]', fontsize=12)
    ax_main.set_title('Scalability: Mesh Overlap Query Time', fontsize=14, fontweight='bold')
    ax_main.set_yscale('log')
    ax_main.legend(fontsize=12)
    ax_main.grid(True, which="both", ls="-", alpha=0.2)
    ax_main.set_xticks(counts)

    plt.tight_layout()
    output_path = FIGURES_DIR / "mesh_overlap_scalability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Also save PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Mesh Overlap Nu Scalability Experiment")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per method")
    parser.add_argument("--grid-resolution", type=int, default=20, help="Grid resolution for RaySpace")
    args = parser.parse_args()
    
    results = run_experiment(args.runs, args.grid_resolution)
    
    if results and results["counts"]:
        print("\nResults Summary:")
        print(f"{'Nu':<10} {'Exact (ms)':<15} {'Estimated (ms)':<15} {'TDBase (ms)':<15}")
        for i, n in enumerate(results["counts"]):
            ex = results['exact']['mean'][i]
            est = results['estimated']['mean'][i]
            td = results['tdbase']['mean'][i]
            td_str = f"{td:.2f}" if td else "N/A"
            print(f"{n:<10} {ex:<15.2f} {est:<15.2f} {td_str:<15}")
                
        plot_results(results)
        
        # Save summary to json
        out_json = RUNS_DIR / f"scalability_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_json, 'w') as f:
            clean_results = {}
            for k, v in results.items():
                if isinstance(v, dict):
                    clean_results[k] = {ki: (vi.tolist() if isinstance(vi, np.ndarray) else vi) for ki, vi in v.items()}
                elif isinstance(v, list):
                    clean_results[k] = v
                else:
                    clean_results[k] = v
            json.dump(clean_results, f, indent=4)
        print(f"Raw results saved to {out_json}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
