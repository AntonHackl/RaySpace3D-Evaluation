#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime
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

# Dataset (names provided by user + extensions from benchmark.py)
FILE1 = "nu400_n_nv150_nu400_vs100_r30.dt"
FILE2 = "nu400_v_nv150_nu400_vs100_r30.dt"

def run_experiment(runs, grid_resolution):
    print("--- Starting Breakdown Experiment ---")
    
    # Setup Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"breakdown_{runs}runs_{timestamp}"
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
    
    # Ensure raw files exist
    f1_path = RAW_DIR / FILE1
    f2_path = RAW_DIR / FILE2
    
    if not f1_path.exists():
        print(f"Error: {f1_path} does not exist.")
        return
    if not f2_path.exists():
        print(f"Error: {f2_path} does not exist.")
        return

    # Check/Run Preprocessing
    print("Checking preprocessing...")
    # Log preprocessing to same dir, or a shared one? 
    # Usually preprocessing is one-off, but we can log it to the run dir for completeness if it happens.
    for f in [f1_path, f2_path]:
        if not exact_adapter.check_preprocessed(str(f)):
            print(f"Preprocessing {f.name}...")
            exact_adapter.preprocess_from_source(str(f), str(f), log_dir=str(run_log_dir))
        else:
            print(f"Already preprocessed: {f.name}")

    # Run Benchmark
    results = {}
    
    print(f"\nRunning Exact Mode ({runs} runs)...")
    res_exact = exact_adapter.run_overlap(
        str(f1_path), 
        str(f2_path), 
        runs,
        log_dir=str(run_log_dir)
    )
    if "error" in res_exact:
        print(f"Error in exact run: {res_exact['error']}")
        return
    results["Exact"] = res_exact
    
    print(f"\nRunning Estimated Mode ({runs} runs)...")
    res_est = estimated_adapter.run_overlap(
        str(f1_path), 
        str(f2_path), 
        runs,
        log_dir=str(run_log_dir)
    )
    if "error" in res_est:
        print(f"Error in estimated run: {res_est['error']}")
        return
    results["Estimated"] = res_est
    
    return results

def plot_results(results):
    print("\nPlotting results...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    modes = list(results.keys()) # ["Exact", "Estimated"]
    
    # Identify all unique phases across all modes
    all_phases = set()
    for mode in modes:
        if "breakdown" in results[mode]:
            all_phases.update(results[mode]["breakdown"].keys())
            
    print(f"Phases found: {all_phases}")
    
    # Map raw phase names to readable labels
    phase_mapping = {
        "selectivity estimation_": "Selectivity Est.",
        "execute hash query_": "Hash Query",
        "query_": "Ray Query",
        "gpu deduplication_": "Deduplication",
        "download results_": "Download"
    }
    
    # Filter and sort phases for stacking order
    # Bottom to top: Estimation -> Query/Hash -> Dedup -> Download
    ordered_phases_raw = [
        "selectivity estimation_",
        "query_",
        "execute hash query_",
        "gpu deduplication_",
        "download results_"
    ]
    
    # Keep only phases that are present
    active_phases = [p for p in ordered_phases_raw if p in all_phases]
    
    # Check if we have phases that are not in our ordered list (unexpected keys)
    for p in all_phases:
        if p not in ordered_phases_raw:
            print(f"Warning: Unexpected phase key found: '{p}'")
            active_phases.append(p)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.5
    indices = np.arange(len(modes))
    
    bottoms = np.zeros(len(modes))
    
    colors = {
        "selectivity estimation_": "#ff9999", # Red-ish
        "query_": "#66b3ff",              # Blue-ish
        "execute hash query_": "#3399ff",   # Darker Blue
        "gpu deduplication_": "#99ff99",    # Green-ish
        "download results_": "#ffcc99"      # Orange-ish
    }
    
    # Plot bars
    for phase in active_phases:
        values = []
        for mode in modes:
            val = results[mode]["breakdown"].get(phase, 0.0)
            values.append(val)
        
        label = phase_mapping.get(phase, phase.replace("_", " ").strip().title())
        color = colors.get(phase, None)
        
        # If value is very small but not zero, force a minimum visualization height?
        # Or print it.
        # Here we just plot.
        
        ax.bar(indices, values, bar_width, bottom=bottoms, label=label, color=color, edgecolor='white')
        bottoms += np.array(values)
    
    ax.set_xlabel('Method')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Mesh Overlap Query Time Breakdown (nu400)')
    ax.set_xticks(indices)
    ax.set_xticklabels(modes)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    output_path = FIGURES_DIR / "mesh_overlap_breakdown.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Mesh Overlap Breakdown Experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per method")
    parser.add_argument("--grid-resolution", type=int, default=10, help="Grid resolution for RaySpace")
    args = parser.parse_args()
    
    results = run_experiment(args.runs, args.grid_resolution)
    
    if results:
        print("\nResults Summary:")
        for mode, data in results.items():
            print(f"{mode}: Total {data['mean']:.2f} ms")
            print("  Breakdown:")
            for k, v in data['breakdown'].items():
                print(f"    {k}: {v:.2f} ms")
                
        plot_results(results)

if __name__ == "__main__":
    main()
