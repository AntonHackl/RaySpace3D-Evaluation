
import os
import sys
import json
import argparse
from pathlib import Path
import subprocess

# Add current directory to path to import adapters
# Add current directory to path to import adapters
sys.path.append(str(Path(__file__).parent))
from adapters.raytracer_adapter import RaytracerAdapter
from adapters.tdbase_adapter import TDBaseAdapter
from adapters.cgal_adapter import CGALAdapter

# Configuration
SELECTIVITIES = [0.0001, 0.0005, 0.001, 0.005, 0.01]
NUM_CUBES = 50000
MIN_SIZE = 1
MAX_SIZE = 4
GRID_CELL_SIZE = 5
TIMEOUT_SECONDS = 120.0  # 2 minutes timeout per run

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
RAYSPACE_DIR = WORKSPACE_ROOT / "src/RaySpace3D"
TDBASE_DIR = WORKSPACE_ROOT / "baselines/RaySpace3DBaselines/tdbase"
CGAL_DIR = WORKSPACE_ROOT / "baselines/RaySpace3DBaselines/CGAL"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw" / "selectivity_test"
PREPROCESSED_DIR = DATA_DIR / "preprocessed" / "selectivity_test"
TIMINGS_DIR = DATA_DIR / "timings" / "selectivity_test"
RESULTS_DIR = Path(__file__).parent / "results" / "selectivity_test"

GENERATOR_SCRIPT = RAYSPACE_DIR / "scripts/generate_cubes_by_selectivity.py"

def compute_universe_for_selectivity(target_selectivity, min_size, max_size):
    avg_size = (min_size + max_size) / 2.0
    if target_selectivity <= 0:
        raise ValueError("Target selectivity must be positive")
    universe_extent = (2.0 * avg_size) / (target_selectivity ** (1.0/3.0))
    return universe_extent

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_results = []

    for selectivity in SELECTIVITIES:
        print(f"\n{'='*60}")
        print(f"Processing Selectivity: {selectivity}")
        print(f"{'='*60}")

        # 1. Compute Universe and Grid Resolution
        universe_extent = compute_universe_for_selectivity(selectivity, MIN_SIZE, MAX_SIZE)
        grid_resolution = int(round(universe_extent / GRID_CELL_SIZE))
        if grid_resolution < 1: grid_resolution = 1
        
        print(f"Universe Extent: {universe_extent:.2f}")
        print(f"Grid Resolution: {grid_resolution} (Cell Size: {universe_extent/grid_resolution:.2f})")

        # 2. Generate Data
        file_suffix = str(selectivity).replace('.', '_')
        obj_a = RAW_DIR / f"cubes_a_sel_{file_suffix}.obj"
        obj_b = RAW_DIR / f"cubes_b_sel_{file_suffix}.obj"
        
        # .dt paths for consistent naming in adapter
        dt_a = obj_a.with_suffix('.dt') 
        dt_b = obj_b.with_suffix('.dt')

        if not obj_a.exists() or not obj_b.exists():
            print("Generating cubes...")
            cmd = [
                "python3", str(GENERATOR_SCRIPT),
                "--num-cubes-a", str(NUM_CUBES),
                "--num-cubes-b", str(NUM_CUBES),
                "--min-size", str(MIN_SIZE),
                "--max-size", str(MAX_SIZE),
                "--selectivity", str(selectivity),
                "--output-a", str(obj_a),
                "--output-b", str(obj_b),
                "--seed", "42"
            ]
            subprocess.run(cmd, check=True)
        else:
            print("Files already exist, skipping generation.")

        # 3. Setup Adapter
        adapter = RaytracerAdapter(
            str(RAYSPACE_DIR), 
            mode="exact", 
            preprocessed_dir=str(PREPROCESSED_DIR),
            timings_dir=str(TIMINGS_DIR),
            grid_resolution=grid_resolution,
            warmup_runs=2
        )

        # 4. Preprocess
        # RaytracerAdapter.check_preprocessed() checks if .pre exists based on input filename
        # We assume if the file exists it was generated with the correct parameters (grid size).
        # To be safe, we could delete it, but for now we trust the flow or use distinct filenames if parameters changed repeatedly (they don't here).
        print("Ensuring preprocessed files (Raytracer)...")
        adapter.preprocess_from_source(str(obj_a), str(dt_a))
        adapter.preprocess_from_source(str(obj_b), str(dt_b))

        # Setup TDBase & Preprocess
        tdbase_adapter = TDBaseAdapter(
            str(TDBASE_DIR),
            preprocessed_dir=str(PREPROCESSED_DIR)
        )
        print("Ensuring preprocessed files (TDBase)...")
        # For TDBase, we convert obj to dt using the tool
        # TDBaseAdapter uses preprocess_from_source(obj, dt)
        tdbase_adapter.preprocess_from_source(str(obj_a), str(dt_a))
        tdbase_adapter.preprocess_from_source(str(obj_b), str(dt_b))

        # Setup CGAL
        cgal_adapter = CGALAdapter(
            str(CGAL_DIR),
            preprocessed_dir=str(PREPROCESSED_DIR)
        )
        # CGAL uses the same preprocessed files as Raytracer (.pre), which are already generated.

        # 5. Run Benchmark
        print("Running benchmark...")
        
        # Test both Exact, Estimated, TDBase, and CGAL
        modes = ["exact", "estimated", "tdbase", "cgal"]
        res_per_sel = {
            "selectivity": selectivity, 
            "grid_resolution": grid_resolution, 
            "universe": universe_extent,
            "num_cubes": NUM_CUBES
        }
        
        for mode in modes:
            current_adapter = None
            if mode == "tdbase":
                current_adapter = tdbase_adapter
                # Update name for consistent result key
                current_adapter.name = "tdbase"
            elif mode == "cgal":
                current_adapter = cgal_adapter
                current_adapter.name = "cgal"
            else:
                adapter.mode = mode
                # Update executable manually as correct binary depends on mode
                if mode == "exact":
                    adapter.executable = adapter.rayspace_dir / "query/build/bin/raytracer_mesh_overlap"
                    adapter.name = "exact" # For result key consistency in json
                elif mode == "estimated":
                    adapter.executable = adapter.rayspace_dir / "query/build/bin/raytracer_overlap_estimated"
                    adapter.name = "estimated"
                current_adapter = adapter
                
            results = current_adapter.run_overlap(
                str(obj_a),
                str(obj_b),
                num_runs=5,
                timeout=TIMEOUT_SECONDS
            )
            
            # Use mode name as key in results
            result_key = mode
            
            if "error" in results:
                print(f"[{result_key}] Error: {results['error']}")
                res_per_sel[result_key] = {"error": results['error']}
            else:
                breakdown = results.get("breakdown", {})
                mean_time = results['mean']
                print(f"[{result_key}] Mean Time: {mean_time:.4f} ms")
                
                # Check for estimated mode to include download time in mean?
                # The mean reported by run_overlap covers the total time measured by Python or the tool.
                # In RaytracerAdapter, it sums up phases.
                # The breakdown should already be correct.
                
                res_per_sel[result_key] = {
                    "mean_ms": mean_time,
                    "std_ms": results['std'],
                    "intersections": results.get("num_intersections", 0),
                    "breakdown": breakdown
                }

        summary_results.append(res_per_sel)

    # Save summary
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
