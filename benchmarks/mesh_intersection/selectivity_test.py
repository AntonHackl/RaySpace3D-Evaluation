#!/usr/bin/env python3
import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    canonical_cube_pair_paths,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
    compute_universe_for_selectivity,
)
from benchmarks.mesh_intersection.adapters.cgal_adapter import CGALIntersectionAdapter
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter


SELECTIVITIES = [0.0001, 0.0005, 0.001, 0.005, 0.01]
NUM_CUBES = 50000
MIN_SIZE = 1
MAX_SIZE = 4
GRID_CELL_SIZE = 5
TIMEOUT_SECONDS = 120.0

RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
CGAL_BASE_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"
RESULTS_DIR = SCRIPT_DIR / "results" / "selectivity_test"
RUNS_DIR = SCRIPT_DIR / "runs"


def main():
    parser = argparse.ArgumentParser(description="Selectivity Benchmark for Mesh Intersection")
    parser.add_argument("--approaches", type=str, nargs="+",
                        default=["estimated", "cgal"],
                        choices=["estimated", "estimate_only", "cgal"],
                        help="Approaches to run")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per selectivity")
    args = parser.parse_args()

    dirs = get_shared_data_dirs("selectivity")
    raw_dir = dirs["raw"]
    preprocessed_dir = dirs["preprocessed"]
    timings_dir = dirs["timings"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    
    cgal_adapter = CGALIntersectionAdapter(str(CGAL_BASE_DIR), preprocessed_dir=str(preprocessed_dir))

    summary_results = []

    for selectivity in SELECTIVITIES:
        print(f"\n{'=' * 60}")
        print(f"Processing Selectivity: {selectivity}")
        print(f"{'=' * 60}")

        universe_extent = compute_universe_for_selectivity(selectivity, MIN_SIZE, MAX_SIZE)
        grid_resolution = int(round(universe_extent / GRID_CELL_SIZE))
        if grid_resolution < 1:
            grid_resolution = 1

        print(f"Universe Extent: {universe_extent:.2f}")
        print(f"Grid Resolution: {grid_resolution} (Cell Size: {universe_extent / grid_resolution:.2f})")

        obj_a, obj_b = canonical_cube_pair_paths(
            raw_dir,
            num_cubes_a=NUM_CUBES,
            num_cubes_b=NUM_CUBES,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            selectivity=selectivity,
            seed=42,
            grid_resolution=grid_resolution,
        )

        dt_a = obj_a.with_suffix('.dt')
        dt_b = obj_b.with_suffix('.dt')

        ensure_cube_pair_dataset(
            obj_a,
            obj_b,
            num_cubes_a=NUM_CUBES,
            num_cubes_b=NUM_CUBES,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            selectivity=selectivity,
            seed=42,
        )

        adapter = RaytracerIntersectionAdapter(
            str(RAYSPACE_DIR),
            mode="estimated",
            preprocessed_dir=str(preprocessed_dir),
            timings_dir=str(timings_dir),
            grid_resolution=grid_resolution,
            warmup_runs=2
        )

        print("Ensuring preprocessed files (Raytracer)...")
        adapter.preprocess_from_source(str(obj_a), str(dt_a))
        adapter.preprocess_from_source(str(obj_b), str(dt_b))

        res_per_sel = {
            "selectivity": selectivity,
            "grid_resolution": grid_resolution,
            "universe": universe_extent,
            "num_cubes": NUM_CUBES
        }

        for mode in args.approaches:
            if mode == "cgal":
                cgal_results = cgal_adapter.run_intersection(
                    str(obj_a),
                    str(obj_b),
                    num_runs=args.runs,
                    timeout=TIMEOUT_SECONDS,
                )
                if "error" in cgal_results:
                    print(f"[cgal] Error: {cgal_results['error']}")
                    res_per_sel["cgal"] = {"error": cgal_results["error"]}
                else:
                    print(f"[cgal] Mean Time: {cgal_results['mean']:.4f} ms")
                    res_per_sel["cgal"] = {
                        "mean_ms": cgal_results["mean"],
                        "std_ms": cgal_results["std"],
                        "intersections": cgal_results.get("num_intersections", 0),
                    }
                continue

            adapter.mode = mode
            query_bin_dir = Path(str(RAYSPACE_DIR)) / "query" / "build" / "bin"
            if mode in ("estimated", "estimate_only"):
                adapter.executable = query_bin_dir / "raytracer_intersection_estimated"
                adapter.name = f"Raytracer_{mode}"

            results = adapter.run_intersection(
                str(obj_a),
                str(obj_b),
                num_runs=args.runs,
                timeout=TIMEOUT_SECONDS
            )

            result_key = mode
            if "error" in results:
                print(f"[{result_key}] Error: {results['error']}")
                res_per_sel[result_key] = {"error": results["error"]}
            else:
                mean_time = results["mean"]
                print(f"[{result_key}] Mean Time: {mean_time:.4f} ms")
                res_per_sel[result_key] = {
                    "mean_ms": mean_time,
                    "std_ms": results["std"],
                    "intersections": results.get("num_intersections", 0),
                    "breakdown": results.get("breakdown", {})
                }

        summary_results.append(res_per_sel)

    # Save to conventional results path
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4)
    print(f"\nSummary saved to {summary_path}")

    # Also save with timestamp to runs/ directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    runs_path = RUNS_DIR / f"intersection_selectivity_{timestamp}.json"
    
    # Bundle metadata with results for the runs/ version
    full_output = {
        "metadata": {
            "timestamp": timestamp,
            "selectivities": SELECTIVITIES,
            "num_cubes": NUM_CUBES,
            "runs": args.runs,
            "approaches": args.approaches
        },
        "results": summary_results
    }
    
    with open(runs_path, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, indent=4)
    print(f"Detailed run log saved to {runs_path}")


if __name__ == "__main__":
    main()
