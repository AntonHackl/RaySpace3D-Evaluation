#!/usr/bin/env python3
import argparse
import json
import subprocess
from pathlib import Path

from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter


SELECTIVITIES = [0.0001, 0.0005, 0.001, 0.005, 0.01]
NUM_CUBES = 50000
MIN_SIZE = 1
MAX_SIZE = 4
GRID_CELL_SIZE = 5
TIMEOUT_SECONDS = 120.0

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw" / "selectivity_test"
PREPROCESSED_DIR = DATA_DIR / "preprocessed" / "selectivity_test"
TIMINGS_DIR = DATA_DIR / "timings" / "selectivity_test"
RESULTS_DIR = SCRIPT_DIR / "results" / "selectivity_test"

GENERATOR_SCRIPT = RAYSPACE_DIR / "scripts" / "generate_cubes_by_selectivity.py"


def compute_universe_for_selectivity(target_selectivity, min_size, max_size):
    avg_size = (min_size + max_size) / 2.0
    if target_selectivity <= 0:
        raise ValueError("Target selectivity must be positive")
    universe_extent = (2.0 * avg_size) / (target_selectivity ** (1.0 / 3.0))
    return universe_extent


def main():
    parser = argparse.ArgumentParser(description="Selectivity Benchmark for Mesh Intersection")
    parser.add_argument("--approaches", type=str, nargs="+",
                        default=["two_pass", "estimated"],
                        choices=["two_pass", "estimated", "estimate_only"],
                        help="Approaches to run")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per selectivity")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

        file_suffix = str(selectivity).replace('.', '_')
        obj_a = RAW_DIR / f"cubes_a_sel_{file_suffix}.obj"
        obj_b = RAW_DIR / f"cubes_b_sel_{file_suffix}.obj"

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

        adapter = RaytracerIntersectionAdapter(
            str(RAYSPACE_DIR),
            mode="two_pass",
            preprocessed_dir=str(PREPROCESSED_DIR),
            timings_dir=str(TIMINGS_DIR),
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
            adapter.mode = mode
            query_bin_dir = Path(str(RAYSPACE_DIR)) / "query" / "build" / "bin"
            if mode == "two_pass":
                adapter.executable = query_bin_dir / "raytracer_mesh_intersection"
                adapter.name = f"Raytracer_{mode}"
            elif mode in ("estimated", "estimate_only"):
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

    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=4)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
