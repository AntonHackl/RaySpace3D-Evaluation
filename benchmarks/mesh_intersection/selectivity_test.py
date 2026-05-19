#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    build_selectivity_sweep,
    canonical_cube_pair_paths,
    create_benchmark_run_layout,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
    compute_universe_for_selectivity,
    write_json,
)
from benchmarks.mesh_intersection.adapters.cgal_adapter import CGALIntersectionAdapter
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter


SELECTIVITIES = build_selectivity_sweep()
NUM_CUBES = 50000
MIN_SIZE = 1
MAX_SIZE = 4
GRID_CELL_SIZE = 5
TIMEOUT_SECONDS = 120.0

RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
CGAL_BASE_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"
RESULTS_DIR = SCRIPT_DIR / "results" / "selectivity_test"


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
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "intersection_selectivity")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    cgal_adapter = CGALIntersectionAdapter(str(CGAL_BASE_DIR), preprocessed_dir=str(preprocessed_dir))

    summary_results = []

    for selectivity in SELECTIVITIES:
        print(f"\n{'=' * 60}")
        print(f"Processing Selectivity: {selectivity}")
        print(f"{'=' * 60}")

        universe_extent = compute_universe_for_selectivity(selectivity, MIN_SIZE, MAX_SIZE)
        grid_cell_size = int(round(universe_extent / GRID_CELL_SIZE))
        if grid_cell_size < 1:
            grid_cell_size = 1

        print(f"Universe Extent: {universe_extent:.2f}")
        print(f"Grid Resolution: {grid_cell_size} (Cell Size: {universe_extent / grid_cell_size:.2f})")

        obj_a, obj_b = canonical_cube_pair_paths(
            raw_dir,
            num_cubes_a=NUM_CUBES,
            num_cubes_b=NUM_CUBES,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            selectivity=selectivity,
            seed=42,
            grid_cell_size=grid_cell_size,
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
            grid_cell_size=grid_cell_size,
            warmup_runs=2
        )

        run_log_dir = run_layout["logs_dir"]

        print("Ensuring preprocessed files (Raytracer)...")
        adapter.preprocess_from_source(str(obj_a), str(dt_a), log_dir=str(run_log_dir))
        adapter.preprocess_from_source(str(obj_b), str(dt_b), log_dir=str(run_log_dir))

        res_per_sel = {
            "selectivity": selectivity,
            "grid_cell_size": grid_cell_size,
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
                    log_dir=str(run_log_dir),
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
                timeout=TIMEOUT_SECONDS,
                log_dir=str(run_log_dir),
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

    # Keep legacy summary path for compatibility.
    summary_path = RESULTS_DIR / "summary.json"
    write_json(summary_path, summary_results)
    print(f"\nSummary saved to {summary_path}")

    # Canonical per-run artifact.
    full_output = {
        "metadata": {
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "selectivities": SELECTIVITIES,
            "num_cubes": NUM_CUBES,
            "runs": args.runs,
            "approaches": args.approaches
        },
        "results": summary_results
    }

    runs_path = run_layout["results_json"]
    write_json(runs_path, full_output)
    print(f"Detailed run log saved to {runs_path}")


if __name__ == "__main__":
    main()
