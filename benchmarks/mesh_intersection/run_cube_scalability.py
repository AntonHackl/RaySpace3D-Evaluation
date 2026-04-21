#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    RAYSPACE_DIR,
    canonical_cube_pair_paths,
    compute_universe_for_selectivity,
    create_benchmark_run_layout,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.mesh_intersection.adapters.cgal_adapter import CGALIntersectionAdapter
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter


SCRIPT_DIR = Path(__file__).resolve().parent
CGAL_BASE_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"


DEFAULT_COUNTS = [200000, 400000, 600000, 1000000]


def main():
    parser = argparse.ArgumentParser(description="Cube scalability benchmark for mesh intersection")
    parser.add_argument("--counts", type=int, nargs="+", default=DEFAULT_COUNTS,
                        help="Cube counts for dataset B")
    parser.add_argument("--fixed-count", type=float, default=1.00000,
                        help="Cube count for fixed dataset A")
    parser.add_argument("--selectivity", type=float, default=0.001,
                        help="Target selectivity for generated datasets")
    parser.add_argument("--min-size", type=float, default=1.0)
    parser.add_argument("--max-size", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-cell-size", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--approaches", type=str, nargs="+",
                        default=["estimated", "cgal"],
                        choices=["estimated", "estimate_only", "cgal"])
    args = parser.parse_args()

    dirs = get_shared_data_dirs("cube_scalability")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "intersection_cube_scalability")

    universe_extent = compute_universe_for_selectivity(args.selectivity, args.min_size, args.max_size)

    estimated_adapter = RaytracerIntersectionAdapter(
        str(RAYSPACE_DIR), mode="estimated", preprocessed_dir=str(dirs["preprocessed"]),
        timings_dir=str(dirs["timings"]), grid_cell_size=args.grid_cell_size, warmup_runs=args.warmup_runs,
    )
    cgal_adapter = CGALIntersectionAdapter(str(CGAL_BASE_DIR), preprocessed_dir=str(dirs["preprocessed"]))

    results = []
    for count_b in args.counts:
        fixed_a, ds_b = canonical_cube_pair_paths(
            dirs["raw"],
            num_cubes_a=args.fixed_count,
            num_cubes_b=count_b,
            min_size=args.min_size,
            max_size=args.max_size,
            selectivity=args.selectivity,
            seed=args.seed,
            grid_cell_size=args.grid_cell_size,
        )

        ensure_cube_pair_dataset(
            fixed_a,
            ds_b,
            num_cubes_a=args.fixed_count,
            num_cubes_b=count_b,
            min_size=args.min_size,
            max_size=args.max_size,
            selectivity=args.selectivity,
            seed=args.seed,
        )

        for file_path in (fixed_a, ds_b):
            if not estimated_adapter.check_preprocessed(str(file_path)):
                estimated_adapter.preprocess_from_source(str(file_path), str(file_path))

        entry = {
            "count_a": args.fixed_count,
            "count_b": count_b,
            "selectivity": args.selectivity,
            "universe_extent": universe_extent,
            "size_bytes_a": fixed_a.stat().st_size if fixed_a.exists() else 0,
            "size_bytes_b": ds_b.stat().st_size if ds_b.exists() else 0,
        }

        if "estimated" in args.approaches:
            res = estimated_adapter.run_intersection(str(fixed_a), str(ds_b), args.runs, timeout=args.timeout)
            entry["estimated"] = res

        if "estimate_only" in args.approaches:
            estimated_adapter.mode = "estimate_only"
            estimated_adapter.name = "Raytracer_estimate_only"
            estimated_adapter.executable = estimated_adapter.rayspace_dir / "query" / "build" / "bin" / "raytracer_intersection_estimated"
            res = estimated_adapter.run_intersection(str(fixed_a), str(ds_b), args.runs, timeout=args.timeout)
            entry["estimate_only"] = res

        if "cgal" in args.approaches:
            res = cgal_adapter.run_intersection(str(fixed_a), str(ds_b), args.runs, timeout=args.timeout)
            entry["cgal"] = res

        results.append(entry)
        print(f"count_b={count_b}: done")

    payload = {
        "metadata": {
            "scenario": "cube_scalability",
            "query_type": "intersection",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "counts": args.counts,
            "fixed_count": args.fixed_count,
            "selectivity": args.selectivity,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "seed": args.seed,
            "grid_cell_size": args.grid_cell_size,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "approaches": args.approaches,
            "shared_data_root": str(dirs["root"]),
        },
        "results": results,
    }

    out = run_layout["results_json"]
    write_json(out, payload)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
