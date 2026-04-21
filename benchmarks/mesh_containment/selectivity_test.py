#!/usr/bin/env python3
import argparse
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
from benchmarks.mesh_containment.adapters.cgal_adapter import CGALContainmentAdapter
from benchmarks.mesh_containment.adapters.raytracer_adapter import RaytracerContainmentAdapter


RESULTS_DIR = SCRIPT_DIR / "results"
CGAL_BASE_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"


DEFAULT_SELECTIVITIES = [0.0001, 0.0005, 0.001, 0.005, 0.01]


def main():
    parser = argparse.ArgumentParser(description="Selectivity benchmark for mesh containment")
    parser.add_argument("--selectivities", type=float, nargs="+", default=DEFAULT_SELECTIVITIES)
    parser.add_argument("--num-cubes", type=int, default=50000)
    parser.add_argument("--min-size", type=float, default=1.0)
    parser.add_argument("--max-size", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-cell-size", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--include-overlap-pairs", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--approaches", type=str, nargs="+", default=["raytracer", "cgal"], choices=["raytracer", "cgal"])
    args = parser.parse_args()

    dirs = get_shared_data_dirs("selectivity")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "containment_selectivity")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raytracer = RaytracerContainmentAdapter(
        str(RAYSPACE_DIR), preprocessed_dir=str(dirs["preprocessed"]),
        timings_dir=str(dirs["timings"]), grid_cell_size=10, warmup_runs=args.warmup_runs,
        include_overlap_pairs=args.include_overlap_pairs,
    )
    cgal = CGALContainmentAdapter(str(CGAL_BASE_DIR), preprocessed_dir=str(dirs["preprocessed"]))

    summary = []
    for selectivity in args.selectivities:
        universe_extent = compute_universe_for_selectivity(selectivity, args.min_size, args.max_size)
        grid_cell_size = max(1, int(round(universe_extent / args.grid_cell_size)))
        raytracer.grid_cell_size = grid_cell_size

        obj_a, obj_b = canonical_cube_pair_paths(
            dirs["raw"],
            num_cubes_a=args.num_cubes,
            num_cubes_b=args.num_cubes,
            min_size=args.min_size,
            max_size=args.max_size,
            selectivity=selectivity,
            seed=args.seed,
            grid_cell_size=grid_cell_size,
        )

        ensure_cube_pair_dataset(
            obj_a,
            obj_b,
            num_cubes_a=args.num_cubes,
            num_cubes_b=args.num_cubes,
            min_size=args.min_size,
            max_size=args.max_size,
            selectivity=selectivity,
            seed=args.seed,
        )

        for file_path in (obj_a, obj_b):
            if not raytracer.check_preprocessed(str(file_path)):
                raytracer.preprocess_from_source(str(file_path), str(file_path))

        row = {
            "selectivity": selectivity,
            "grid_cell_size": grid_cell_size,
            "universe_extent": universe_extent,
            "num_cubes": args.num_cubes,
            "size_bytes_a": obj_a.stat().st_size if obj_a.exists() else 0,
            "size_bytes_b": obj_b.stat().st_size if obj_b.exists() else 0,
        }

        if "raytracer" in args.approaches:
            row["raytracer"] = raytracer.run_containment(str(obj_a), str(obj_b), args.runs, timeout=args.timeout)
        if "cgal" in args.approaches:
            row["cgal"] = cgal.run_containment(str(obj_a), str(obj_b), args.runs, timeout=args.timeout)

        summary.append(row)
        print(f"selectivity={selectivity}: done")

    payload = {
        "metadata": {
            "scenario": "selectivity",
            "query_type": "containment",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "selectivities": args.selectivities,
            "num_cubes": args.num_cubes,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "seed": args.seed,
            "grid_cell_size": args.grid_cell_size,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "include_overlap_pairs": args.include_overlap_pairs,
            "timeout_seconds": args.timeout,
            "approaches": args.approaches,
            "shared_data_root": str(dirs["root"]),
        },
        "results": summary,
    }

    out_runs = run_layout["results_json"]
    write_json(out_runs, payload)
    print(f"Saved run log: {out_runs}")

    # Keep legacy summary path for compatibility.
    out_summary = RESULTS_DIR / "summary.json"
    write_json(out_summary, summary)
    print(f"Saved summary: {out_summary}")


if __name__ == "__main__":
    main()
