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
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
    timestamp_tag,
)
from benchmarks.mesh_containment.adapters.cgal_adapter import CGALContainmentAdapter
from benchmarks.mesh_containment.adapters.raytracer_adapter import RaytracerContainmentAdapter


RUNS_DIR = SCRIPT_DIR / "runs"
CGAL_BASE_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"


DEFAULT_COUNTS = [200000, 400000, 600000, 1000000]


def main():
    parser = argparse.ArgumentParser(description="Cube scalability benchmark for mesh containment")
    parser.add_argument("--counts", type=int, nargs="+", default=DEFAULT_COUNTS)
    parser.add_argument("--fixed-count", type=int, default=200000)
    parser.add_argument("--selectivity", type=float, default=0.001)
    parser.add_argument("--min-size", type=float, default=1.0)
    parser.add_argument("--max-size", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-resolution", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--approaches", type=str, nargs="+", default=["raytracer", "cgal"], choices=["raytracer", "cgal"])
    args = parser.parse_args()

    dirs = get_shared_data_dirs("cube_scalability")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    universe_extent = compute_universe_for_selectivity(args.selectivity, args.min_size, args.max_size)

    raytracer = RaytracerContainmentAdapter(
        str(RAYSPACE_DIR), preprocessed_dir=str(dirs["preprocessed"]),
        timings_dir=str(dirs["timings"]), grid_resolution=args.grid_resolution, warmup_runs=args.warmup_runs,
    )
    cgal = CGALContainmentAdapter(str(CGAL_BASE_DIR), preprocessed_dir=str(dirs["preprocessed"]))

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
            grid_resolution=args.grid_resolution,
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
            if not raytracer.check_preprocessed(str(file_path)):
                raytracer.preprocess_from_source(str(file_path), str(file_path))

        row = {
            "count_a": args.fixed_count,
            "count_b": count_b,
            "selectivity": args.selectivity,
            "universe_extent": universe_extent,
            "size_bytes_a": fixed_a.stat().st_size if fixed_a.exists() else 0,
            "size_bytes_b": ds_b.stat().st_size if ds_b.exists() else 0,
        }

        if "raytracer" in args.approaches:
            row["raytracer"] = raytracer.run_containment(str(fixed_a), str(ds_b), args.runs, timeout=args.timeout)
        if "cgal" in args.approaches:
            row["cgal"] = cgal.run_containment(str(fixed_a), str(ds_b), args.runs, timeout=args.timeout)

        results.append(row)
        print(f"count_b={count_b}: done")

    payload = {
        "metadata": {
            "scenario": "cube_scalability",
            "query_type": "containment",
            "timestamp": timestamp_tag(),
            "counts": args.counts,
            "fixed_count": args.fixed_count,
            "selectivity": args.selectivity,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "seed": args.seed,
            "grid_resolution": args.grid_resolution,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "approaches": args.approaches,
            "shared_data_root": str(dirs["root"]),
        },
        "results": results,
    }

    out = RUNS_DIR / f"containment_cube_scalability_{timestamp_tag()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
