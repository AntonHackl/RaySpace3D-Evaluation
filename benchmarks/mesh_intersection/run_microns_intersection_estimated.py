#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    RAYSPACE_DIR,
    canonical_microns_aggregated_paths,
    create_benchmark_run_layout,
    ensure_microns_splits,
    ensure_microns_aggregated_meshes,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter


def main():
    parser = argparse.ArgumentParser(description="MICrONS subset estimated benchmark for mesh intersection")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8, 16],
                        help="MICrONS subset sizes in GB to benchmark")
    parser.add_argument("--source-root", type=str, 
                        default=str(REPO_ROOT / "datasets_scripts" / "microns_data"),
                        help="Root directory containing MICrONS GLB subset folders")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--grid-cell-size", type=float, default=5.0)
    parser.add_argument("--overlap-max-iterations", type=int, default=100)
    parser.add_argument("--containment-max-iterations", type=int, default=100)
    parser.add_argument("--hash-load-factor", type=float, default=0.5)
    parser.add_argument("--query-direction", type=str, default="both", choices=["both", "mesh1tomesh2", "mesh2tomesh1"])
    args = parser.parse_args()

    dirs = get_shared_data_dirs("microns_intersection_estimated")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "intersection_microns_estimated")

    splits_dir = dirs["root"] / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_root)

    estimated_adapter = RaytracerIntersectionAdapter(
        str(RAYSPACE_DIR), mode="estimated", preprocessed_dir=str(dirs["preprocessed"]),
        timings_dir=str(dirs["timings"]), grid_cell_size=args.grid_cell_size, warmup_runs=args.warmup_runs,
    )

    extra_args = [
        "--overlap-max-iterations", str(args.overlap_max_iterations),
        "--containment-max-iterations", str(args.containment_max_iterations),
        "--hash-load-factor", str(args.hash_load_factor),
        "--query-direction", args.query_direction,
    ]

    results = []
    for size_gb in args.sizes:
        print(f"--- Preparing MICrONS {size_gb}GB dataset ---")
        split_a, split_b = ensure_microns_splits(size_gb, source_root, splits_dir)
        
        agg_a, agg_b = canonical_microns_aggregated_paths(dirs["raw"], size_gb)
        ensure_microns_aggregated_meshes(split_a, split_b, agg_a, agg_b)

        for file_path in (agg_a, agg_b):
            if not estimated_adapter.check_preprocessed(str(file_path)):
                estimated_adapter.preprocess_from_source(str(file_path), str(file_path))

        entry = {
            "size_gb": size_gb,
            "manifest_a": str(split_a.resolve()),
            "manifest_b": str(split_b.resolve()),
            "source_dir": str((source_root / f"microns_region_{size_gb}gb_glb").resolve()),
            "size_bytes_a": agg_a.stat().st_size if agg_a.exists() else 0,
            "size_bytes_b": agg_b.stat().st_size if agg_b.exists() else 0,
        }

        res = estimated_adapter.run_intersection(
            str(agg_a), str(agg_b), args.runs, timeout=args.timeout, extra_args=extra_args
        )
        entry["estimated"] = res

        results.append(entry)
        print(f"size_gb={size_gb}: done")

    payload = {
        "metadata": {
            "scenario": "microns_intersection_estimated",
            "query_type": "intersection",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "sizes": args.sizes,
            "source_root": str(source_root.resolve()),
            "grid_cell_size": args.grid_cell_size,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "overlap_max_iterations": args.overlap_max_iterations,
            "containment_max_iterations": args.containment_max_iterations,
            "hash_load_factor": args.hash_load_factor,
            "query_direction": args.query_direction,
            "approaches": ["estimated"],
            "shared_data_root": str(dirs["root"]),
            "split_rule": "alternating_50_50",
        },
        "results": results,
    }

    out = run_layout["results_json"]
    write_json(out, payload)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
