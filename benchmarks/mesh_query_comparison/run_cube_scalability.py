#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    canonical_cube_pair_paths,
    create_benchmark_run_layout,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.mesh_query_comparison.core import (
    add_query_selection_arguments,
    build_intersection_extra_args,
    build_raytracer_query_adapters,
    ensure_preprocessed,
    generate_query_comparison_figures,
    resolve_queries,
    sanitize_case_token,
    run_selected_queries,
)


DEFAULT_CUBE_COUNTS = [200000, 400000, 600000, 1000000]


def main():
    parser = argparse.ArgumentParser(description="Cube scalability benchmark for mesh query comparison")
    parser.add_argument("--num-cubes-b", type=int, nargs="+", default=DEFAULT_CUBE_COUNTS)
    parser.add_argument("--num-cubes-a", type=int, default=1000)
    parser.add_argument("--min-size", type=float, default=1.0)
    parser.add_argument("--max-size", type=float, default=2.0)
    parser.add_argument("--selectivity", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-cell-size", type=float, default=1.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)

    add_query_selection_arguments(parser)

    parser.add_argument("--overlap-mode", type=str, default="direct_estimation", choices=["direct_estimation"])
    parser.add_argument("--intersection-mode", type=str, default="estimated", choices=["estimated", "estimate_only"])
    parser.add_argument("--overlap-query-direction", type=str, default="both", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"])
    parser.add_argument("--intersection-query-direction", type=str, default="both", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"])
    parser.add_argument("--overlap-max-iterations", type=float, default=100.0)
    parser.add_argument("--containment-max-iterations", type=int, default=512)
    parser.add_argument("--hash-load-factor", type=float, default=0.5)
    parser.add_argument("--enable-profiling-stats", action="store_true")
    parser.add_argument("--use-anyhit-point-in-mesh", action="store_true")
    parser.add_argument("--include-overlap-pairs", action="store_true")
    args = parser.parse_args()

    queries = resolve_queries(args.queries, args.approaches)

    shared_dirs = get_shared_data_dirs("cube_scalability")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "query_comparison_cube_scalability")
    logs_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])

    adapters = build_raytracer_query_adapters(
        repo_root=REPO_ROOT,
        shared_dirs=shared_dirs,
        grid_cell_size=args.grid_cell_size,
        warmup_runs=args.warmup_runs,
        overlap_mode=args.overlap_mode,
        intersection_mode=args.intersection_mode,
        include_overlap_pairs=args.include_overlap_pairs,
        use_anyhit_point_in_mesh=args.use_anyhit_point_in_mesh,
        overlap_max_iterations=int(args.overlap_max_iterations),
    )

    intersection_extra_args = build_intersection_extra_args(
        overlap_max_iterations=args.overlap_max_iterations,
        containment_max_iterations=args.containment_max_iterations,
        hash_load_factor=args.hash_load_factor,
        enable_profiling_stats=args.enable_profiling_stats,
        intersection_query_direction=args.intersection_query_direction,
    )

    results = []
    case_labels = []
    for num_cubes_b in args.num_cubes_b:
        mesh_a, mesh_b = canonical_cube_pair_paths(
            shared_dirs["raw"],
            num_cubes_a=args.num_cubes_a,
            num_cubes_b=num_cubes_b,
            min_size=args.min_size,
            max_size=args.max_size,
            selectivity=args.selectivity,
            seed=args.seed,
            grid_cell_size=args.grid_cell_size,
        )
        ensure_cube_pair_dataset(
            mesh_a,
            mesh_b,
            num_cubes_a=args.num_cubes_a,
            num_cubes_b=num_cubes_b,
            min_size=args.min_size,
            max_size=args.max_size,
            selectivity=args.selectivity,
            seed=args.seed,
        )

        case_label = f"na{args.num_cubes_a}_nb{num_cubes_b}"
        case_log_dir = logs_dir / sanitize_case_token(case_label)

        ensure_preprocessed(adapters, [mesh_a, mesh_b], log_dir=case_log_dir)

        row = {
            "num_cubes_a": args.num_cubes_a,
            "num_cubes_b": num_cubes_b,
            "mesh1": str(mesh_a),
            "mesh2": str(mesh_b),
            "size_bytes1": mesh_a.stat().st_size if mesh_a.exists() else 0,
            "size_bytes2": mesh_b.stat().st_size if mesh_b.exists() else 0,
            "selectivity": args.selectivity,
        }
        row.update(
            run_selected_queries(
                adapters=adapters,
                queries=queries,
                mesh1=mesh_a,
                mesh2=mesh_b,
                runs=args.runs,
                timeout=args.timeout,
                overlap_query_direction=args.overlap_query_direction,
                intersection_extra_args=intersection_extra_args,
                log_dir=case_log_dir,
            )
        )

        results.append(row)
        case_labels.append(case_label)
        print(f"num_cubes_b={num_cubes_b}: done")

    generate_query_comparison_figures(
        results_rows=results,
        queries=queries,
        case_labels=case_labels,
        figures_dir=figures_dir,
        title_prefix="Cube scalability",
        x_axis_label="Dataset case",
    )

    payload = {
        "metadata": {
            "scenario": "cube_scalability",
            "query_type": "mesh_query_comparison",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "num_cubes_a": args.num_cubes_a,
            "num_cubes_b": args.num_cubes_b,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "selectivity": args.selectivity,
            "seed": args.seed,
            "grid_cell_size": args.grid_cell_size,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "queries": queries,
            "query_implementations": {
                "overlap": f"raytracer_{args.overlap_mode}",
                "intersection": f"raytracer_{args.intersection_mode}",
                "containment": "raytracer_containment",
            },
            "intersection_query_direction": args.intersection_query_direction,
            "overlap_query_direction": args.overlap_query_direction,
            "overlap_max_iterations": args.overlap_max_iterations,
            "containment_max_iterations": args.containment_max_iterations,
            "hash_load_factor": args.hash_load_factor,
            "enable_profiling_stats": args.enable_profiling_stats,
            "use_anyhit_point_in_mesh": args.use_anyhit_point_in_mesh,
            "include_overlap_pairs": args.include_overlap_pairs,
            "shared_data_root": str(shared_dirs["root"]),
        },
        "results": results,
    }

    out = Path(run_layout["results_json"])
    write_json(out, payload)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
