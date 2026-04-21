#!/usr/bin/env python3
import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
MESH_QUERY_COMPARISON_DIR = REPO_ROOT / "benchmarks" / "mesh_query_comparison"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated. Use benchmarks/mesh_query_comparison/* runners instead. "
            "This wrapper forwards to the new benchmark family for nu/cube scenarios."
        )
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--grid-res", type=float, default=1.0)
    parser.add_argument("--dataset-set", choices=["nu", "cube"], default="nu")
    parser.add_argument("--intersection-query-direction", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"], default="both")
    parser.add_argument("--overlap-max-iterations", type=float, default=1.00)
    parser.add_argument("--containment-max-iterations", type=int, default=512)
    parser.add_argument("--hash-load-factor", type=float, default=0.5)
    parser.add_argument("--enable-profiling-stats", action="store_true")
    args = parser.parse_args()

    target_script = MESH_QUERY_COMPARISON_DIR / (
        "run_nu_scalability.py" if args.dataset_set == "nu" else "run_cube_scalability.py"
    )
    cmd = [
        sys.executable,
        str(target_script),
        "--runs",
        str(args.runs),
        "--grid-cell-size",
        str(args.grid_res),
        "--intersection-query-direction",
        args.intersection_query_direction,
        "--overlap-max-iterations",
        str(args.overlap_max_iterations),
        "--containment-max-iterations",
        str(args.containment_max_iterations),
        "--hash-load-factor",
        str(args.hash_load_factor),
    ]
    if args.enable_profiling_stats:
        cmd.append("--enable-profiling-stats")

    print("[DEPRECATED] intersection_vs_overlap is obsolete.")
    print("[DEPRECATED] Forwarding to mesh_query_comparison with the standard three-query default.")
    print("[DEPRECATED] For full coverage, use:")
    print("[DEPRECATED]   benchmarks/mesh_query_comparison/run_nu_scalability.py")
    print("[DEPRECATED]   benchmarks/mesh_query_comparison/run_cube_scalability.py")
    print("[DEPRECATED]   benchmarks/mesh_query_comparison/selectivity_test.py")
    print("[DEPRECATED]   benchmarks/mesh_query_comparison/run_mesh_complexity_benchmark.py")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
