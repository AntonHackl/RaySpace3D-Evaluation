#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    canonical_nu_pair_paths,
    create_benchmark_run_layout,
    ensure_nu_pair_dataset,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.mesh_containment.adapters.cgal_adapter import CGALContainmentAdapter
from benchmarks.mesh_containment.adapters.raytracer_adapter import RaytracerContainmentAdapter


CGAL_BASE_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"
LEGACY_OVERLAP_RAW_DIR = REPO_ROOT / "benchmarks" / "mesh_overlap" / "data" / "raw"
DEFAULT_NU_COUNTS = [200, 400, 600, 800]


def main():
    parser = argparse.ArgumentParser(description="Nu scalability benchmark for mesh containment")
    parser.add_argument("--nu", type=int, nargs="+", default=DEFAULT_NU_COUNTS)
    parser.add_argument("--grid-resolution", type=int, default=20)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--include-overlap-pairs", action="store_true")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=["raytracer", "cgal"],
        choices=["raytracer", "cgal"],
    )
    parser.add_argument("--threads", type=int, default=None, help="CGAL threads")
    args = parser.parse_args()

    shared_dirs = get_shared_data_dirs("nu_scalability")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "containment_nu_scalability")

    raytracer = RaytracerContainmentAdapter(
        str(REPO_ROOT / "src" / "RaySpace3D"),
        preprocessed_dir=str(shared_dirs["preprocessed"]),
        timings_dir=str(shared_dirs["timings"]),
        grid_resolution=args.grid_resolution,
        warmup_runs=args.warmup_runs,
        include_overlap_pairs=args.include_overlap_pairs,
    )
    cgal = CGALContainmentAdapter(
        str(CGAL_BASE_DIR),
        preprocessed_dir=str(shared_dirs["preprocessed"]),
        threads=args.threads,
    )

    results = []
    for nu in args.nu:
        n_path, v_path = canonical_nu_pair_paths(shared_dirs["raw"], nu=nu)
        ensure_nu_pair_dataset(n_path, v_path, legacy_raw_dirs=[LEGACY_OVERLAP_RAW_DIR])

        for f_path in (n_path, v_path):
            if not raytracer.check_preprocessed(str(f_path)):
                raytracer.preprocess_from_source(str(f_path), str(f_path))

        row = {
            "nu": nu,
            "mesh1": str(v_path),
            "mesh2": str(n_path),
            "size_bytes1": v_path.stat().st_size if v_path.exists() else 0,
            "size_bytes2": n_path.stat().st_size if n_path.exists() else 0,
        }

        if "raytracer" in args.approaches:
            row["raytracer"] = raytracer.run_containment(str(v_path), str(n_path), args.runs, timeout=args.timeout)

        if "cgal" in args.approaches:
            row["cgal"] = cgal.run_containment(str(v_path), str(n_path), args.runs, timeout=args.timeout)

        results.append(row)
        print(f"nu={nu}: done")

    payload = {
        "metadata": {
            "scenario": "nu_scalability",
            "query_type": "containment",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "nu": args.nu,
            "grid_resolution": args.grid_resolution,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "include_overlap_pairs": args.include_overlap_pairs,
            "timeout_seconds": args.timeout,
            "approaches": args.approaches,
            "shared_data_root": str(shared_dirs["root"]),
        },
        "results": results,
    }

    out = Path(run_layout["results_json"])
    write_json(out, payload)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
