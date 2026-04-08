#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from benchmarks.mesh_containment.adapters import (
    CGALContainmentAdapter,
    RaytracerContainmentAdapter,
)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


DATASETS = {
    "validation": ("validation_a.obj", "validation_b.obj"),
    "cubes_100k": ("cubes_100k.obj", "cubes_100k_v2.obj"),
    "cubes_100k_sel02": ("cubes_100k_s002_real.obj", "cubes_100k_s002_real_v2.obj"),
}
DEFAULT_DATASET = "validation"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
CGAL_BASE_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/CGAL"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
RUNS_DIR = SCRIPT_DIR / "runs"


def print_results(adapter_name, results):
    if "error" in results:
        print(f"[{adapter_name}] Failed: {results['error']}")
    else:
        print(f"[{adapter_name}] Results ({len(results['raw_times'])} runs):")
        print(f"  Mean: {results['mean']:.4f} ms")
        print(f"  Min:  {results['min']:.4f} ms")
        print(f"  Max:  {results['max']:.4f} ms")
        print(f"  Std:  {results['std']:.4f} ms")
        if "num_containments" in results:
            print(f"  Containment Pairs: {results['num_containments']}")


def main():
    parser = argparse.ArgumentParser(description="Mesh Containment Benchmark")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per adapter")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        choices=list(DATASETS.keys()),
        help=f"Dataset configuration: {', '.join(DATASETS.keys())}",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        default=["raytracer", "cgal"],
        choices=["raytracer", "cgal"],
        help="Approaches to run",
    )
    parser.add_argument("--file1", type=str, default=None, help="First dataset file (overrides --dataset)")
    parser.add_argument("--file2", type=str, default=None, help="Second dataset file (overrides --dataset)")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR), help="Directory containing raw data files")
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=str(PREPROCESSED_DIR),
        help="Directory for preprocessed files",
    )
    parser.add_argument("--timings-dir", type=str, default=str(TIMINGS_DIR), help="Directory for timing JSON files")
    parser.add_argument("--grid-resolution", type=int, default=10, help="Grid resolution for preprocessing")
    parser.add_argument("--raytracer-warmup-runs", type=int, default=1, help="Warmup iterations per run")
    parser.add_argument(
        "--include-overlap-pairs",
        action="store_true",
        help="Include overlap/touch pairs in Raytracer containment output (union of overlap + strict containment)",
    )
    parser.add_argument("--cgal-dir", type=str, default=str(CGAL_BASE_DIR), help="Path to CGAL baseline directory")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads for CGAL")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout per run in seconds")
    parser.add_argument("--log-dir", type=str, default=str(RUNS_DIR / "logs"), help="Directory to write run logs")
    parser.add_argument("--no-logs", action="store_true", help="Disable writing logs to files")

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    preprocessed_dir = Path(args.preprocessed_dir)
    timings_dir = Path(args.timings_dir)

    if args.file1 and args.file2:
        file1_path = raw_dir / args.file1
        file2_path = raw_dir / args.file2
        file1_source = file1_path
        file2_source = file2_path
    else:
        file1, file2 = DATASETS[args.dataset]
        file1_path = raw_dir / file1
        file2_path = raw_dir / file2
        file1_source = file1_path
        file2_source = file2_path

    if not file1_path.exists():
        print(f"Warning: File not found: {file1_path}")
    if not file2_path.exists():
        print(f"Warning: File not found: {file2_path}")

    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    timings_dir.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.dataset}_{args.runs}runs_{timestamp}"

    run_log_dir = None
    benchmark_log_file = None
    tee_file_handle = None

    if not args.no_logs:
        run_log_dir = Path(args.log_dir) / run_name
        run_log_dir.mkdir(parents=True, exist_ok=True)
        benchmark_log_file = run_log_dir / "benchmark.log"
        tee_file_handle = open(benchmark_log_file, "w", encoding="utf-8")
        sys.stdout = _Tee(sys.stdout, tee_file_handle)
        sys.stderr = _Tee(sys.stderr, tee_file_handle)

    try:
        adapters = []
        if "raytracer" in args.approaches:
            adapters.append(
                RaytracerContainmentAdapter(
                    str(RAYSPACE_DIR),
                    preprocessed_dir=str(preprocessed_dir),
                    timings_dir=str(timings_dir),
                    grid_resolution=args.grid_resolution,
                    warmup_runs=args.raytracer_warmup_runs,
                    include_overlap_pairs=args.include_overlap_pairs,
                )
            )
        if "cgal" in args.approaches:
            adapters.append(
                CGALContainmentAdapter(
                    str(args.cgal_dir),
                    preprocessed_dir=str(preprocessed_dir),
                    threads=args.threads,
                )
            )

        needs_preprocess = any(a in args.approaches for a in ["raytracer", "cgal"])
        if needs_preprocess:
            preprocess_adapter = RaytracerContainmentAdapter(
                str(RAYSPACE_DIR),
                preprocessed_dir=str(preprocessed_dir),
                timings_dir=str(timings_dir),
                grid_resolution=args.grid_resolution,
                warmup_runs=0,
                include_overlap_pairs=args.include_overlap_pairs,
            )
            print("\n--- Ensuring datasets are preprocessed ---")
            for f_dt, f_src in [(file1_path, file1_source), (file2_path, file2_source)]:
                if not preprocess_adapter.check_preprocessed(str(f_dt)):
                    preprocess_adapter.preprocess_from_source(
                        str(f_src),
                        str(f_dt),
                        log_dir=str(run_log_dir) if run_log_dir else None,
                    )
                else:
                    print(f"Dataset already preprocessed: {f_dt.name}")

        print(f"\n--- Running Benchmark (Runs: {args.runs}) ---")
        print(f"Dataset 1: {file1_path}")
        print(f"Dataset 2: {file2_path}")

        all_results = {}
        ssot_stats = {"num_obj1": 0, "num_obj2": 0, "num_containments": 0}

        for adapter in adapters:
            print(f"\nRunning {adapter.name}...")
            results = adapter.run_containment(
                str(file1_path),
                str(file2_path),
                args.runs,
                timeout=args.timeout,
                log_dir=str(run_log_dir) if run_log_dir else None,
            )
            print_results(adapter.name, results)
            all_results[adapter.name] = results

            if adapter.name == "Raytracer" and "error" not in results:
                ssot_stats["num_obj1"] = results.get("num_obj1", 0)
                ssot_stats["num_obj2"] = results.get("num_obj2", 0)
                ssot_stats["num_containments"] = results.get("num_containments", 0)

        if ssot_stats["num_obj1"] > 0 and ssot_stats["num_obj2"] > 0:
            cross_product_size = ssot_stats["num_obj1"] * ssot_stats["num_obj2"]
            selectivity = ssot_stats["num_containments"] / cross_product_size if cross_product_size > 0 else 0.0
            print("\n--- Containment Statistics (SSOT) ---")
            print(f"  Objects 1:      {ssot_stats['num_obj1']}")
            print(f"  Objects 2:      {ssot_stats['num_obj2']}")
            print(f"  Cross Product:  {cross_product_size}")
            print(f"  Containments:   {ssot_stats['num_containments']}")
            print(f"  Selectivity:    {selectivity:.8f}")
        else:
            cross_product_size = 0
            selectivity = 0.0

        output_file = RUNS_DIR / f"{run_name}.json"
        json_results = {
            "metadata": {
                "timestamp": timestamp,
                "dataset": args.dataset,
                "file1": file1_path.name,
                "file2": file2_path.name,
                "num_runs": args.runs,
                "run_name": run_name,
                "log_dir": str(run_log_dir) if run_log_dir else None,
                "benchmark_log": str(benchmark_log_file) if benchmark_log_file else None,
                "num_obj1": int(ssot_stats["num_obj1"]),
                "num_obj2": int(ssot_stats["num_obj2"]),
                "size_bytes1": file1_path.stat().st_size if file1_path.exists() else 0,
                "size_bytes2": file2_path.stat().st_size if file2_path.exists() else 0,
                "cross_product_size": int(cross_product_size),
                "num_containments": int(ssot_stats["num_containments"]),
                "selectivity": float(selectivity),
            },
            "results": all_results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(json_results, f, indent=2)

        print(f"\nSaved results to {output_file}")
    finally:
        if tee_file_handle:
            tee_file_handle.flush()
            tee_file_handle.close()
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
