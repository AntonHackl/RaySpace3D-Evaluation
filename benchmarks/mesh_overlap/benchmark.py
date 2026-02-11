#!/usr/bin/env python3
import time
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from adapters import TDBaseAdapter, CGALAdapter, RaytracerAdapter
import numpy as np


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()

# Dataset configurations
# Format: (file1_dt, file2_dt, file1_source, file2_source)
# source files are used for RaySpace/CGAL preprocessing, dt files for TDBase queries
# If source is None, use the dt file
DATASETS = {
    "small": ("test_small_n_nv15_nu30_vs100_r30.dt", "test_small_v_nv15_nu30_vs100_r30.dt", None, None),
    "medium": ("medium_n_nv150_nu200_vs100_r30.dt", "medium_v_nv150_nu200_vs100_r30.dt", None, None),
    "nu200": ("nu200_n_nv150_nu200_vs100_r30.dt", "nu200_v_nv150_nu200_vs100_r30.dt", None, None),
    "nu400": ("nu400_n_nv150_nu400_vs100_r30.dt", "nu400_v_nv150_nu400_vs100_r30.dt", None, None),
    "nuclei_join": ("medium_n_nv150_nu200_vs100_r30.dt", "medium_n2_n_nv150_nu200_vs100_r30.dt", None, None),
    "tdbase_intersect": ("tdbase_intersect_n_nv50_nu200_vs100_r30.dt", "tdbase_intersect_v_nv50_nu200_vs100_r30.dt", None, None),
    "cubes_100k": ("cubes_100k.dt", "cubes_100k_v2.dt", "cubes_100k.obj", "cubes_100k_v2.obj"),
    # Generated via RaySpace3D/scripts/generate_cubes_by_selectivity.py
    # Targeted ~0.2% selectivity under the raytracer overlap query (empirically ~0.21%).
    "cubes_100k_sel02": ("cubes_100k_s002_real.obj", "cubes_100k_s002_real_v2.obj", "cubes_100k_s002_real.obj", "cubes_100k_s002_real_v2.obj"),
    "cubes_1m": ("cubes_1m.obj", "cubes_1m_v2.obj", "cubes_1m.obj", "cubes_1m_v2.obj"),
    # Generated via RaySpace3D/scripts/generate_cubes_by_selectivity.py (streaming mode)
    # Targeted ~0.5% selectivity (0.005)
    "cubes_1m_sel05": ("cubes_1m_s005.obj", "cubes_1m_s005_v2.obj", "cubes_1m_s005.obj", "cubes_1m_s005_v2.obj"),
}
DEFAULT_DATASET = "small"

# Paths to systems (relative to workspace root or this script)
# Script is in mesh_overlap_benchmark/
# Workspace root is ..
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
TDBASE_DIR = WORKSPACE_ROOT / "tdbase"
CGAL_BASE_DIR = WORKSPACE_ROOT / "RaySpace3DBaslines/CGAL" 
RAYSPACE_DIR = WORKSPACE_ROOT / "RaySpace3D"
DATA_DIR = WORKSPACE_ROOT / "mesh_overlap_benchmark" / "data"
RAW_DIR = DATA_DIR / "raw" 
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
RUNS_DIR = WORKSPACE_ROOT / "mesh_overlap_benchmark" / "runs"

def print_results(adapter_name, results):
    if "error" in results:
        print(f"[{adapter_name}] Failed: {results['error']}")
    else:
        print(f"[{adapter_name}] Results ({len(results['raw_times'])} runs):")
        print(f"  Mean: {results['mean']:.4f} ms")
        print(f"  Min:  {results['min']:.4f} ms")
        print(f"  Max:  {results['max']:.4f} ms")
        print(f"  Std:  {results['std']:.4f} ms")

def main():
    parser = argparse.ArgumentParser(description="Mesh Overlap Benchmark")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per adapter")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, 
                        choices=list(DATASETS.keys()),
                        help=f"Dataset configuration to use: {', '.join(DATASETS.keys())}")
    parser.add_argument("--approaches", type=str, nargs="+", default=["cgal", "tdbase", "raytracer_exact", "raytracer_estimated"],
                        choices=["cgal", "tdbase", "raytracer_exact", "raytracer_estimated", "raytracer_estimate_only"],
                        help="Approaches to run (default: all)")
    parser.add_argument("--file1", type=str, default=None, help="First dataset file (overrides --dataset)")
    parser.add_argument("--file2", type=str, default=None, help="Second dataset file (overrides --dataset)")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR), help="Directory containing raw data files (default: mesh_overlap_benchmark/data/raw)")
    parser.add_argument("--preprocessed-dir", type=str, default=str(PREPROCESSED_DIR), help="Directory for preprocessed files (default: mesh_overlap_benchmark/data/preprocessed)")
    parser.add_argument("--timings-dir", type=str, default=str(TIMINGS_DIR), help="Directory for timing JSON files (default: mesh_overlap_benchmark/data/timings)")
    parser.add_argument("--grid-resolution", type=int, default=10, help="Grid resolution for RaySpace preprocessing (default: 10)")
    parser.add_argument("--raytracer-warmup-runs", type=int, default=1, help="Warmup iterations per raytracer invocation (default: 1; set 0 to disable)")
    parser.add_argument("--timeout", type=float, default=120.0, help="Timeout for query execution in seconds (default: 120.0)")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads for parallel approaches (default: all available)")
    parser.add_argument("--log-dir", type=str, default=str(RUNS_DIR / "logs"), help="Directory to write run logs (default: mesh_overlap_benchmark/runs/logs)")
    parser.add_argument("--no-logs", action="store_true", help="Disable writing benchmark/adapters logs to files")
    
    args = parser.parse_args()

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    raw_dir = Path(args.raw_dir)
    preprocessed_dir = Path(args.preprocessed_dir)
    timings_dir = Path(args.timings_dir)
    
    # Determine which files to use
    if args.file1 and args.file2:
        file1_path = raw_dir / args.file1
        file2_path = raw_dir / args.file2
        # For custom files, use the same files for preprocessing
        file1_source = file1_path
        file2_source = file2_path
    else:
        file1, file2, src1, src2 = DATASETS[args.dataset]
        file1_path = raw_dir / file1
        file2_path = raw_dir / file2
        # Use source files for preprocessing if specified, otherwise use dt files
        file1_source = raw_dir / src1 if src1 else file1_path
        file2_source = raw_dir / src2 if src2 else file2_path
    
    # Check if input files exist
    if not file1_path.exists():
        print(f"Warning: File not found: {file1_path}")
    
    if not file2_path.exists():
        print(f"Warning: File not found: {file2_path}")

    # Ensure directories exist
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    timings_dir.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Prepare run output names early so we can log the whole run
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
        # Initialize requested adapters
        adapters = []
        if "cgal" in args.approaches:
            adapters.append(CGALAdapter(str(CGAL_BASE_DIR), preprocessed_dir=str(preprocessed_dir), threads=args.threads))
        if "tdbase" in args.approaches:
            adapters.append(TDBaseAdapter(str(TDBASE_DIR)))
        if "raytracer_exact" in args.approaches:
            adapters.append(RaytracerAdapter(str(RAYSPACE_DIR), mode="exact", preprocessed_dir=str(preprocessed_dir), timings_dir=str(timings_dir), grid_resolution=args.grid_resolution, warmup_runs=args.raytracer_warmup_runs))
        if "raytracer_estimated" in args.approaches:
            adapters.append(RaytracerAdapter(str(RAYSPACE_DIR), mode="estimated", preprocessed_dir=str(preprocessed_dir), timings_dir=str(timings_dir), grid_resolution=args.grid_resolution, warmup_runs=args.raytracer_warmup_runs))
        if "raytracer_estimate_only" in args.approaches:
            adapters.append(RaytracerAdapter(str(RAYSPACE_DIR), mode="estimate_only", preprocessed_dir=str(preprocessed_dir), timings_dir=str(timings_dir), grid_resolution=args.grid_resolution, warmup_runs=args.raytracer_warmup_runs))

        all_results = {}
        ssot_stats = {"num_obj1": 0, "num_obj2": 0, "num_intersections": 0}
        ssot_requested = ("raytracer_exact" in args.approaches) or ("raytracer_estimated" in args.approaches)

        # Preprocessing check for Raytracer and CGAL (use source files)
        rt_adapter = next((a for a in adapters if isinstance(a, RaytracerAdapter)), None)
        if rt_adapter:
            print("\n--- Ensuring datasets are preprocessed ---")
            for f_dt, f_src in [(file1_path, file1_source), (file2_path, file2_source)]:
                # Check if preprocessing exists based on dt filename
                if not rt_adapter.check_preprocessed(str(f_dt)):
                    # Preprocess using source file
                    rt_adapter.preprocess_from_source(str(f_src), str(f_dt), log_dir=str(run_log_dir) if run_log_dir else None)
                else:
                    print(f"Dataset already preprocessed: {f_dt.name}")

        print(f"\n--- Running Benchmark (Runs: {args.runs}) ---")
        print(f"Dataset 1: {file1_path}")
        print(f"Dataset 2: {file2_path}")

        for adapter in adapters:
            print(f"\nRunning {adapter.name}...")
            results = adapter.run_overlap(
                str(file1_path),
                str(file2_path),
                args.runs,
                timeout=args.timeout,
                log_dir=str(run_log_dir) if run_log_dir else None,
            )
            print_results(adapter.name, results)
            all_results[adapter.name] = results
            
            # Capture SSOT stats if this is a raytracer
            if "Raytracer" in adapter.name and "error" not in results:
                # Prefer exact if available, or if this is the first one we find
                if adapter.name == "Raytracer_exact" or ssot_stats["num_obj1"] == 0:
                    ssot_stats["num_obj1"] = results.get("num_obj1", 0)
                    ssot_stats["num_obj2"] = results.get("num_obj2", 0)
                    ssot_stats["num_intersections"] = results.get("num_intersections", 0)

        # If no raytracer SSOT was run, but we need SSOT, run it once.
        # Note: estimate-only mode intentionally does NOT compute SSOT.
        if ssot_requested and ssot_stats["num_obj1"] == 0:
            print("\n--- Collecting Ground Truth Stats from Raytracer (SSOT) ---")
            temp_rt = RaytracerAdapter(str(RAYSPACE_DIR), mode="exact", preprocessed_dir=str(preprocessed_dir), timings_dir=str(timings_dir), warmup_runs=args.raytracer_warmup_runs)
            # Ensure preprocessed for SSOT run
            for f_dt, f_src in [(file1_path, file1_source), (file2_path, file2_source)]:
                if not temp_rt.check_preprocessed(str(f_dt)):
                    temp_rt.preprocess_from_source(str(f_src), str(f_dt), log_dir=str(run_log_dir) if run_log_dir else None)
            
            gt_res = temp_rt.run_overlap(
                str(file1_path),
                str(file2_path),
                1,
                timeout=args.timeout,
                log_dir=str(run_log_dir) if run_log_dir else None,
            )
            if "error" not in gt_res:
                ssot_stats["num_obj1"] = gt_res.get("num_obj1", 0)
                ssot_stats["num_obj2"] = gt_res.get("num_obj2", 0)
                ssot_stats["num_intersections"] = gt_res.get("num_intersections", 0)

        cross_product_size = ssot_stats["num_obj1"] * ssot_stats["num_obj2"]
        selectivity = ssot_stats["num_intersections"] / cross_product_size if cross_product_size > 0 else 0.0

        if ssot_requested:
            print(f"\n--- Join Statistics (SSOT) ---")
            print(f"  Objects 1:     {ssot_stats['num_obj1']}")
            print(f"  Objects 2:     {ssot_stats['num_obj2']}")
            print(f"  Cross Product: {cross_product_size}")
            print(f"  Intersections: {ssot_stats['num_intersections']}")
            print(f"  Selectivity:   {selectivity:.8f}")
        else:
            print("\n--- Join Statistics (SSOT) ---")
            print("  SSOT not computed (no exact/estimated join was requested).")

        # Save results to runs directory with timestamp
        output_file = RUNS_DIR / f"{run_name}.json"

        # Convert numpy types to native python for json serialization
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
                "cross_product_size": int(cross_product_size),
                "num_intersections": int(ssot_stats["num_intersections"]),
                "selectivity": float(selectivity)
            },
            "results": {}
        }

        for name, res in all_results.items():
            if "error" in res:
                json_results["results"][name] = res
            else:
                json_results["results"][name] = {
                    k: (
                        float(v)
                        if isinstance(v, (np.floating, float))
                        else [float(x) for x in v]
                        if isinstance(v, list)
                        else v
                    )
                    for k, v in res.items()
                }

        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=4)
        print(f"\nResults saved to {output_file}")
    finally:
        # Restore streams before closing the tee log file (prevents flush-on-exit issues)
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        if tee_file_handle is not None:
            tee_file_handle.flush()
            tee_file_handle.close()

if __name__ == "__main__":
    main()
