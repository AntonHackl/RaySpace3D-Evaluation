"""
Hash Contention Benchmark for Direct Estimation Overlap Query.

Sweeps over hash table sizes expressed as multiples of the true result count,
measuring query time and contention for each size.

First, a baseline run is executed with a configurable fraction of currently
free GPU memory (default: 90%) for the hash table.

Then, for each multiplier (default: 10x, 5x, 2x, 1.5x, 1x, 0.8x):
  - Timing run  : N measured runs, no contention tracking
  - Contention run: 1 run, contention tracking enabled

Saves a JSON results file and optionally auto-launches the visualization.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json

# ---------------------------------------------------------------------------
# Repo-relative constants (identical convention to benchmark.py)
# ---------------------------------------------------------------------------
SCRIPT_DIR  = Path(__file__).resolve().parent
REPO_ROOT   = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src" / "RaySpace3D"
DATA_DIR    = SCRIPT_DIR / "data"
RAW_DIR     = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
RUNS_DIR    = SCRIPT_DIR / "runs"
GENERATE_CUBES_SCRIPT = RAYSPACE_DIR / "scripts" / "generate_cubes_by_selectivity.py"
BUILD_SCRIPT = REPO_ROOT / "build_all.sh"

# Import the adapter (repo package layout)
sys.path.insert(0, str(REPO_ROOT))
from benchmarks.mesh_overlap.adapters.raytracer_adapter import RaytracerAdapter

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Hash contention benchmark for the direct estimation overlap query",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-cubes",      type=int,   default=100_000,
                        help="Number of cubes in each dataset")
    parser.add_argument("--selectivity",    type=float, default=0.001,
                        help="Target pair selectivity for dataset generation")
    parser.add_argument("--min-size",       type=float, default=1.0,
                        help="Minimum cube edge length")
    parser.add_argument("--max-size",       type=float, default=2.0,
                        help="Maximum cube edge length")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Random seed for dataset generation")
    parser.add_argument("--timing-runs",    type=int,   default=5,
                        help="Number of measured timing runs per hash-table setting")
    parser.add_argument("--timeout",        type=float, default=120.0,
                        help="Per-query timeout in seconds for direct estimation runs")
    parser.add_argument("--warmup-runs",    type=int,   default=2,
                        help="Number of warmup runs inside the binary")
    parser.add_argument("--grid-resolution",type=int,   default=10,
                        help="Grid resolution for preprocessing")
    parser.add_argument("--multipliers",    type=float, nargs="+",
                        default=[100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.5, 1.0, 0.8],
                        help="Hash table size multipliers (relative to true result count)")
    parser.add_argument("--gpu-auto-free-mem-fraction", type=float, default=0.9,
                        help="Fraction of currently free GPU memory to use for the gpu_auto hash-table step")
    parser.add_argument("--output-dir",     type=str,   default=str(RUNS_DIR),
                        help="Directory for output JSON files")
    parser.add_argument("--skip-rebuild",   action="store_true",
                        help="Skip rebuilding the binary")
    parser.add_argument("--skip-generate",  action="store_true",
                        help="Skip dataset generation if OBJ files already exist")
    parser.add_argument("--skip-preprocess",action="store_true",
                        help="Skip preprocessing if .pre files already exist")
    parser.add_argument("--no-visualize",   action="store_true",
                        help="Do not automatically launch visualization")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd, desc):
    """Run a subprocess command, stream output, raise on failure."""
    print(f"\n>>> {desc}")
    print("    " + " ".join(str(c) for c in cmd))
    result = subprocess.run([str(c) for c in cmd], check=True)
    return result


def make_adapter(rayspace_dir, preprocessed_dir, timings_dir, grid_resolution,
                 warmup_runs, hash_table_size=None, track_hash_contention=False,
                 hash_table_free_mem_fraction=0.9):
    return RaytracerAdapter(
        rayspace_dir=str(rayspace_dir),
        mode="direct_estimation",
        preprocessed_dir=str(preprocessed_dir),
        timings_dir=str(timings_dir),
        grid_resolution=grid_resolution,
        warmup_runs=warmup_runs,
        track_hash_contention=track_hash_contention,
        hash_table_size=hash_table_size,
        hash_table_free_mem_fraction=hash_table_free_mem_fraction,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_hash_contention")

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Rebuild ------------------------------------------------
    if not args.skip_rebuild:
        run_cmd(
            [str(BUILD_SCRIPT), "--only", "query"],
            "Rebuilding query binary",
        )
    else:
        print("[skip] Rebuild step skipped.")

    # ---- Step 2: Generate datasets --------------------------------------
    dataset_a = RAW_DIR / "cubes_100k_contention_a.obj"
    dataset_b = RAW_DIR / "cubes_100k_contention_b.obj"

    if args.skip_generate and dataset_a.exists() and dataset_b.exists():
        print(f"[skip] Dataset files already exist:\n  {dataset_a}\n  {dataset_b}")
    else:
        run_cmd(
            [
                sys.executable,
                str(GENERATE_CUBES_SCRIPT),
                "--num-cubes-a", str(args.num_cubes),
                "--num-cubes-b", str(args.num_cubes),
                "--min-size",    str(args.min_size),
                "--max-size",    str(args.max_size),
                "--selectivity", str(args.selectivity),
                "--output-a",    str(dataset_a),
                "--output-b",    str(dataset_b),
                "--seed",        str(args.seed),
            ],
            f"Generating {args.num_cubes:,} cube datasets at selectivity={args.selectivity}",
        )

    # ---- Step 3: Preprocess ---------------------------------------------
    pre_a = PREPROCESSED_DIR / "cubes_100k_contention_a.pre"
    pre_b = PREPROCESSED_DIR / "cubes_100k_contention_b.pre"

    if args.skip_preprocess and pre_a.exists() and pre_b.exists():
        print(f"[skip] Preprocessed files already exist:\n  {pre_a}\n  {pre_b}")
    else:
        adapter_pre = make_adapter(
            RAYSPACE_DIR, PREPROCESSED_DIR, TIMINGS_DIR, args.grid_resolution, warmup_runs=0
        )
        print("\n>>> Preprocessing dataset A ...")
        adapter_pre.preprocess_from_source(str(dataset_a), str(dataset_a))
        print("\n>>> Preprocessing dataset B ...")
        adapter_pre.preprocess_from_source(str(dataset_b), str(dataset_b))

    # ---- Step 4: Discovery run (find true result count) -----------------
    print("\n" + "=" * 60)
    print("STEP 4: Discovery run to find true result count")
    print("=" * 60)
    adapter_discovery = make_adapter(
        RAYSPACE_DIR, PREPROCESSED_DIR, TIMINGS_DIR,
        args.grid_resolution, warmup_runs=args.warmup_runs,
    )
    discovery_result = adapter_discovery.run_overlap(
        str(dataset_a), str(dataset_b), num_runs=1, timeout=args.timeout
    )
    if "error" in discovery_result:
        print(f"ERROR in discovery run: {discovery_result['error']}", file=sys.stderr)
        sys.exit(1)

    true_result_count = discovery_result["num_intersections"]
    print(f"\nTrue result count: {true_result_count:,} pairs")

    if true_result_count <= 0:
        print("WARNING: True result count is 0. Hash table multipliers will use minimum size of 1024.")

    # ---- Step 5: Hash-size sweep ----------------------------------------
    results = []

    sweep_configs = [{
        "kind": "gpu_auto",
        "label": f"GPU {args.gpu_auto_free_mem_fraction * 100:.0f}% free memory",
        "multiplier": None,
        "hash_size": None,
        "hash_table_free_mem_fraction": args.gpu_auto_free_mem_fraction,
    }]
    for multiplier in args.multipliers:
        sweep_configs.append({
            "kind": "multiplier",
            "label": f"{multiplier:.1f}x",
            "multiplier": multiplier,
            "hash_size": max(1024, int(true_result_count * multiplier)) if true_result_count > 0 else 1024,
            "hash_table_free_mem_fraction": None,
        })

    for cfg in sweep_configs:
        hash_size = cfg["hash_size"]
        print("\n" + "=" * 60)
        if cfg["kind"] == "gpu_auto":
            print(f"Hash size mode: {cfg['label']} (auto)")
        else:
            print(f"Multiplier: {cfg['multiplier']:.1f}x  →  hash_table_size = {hash_size:,}")
        print("=" * 60)

        # --- Timing run (no contention tracking) ---
        print(f"  [timing] {args.timing_runs} runs, no contention tracking ...")
        adapter_timing = make_adapter(
            RAYSPACE_DIR, PREPROCESSED_DIR, TIMINGS_DIR,
            args.grid_resolution, warmup_runs=args.warmup_runs,
            hash_table_size=hash_size,
            track_hash_contention=False,
            hash_table_free_mem_fraction=cfg["hash_table_free_mem_fraction"],
        )
        timing_result = adapter_timing.run_overlap(
            str(dataset_a), str(dataset_b), num_runs=args.timing_runs, timeout=args.timeout
        )
        if "error" in timing_result:
            print(f"  ERROR in timing run: {timing_result['error']}", file=sys.stderr)
            timing_result = {
                "mean": None, "std": None, "raw_times": [],
                "num_intersections": None, "actual_hash_table_size": hash_size or 0,
            }

        # --- Contention run (1 run, tracking enabled) ---
        print(f"  [contention] 1 run, contention tracking ...")
        adapter_contention = make_adapter(
            RAYSPACE_DIR, PREPROCESSED_DIR, TIMINGS_DIR,
            args.grid_resolution, warmup_runs=args.warmup_runs,
            hash_table_size=hash_size,
            track_hash_contention=True,
            hash_table_free_mem_fraction=cfg["hash_table_free_mem_fraction"],
        )
        contention_result = adapter_contention.run_overlap(
            str(dataset_a), str(dataset_b), num_runs=1, timeout=args.timeout
        )
        if "error" in contention_result:
            print(f"  ERROR in contention run: {contention_result['error']}", file=sys.stderr)
            contention_result = {
                "hash_accesses": None, "hash_contentions": None,
                "contention_pct": None, "num_intersections": None,
                "actual_hash_table_size": hash_size or 0,
            }

        entry = {
            "size_kind":             cfg["kind"],
            "size_label":            cfg["label"],
            "multiplier":            cfg["multiplier"],
            "hash_table_size":       hash_size,
            "hash_table_free_mem_fraction": cfg["hash_table_free_mem_fraction"],
            "actual_hash_table_size_timing":    timing_result.get("actual_hash_table_size", hash_size or 0),
            "actual_hash_table_size_contention": contention_result.get("actual_hash_table_size", hash_size or 0),
            "mean_time_ms":          timing_result.get("mean"),
            "std_time_ms":           timing_result.get("std"),
            "raw_times_ms":          timing_result.get("raw_times", []),
            "pairs_found_timing":    timing_result.get("num_intersections"),
            "pairs_found_contention": contention_result.get("num_intersections"),
            "hash_accesses":         contention_result.get("hash_accesses"),
            "hash_contentions":      contention_result.get("hash_contentions"),
            "contention_pct":        contention_result.get("contention_pct"),
        }
        results.append(entry)

        effective_hash_size = entry["actual_hash_table_size_timing"] or entry["hash_table_size"]
        print(f"  effective hash table size = {effective_hash_size}")

        print(f"  mean_time_ms    = {entry['mean_time_ms']}")
        print(f"  pairs_found     = {entry['pairs_found_timing']}")
        print(f"  hash_accesses   = {entry['hash_accesses']}")
        print(f"  hash_contentions= {entry['hash_contentions']}")
        print(f"  contention_pct  = {entry['contention_pct']}")

    # ---- Step 6: Save JSON ----------------------------------------------
    output_json = Path(run_layout["results_json"])
    timestamp = run_layout["timestamp"]

    output = {
        "metadata": {
            "timestamp":       timestamp,
            "run_name":        run_layout["run_name"],
            "run_dir":         str(run_layout["run_dir"]),
            "num_cubes":       args.num_cubes,
            "selectivity":     args.selectivity,
            "min_size":        args.min_size,
            "max_size":        args.max_size,
            "seed":            args.seed,
            "true_result_count": true_result_count,
            "timing_runs":     args.timing_runs,
            "timeout_seconds": args.timeout,
            "warmup_runs":     args.warmup_runs,
            "grid_resolution": args.grid_resolution,
            "multipliers":     args.multipliers,
            "gpu_auto_free_mem_fraction": args.gpu_auto_free_mem_fraction,
            "includes_gpu_auto_step": True,
            "dataset_a":       str(dataset_a),
            "dataset_b":       str(dataset_b),
        },
        "results": results,
    }

    write_json(output_json, output)
    # Keep legacy output-dir alias for compatibility.
    legacy_latest = output_dir / "hash_contention_benchmark_latest.json"
    write_json(legacy_latest, output)
    print(f"\nResults saved to: {output_json}")

    # ---- Step 7: Visualize ----------------------------------------------
    if not args.no_visualize:
        vis_script = SCRIPT_DIR / "visualize_hash_contention_benchmark.py"
        if vis_script.exists():
            try:
                run_cmd(
                    [sys.executable, str(vis_script), "--input", str(output_json)],
                    "Running visualization",
                )
            except subprocess.CalledProcessError as e:
                print(f"WARNING: Visualization failed: {e}")
        else:
            print(f"[skip] Visualization script not found: {vis_script}")

    print("\nDone.")


if __name__ == "__main__":
    main()
