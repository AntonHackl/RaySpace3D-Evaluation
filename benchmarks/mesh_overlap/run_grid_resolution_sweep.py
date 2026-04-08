"""
Grid-resolution sweep benchmark for overlap direct estimation.

Runs direct-estimation overlap query across multiple grid resolutions and captures:
- Query timing (mean/std/raw)
- Estimated pairs after replication correction
- Estimated pairs after load-factor application
- Hash table size estimate

Also computes a single exact-overlap ground truth count for factor-based comparison.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo-relative constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src" / "RaySpace3D"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
RUNS_DIR = SCRIPT_DIR / "runs"
GENERATE_CUBES_SCRIPT = RAYSPACE_DIR / "scripts" / "generate_cubes_by_selectivity.py"
BUILD_SCRIPT = REPO_ROOT / "build_all.sh"

sys.path.insert(0, str(REPO_ROOT))
from benchmarks.mesh_overlap.adapters.raytracer_adapter import RaytracerAdapter


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Grid-resolution sweep benchmark for overlap direct estimation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-cubes", type=int, default=100_000,
                        help="Number of cubes in each dataset")
    parser.add_argument("--cube-size", type=float, default=5.0,
                        help="Fixed cube edge length for both datasets")
    parser.add_argument("--universe", type=float, default=100.0,
                        help="Target cubic universe extent")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for dataset generation")
    parser.add_argument("--timing-runs", type=int, default=5,
                        help="Measured timing runs per grid resolution")
    parser.add_argument("--timeout", type=float, default=240.0,
                        help="Per-query timeout in seconds")
    parser.add_argument("--warmup-runs", type=int, default=2,
                        help="Warmup runs inside query binaries")
    parser.add_argument("--grid-resolutions", type=int, nargs="+",
                        default=[1, 5, 10, 20, 50, 100],
                        help="Grid resolutions to sweep")
    parser.add_argument("--groundtruth-grid-resolution", type=int, default=10,
                        help="Preprocessing grid resolution used for exact-overlap ground truth run")
    parser.add_argument("--output-dir", type=str, default=str(RUNS_DIR),
                        help="Directory for run outputs")
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="Skip rebuilding query binaries")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip dataset generation if OBJ files already exist")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing if expected .pre files already exist")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Run selectivity estimation only for sweep points (skip actual overlap query)")
    parser.add_argument("--no-alpha-correction", action="store_true",
                        help="Disable alpha correction in raytracer_overlap_direct_estimation")
    parser.add_argument("--hash-load-factor", type=float, default=0.5,
                        help="Load factor used to derive hash size from corrected estimate")
    parser.add_argument("--run-types", type=str, nargs="+", default=["cubes", "nu"],
                        choices=["cubes", "nu"],
                        help="Dataset run types to execute and merge into one result JSON")
    parser.add_argument("--cube-dataset-a", type=str, default="cubes_100k_size5_u100_a.obj",
                        help="Cube dataset A filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--cube-dataset-b", type=str, default="cubes_100k_size5_u100_b.obj",
                        help="Cube dataset B filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--nu-dataset-a", type=str, default="tdbase_n_nv150_nu800_n_nv150_nu800_vs100_r30.dt",
                        help="NU dataset A filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--nu-dataset-b", type=str, default="tdbase_n_nv150_nu800_v_nv150_nu800_vs100_r30.dt",
                        help="NU dataset B filename in benchmarks/mesh_overlap/data/raw")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_cmd(cmd, desc):
    print(f"\n>>> {desc}")
    print("    " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def make_adapter(
    mode,
    preprocessed_dir,
    timings_dir,
    grid_resolution,
    warmup_runs,
    use_alpha_correction=True,
    track_hash_contention=False,
):
    return RaytracerAdapter(
        rayspace_dir=str(RAYSPACE_DIR),
        mode=mode,
        preprocessed_dir=str(preprocessed_dir),
        timings_dir=str(timings_dir),
        grid_resolution=grid_resolution,
        warmup_runs=warmup_runs,
        use_alpha_correction=use_alpha_correction,
        track_hash_contention=track_hash_contention,
    )


def compute_selectivity_for_universe(cube_size, universe_extent):
    # P(overlap) ≈ (2s / U)^3 for equal-size cubes in a cubic universe.
    return ((2.0 * cube_size) / universe_extent) ** 3


def safe_div(num, denom):
    if denom and denom > 0:
        return float(num) / float(denom)
    return None


def compute_hash_table_size_from_estimate(estimated_pairs, load_factor):
    if estimated_pairs is None:
        return None

    if load_factor <= 0.0:
        raise ValueError("hash-load-factor must be > 0")

    target = int(float(estimated_pairs) / float(load_factor))
    if target < 1024:
        target = 1024
    if target > 2147483648:
        target = 2147483648
    if target % 2 == 0:
        target += 1
    return target


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(args.output_dir) / f"grid_resolution_sweep_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pre_root = run_dir / "preprocessed"
    timings_root = run_dir / "timings"
    logs_root = run_dir / "logs"
    figures_root = run_dir / "figures"
    for p in (pre_root, timings_root, logs_root, figures_root, RAW_DIR):
        p.mkdir(parents=True, exist_ok=True)

    if not args.skip_rebuild:
        run_cmd([str(BUILD_SCRIPT), "--only", "query"], "Rebuilding query binary")
    else:
        print("[skip] Rebuild step skipped.")

    selectivity = compute_selectivity_for_universe(args.cube_size, args.universe)

    def run_dataset_sweep(run_type, dataset_a, dataset_b, display_name):
        print("\n" + "#" * 80)
        print(f"RUN TYPE: {run_type} ({display_name})")
        print("#" * 80)

        if not dataset_a.exists() or not dataset_b.exists():
            missing = [str(p) for p in (dataset_a, dataset_b) if not p.exists()]
            print(f"ERROR: Missing dataset files for run_type={run_type}: {missing}", file=sys.stderr)
            sys.exit(1)

        if run_type == "cubes":
            if args.skip_generate and dataset_a.exists() and dataset_b.exists():
                print(f"[skip] Cube dataset files already exist:\n  {dataset_a}\n  {dataset_b}")
            else:
                run_cmd(
                    [
                        sys.executable,
                        str(GENERATE_CUBES_SCRIPT),
                        "--num-cubes-a", str(args.num_cubes),
                        "--num-cubes-b", str(args.num_cubes),
                        "--min-size", str(args.cube_size),
                        "--max-size", str(args.cube_size),
                        "--selectivity", str(selectivity),
                        "--output-a", str(dataset_a),
                        "--output-b", str(dataset_b),
                        "--seed", str(args.seed),
                    ],
                    (
                        f"Generating cube datasets ({args.num_cubes:,} cubes each, "
                        f"size={args.cube_size}, target universe={args.universe})"
                    ),
                )

        # Ground-truth run (exact overlap) once per dataset pair.
        print("\n" + "=" * 80)
        print(f"STEP: Computing exact-overlap ground truth ({run_type})")
        print("=" * 80)

        dataset_pre_root = pre_root / run_type
        dataset_timings_root = timings_root / run_type
        dataset_logs_root = logs_root / run_type
        dataset_pre_root.mkdir(parents=True, exist_ok=True)
        dataset_timings_root.mkdir(parents=True, exist_ok=True)
        dataset_logs_root.mkdir(parents=True, exist_ok=True)

        gt_pre_dir = dataset_pre_root / "groundtruth_exact"
        gt_timings_dir = dataset_timings_root / "groundtruth_exact"
        gt_logs_dir = dataset_logs_root / "groundtruth_exact"
        gt_adapter = make_adapter(
            mode="exact",
            preprocessed_dir=gt_pre_dir,
            timings_dir=gt_timings_dir,
            grid_resolution=args.groundtruth_grid_resolution,
            warmup_runs=args.warmup_runs,
            use_alpha_correction=True,
            track_hash_contention=False,
        )

        if not args.skip_preprocess:
            print("Preprocessing ground-truth dataset A...")
            gt_adapter.preprocess_from_source(str(dataset_a), str(dataset_a), log_dir=str(gt_logs_dir))
            print("Preprocessing ground-truth dataset B...")
            gt_adapter.preprocess_from_source(str(dataset_b), str(dataset_b), log_dir=str(gt_logs_dir))
        else:
            print("[skip] Ground-truth preprocessing skipped.")

        gt_result = gt_adapter.run_overlap(
            str(dataset_a),
            str(dataset_b),
            num_runs=1,
            timeout=args.timeout,
            log_dir=str(gt_logs_dir),
            estimate_only=False,
        )
        if "error" in gt_result:
            print(f"ERROR during ground-truth run ({run_type}): {gt_result['error']}", file=sys.stderr)
            sys.exit(1)

        ground_truth_pairs = int(gt_result.get("num_intersections", 0))
        print(f"Exact ground truth pairs ({run_type}): {ground_truth_pairs:,}")

        results = []
        for grid_res in args.grid_resolutions:
            print("\n" + "=" * 80)
            print(f"{run_type} | Grid resolution: {grid_res}")
            print("=" * 80)

            pre_dir = dataset_pre_root / f"grid_{grid_res}"
            timing_dir = dataset_timings_root / f"grid_{grid_res}"
            log_dir = dataset_logs_root / f"grid_{grid_res}"

            adapter_pre = make_adapter(
                mode="direct_estimation",
                preprocessed_dir=pre_dir,
                timings_dir=timing_dir,
                grid_resolution=grid_res,
                warmup_runs=0,
                use_alpha_correction=not args.no_alpha_correction,
                track_hash_contention=False,
            )

            pre_a = pre_dir / dataset_a.with_suffix(".pre").name
            pre_b = pre_dir / dataset_b.with_suffix(".pre").name
            should_preprocess = not (args.skip_preprocess and pre_a.exists() and pre_b.exists())

            if should_preprocess:
                print("Preprocessing dataset A...")
                adapter_pre.preprocess_from_source(str(dataset_a), str(dataset_a), log_dir=str(log_dir))
                print("Preprocessing dataset B...")
                adapter_pre.preprocess_from_source(str(dataset_b), str(dataset_b), log_dir=str(log_dir))
            else:
                print(f"[skip] Preprocessed files already exist for grid={grid_res}")

            print(f"[timing] running {args.timing_runs} measured runs ({'estimation only' if args.estimate_only else 'full query'})...")
            adapter_timing = make_adapter(
                mode="direct_estimation",
                preprocessed_dir=pre_dir,
                timings_dir=timing_dir,
                grid_resolution=grid_res,
                warmup_runs=args.warmup_runs,
                use_alpha_correction=not args.no_alpha_correction,
                track_hash_contention=False,
            )
            timing_result = adapter_timing.run_overlap(
                str(dataset_a),
                str(dataset_b),
                num_runs=args.timing_runs,
                timeout=args.timeout,
                log_dir=str(log_dir / "timing"),
                estimate_only=args.estimate_only,
            )

            metrics_result = timing_result

            entry = {
                "grid_resolution": grid_res,
                "timing": {
                    "mean_time_ms": None if "error" in timing_result else timing_result.get("mean"),
                    "std_time_ms": None if "error" in timing_result else timing_result.get("std"),
                    "raw_times_ms": [] if "error" in timing_result else timing_result.get("raw_times", []),
                    "error": timing_result.get("error"),
                },
                "pairs": {
                    "raw_estimated_pairs_before_replication_correction": None if "error" in metrics_result else metrics_result.get("raw_estimated_pairs"),
                    "raw_estimated_pairs_after_replication_correction": None if "error" in metrics_result else metrics_result.get("final_estimated_pairs"),
                    "estimated_pairs_after_load_factor_application": None,
                    "ground_truth_pairs": ground_truth_pairs,
                    "replication_corrected_vs_ground_truth_factor": None,
                    "load_factor_applied_vs_ground_truth_factor": None,
                },
                "hash": {
                    "estimated_hash_table_size": None,
                    "hash_load_factor": args.hash_load_factor,
                    "hash_accesses": None if "error" in metrics_result else metrics_result.get("hash_accesses"),
                    "hash_contentions": None if "error" in metrics_result else metrics_result.get("hash_contentions"),
                    "contention_pct": None if "error" in metrics_result else metrics_result.get("contention_pct"),
                },
                "metrics_error": metrics_result.get("error"),
            }

            final_pairs = entry["pairs"]["raw_estimated_pairs_after_replication_correction"]
            load_factor_pairs = None

            if final_pairs is not None:
                load_factor_pairs = int(float(final_pairs) / float(args.hash_load_factor))

            entry["pairs"]["estimated_pairs_after_load_factor_application"] = load_factor_pairs
            entry["hash"]["estimated_hash_table_size"] = compute_hash_table_size_from_estimate(final_pairs, args.hash_load_factor)

            if final_pairs is not None:
                entry["pairs"]["replication_corrected_vs_ground_truth_factor"] = safe_div(final_pairs, ground_truth_pairs)
            if load_factor_pairs is not None:
                entry["pairs"]["load_factor_applied_vs_ground_truth_factor"] = safe_div(load_factor_pairs, ground_truth_pairs)

            results.append(entry)

            print(
                "Result summary: "
                f"time_ms={entry['timing']['mean_time_ms']}, "
                f"replication_corrected_pairs={final_pairs}, "
                f"load_factor_applied_pairs={load_factor_pairs}, "
                f"hash_size_estimate={entry['hash']['estimated_hash_table_size']}"
            )

        dataset_output = {
            "run_type": run_type,
            "display_name": display_name,
            "metadata": {
                "timestamp": timestamp,
                "run_dir": str(run_dir),
                "run_type": run_type,
                "display_name": display_name,
                "num_cubes": args.num_cubes,
                "cube_size": args.cube_size,
                "target_universe": args.universe,
                "selectivity_for_target_universe": selectivity if run_type == "cubes" else None,
                "seed": args.seed,
                "timing_runs": args.timing_runs,
                "timeout_seconds": args.timeout,
                "warmup_runs": args.warmup_runs,
                "grid_resolutions": args.grid_resolutions,
                "groundtruth_grid_resolution": args.groundtruth_grid_resolution,
                "estimate_only": args.estimate_only,
                "alpha_correction_enabled": not args.no_alpha_correction,
                "hash_load_factor": args.hash_load_factor,
                "dataset_a": str(dataset_a),
                "dataset_b": str(dataset_b),
                "ground_truth_pairs_exact_overlap": ground_truth_pairs,
            },
            "results": results,
        }
        return dataset_output

    run_types = list(dict.fromkeys(args.run_types))
    merged_runs = []

    if "cubes" in run_types:
        cubes_a = RAW_DIR / args.cube_dataset_a
        cubes_b = RAW_DIR / args.cube_dataset_b
        merged_runs.append(run_dataset_sweep("cubes", cubes_a, cubes_b, "Uniform cubes"))

    if "nu" in run_types:
        nu_a = RAW_DIR / args.nu_dataset_a
        nu_b = RAW_DIR / args.nu_dataset_b
        merged_runs.append(run_dataset_sweep("nu", nu_a, nu_b, "NU (tdbase nv150 nu800 large)"))

    output = {
        "metadata": {
            "timestamp": timestamp,
            "run_dir": str(run_dir),
            "run_types": run_types,
            "timing_runs": args.timing_runs,
            "timeout_seconds": args.timeout,
            "warmup_runs": args.warmup_runs,
            "grid_resolutions": args.grid_resolutions,
            "groundtruth_grid_resolution": args.groundtruth_grid_resolution,
            "estimate_only": args.estimate_only,
            "alpha_correction_enabled": not args.no_alpha_correction,
            "hash_load_factor": args.hash_load_factor,
        },
        "runs": merged_runs,
    }

    output_json = run_dir / "grid_resolution_sweep_results.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    latest_json = Path(args.output_dir) / "grid_resolution_sweep_latest.json"
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved results:\n  {output_json}\n  {latest_json}")

    vis_script = SCRIPT_DIR / "visualize_grid_resolution_sweep.py"
    if vis_script.exists():
        try:
            for run_data in merged_runs:
                run_type = run_data["run_type"]
                run_cmd(
                    [
                        sys.executable,
                        str(vis_script),
                        "--input", str(output_json),
                        "--output-dir", str(figures_root),
                        "--run-type", run_type,
                        "--output-stem", f"grid_resolution_sweep_{run_type}",
                    ],
                    f"Running visualization ({run_type})",
                )
        except subprocess.CalledProcessError as exc:
            print(f"WARNING: Visualization failed: {exc}")
    else:
        print(f"[skip] Visualization script not found: {vis_script}")

    print("\nDone.")


if __name__ == "__main__":
    main()
