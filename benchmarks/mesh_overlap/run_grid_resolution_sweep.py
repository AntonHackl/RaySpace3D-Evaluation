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
GENERATE_PLANE_CUBES_SCRIPT = RAYSPACE_DIR / "scripts" / "generate_plane_scatter_cube_pair.py"
GENERATE_R1_ADVERSARIAL_CUBES_SCRIPT = RAYSPACE_DIR / "scripts" / "generate_resolution1_adversarial_cube_pair.py"
GENERATE_MC_ADVERSARIAL_CUBES_SCRIPT = RAYSPACE_DIR / "scripts" / "generate_multi_clump_adversarial.py"
VISUALIZE_PLANE_CUBES_SCRIPT = RAYSPACE_DIR / "scripts" / "visualize_plane_scatter_cubes.py"
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
                        choices=["cubes", "nu", "plane", "r1_adversarial", "multi_clump"],
                        help="Dataset run types to execute and merge into one result JSON")
    parser.add_argument("--cube-dataset-a", type=str, default="cubes_100k_size5_u100_a.obj",
                        help="Cube dataset A filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--cube-dataset-b", type=str, default="cubes_100k_size5_u100_b.obj",
                        help="Cube dataset B filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--nu-dataset-a", type=str, default="tdbase_n_nv150_nu800_n_nv150_nu800_vs100_r30.dt",
                        help="NU dataset A filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--nu-dataset-b", type=str, default="tdbase_n_nv150_nu800_v_nv150_nu800_vs100_r30.dt",
                        help="NU dataset B filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--plane-num-cubes", type=int, default=50_000,
                        help="Number of cubes in each plane dataset")
    parser.add_argument("--plane-cube-size", type=float, default=5.0,
                        help="Fixed cube edge length for plane datasets")
    parser.add_argument("--plane-xy-extent", type=float, default=120.0,
                        help="Extent for x/y center placement of plane datasets")
    parser.add_argument("--plane-z-base", type=float, default=60.0,
                        help="Shared z baseline where both plane datasets align")
    parser.add_argument("--plane-z-span", type=float, default=80.0,
                        help="Total z span from low to high along the plane direction")
    parser.add_argument("--plane-z-noise-sigma", type=float, default=2.0,
                        help="Gaussian z scatter sigma for plane datasets")
    parser.add_argument("--plane-direction-x", type=float, default=1.0,
                        help="Plane sweep direction x component")
    parser.add_argument("--plane-direction-y", type=float, default=1.0,
                        help="Plane sweep direction y component")
    parser.add_argument("--plane-dataset-a", type=str, default="cubes_plane_50k_a.obj",
                        help="Plane dataset A filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--plane-dataset-b", type=str, default="cubes_plane_50k_b.obj",
                        help="Plane dataset B filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--plane-metadata-json", type=str, default="cubes_plane_50k_metadata.json",
                        help="Plane dataset metadata filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--plane-visualize", action="store_true",
                        help="Generate a validation visualization for plane datasets")
    parser.add_argument("--r1-num-cubes", type=int, default=50_000,
                        help="Number of cubes in each resolution-1 adversarial dataset")
    parser.add_argument("--r1-cube-size", type=float, default=5.0,
                        help="Fixed cube edge length for resolution-1 adversarial datasets")
    parser.add_argument("--r1-world-size", type=float, default=2000.0,
                        help="World size used by resolution-1 adversarial generator")
    parser.add_argument("--r1-cluster-radius", type=float, default=30.0,
                        help="Cluster radius for resolution-1 adversarial generator")
    parser.add_argument("--r1-cluster-margin", type=float, default=220.0,
                        help="Cluster center margin to world boundaries for resolution-1 adversarial generator")
    parser.add_argument("--r1-anchors-per-dataset", type=int, default=2,
                        help="Number of AABB-inflating anchors per dataset for resolution-1 adversarial generator")
    parser.add_argument("--r1-dataset-a", type=str, default="cubes_r1_adversarial_50k_a.obj",
                        help="Resolution-1 adversarial dataset A filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--r1-dataset-b", type=str, default="cubes_r1_adversarial_50k_b.obj",
                        help="Resolution-1 adversarial dataset B filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--r1-metadata-json", type=str, default="cubes_r1_adversarial_50k_metadata.json",
                        help="Resolution-1 adversarial metadata filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--mc-num-cubes", type=int, default=50_000,
                        help="Number of cubes in each multi-clump adversarial dataset")
    parser.add_argument("--mc-num-clusters", type=int, default=50,
                        help="Number of clusters in multi-clump adversarial generator")
    parser.add_argument("--mc-cluster-radius", type=float, default=2.0,
                        help="Cluster radius for multi-clump adversarial generator")
    parser.add_argument("--mc-span", type=float, default=1000.0,
                        help="World span for multi-clump adversarial generator")
    parser.add_argument("--mc-cube-size", type=float, default=0.5,
                        help="Fixed cube edge length for multi-clump adversarial datasets")
    parser.add_argument("--mc-dataset-a", type=str, default="cubes_multi_clump_50k_a.obj",
                        help="Multi-clump adversarial dataset A filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--mc-dataset-b", type=str, default="cubes_multi_clump_50k_b.obj",
                        help="Multi-clump adversarial dataset B filename in benchmarks/mesh_overlap/data/raw")
    parser.add_argument("--mc-metadata-json", type=str, default="cubes_multi_clump_50k_metadata.json",
                        help="Multi-clump adversarial metadata filename in benchmarks/mesh_overlap/data/raw")
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
        elif run_type == "plane":
            metadata_path = RAW_DIR / args.plane_metadata_json
            if args.skip_generate and dataset_a.exists() and dataset_b.exists():
                print(f"[skip] Plane dataset files already exist:\n  {dataset_a}\n  {dataset_b}")
            else:
                run_cmd(
                    [
                        sys.executable,
                        str(GENERATE_PLANE_CUBES_SCRIPT),
                        "--num-cubes-a", str(args.plane_num_cubes),
                        "--num-cubes-b", str(args.plane_num_cubes),
                        "--cube-size", str(args.plane_cube_size),
                        "--xy-extent", str(args.plane_xy_extent),
                        "--z-base", str(args.plane_z_base),
                        "--z-span", str(args.plane_z_span),
                        "--z-noise-sigma", str(args.plane_z_noise_sigma),
                        "--direction-x", str(args.plane_direction_x),
                        "--direction-y", str(args.plane_direction_y),
                        "--output-a", str(dataset_a),
                        "--output-b", str(dataset_b),
                        "--metadata-json", str(metadata_path),
                        "--seed", str(args.seed),
                    ],
                    (
                        "Generating plane-scatter cube datasets "
                        f"({args.plane_num_cubes:,} cubes each, size={args.plane_cube_size})"
                    ),
                )

            if args.plane_visualize:
                plane_plot = figures_root / f"plane_dataset_validation_{run_type}_{timestamp}.png"
                run_cmd(
                    [
                        sys.executable,
                        str(VISUALIZE_PLANE_CUBES_SCRIPT),
                        "--dataset-a", str(dataset_a),
                        "--dataset-b", str(dataset_b),
                        "--output", str(plane_plot),
                        "--direction-x", str(args.plane_direction_x),
                        "--direction-y", str(args.plane_direction_y),
                        "--seed", str(args.seed),
                    ],
                    f"Visualizing plane datasets ({run_type})",
                )
        elif run_type == "r1_adversarial":
            metadata_path = RAW_DIR / args.r1_metadata_json
            if args.skip_generate and dataset_a.exists() and dataset_b.exists():
                print(f"[skip] Resolution-1 adversarial dataset files already exist:\n  {dataset_a}\n  {dataset_b}")
            else:
                run_cmd(
                    [
                        sys.executable,
                        str(GENERATE_R1_ADVERSARIAL_CUBES_SCRIPT),
                        "--num-cubes-a", str(args.r1_num_cubes),
                        "--num-cubes-b", str(args.r1_num_cubes),
                        "--cube-size", str(args.r1_cube_size),
                        "--world-size", str(args.r1_world_size),
                        "--cluster-radius", str(args.r1_cluster_radius),
                        "--cluster-margin", str(args.r1_cluster_margin),
                        "--anchors-per-dataset", str(args.r1_anchors_per_dataset),
                        "--output-a", str(dataset_a),
                        "--output-b", str(dataset_b),
                        "--metadata-json", str(metadata_path),
                        "--seed", str(args.seed),
                    ],
                    (
                        "Generating resolution-1 adversarial cube datasets "
                        f"({args.r1_num_cubes:,} cubes each, size={args.r1_cube_size})"
                    ),
                )
        elif run_type == "multi_clump":
            metadata_path = RAW_DIR / args.mc_metadata_json
            if args.skip_generate and dataset_a.exists() and dataset_b.exists():
                print(f"[skip] Multi-clump adversarial dataset files already exist:\n  {dataset_a}\n  {dataset_b}")
            else:
                run_cmd(
                    [
                        sys.executable,
                        str(GENERATE_MC_ADVERSARIAL_CUBES_SCRIPT),
                        "--count", str(args.mc_num_cubes),
                        "--clusters", str(args.mc_num_clusters),
                        "--cluster-radius", str(args.mc_cluster_radius),
                        "--span", str(args.mc_span),
                        "--cube-size", str(args.mc_cube_size),
                        "--output-a", str(dataset_a),
                        "--output-b", str(dataset_b),
                        "--seed", str(args.seed),
                    ],
                    (
                        "Generating multi-clump adversarial cube datasets "
                        f"({args.mc_num_cubes:,} cubes each, size={args.mc_cube_size})"
                    ),
                )

            # Always visualize geometric distribution
            mc_plot = figures_root / f"multi_clump_geometry_{timestamp}.png"
            run_cmd(
                [
                    sys.executable,
                    str(VISUALIZE_PLANE_CUBES_SCRIPT),
                    "--dataset-a", str(dataset_a),
                    "--dataset-b", str(dataset_b),
                    "--output", str(mc_plot),
                    "--direction-x", "1.0",
                    "--direction-y", "1.0",
                    "--seed", str(args.seed),
                ],
                f"Visualizing multi-clump datasets ({run_type})",
            )

        if not dataset_a.exists() or not dataset_b.exists():
            missing = [str(p) for p in (dataset_a, dataset_b) if not p.exists()]
            print(f"ERROR: Missing dataset files for run_type={run_type}: {missing}", file=sys.stderr)
            sys.exit(1)

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
                "num_cubes": args.num_cubes if run_type == "cubes" else (args.plane_num_cubes if run_type == "plane" else (args.r1_num_cubes if run_type == "r1_adversarial" else None)),
                "cube_size": args.cube_size if run_type == "cubes" else (args.plane_cube_size if run_type == "plane" else (args.r1_cube_size if run_type == "r1_adversarial" else None)),
                "target_universe": args.universe if run_type == "cubes" else None,
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
                "plane_xy_extent": args.plane_xy_extent if run_type == "plane" else None,
                "plane_z_base": args.plane_z_base if run_type == "plane" else None,
                "plane_z_span": args.plane_z_span if run_type == "plane" else None,
                "plane_z_noise_sigma": args.plane_z_noise_sigma if run_type == "plane" else None,
                "plane_direction": [args.plane_direction_x, args.plane_direction_y] if run_type == "plane" else None,
                "r1_world_size": args.r1_world_size if run_type == "r1_adversarial" else None,
                "r1_cluster_radius": args.r1_cluster_radius if run_type == "r1_adversarial" else None,
                "r1_cluster_margin": args.r1_cluster_margin if run_type == "r1_adversarial" else None,
                "r1_anchors_per_dataset": args.r1_anchors_per_dataset if run_type == "r1_adversarial" else None,
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

    if "plane" in run_types:
        plane_a = RAW_DIR / args.plane_dataset_a
        plane_b = RAW_DIR / args.plane_dataset_b
        merged_runs.append(run_dataset_sweep("plane", plane_a, plane_b, "Opposing plane-scatter cubes"))

    if "r1_adversarial" in run_types:
        r1_a = RAW_DIR / args.r1_dataset_a
        r1_b = RAW_DIR / args.r1_dataset_b
        merged_runs.append(run_dataset_sweep("r1_adversarial", r1_a, r1_b, "Resolution-1 adversarial clusters"))

    if "multi_clump" in run_types:
        mc_a = RAW_DIR / args.mc_dataset_a
        mc_b = RAW_DIR / args.mc_dataset_b
        merged_runs.append(run_dataset_sweep("multi_clump", mc_a, mc_b, "Multi-clump adversarial clusters"))

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
