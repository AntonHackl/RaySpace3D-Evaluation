#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    RAYSPACE_DIR,
    canonical_cube_pair_paths,
    create_benchmark_run_layout,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter

def _safe_speedup(closest_ms: float, anyhit_ms: float):
    if anyhit_ms <= 0.0:
        return None
    return closest_ms / anyhit_ms


def _extract_query_ms_from_timing_json(timing_json: Path) -> float:
    with open(timing_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    phases = data.get("phases", {})
    total = 0.0
    for key, phase in phases.items():
        k = key.lower()
        if k.startswith("raytrace_overlap_hash_mesh1tomesh2"):
            total += float(phase.get("duration_ms", 0.0))
        elif k.startswith("raytrace_overlap_hash_mesh2tomesh1"):
            total += float(phase.get("duration_ms", 0.0))
        elif k.startswith("raytrace_containment_hash_mesh1tomesh2"):
            total += float(phase.get("duration_ms", 0.0))
        elif k.startswith("raytrace_containment_hash_mesh2tomesh1"):
            total += float(phase.get("duration_ms", 0.0))
        elif k.startswith("compact_hash_table_pairs"):
            total += float(phase.get("duration_ms", 0.0))
    return total


def _run_mode(
    adapter: RaytracerIntersectionAdapter,
    mesh_a: Path,
    mesh_b: Path,
    runs: int,
    timeout_s: float,
    timing_dir: Path,
    use_anyhit: bool,
):
    executable = adapter.executable
    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")

    pre_a = adapter.preprocessed_dir / mesh_a.with_suffix(".pre").name
    pre_b = adapter.preprocessed_dir / mesh_b.with_suffix(".pre").name
    in_a = str(pre_a if pre_a.exists() else mesh_a)
    in_b = str(pre_b if pre_b.exists() else mesh_b)

    raw_times_ms = []
    pair_counts = []

    for run_idx in range(runs):
        timing_json = timing_dir / f"compare_{'anyhit' if use_anyhit else 'closest'}_{run_idx}.json"
        cmd = [
            str(executable),
            "--mesh1", in_a,
            "--mesh2", in_b,
            "--output", str(timing_json),
        ]
        if use_anyhit:
            cmd.append("--use-anyhit-containment")

        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=True,
        )
        output = cp.stdout + cp.stderr
        raw_times_ms.append(_extract_query_ms_from_timing_json(timing_json))

        num_pairs = None
        for line in output.splitlines():
            if "Actual Intersection Pairs:" in line:
                num_pairs = int(line.split(":", 1)[1].strip())
                break
        pair_counts.append(num_pairs)

    return {
        "mean": statistics.mean(raw_times_ms),
        "min": min(raw_times_ms),
        "max": max(raw_times_ms),
        "std": statistics.pstdev(raw_times_ms),
        "raw_times": raw_times_ms,
        "num_intersections": pair_counts[-1],
        "pair_counts_per_run": pair_counts,
        "timing_metric": "overlap+containment+compact from timing counters",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare RaySpace intersection performance: closest-hit containment vs any-hit containment"
    )
    parser.add_argument("--num-cubes-a", type=int, default=20000)
    parser.add_argument("--num-cubes-b", type=int, default=20000)
    parser.add_argument("--selectivity", type=float, default=0.001)
    parser.add_argument("--min-size", type=float, default=1.0)
    parser.add_argument("--max-size", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-resolution", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1800.0)
    args = parser.parse_args()

    dirs = get_shared_data_dirs("mesh_intersection_anyhit_compare")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "mesh_intersection_closest_vs_anyhit")

    mesh_a, mesh_b = canonical_cube_pair_paths(
        dirs["raw"],
        num_cubes_a=args.num_cubes_a,
        num_cubes_b=args.num_cubes_b,
        min_size=args.min_size,
        max_size=args.max_size,
        selectivity=args.selectivity,
        seed=args.seed,
        grid_resolution=args.grid_resolution,
    )

    ensure_cube_pair_dataset(
        mesh_a,
        mesh_b,
        num_cubes_a=args.num_cubes_a,
        num_cubes_b=args.num_cubes_b,
        min_size=args.min_size,
        max_size=args.max_size,
        selectivity=args.selectivity,
        seed=args.seed,
    )

    adapter = RaytracerIntersectionAdapter(
        str(RAYSPACE_DIR),
        mode="estimated",
        preprocessed_dir=str(dirs["preprocessed"]),
        timings_dir=str(dirs["timings"]),
        grid_resolution=args.grid_resolution,
        warmup_runs=1,
    )

    for file_path in (mesh_a, mesh_b):
        if not adapter.check_preprocessed(str(file_path)):
            adapter.preprocess_from_source(str(file_path), str(file_path))

    print("Running closest-hit containment benchmark...")
    closest = _run_mode(adapter, mesh_a, mesh_b, args.runs, args.timeout, dirs["timings"], use_anyhit=False)

    print("Running any-hit containment benchmark...")
    anyhit = _run_mode(adapter, mesh_a, mesh_b, args.runs, args.timeout, dirs["timings"], use_anyhit=True)

    if "error" in closest:
        raise RuntimeError(f"closest-hit run failed: {closest['error']}")
    if "error" in anyhit:
        raise RuntimeError(f"any-hit run failed: {anyhit['error']}")

    closest_mean = float(closest["mean"])
    anyhit_mean = float(anyhit["mean"])
    speedup = _safe_speedup(closest_mean, anyhit_mean)

    payload = {
        "metadata": {
            "scenario": "mesh_intersection_anyhit_compare",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "num_cubes_a": args.num_cubes_a,
            "num_cubes_b": args.num_cubes_b,
            "selectivity": args.selectivity,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "seed": args.seed,
            "grid_resolution": args.grid_resolution,
            "runs": args.runs,
            "timeout_seconds": args.timeout,
            "mesh_a": str(mesh_a),
            "mesh_b": str(mesh_b),
        },
        "closest_hit": closest,
        "any_hit": anyhit,
        "comparison": {
            "mean_ms_delta": anyhit_mean - closest_mean,
            "mean_ms_ratio_anyhit_over_closest": (anyhit_mean / closest_mean) if closest_mean > 0.0 else None,
            "closest_over_anyhit_speedup": speedup,
            "faster_mode": "any_hit" if anyhit_mean < closest_mean else "closest_hit",
            "pair_count_equal": closest.get("num_intersections") == anyhit.get("num_intersections"),
            "pair_count_closest": closest.get("num_intersections"),
            "pair_count_anyhit": anyhit.get("num_intersections"),
        },
    }

    out = Path(run_layout["results_json"])
    write_json(out, payload)

    print("=== Comparison Summary ===")
    print(f"Closest-hit mean (ms): {closest_mean:.3f}")
    print(f"Any-hit mean (ms):     {anyhit_mean:.3f}")
    print(f"Delta (any-hit - closest) ms: {anyhit_mean - closest_mean:.3f}")
    if speedup is not None:
        print(f"Closest/AnyHit speedup: {speedup:.4f}x")
    print(f"Pair counts equal: {payload['comparison']['pair_count_equal']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
