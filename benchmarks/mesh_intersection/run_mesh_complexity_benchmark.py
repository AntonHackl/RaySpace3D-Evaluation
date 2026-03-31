#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    RAYSPACE_DIR,
    canonical_sphere_pair_paths,
    count_vertices,
    ensure_sphere_pair_dataset,
    get_shared_data_dirs,
    timestamp_tag,
)
from benchmarks.mesh_intersection.adapters.cgal_adapter import CGALIntersectionAdapter
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter


SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "runs"
CGAL_BASE_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"
SPHERE_TEMPLATE_DIR = REPO_ROOT / "benchmarks" / "mesh_overlap" / "data" / "single_obj_files"


def main():
    parser = argparse.ArgumentParser(description="Mesh complexity benchmark for mesh intersection")
    parser.add_argument("--stages", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument("--num-objects", type=int, default=50000)
    parser.add_argument("--selectivity", type=float, default=0.0005)
    parser.add_argument("--min-size", type=float, default=1.0)
    parser.add_argument("--max-size", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-resolution", type=int, default=20)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--approaches", type=str, nargs="+",
                        default=["estimated", "cgal"],
                        choices=["estimated", "estimate_only", "cgal"])
    args = parser.parse_args()

    dirs = get_shared_data_dirs("mesh_complexity")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    estimated = RaytracerIntersectionAdapter(
        str(RAYSPACE_DIR), mode="estimated", preprocessed_dir=str(dirs["preprocessed"]),
        timings_dir=str(dirs["timings"]), grid_resolution=args.grid_resolution, warmup_runs=args.warmup_runs,
    )
    cgal = CGALIntersectionAdapter(str(CGAL_BASE_DIR), preprocessed_dir=str(dirs["preprocessed"]))

    results = []
    for stage in args.stages:
        template = SPHERE_TEMPLATE_DIR / f"Sphere_Stage_{stage}.obj"
        if not template.exists():
            print(f"[skip] Missing template: {template}")
            continue

        obj_a, obj_b = canonical_sphere_pair_paths(
            dirs["raw"],
            template_name=template.name,
            num_objects=args.num_objects,
            min_size=args.min_size,
            max_size=args.max_size,
            selectivity=args.selectivity,
            seed=args.seed,
            grid_resolution=args.grid_resolution,
        )

        ensure_sphere_pair_dataset(
            obj_a,
            obj_b,
            template_obj=template,
            num_objects=args.num_objects,
            min_size=args.min_size,
            max_size=args.max_size,
            selectivity=args.selectivity,
            seed=args.seed,
        )

        for file_path in (obj_a, obj_b):
            if not estimated.check_preprocessed(str(file_path)):
                estimated.preprocess_from_source(str(file_path), str(file_path))

        row = {
            "stage": stage,
            "vertices_per_mesh": count_vertices(template),
            "num_objects": args.num_objects,
            "selectivity": args.selectivity,
            "size_bytes_a": obj_a.stat().st_size if obj_a.exists() else 0,
            "size_bytes_b": obj_b.stat().st_size if obj_b.exists() else 0,
        }

        if "estimated" in args.approaches:
            row["estimated"] = estimated.run_intersection(str(obj_a), str(obj_b), args.runs, timeout=args.timeout)
        if "estimate_only" in args.approaches:
            estimated.mode = "estimate_only"
            estimated.name = "Raytracer_estimate_only"
            estimated.executable = estimated.rayspace_dir / "query" / "build" / "bin" / "raytracer_intersection_estimated"
            row["estimate_only"] = estimated.run_intersection(str(obj_a), str(obj_b), args.runs, timeout=args.timeout)
        if "cgal" in args.approaches:
            row["cgal"] = cgal.run_intersection(str(obj_a), str(obj_b), args.runs, timeout=args.timeout)

        results.append(row)
        print(f"stage={stage}: done")

    payload = {
        "metadata": {
            "scenario": "mesh_complexity",
            "query_type": "intersection",
            "timestamp": timestamp_tag(),
            "stages": args.stages,
            "num_objects": args.num_objects,
            "selectivity": args.selectivity,
            "min_size": args.min_size,
            "max_size": args.max_size,
            "seed": args.seed,
            "grid_resolution": args.grid_resolution,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "approaches": args.approaches,
            "shared_data_root": str(dirs["root"]),
            "template_dir": str(SPHERE_TEMPLATE_DIR),
        },
        "results": results,
    }

    out = RUNS_DIR / f"intersection_mesh_complexity_{timestamp_tag()}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
