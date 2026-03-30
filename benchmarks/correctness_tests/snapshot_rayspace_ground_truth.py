#!/usr/bin/env python3
import json
import time
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.mesh_overlap.adapters.raytracer_adapter import RaytracerAdapter
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter
from benchmarks.mesh_containment.adapters.raytracer_adapter import RaytracerContainmentAdapter

RAW_DIR = SCRIPT_DIR / "data" / "raw"
PREPROCESSED_DIR = SCRIPT_DIR / "data" / "preprocessed"
TIMINGS_DIR = SCRIPT_DIR / "data" / "timings"
GROUND_TRUTH_DIR = SCRIPT_DIR / "ground_truth"
MANIFEST_PATH = RAW_DIR / "manual_expected_results.json"

RAYSPACE_DIR = REPO_ROOT / "src" / "RaySpace3D"
TIMEOUT_SECONDS = 1800.0


def _load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset manifest at {MANIFEST_PATH}. Run generate_datasets.py first."
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_dt_name(path_str: str) -> str:
    return str(Path(path_str).with_suffix(".dt"))


def main():
    manifest = _load_manifest()
    cubes = manifest["cubes_20k_sel_0_001"]
    mesh_a = cubes["mesh_a"]
    mesh_b = cubes["mesh_b"]

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    GROUND_TRUTH_DIR.mkdir(parents=True, exist_ok=True)

    overlap_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR),
        mode="exact",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_resolution=10,
        warmup_runs=2,
    )
    intersection_adapter = RaytracerIntersectionAdapter(
        str(RAYSPACE_DIR),
        mode="two_pass",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_resolution=10,
        warmup_runs=2,
    )
    containment_adapter = RaytracerContainmentAdapter(
        str(RAYSPACE_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_resolution=10,
        warmup_runs=1,
    )

    overlap_adapter.preprocess_from_source(mesh_a, _as_dt_name(mesh_a))
    overlap_adapter.preprocess_from_source(mesh_b, _as_dt_name(mesh_b))

    intersection_adapter.preprocess_from_source(mesh_a, _as_dt_name(mesh_a))
    intersection_adapter.preprocess_from_source(mesh_b, _as_dt_name(mesh_b))

    containment_adapter.preprocess_from_source(mesh_a, _as_dt_name(mesh_a))
    containment_adapter.preprocess_from_source(mesh_b, _as_dt_name(mesh_b))

    overlap_result = overlap_adapter.run_overlap(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)
    if "error" in overlap_result:
        raise RuntimeError(f"Overlap query failed: {overlap_result['error']}")

    intersection_result = intersection_adapter.run_intersection(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)
    if "error" in intersection_result:
        raise RuntimeError(f"Intersection query failed: {intersection_result['error']}")

    containment_result = containment_adapter.run_containment(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)
    if "error" in containment_result:
        raise RuntimeError(f"Containment query failed: {containment_result['error']}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "metadata": {
            "timestamp": timestamp,
            "dataset": "cubes_20k_sel_0_001",
            "mesh_a": mesh_a,
            "mesh_b": mesh_b,
            "seed": cubes["seed"],
            "num_cubes_a": cubes["num_cubes_a"],
            "num_cubes_b": cubes["num_cubes_b"],
            "selectivity": cubes["selectivity"],
            "rayspace_modes": {
                "overlap": "exact",
                "intersection": "two_pass",
                "containment": "default",
            },
        },
        "ground_truth": {
            "overlap_num_pairs": int(overlap_result.get("num_intersections", 0)),
            "intersection_num_pairs": int(intersection_result.get("num_intersections", 0)),
            "containment_num_pairs": int(containment_result.get("num_containments", 0)),
        },
        "raw_results": {
            "overlap": overlap_result,
            "intersection": intersection_result,
            "containment": containment_result,
        },
    }

    latest_path = GROUND_TRUTH_DIR / "rayspace_20k_sel0_001_current.json"
    timestamped_path = GROUND_TRUTH_DIR / f"rayspace_20k_sel0_001_current_{timestamp}.json"

    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    with open(timestamped_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(f"Wrote: {latest_path}")
    print(f"Wrote: {timestamped_path}")
    print("Ground truth counts:")
    print(f"  overlap: {output['ground_truth']['overlap_num_pairs']}")
    print(f"  intersection: {output['ground_truth']['intersection_num_pairs']}")
    print(f"  containment: {output['ground_truth']['containment_num_pairs']}")


if __name__ == "__main__":
    main()
