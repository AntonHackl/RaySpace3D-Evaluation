#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import time
from typing import Dict
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.mesh_overlap.adapters.raytracer_adapter import RaytracerAdapter
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter
from benchmarks.mesh_containment.adapters.raytracer_adapter import RaytracerContainmentAdapter
from benchmarks.mesh_intersection.adapters.cgal_adapter import CGALIntersectionAdapter
from benchmarks.mesh_containment.adapters.cgal_adapter import CGALContainmentAdapter
from benchmarks.common.adapters.base import run_command_streaming

RAW_DIR = SCRIPT_DIR / "data" / "raw"
PREPROCESSED_DIR = SCRIPT_DIR / "data" / "preprocessed"
TIMINGS_DIR = SCRIPT_DIR / "data" / "timings"
RUNS_DIR = SCRIPT_DIR / "runs"
MANIFEST_PATH = RAW_DIR / "manual_expected_results.json"

RAYSPACE_DIR = REPO_ROOT / "src" / "RaySpace3D"
CGAL_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"
TIMEOUT_SECONDS = 1800.0
ESTIMATED_RELATIVE_TOLERANCE = 0.05
CGAL_QUERY_RELATIVE_TOLERANCE = 0.05

# This must match the values produced by snapshot_rayspace_ground_truth.py at the current time.
# After re-snapshotting, update these constants intentionally.
RAYSPACE_20K_GROUND_TRUTH: Dict[str, int] = {
    "overlap_num_pairs": 421752,
    "intersection_num_pairs": 430072,
    "containment_num_pairs": 18705,
}


def _as_dt_name(path_str: str) -> str:
    return str(Path(path_str).with_suffix(".dt"))


def _load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset manifest at {MANIFEST_PATH}. Run generate_datasets.py first."
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _to_pair_set(pairs):
    return {(int(a), int(b)) for a, b in pairs}


def _parse_count_from_output(output: str):
    patterns = [
        r"Unique intersecting object pairs:\s*(\d+)",
        r"Unique object pairs:\s*(\d+)",
        r"Total Overlaps:\s*(\d+)",
        r"Overlap pairs:\s*(\d+)",
        r"Containment pairs \(B in A\):\s*(\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return int(match.group(1))
    return None


def _run_cgal_style_overlap_with_count(name: str, executable: Path, mesh_a: str, mesh_b: str):
    p1 = PREPROCESSED_DIR / Path(mesh_a).with_suffix(".pre").name
    p2 = PREPROCESSED_DIR / Path(mesh_b).with_suffix(".pre").name

    if not executable.exists():
        return {"error": f"Executable not found: {executable}"}
    if not p1.exists() or not p2.exists():
        return {"error": f"Missing preprocessed files: {p1} or {p2}"}

    stdout_text, stderr_text = run_command_streaming(
        [str(executable), str(p1), str(p2)],
        timeout=TIMEOUT_SECONDS,
        log_path=None,
        prefix=f"[{name}]",
    )
    output = stdout_text + stderr_text
    count = _parse_count_from_output(output)
    if count is None:
        return {"error": f"Could not parse pair count from {name} output"}

    timing_ms = None
    timing_match = re.search(r"Query Time:.*?\(([\d.]+) ms\)", output)
    if timing_match:
        timing_ms = float(timing_match.group(1))

    return {
        "num_pairs": count,
        "timing_ms": timing_ms,
    }


def _prepare_preprocessed(mesh_a: str, mesh_b: str):
    overlap_prep = RaytracerAdapter(
        str(RAYSPACE_DIR),
        mode="exact",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_resolution=10,
        warmup_runs=1,
    )
    overlap_prep.preprocess_from_source(mesh_a, _as_dt_name(mesh_a))
    overlap_prep.preprocess_from_source(mesh_b, _as_dt_name(mesh_b))


def _build_check(found: int, expected: int, approximate: bool = False, tolerance: float = ESTIMATED_RELATIVE_TOLERANCE):
    if approximate:
        if expected <= 0:
            rel_error = 0.0 if found == expected else 1.0
        else:
            rel_error = abs(found - expected) / float(expected)
        return {
            "pass": rel_error <= tolerance,
            "expected": expected,
            "found": found,
            "approximate": True,
            "relative_error": rel_error,
            "tolerance": tolerance,
        }

    return {
        "pass": found == expected,
        "expected": expected,
        "found": found,
    }


def run_overlap_checks(manifest, approaches):
    manual = manifest["manual"]["overlap"]
    cubes = manifest["cubes_20k_sel_0_001"]

    expected_manual_pairs = _to_pair_set(manual["expected_overlap_pairs"])
    expected_manual_count = len(expected_manual_pairs)

    results = {
        "manual": {},
        "cubes_20k": {},
    }

    for dataset_name, pair in (
        ("manual", (manual["mesh_a"], manual["mesh_b"])),
        ("cubes_20k", (cubes["mesh_a"], cubes["mesh_b"])),
    ):
        mesh_a, mesh_b = pair
        _prepare_preprocessed(mesh_a, mesh_b)

        if "rayspace" in approaches:
            exact_adapter = RaytracerAdapter(
                str(RAYSPACE_DIR),
                mode="exact",
                preprocessed_dir=str(PREPROCESSED_DIR),
                timings_dir=str(TIMINGS_DIR),
                grid_resolution=10,
                warmup_runs=2,
            )
            exact_out = exact_adapter.run_overlap(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)

            direct_adapter = RaytracerAdapter(
                str(RAYSPACE_DIR),
                mode="direct_estimation",
                preprocessed_dir=str(PREPROCESSED_DIR),
                timings_dir=str(TIMINGS_DIR),
                grid_resolution=10,
                warmup_runs=2,
            )
            direct_out = direct_adapter.run_overlap(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)

            if "error" in exact_out:
                results[dataset_name]["rayspace_exact"] = {"pass": False, "error": exact_out["error"]}
            else:
                found = int(exact_out.get("num_intersections", 0))
                expected = expected_manual_count if dataset_name == "manual" else int(RAYSPACE_20K_GROUND_TRUTH["overlap_num_pairs"])
                results[dataset_name]["rayspace_exact"] = _build_check(found, expected)

            if "error" in direct_out:
                results[dataset_name]["rayspace_direct_estimation"] = {"pass": False, "error": direct_out["error"]}
            else:
                found = int(direct_out.get("num_intersections", 0))
                expected = expected_manual_count if dataset_name == "manual" else int(RAYSPACE_20K_GROUND_TRUTH["overlap_num_pairs"])
                results[dataset_name]["rayspace_direct_estimation"] = _build_check(found, expected)

        if "cgal" in approaches:
            cgal_out = _run_cgal_style_overlap_with_count(
                "CGAL",
                CGAL_DIR / "build" / "cgal_overlap",
                mesh_a,
                mesh_b,
            )
            if "error" in cgal_out:
                results[dataset_name]["cgal"] = {"pass": False, "error": cgal_out["error"]}
            else:
                found = int(cgal_out["num_pairs"])
                expected = expected_manual_count if dataset_name == "manual" else int(RAYSPACE_20K_GROUND_TRUTH["overlap_num_pairs"])
                results[dataset_name]["cgal"] = {
                    "pass": found == expected,
                    "expected": expected,
                    "found": found,
                    "timing_ms": cgal_out["timing_ms"],
                }

        if "touch" in approaches:
            touch_out = _run_cgal_style_overlap_with_count(
                "TOUCH",
                CGAL_DIR / "build" / "cgal_touch",
                mesh_a,
                mesh_b,
            )
            if "error" in touch_out:
                results[dataset_name]["touch"] = {"pass": False, "error": touch_out["error"]}
            else:
                found = int(touch_out["num_pairs"])
                expected = expected_manual_count if dataset_name == "manual" else int(RAYSPACE_20K_GROUND_TRUTH["overlap_num_pairs"])
                results[dataset_name]["touch"] = {
                    "pass": found == expected,
                    "expected": expected,
                    "found": found,
                    "timing_ms": touch_out["timing_ms"],
                }

    return results


def run_intersection_checks(manifest, approaches):
    filtered_approaches = [a for a in approaches if a != "touch"]

    manual = manifest["manual"]["intersection"]
    cubes = manifest["cubes_20k_sel_0_001"]

    expected_manual_pairs = _to_pair_set(manual["expected_intersection_pairs"])
    expected_manual_count = len(expected_manual_pairs)

    results = {
        "manual": {},
        "cubes_20k": {},
    }

    for dataset_name, pair in (
        ("manual", (manual["mesh_a"], manual["mesh_b"])),
        ("cubes_20k", (cubes["mesh_a"], cubes["mesh_b"])),
    ):
        mesh_a, mesh_b = pair
        _prepare_preprocessed(mesh_a, mesh_b)

        if "rayspace" in filtered_approaches:
            estimated_adapter = RaytracerIntersectionAdapter(
                str(RAYSPACE_DIR),
                mode="estimated",
                preprocessed_dir=str(PREPROCESSED_DIR),
                timings_dir=str(TIMINGS_DIR),
                grid_resolution=10,
                warmup_runs=2,
            )
            estimated_out = estimated_adapter.run_intersection(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)

            if "error" in estimated_out:
                results[dataset_name]["rayspace_estimated"] = {"pass": False, "error": estimated_out["error"]}
            else:
                found = int(estimated_out.get("num_intersections", 0))
                expected = expected_manual_count if dataset_name == "manual" else int(RAYSPACE_20K_GROUND_TRUTH["intersection_num_pairs"])
                results[dataset_name]["rayspace_estimated"] = _build_check(found, expected, approximate=True)

        if "cgal" in filtered_approaches:
            adapter = CGALIntersectionAdapter(
                str(CGAL_DIR),
                preprocessed_dir=str(PREPROCESSED_DIR),
            )
            out = adapter.run_intersection(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)
            if "error" in out:
                results[dataset_name]["cgal"] = {"pass": False, "error": out["error"]}
            else:
                found = int(out.get("num_intersections", 0))
                expected = expected_manual_count if dataset_name == "manual" else int(RAYSPACE_20K_GROUND_TRUTH["intersection_num_pairs"])
                results[dataset_name]["cgal"] = _build_check(
                    found,
                    expected,
                    approximate=True,
                    tolerance=CGAL_QUERY_RELATIVE_TOLERANCE,
                )

    return results


def run_containment_checks(manifest, approaches, use_anyhit_point_in_mesh: bool = False):
    filtered_approaches = [a for a in approaches if a != "touch"]

    manual = manifest["manual"]["containment"]
    cubes = manifest["cubes_20k_sel_0_001"]

    expected_manual_pairs = _to_pair_set(manual["expected_containment_pairs"])
    expected_manual_count = len(expected_manual_pairs)

    results = {
        "manual": {},
        "cubes_20k": {},
    }

    for dataset_name, pair in (
        ("manual", (manual["mesh_a"], manual["mesh_b"])),
        ("cubes_20k", (cubes["mesh_a"], cubes["mesh_b"])),
    ):
        mesh_a, mesh_b = pair
        _prepare_preprocessed(mesh_a, mesh_b)

        if "rayspace" in filtered_approaches:
            adapter = RaytracerContainmentAdapter(
                str(RAYSPACE_DIR),
                preprocessed_dir=str(PREPROCESSED_DIR),
                timings_dir=str(TIMINGS_DIR),
                grid_resolution=10,
                warmup_runs=1,
                use_anyhit_point_in_mesh=use_anyhit_point_in_mesh,
            )
            out = adapter.run_containment(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)
            if "error" in out:
                results[dataset_name]["rayspace_exact"] = {"pass": False, "error": out["error"]}
            else:
                found = int(out.get("num_containments", 0))
                expected = expected_manual_count if dataset_name == "manual" else int(RAYSPACE_20K_GROUND_TRUTH["containment_num_pairs"])
                results[dataset_name]["rayspace_exact"] = _build_check(found, expected)

        if "cgal" in filtered_approaches:
            adapter = CGALContainmentAdapter(
                str(CGAL_DIR),
                preprocessed_dir=str(PREPROCESSED_DIR),
            )
            out = adapter.run_containment(mesh_a, mesh_b, num_runs=1, timeout=TIMEOUT_SECONDS)
            if "error" in out:
                results[dataset_name]["cgal"] = {"pass": False, "error": out["error"]}
            else:
                found = int(out.get("num_containments", 0))
                expected = expected_manual_count if dataset_name == "manual" else int(RAYSPACE_20K_GROUND_TRUTH["containment_num_pairs"])
                results[dataset_name]["cgal"] = _build_check(
                    found,
                    expected,
                    approximate=True,
                    tolerance=CGAL_QUERY_RELATIVE_TOLERANCE,
                )

    return results


def main():
    parser = argparse.ArgumentParser(description="Run correctness checks for overlap/intersection/containment.")
    parser.add_argument(
        "--operations",
        nargs="+",
        default=["overlap", "intersection", "containment"],
        choices=["overlap", "intersection", "containment"],
        help="Operations to test.",
    )
    parser.add_argument(
        "--approaches",
        nargs="+",
        default=["rayspace", "cgal", "touch"],
        choices=["rayspace", "cgal", "touch"],
        help="Approaches to run. TOUCH is overlap-only.",
    )
    parser.add_argument(
        "--use-anyhit-point-in-mesh",
        action="store_true",
        help="Use AnyHit point-in-mesh mode for RaySpace containment runs",
    )
    args = parser.parse_args()

    manifest = _load_manifest()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "metadata": {
            "timestamp": time.strftime("%Y%m%d_%H%M%S"),
            "operations": args.operations,
            "approaches": args.approaches,
            "notes": [
                "TDBase intentionally excluded from this correctness suite.",
                "Touching-only geometries are intentionally excluded from datasets.",
                "RaySpace overlap executes exact + direct_estimation; intersection executes estimated only.",
                "RaySpace containment has no estimated binary in this codebase and is run in exact mode only.",
                f"RaySpace containment AnyHit point-in-mesh: {'enabled' if args.use_anyhit_point_in_mesh else 'disabled'}.",
                f"Estimated-mode tolerance: {ESTIMATED_RELATIVE_TOLERANCE:.2%} relative error.",
                f"CGAL query tolerance (intersection/containment): {CGAL_QUERY_RELATIVE_TOLERANCE:.2%} relative error.",
            ],
        },
        "results": {},
    }

    if "overlap" in args.operations:
        summary["results"]["overlap"] = run_overlap_checks(manifest, args.approaches)
    if "intersection" in args.operations:
        summary["results"]["intersection"] = run_intersection_checks(manifest, args.approaches)
    if "containment" in args.operations:
        summary["results"]["containment"] = run_containment_checks(
            manifest,
            args.approaches,
            use_anyhit_point_in_mesh=args.use_anyhit_point_in_mesh,
        )

    all_checks = []
    for operation_result in summary["results"].values():
        for dataset_result in operation_result.values():
            for approach_result in dataset_result.values():
                all_checks.append(bool(approach_result.get("pass", False)))

    summary["status"] = "PASS" if all_checks and all(all_checks) else "FAIL"

    output_path = RUNS_DIR / f"correctness_{summary['metadata']['timestamp']}.json"
    latest_path = RUNS_DIR / "correctness_latest.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote: {output_path}")
    print(f"Wrote: {latest_path}")
    print(f"Overall status: {summary['status']}")

    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
