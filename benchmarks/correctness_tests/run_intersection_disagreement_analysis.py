#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.adapters.base import run_command_streaming
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter

RAW_DIR = SCRIPT_DIR / "data" / "raw"
PREPROCESSED_DIR = SCRIPT_DIR / "data" / "preprocessed"
TIMINGS_DIR = SCRIPT_DIR / "data" / "timings"
RUNS_DIR = SCRIPT_DIR / "runs"
MANIFEST_PATH = RAW_DIR / "manual_expected_results.json"

RAYSPACE_DIR = REPO_ROOT / "src" / "RaySpace3D"
CGAL_DIR = REPO_ROOT / "baselines" / "RaySpace3DBaselines" / "CGAL"
CGAL_EXEC = CGAL_DIR / "build" / "cgal_intersection"
TIMEOUT_SECONDS = 1800.0


@dataclass(frozen=True)
class AABB:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float


def _as_dt_name(path_str: str) -> str:
    return str(Path(path_str).with_suffix(".dt"))


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing dataset manifest at {MANIFEST_PATH}. Run generate_datasets.py first."
        )
    with open(MANIFEST_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_pairs_csv(path: Path) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    if not path.exists():
        return pairs

    with open(path, "r", encoding="utf-8") as handle:
        first = handle.readline().strip()
        if first:
            parts = first.split(",")
            if len(parts) == 2:
                try:
                    pairs.add((int(parts[0]), int(parts[1])))
                except ValueError:
                    pass

        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            try:
                pairs.add((int(parts[0]), int(parts[1])))
            except ValueError:
                continue

    return pairs


def _prepare_preprocessed(mesh_a: str, mesh_b: str, grid_resolution: int):
    prep = RaytracerIntersectionAdapter(
        str(RAYSPACE_DIR),
        mode="estimated",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_resolution=grid_resolution,
        warmup_runs=1,
    )
    prep.preprocess_from_source(mesh_a, _as_dt_name(mesh_a))
    prep.preprocess_from_source(mesh_b, _as_dt_name(mesh_b))


def _run_rayspace_intersection_pairs(
    pre_mesh_a: Path,
    pre_mesh_b: Path,
    output_csv: Path,
    output_timing_json: Path,
    run_dir: Path,
    warmup_runs: int,
) -> Dict[str, Any]:
    executable = RAYSPACE_DIR / "query" / "build" / "bin" / "raytracer_intersection_estimated"
    if not executable.exists():
        return {"error": f"Executable not found: {executable}"}

    cmd = [
        str(executable),
        "--mesh1",
        str(pre_mesh_a),
        "--mesh2",
        str(pre_mesh_b),
        "--output",
        str(output_timing_json),
    ]

    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=str(run_dir),
            check=True,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"RaySpace intersection timed out ({TIMEOUT_SECONDS}s)"}
    except subprocess.CalledProcessError as exc:
        return {"error": f"RaySpace intersection failed with exit code {exc.returncode}: {exc.stderr}"}

    output = cp.stdout + cp.stderr

    timing_match = re.search(r"Query Time:.*?\(([\d.]+) ms\)", output)
    count_match = re.search(r"Actual Intersection Pairs:\s*(\d+)", output)

    # Estimated intersection currently exposes counts/timings, but not pair-level CSV export.
    pairs: Set[Tuple[int, int]] = set()

    return {
        "pairs": pairs,
        "pairs_csv": None,
        "num_pairs_from_csv": len(pairs),
        "num_pairs_reported": int(count_match.group(1)) if count_match else None,
        "timing_ms": float(timing_match.group(1)) if timing_match else None,
        "stdout_stderr": output,
    }


def _run_cgal_intersection_pairs(
    pre_mesh_a: Path,
    pre_mesh_b: Path,
    output_csv: Path,
    threads: Optional[int],
    log_path: Path,
) -> Dict[str, Any]:
    if not CGAL_EXEC.exists():
        return {"error": f"Executable not found: {CGAL_EXEC}"}

    if output_csv.exists():
        output_csv.unlink()

    cmd = [str(CGAL_EXEC), str(pre_mesh_a), str(pre_mesh_b)]
    if threads and threads > 0:
        cmd.append(str(threads))
    cmd.extend(["--output-csv", str(output_csv)])

    try:
        stdout_text, stderr_text = run_command_streaming(
            cmd,
            timeout=TIMEOUT_SECONDS,
            log_path=str(log_path),
            prefix="[CGAL]",
        )
    except Exception as exc:
        return {"error": f"CGAL intersection failed: {exc}"}

    output = stdout_text + stderr_text
    if not output_csv.exists():
        return {
            "error": (
                "CGAL intersection did not produce a pair CSV. Ensure CGAL is rebuilt with "
                "intersection_query.cpp support for --output-csv."
            ),
            "stdout_stderr": output,
        }

    count_match = re.search(r"Unique intersecting object pairs:\s*(\d+)", output)
    timing_match = re.search(r"Query Time:.*?\(([\d.]+) ms\)", output)

    pairs = _read_pairs_csv(output_csv)
    return {
        "pairs": pairs,
        "pairs_csv": str(output_csv),
        "num_pairs_from_csv": len(pairs),
        "num_pairs_reported": int(count_match.group(1)) if count_match else None,
        "timing_ms": float(timing_match.group(1)) if timing_match else None,
    }


def _is_intersection(a: AABB, b: AABB, eps: float = 1e-9) -> bool:
    overlap_x = min(a.max_x, b.max_x) - max(a.min_x, b.min_x)
    overlap_y = min(a.max_y, b.max_y) - max(a.min_y, b.min_y)
    overlap_z = min(a.max_z, b.max_z) - max(a.min_z, b.min_z)
    return overlap_x > eps and overlap_y > eps and overlap_z > eps


def _parse_obj_aabbs(path: Path) -> Dict[int, AABB]:
    object_points: Dict[int, List[Tuple[float, float, float]]] = {}
    current_obj: Optional[int] = None
    fallback_obj = 0

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("o "):
                name = line[2:].strip()
                match = re.search(r"(\d+)$", name)
                if match:
                    current_obj = int(match.group(1))
                else:
                    current_obj = fallback_obj
                    fallback_obj += 1
                object_points.setdefault(current_obj, [])
                continue

            if line.startswith("v "):
                if current_obj is None:
                    current_obj = fallback_obj
                    fallback_obj += 1
                    object_points.setdefault(current_obj, [])

                parts = line.split()
                if len(parts) < 4:
                    continue
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                object_points[current_obj].append((x, y, z))

    boxes: Dict[int, AABB] = {}
    for obj_id, points in object_points.items():
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        zs = [p[2] for p in points]
        boxes[obj_id] = AABB(
            min_x=min(xs),
            min_y=min(ys),
            min_z=min(zs),
            max_x=max(xs),
            max_y=max(ys),
            max_z=max(zs),
        )

    return boxes


def _sample_disagreements(
    only_rs: List[Tuple[int, int]],
    only_cgal: List[Tuple[int, int]],
    max_eval_pairs: int,
    seed: int,
) -> List[Tuple[Tuple[int, int], str]]:
    total = len(only_rs) + len(only_cgal)
    sample_size = min(max_eval_pairs, total)
    if sample_size <= 0:
        return []

    rng = random.Random(seed)

    target_rs = int(round(sample_size * (len(only_rs) / total))) if total else 0
    target_rs = min(target_rs, len(only_rs))
    target_cgal = min(sample_size - target_rs, len(only_cgal))

    while target_rs + target_cgal < sample_size:
        if target_rs < len(only_rs):
            target_rs += 1
        elif target_cgal < len(only_cgal):
            target_cgal += 1
        else:
            break

    sampled_rs = rng.sample(only_rs, target_rs) if target_rs > 0 else []
    sampled_cgal = rng.sample(only_cgal, target_cgal) if target_cgal > 0 else []

    tagged = [(pair, "only_rayspace") for pair in sampled_rs]
    tagged.extend((pair, "only_cgal") for pair in sampled_cgal)
    rng.shuffle(tagged)
    return tagged


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run RaySpace estimated intersection and CGAL intersection on cubes_20k, then "
            "sample pair disagreements and adjudicate with a cube AABB intersection predicate."
        )
    )
    parser.add_argument("--max-eval-pairs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None, help="CGAL threads")
    parser.add_argument("--grid-resolution", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--mesh-a", type=str, default=None, help="Override mesh A path")
    parser.add_argument("--mesh-b", type=str, default=None, help="Override mesh B path")
    args = parser.parse_args()

    if args.max_eval_pairs <= 0:
        raise ValueError("--max-eval-pairs must be positive")

    manifest = _load_manifest()
    default_cfg = manifest["cubes_20k_sel_0_001"]
    mesh_a = args.mesh_a or str(default_cfg["mesh_a"])
    mesh_b = args.mesh_b or str(default_cfg["mesh_b"])

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    run_name = f"intersection_disagreement_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    rs_pairs_csv = run_dir / "rayspace_intersection_pairs.csv"
    rs_timing_json = run_dir / "rayspace_intersection_timing.json"
    cgal_pairs_csv = run_dir / "cgal_intersection_pairs.csv"
    cgal_log = run_dir / "cgal_intersection.log"
    details_csv = run_dir / "sampled_pair_decisions.csv"
    report_json = run_dir / "summary.json"
    latest_json = RUNS_DIR / "intersection_disagreement_latest.json"

    print("Preparing preprocessed inputs...")
    _prepare_preprocessed(mesh_a, mesh_b, args.grid_resolution)

    pre_a = PREPROCESSED_DIR / Path(mesh_a).with_suffix(".pre").name
    pre_b = PREPROCESSED_DIR / Path(mesh_b).with_suffix(".pre").name

    print("Running RaySpace intersection (estimated mode)...")
    rs_result = _run_rayspace_intersection_pairs(
        pre_a,
        pre_b,
        rs_pairs_csv,
        rs_timing_json,
        run_dir,
        args.warmup_runs,
    )
    if "error" in rs_result:
        raise RuntimeError(f"RaySpace intersection failed: {rs_result['error']}")

    print("Running CGAL intersection with pair export...")
    cgal_result = _run_cgal_intersection_pairs(pre_a, pre_b, cgal_pairs_csv, args.threads, cgal_log)
    if "error" in cgal_result:
        raise RuntimeError(cgal_result["error"])

    rs_pairs = rs_result["pairs"]
    cgal_pairs = cgal_result["pairs"]

    if rs_pairs:
        only_rs = sorted(rs_pairs - cgal_pairs)
        only_cgal = sorted(cgal_pairs - rs_pairs)
        agreed = len(rs_pairs & cgal_pairs)
        sampled = _sample_disagreements(only_rs, only_cgal, args.max_eval_pairs, args.seed)
    else:
        only_rs = []
        only_cgal = []
        agreed = 0
        sampled = []

    if sampled:
        print("Building cube AABB maps for truth adjudication...")
        aabbs_a = _parse_obj_aabbs(Path(mesh_a))
        aabbs_b = _parse_obj_aabbs(Path(mesh_b))
    else:
        aabbs_a = {}
        aabbs_b = {}

    decisions: List[Dict[str, object]] = []
    rayspace_correct = 0
    cgal_correct = 0
    inconclusive = 0

    for (a_id, b_id), bucket in sampled:
        a_box = aabbs_a.get(a_id)
        b_box = aabbs_b.get(b_id)
        if a_box is None or b_box is None:
            winner = "inconclusive"
            truth = None
            inconclusive += 1
        else:
            truth = _is_intersection(a_box, b_box)
            if bucket == "only_rayspace":
                winner = "rayspace" if truth else "cgal"
            else:
                winner = "cgal" if truth else "rayspace"
            if winner == "rayspace":
                rayspace_correct += 1
            elif winner == "cgal":
                cgal_correct += 1

        decisions.append(
            {
                "a_object_id": a_id,
                "b_object_id": b_id,
                "bucket": bucket,
                "truth_intersection": truth,
                "winner": winner,
            }
        )

    with open(details_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "a_object_id",
                "b_object_id",
                "bucket",
                "truth_intersection",
                "winner",
            ],
        )
        writer.writeheader()
        writer.writerows(decisions)

    summary = {
        "metadata": {
            "run_name": run_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mesh_a": mesh_a,
            "mesh_b": mesh_b,
            "max_eval_pairs": args.max_eval_pairs,
            "seed": args.seed,
            "threads": args.threads,
            "rayspace_mode": "estimated",
        },
        "pair_counts": {
            "rayspace_pairs": len(rs_pairs),
            "rayspace_pairs_reported": rs_result.get("num_pairs_reported"),
            "cgal_pairs": len(cgal_pairs),
            "agreed_pairs": agreed,
            "only_rayspace": len(only_rs),
            "only_cgal": len(only_cgal),
            "total_disagreements": len(only_rs) + len(only_cgal),
        },
        "sample": {
            "sampled_pairs": len(sampled),
            "sampled_only_rayspace": sum(1 for _, b in sampled if b == "only_rayspace"),
            "sampled_only_cgal": sum(1 for _, b in sampled if b == "only_cgal"),
        },
        "adjudication": {
            "rayspace_correct": rayspace_correct,
            "cgal_correct": cgal_correct,
            "inconclusive": inconclusive,
            "evaluated": len(sampled) - inconclusive,
            "rayspace_correct_rate": (
                rayspace_correct / (len(sampled) - inconclusive)
                if (len(sampled) - inconclusive) > 0
                else None
            ),
            "cgal_correct_rate": (
                cgal_correct / (len(sampled) - inconclusive)
                if (len(sampled) - inconclusive) > 0
                else None
            ),
        },
        "artifacts": {
            "rayspace_pairs_csv": rs_result.get("pairs_csv"),
            "rayspace_timing_json": str(rs_timing_json),
            "cgal_pairs_csv": str(cgal_pairs_csv),
            "cgal_log": str(cgal_log),
            "sampled_decisions_csv": str(details_csv),
        },
        "notes": [
            "Intersection benchmarks now run RaySpace estimated mode only.",
            "Pair-level disagreement sampling requires RaySpace pair CSV export, which is unavailable in the current estimated binary.",
        ],
    }

    with open(report_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(latest_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\n=== Intersection Disagreement Analysis ===")
    print(f"RaySpace pairs:        {len(rs_pairs)}")
    print(f"CGAL pairs:            {len(cgal_pairs)}")
    print(f"Agreed pairs:          {agreed}")
    print(f"Only RaySpace:         {len(only_rs)}")
    print(f"Only CGAL:             {len(only_cgal)}")
    if rs_result.get("num_pairs_reported") is not None:
        print(f"RaySpace reported:     {rs_result['num_pairs_reported']}")
    print(f"Sampled disagreements: {len(sampled)}")
    print(f"RaySpace correct:      {rayspace_correct}")
    print(f"CGAL correct:          {cgal_correct}")
    print(f"Inconclusive:          {inconclusive}")
    print(f"Summary JSON:          {report_json}")
    print(f"Latest JSON:           {latest_json}")
    print(f"Details CSV:           {details_csv}")


if __name__ == "__main__":
    main()
