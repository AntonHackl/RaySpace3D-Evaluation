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
from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json, write_latest_json_alias
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


@dataclass(frozen=True)
class ObjectHitStats:
    ray_hits: int
    target_count: int


PairHitMap = Dict[str, Dict[Tuple[int, int], int]]


def _axis_overlap(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    return min(a_max, b_max) - max(a_min, b_min)


def _write_bbox_report(
    output_txt: Path,
    sampled_pairs: List[Tuple[int, int]],
    aabbs_a: Dict[int, AABB],
    aabbs_b: Dict[int, AABB],
    mesh1_to_mesh2_hits: Optional[Dict[int, ObjectHitStats]] = None,
    mesh2_to_mesh1_hits: Optional[Dict[int, ObjectHitStats]] = None,
    pair_hits: Optional[PairHitMap] = None,
) -> None:
    with open(output_txt, "w", encoding="utf-8") as handle:
        handle.write("RaySpace-only Pair Inspection (Bounding Boxes)\n")
        handle.write("=================================================\n\n")
        for idx, (a_id, b_id) in enumerate(sampled_pairs, start=1):
            a = aabbs_a.get(a_id)
            b = aabbs_b.get(b_id)
            handle.write(f"Pair {idx}: A={a_id}, B={b_id}\n")
            if a is None or b is None:
                handle.write("  Missing AABB data for one or both objects.\n\n")
                continue

            ox = _axis_overlap(a.min_x, a.max_x, b.min_x, b.max_x)
            oy = _axis_overlap(a.min_y, a.max_y, b.min_y, b.max_y)
            oz = _axis_overlap(a.min_z, a.max_z, b.min_z, b.max_z)

            handle.write(
                f"  A bbox: min=({a.min_x:.6f}, {a.min_y:.6f}, {a.min_z:.6f}) "
                f"max=({a.max_x:.6f}, {a.max_y:.6f}, {a.max_z:.6f})\n"
            )
            handle.write(
                f"  B bbox: min=({b.min_x:.6f}, {b.min_y:.6f}, {b.min_z:.6f}) "
                f"max=({b.max_x:.6f}, {b.max_y:.6f}, {b.max_z:.6f})\n"
            )
            handle.write(f"  Axis overlaps: dx={ox:.6f}, dy={oy:.6f}, dz={oz:.6f}\n")
            handle.write(
                "  AABB intersects (strict): "
                f"{(ox > 0.0 and oy > 0.0 and oz > 0.0)}\n\n"
            )

            a_hits = (mesh1_to_mesh2_hits or {}).get(a_id)
            b_hits = (mesh2_to_mesh1_hits or {}).get(b_id)
            if a_hits is not None:
                handle.write(
                    "  A as source (mesh1->mesh2): "
                    f"ray_hits={a_hits.ray_hits}, target_count={a_hits.target_count}\n"
                )
            else:
                handle.write("  A as source (mesh1->mesh2): no hit tracking data\n")
            if b_hits is not None:
                handle.write(
                    "  B as source (mesh2->mesh1): "
                    f"ray_hits={b_hits.ray_hits}, target_count={b_hits.target_count}\n\n"
                )
            else:
                handle.write("  B as source (mesh2->mesh1): no hit tracking data\n\n")

            pair_hit_a_to_b = (pair_hits or {}).get("mesh1_to_mesh2", {}).get((a_id, b_id))
            pair_hit_b_to_a = (pair_hits or {}).get("mesh2_to_mesh1", {}).get((b_id, a_id))
            if pair_hit_a_to_b is not None:
                handle.write(f"  Specific pair hits A->B: {pair_hit_a_to_b}\n")
            else:
                handle.write("  Specific pair hits A->B: no data\n")
            if pair_hit_b_to_a is not None:
                handle.write(f"  Specific pair hits B->A: {pair_hit_b_to_a}\n\n")
            else:
                handle.write("  Specific pair hits B->A: no data\n\n")


def _plot_pair_bbox_projections(
    output_png: Path,
    pair: Tuple[int, int],
    a_box: AABB,
    b_box: AABB,
) -> None:
    import importlib

    plt = importlib.import_module("matplotlib.pyplot")
    patches = importlib.import_module("matplotlib.patches")
    Rectangle = patches.Rectangle

    a_id, b_id = pair
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    projections = [
        ("XY", (a_box.min_x, a_box.min_y), a_box.max_x - a_box.min_x, a_box.max_y - a_box.min_y,
         (b_box.min_x, b_box.min_y), b_box.max_x - b_box.min_x, b_box.max_y - b_box.min_y),
        ("XZ", (a_box.min_x, a_box.min_z), a_box.max_x - a_box.min_x, a_box.max_z - a_box.min_z,
         (b_box.min_x, b_box.min_z), b_box.max_x - b_box.min_x, b_box.max_z - b_box.min_z),
        ("YZ", (a_box.min_y, a_box.min_z), a_box.max_y - a_box.min_y, a_box.max_z - a_box.min_z,
         (b_box.min_y, b_box.min_z), b_box.max_y - b_box.min_y, b_box.max_z - b_box.min_z),
    ]

    for ax, (title, a_xy, a_w, a_h, b_xy, b_w, b_h) in zip(axes, projections):
        rect_a = Rectangle(a_xy, a_w, a_h, fill=False, linewidth=2.0, edgecolor="tab:blue", label="A")
        rect_b = Rectangle(b_xy, b_w, b_h, fill=False, linewidth=2.0, edgecolor="tab:orange", label="B")
        ax.add_patch(rect_a)
        ax.add_patch(rect_b)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)

        xs = [a_xy[0], a_xy[0] + a_w, b_xy[0], b_xy[0] + b_w]
        ys = [a_xy[1], a_xy[1] + a_h, b_xy[1], b_xy[1] + b_h]
        x_pad = (max(xs) - min(xs)) * 0.1 + 1e-6
        y_pad = (max(ys) - min(ys)) * 0.1 + 1e-6
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    ox = _axis_overlap(a_box.min_x, a_box.max_x, b_box.min_x, b_box.max_x)
    oy = _axis_overlap(a_box.min_y, a_box.max_y, b_box.min_y, b_box.max_y)
    oz = _axis_overlap(a_box.min_z, a_box.max_z, b_box.min_z, b_box.max_z)
    fig.suptitle(
        f"Pair A={a_id}, B={b_id} | overlaps dx={ox:.4f}, dy={oy:.4f}, dz={oz:.4f}",
        fontsize=11,
    )
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=180)
    plt.close(fig)


def _visualize_pairs(
    output_dir: Path,
    sampled_pairs: List[Tuple[int, int]],
    aabbs_a: Dict[int, AABB],
    aabbs_b: Dict[int, AABB],
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[str] = []
    for idx, pair in enumerate(sampled_pairs, start=1):
        a_id, b_id = pair
        a = aabbs_a.get(a_id)
        b = aabbs_b.get(b_id)
        if a is None or b is None:
            continue
        out_png = output_dir / f"pair_{idx:02d}_A{a_id}_B{b_id}.png"
        _plot_pair_bbox_projections(out_png, pair, a, b)
        generated.append(str(out_png))
    return generated


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


def _read_object_hits_csv(path: Path) -> Dict[str, Dict[int, ObjectHitStats]]:
    per_direction: Dict[str, Dict[int, ObjectHitStats]] = {
        "mesh1_to_mesh2": {},
        "mesh2_to_mesh1": {},
    }
    if not path.exists():
        return per_direction

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            direction = row.get("direction", "").strip()
            if direction not in per_direction:
                continue
            try:
                source_object_id = int(row.get("source_object_id", ""))
                ray_hits = int(row.get("ray_hits", ""))
                target_count = int(row.get("target_count", ""))
            except ValueError:
                continue
            per_direction[direction][source_object_id] = ObjectHitStats(
                ray_hits=ray_hits,
                target_count=target_count,
            )

    return per_direction


def _read_pair_hits_csv(path: Path) -> PairHitMap:
    per_direction: PairHitMap = {
        "mesh1_to_mesh2": {},
        "mesh2_to_mesh1": {},
    }
    if not path.exists():
        return per_direction

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            direction = row.get("direction", "").strip()
            if direction not in per_direction:
                continue
            try:
                source_object_id = int(row.get("source_object_id", ""))
                target_object_id = int(row.get("target_object_id", ""))
                target_ray_hits = int(row.get("target_ray_hits", ""))
            except ValueError:
                continue
            per_direction[direction][(source_object_id, target_object_id)] = target_ray_hits

    return per_direction


def _prepare_preprocessed(mesh_a: str, mesh_b: str, grid_cell_size: int):
    prep = RaytracerIntersectionAdapter(
        str(RAYSPACE_DIR),
        mode="estimated",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_cell_size=grid_cell_size,
        warmup_runs=1,
    )
    prep.preprocess_from_source(mesh_a, _as_dt_name(mesh_a))
    prep.preprocess_from_source(mesh_b, _as_dt_name(mesh_b))


def _run_rayspace_intersection_pairs(
    pre_mesh_a: Path,
    pre_mesh_b: Path,
    output_csv: Path,
    output_timing_json: Path,
    output_object_hits_csv: Optional[Path],
    output_pair_hits_csv: Optional[Path],
    run_dir: Path,
    warmup_runs: int,
    use_anyhit_containment: bool,
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
        "--pairs-output",
        str(output_csv),
    ]
    if output_pair_hits_csv is not None:
        cmd.extend([
            "--pair-hits-output",
            str(output_pair_hits_csv),
        ])
    if use_anyhit_containment:
        cmd.append("--use-anyhit-containment")

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

    pairs: Set[Tuple[int, int]] = _read_pairs_csv(output_csv)

    return {
        "pairs": pairs,
        "pairs_csv": str(output_csv),
        "object_hits_csv": str(output_object_hits_csv) if output_object_hits_csv is not None else None,
        "pair_hits_csv": str(output_pair_hits_csv) if output_pair_hits_csv is not None else None,
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
    parser.add_argument("--max-eval-pairs", type=float, default=1.0000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=None, help="CGAL threads")
    parser.add_argument("--grid-cell-size", type=float, default=1.0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--mesh-a", type=str, default=None, help="Override mesh A path")
    parser.add_argument("--mesh-b", type=str, default=None, help="Override mesh B path")
    parser.add_argument("--inspect-only-rayspace", type=float, default=1.0,
                        help="Number of random RaySpace-only disagreement pairs to inspect with bbox report and images")
    parser.add_argument("--rayspace-pairs-csv", type=str, default=None,
                        help="Optional existing RaySpace pairs CSV to analyze (skips running RaySpace query)")
    parser.add_argument("--cgal-pairs-csv", type=str, default=None,
                        help="Optional existing CGAL pairs CSV to analyze (skips running CGAL query)")
    parser.add_argument("--rayspace-object-hits-csv", type=str, default=None,
                        help="Optional existing RaySpace object hit tracking CSV")
    parser.add_argument("--rayspace-pair-hits-csv", type=str, default=None,
                        help="Optional existing RaySpace per-pair hit tracking CSV")
    parser.add_argument("--use-anyhit-containment", action="store_true",
                        help="Run RaySpace estimated intersection with AnyHit containment mode")
    args = parser.parse_args()

    if args.max_eval_pairs <= 0:
        raise ValueError("--max-eval-pairs must be positive")

    manifest = _load_manifest()
    default_cfg = manifest["cubes_20k_sel_0_001"]
    mesh_a = args.mesh_a or str(default_cfg["mesh_a"])
    mesh_b = args.mesh_b or str(default_cfg["mesh_b"])

    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "intersection_disagreement")
    run_name = run_layout["run_name"]
    run_dir = Path(run_layout["run_dir"])

    rs_pairs_csv = run_dir / "rayspace_intersection_pairs.csv"
    rs_timing_json = run_dir / "rayspace_intersection_timing.json"
    cgal_pairs_csv = run_dir / "cgal_intersection_pairs.csv"
    rs_object_hits_csv = run_dir / "rayspace_object_hits.csv"
    rs_pair_hits_csv = run_dir / "rayspace_pair_hits.csv"
    cgal_log = run_dir / "cgal_intersection.log"
    details_csv = run_dir / "sampled_pair_decisions.csv"
    report_json = run_dir / "summary.json"
    results_json = Path(run_layout["results_json"])
    latest_json = RUNS_DIR / "intersection_disagreement_latest.json"

    use_existing_pair_csvs = bool(args.rayspace_pairs_csv and args.cgal_pairs_csv)
    if use_existing_pair_csvs:
        print("Using existing pair CSVs for disagreement analysis...")
        rs_pairs = _read_pairs_csv(Path(args.rayspace_pairs_csv))
        cgal_pairs = _read_pairs_csv(Path(args.cgal_pairs_csv))
        rs_result = {
            "pairs": rs_pairs,
            "pairs_csv": str(args.rayspace_pairs_csv),
            "object_hits_csv": args.rayspace_object_hits_csv,
            "pair_hits_csv": args.rayspace_pair_hits_csv,
            "num_pairs_from_csv": len(rs_pairs),
            "num_pairs_reported": len(rs_pairs),
            "timing_ms": None,
        }
        cgal_result = {
            "pairs": cgal_pairs,
            "pairs_csv": str(args.cgal_pairs_csv),
            "num_pairs_from_csv": len(cgal_pairs),
            "num_pairs_reported": len(cgal_pairs),
            "timing_ms": None,
        }
    else:
        print("Preparing preprocessed inputs...")
        _prepare_preprocessed(mesh_a, mesh_b, args.grid_cell_size)

        pre_a = PREPROCESSED_DIR / Path(mesh_a).with_suffix(".pre").name
        pre_b = PREPROCESSED_DIR / Path(mesh_b).with_suffix(".pre").name

        print("Running RaySpace intersection (estimated mode)...")
        rs_result = _run_rayspace_intersection_pairs(
            pre_a,
            pre_b,
            rs_pairs_csv,
            rs_timing_json,
            rs_object_hits_csv,
            rs_pair_hits_csv,
            run_dir,
            args.warmup_runs,
            args.use_anyhit_containment,
        )
        if "error" in rs_result:
            raise RuntimeError(f"RaySpace intersection failed: {rs_result['error']}")

        print("Running CGAL intersection with pair export...")
        cgal_result = _run_cgal_intersection_pairs(pre_a, pre_b, cgal_pairs_csv, args.threads, cgal_log)
        if "error" in cgal_result:
            raise RuntimeError(cgal_result["error"])

    rs_pairs = rs_result["pairs"]
    cgal_pairs = cgal_result["pairs"]
    object_hits_csv_path = rs_result.get("object_hits_csv")
    pair_hits_csv_path = rs_result.get("pair_hits_csv")
    object_hits: Dict[str, Dict[int, ObjectHitStats]] = {
        "mesh1_to_mesh2": {},
        "mesh2_to_mesh1": {},
    }
    pair_hits: PairHitMap = {
        "mesh1_to_mesh2": {},
        "mesh2_to_mesh1": {},
    }
    if object_hits_csv_path:
        object_hits = _read_object_hits_csv(Path(object_hits_csv_path))
    if pair_hits_csv_path:
        pair_hits = _read_pair_hits_csv(Path(pair_hits_csv_path))

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

    inspected_only_rs: List[Tuple[int, int]] = []
    bbox_report_txt = run_dir / "only_rayspace_sample_bboxes.txt"
    visuals_dir = run_dir / "only_rayspace_visualizations"
    generated_visuals: List[str] = []
    if args.inspect_only_rayspace > 0 and only_rs and aabbs_a and aabbs_b:
        inspect_count = min(args.inspect_only_rayspace, len(only_rs))
        inspected_only_rs = random.Random(args.seed + 1001).sample(only_rs, inspect_count)
        _write_bbox_report(
            bbox_report_txt,
            inspected_only_rs,
            aabbs_a,
            aabbs_b,
            object_hits.get("mesh1_to_mesh2"),
            object_hits.get("mesh2_to_mesh1"),
            pair_hits,
        )
        generated_visuals = _visualize_pairs(visuals_dir, inspected_only_rs, aabbs_a, aabbs_b)

    summary = {
        "metadata": {
            "run_name": run_name,
            "timestamp": run_layout["timestamp"],
            "run_dir": str(run_dir),
            "mesh_a": mesh_a,
            "mesh_b": mesh_b,
            "max_eval_pairs": args.max_eval_pairs,
            "seed": args.seed,
            "threads": args.threads,
            "rayspace_mode": "estimated",
            "use_anyhit_containment": args.use_anyhit_containment,
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
            "inspected_only_rayspace_pairs": len(inspected_only_rs),
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
            "rayspace_object_hits_csv": object_hits_csv_path,
            "rayspace_pair_hits_csv": pair_hits_csv_path,
            "rayspace_timing_json": str(rs_timing_json),
            "cgal_pairs_csv": str(cgal_pairs_csv),
            "cgal_log": str(cgal_log),
            "sampled_decisions_csv": str(details_csv),
            "only_rayspace_bbox_report_txt": str(bbox_report_txt) if inspected_only_rs else None,
            "only_rayspace_visualization_dir": str(visuals_dir) if generated_visuals else None,
        },
        "inspected_only_rayspace_pairs": [
            {"a_object_id": int(a), "b_object_id": int(b)} for (a, b) in inspected_only_rs
        ],
        "generated_visualizations": generated_visuals,
        "notes": [
            "Intersection benchmarks now run RaySpace estimated mode only.",
            "Pair-level disagreement sampling requires RaySpace pair CSV export, which is unavailable in the current estimated binary.",
            "Use --rayspace-pairs-csv and --cgal-pairs-csv to inspect historical pair-level disagreement runs.",
            "When object hit tracking CSV is available, inspected RaySpace-only pairs include per-object ray hit and target counts.",
            "When pair hit tracking CSV is available, inspected RaySpace-only pairs include specific A->B and B->A hit counts.",
        ],
    }

    write_json(results_json, summary)
    write_json(report_json, summary)
    write_latest_json_alias(latest_json, summary)

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
    print(f"Results JSON:          {results_json}")
    print(f"Summary JSON:          {report_json}")
    print(f"Latest JSON:           {latest_json}")
    print(f"Details CSV:           {details_csv}")
    if inspected_only_rs:
        print(f"BBox report (.txt):    {bbox_report_txt}")
        print(f"Visualization dir:     {visuals_dir}")


if __name__ == "__main__":
    main()
