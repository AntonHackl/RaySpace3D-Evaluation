#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, Set, Tuple
from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json

# Add current directory to path to import adapters
import sys
sys.path.append(str(Path(__file__).parent))

from adapters.raytracer_adapter import RaytracerAdapter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"

DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
RUNS_DIR = SCRIPT_DIR / "runs"

TIMEOUT_SECONDS = 120.0
DEFAULT_NU_COUNTS = [200, 400, 600, 800]


def find_dataset_files(nu: int):
    candidates_v = list(RAW_DIR.glob(f"*_v_*nu{nu}*.dt"))
    candidates_v = [c for c in candidates_v if "nv150" in c.name]

    candidates_n = list(RAW_DIR.glob(f"*_n_*nu{nu}*.dt"))
    candidates_n = [c for c in candidates_n if "nv150" in c.name and "_n2_" not in c.name]

    if not candidates_v or not candidates_n:
        return None, None

    candidates_v.sort(key=lambda x: len(x.name))
    candidates_n.sort(key=lambda x: len(x.name))
    return candidates_v[0], candidates_n[0]


def read_pairs_csv(path: Path) -> Set[Tuple[int, int]]:
    if not path.exists():
        return set()

    pairs: Set[Tuple[int, int]] = set()
    with open(path, "r") as f:
        header = f.readline()
        if "object_id_mesh1" not in header:
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 2:
                continue
            try:
                a = int(parts[0])
                b = int(parts[1])
            except ValueError:
                continue
            pairs.add((a, b))
    return pairs


def compute_metrics(pred: Set[Tuple[int, int]], gt: Set[Tuple[int, int]]) -> Dict[str, float]:
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_pairs": len(pred),
        "gt_pairs": len(gt),
    }


def run_experiment(runs: int, grid_resolution: int, nu_counts, run_layout):
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = run_layout["timestamp"]
    run_name = run_layout["run_name"]
    run_log_dir = Path(run_layout["logs_dir"])

    direct_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR),
        mode="direct_estimation",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_resolution=grid_resolution,
        warmup_runs=1,
    )

    summary = {
        "timestamp": timestamp,
        "run_name": run_name,
        "run_dir": str(run_layout["run_dir"]),
        "runs": runs,
        "grid_resolution": grid_resolution,
        "results": [],
    }

    for nu in nu_counts:
        f_v_path, f_n_path = find_dataset_files(nu)
        if not f_v_path or not f_n_path:
            print(f"[nu={nu}] Dataset files not found. Skipping.")
            continue

        print(f"\n[nu={nu}] {f_v_path.name} vs {f_n_path.name}")
        direct_adapter.preprocess_from_source(str(f_v_path), str(f_v_path), log_dir=str(run_log_dir))
        direct_adapter.preprocess_from_source(str(f_n_path), str(f_n_path), log_dir=str(run_log_dir))

        pairs_dir = run_log_dir / f"pairs_nu{nu}"
        pairs_dir.mkdir(parents=True, exist_ok=True)

        gt_pairs_path = pairs_dir / "ground_truth_both.csv"
        small_to_large_pairs_path = pairs_dir / "small_to_large.csv"
        large_to_small_pairs_path = pairs_dir / "large_to_small.csv"

        gt_res = direct_adapter.run_overlap(
            str(f_v_path),
            str(f_n_path),
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS,
            query_direction="both",
            pairs_output=str(gt_pairs_path),
        )
        if "error" in gt_res:
            print(f"[nu={nu}] Ground truth run failed: {gt_res['error']}")
            continue

        mesh1_objects = int(gt_res.get("num_obj1", 0))
        mesh2_objects = int(gt_res.get("num_obj2", 0))

        if mesh1_objects <= mesh2_objects:
            dir_small_to_large = "mesh1_to_mesh2"
            dir_large_to_small = "mesh2_to_mesh1"
        else:
            dir_small_to_large = "mesh2_to_mesh1"
            dir_large_to_small = "mesh1_to_mesh2"

        small_to_large_res = direct_adapter.run_overlap(
            str(f_v_path),
            str(f_n_path),
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS,
            query_direction=dir_small_to_large,
            pairs_output=str(small_to_large_pairs_path),
        )
        if "error" in small_to_large_res:
            print(f"[nu={nu}] Small->Large run failed: {small_to_large_res['error']}")
            continue

        large_to_small_res = direct_adapter.run_overlap(
            str(f_v_path),
            str(f_n_path),
            runs,
            log_dir=str(run_log_dir),
            timeout=TIMEOUT_SECONDS,
            query_direction=dir_large_to_small,
            pairs_output=str(large_to_small_pairs_path),
        )
        if "error" in large_to_small_res:
            print(f"[nu={nu}] Large->Small run failed: {large_to_small_res['error']}")
            continue

        gt_pairs = read_pairs_csv(gt_pairs_path)
        pred_small_to_large = read_pairs_csv(small_to_large_pairs_path)
        pred_large_to_small = read_pairs_csv(large_to_small_pairs_path)

        metrics_small_to_large = compute_metrics(pred_small_to_large, gt_pairs)
        metrics_large_to_small = compute_metrics(pred_large_to_small, gt_pairs)

        nu_result = {
            "nu": nu,
            "dataset": {
                "mesh1": f_v_path.name,
                "mesh2": f_n_path.name,
                "mesh1_objects": mesh1_objects,
                "mesh2_objects": mesh2_objects,
                "small_to_large_direction": dir_small_to_large,
                "large_to_small_direction": dir_large_to_small,
            },
            "query_time_ms": {
                "ground_truth_both": {
                    "mean": gt_res["mean"],
                    "std": gt_res["std"],
                },
                "small_to_large_only": {
                    "mean": small_to_large_res["mean"],
                    "std": small_to_large_res["std"],
                },
                "large_to_small_only": {
                    "mean": large_to_small_res["mean"],
                    "std": large_to_small_res["std"],
                },
            },
            "error_metrics": {
                "small_to_large_only": metrics_small_to_large,
                "large_to_small_only": metrics_large_to_small,
            },
        }
        summary["results"].append(nu_result)

        print(
            f"[nu={nu}] time(ms) both={gt_res['mean']:.2f}, "
            f"small->large={small_to_large_res['mean']:.2f}, "
            f"large->small={large_to_small_res['mean']:.2f}"
        )
        print(
            f"[nu={nu}] recall small->large={metrics_small_to_large['recall']:.4f}, "
            f"large->small={metrics_large_to_small['recall']:.4f}"
        )

    return summary


def print_summary(summary):
    results = summary.get("results", [])
    if not results:
        print("No successful results.")
        return

    print("\n=== Query Time Comparison (Direct Estimation) ===")
    print(f"{'Nu':<8} {'Both(ms)':<12} {'S->L(ms)':<12} {'L->S(ms)':<12}")
    print("-" * 48)
    for row in results:
        nu = row["nu"]
        t = row["query_time_ms"]
        print(
            f"{nu:<8} "
            f"{t['ground_truth_both']['mean']:<12.2f} "
            f"{t['small_to_large_only']['mean']:<12.2f} "
            f"{t['large_to_small_only']['mean']:<12.2f}"
        )

    print("\n=== Error Metrics vs Ground Truth (Both Directions) ===")
    print(f"{'Nu':<8} {'Variant':<18} {'Precision':<12} {'Recall':<12} {'F1':<12} {'TP':<8} {'FP':<8} {'FN':<8}")
    print("-" * 98)
    for row in results:
        nu = row["nu"]
        em = row["error_metrics"]
        for variant_key, label in [
            ("small_to_large_only", "small->large"),
            ("large_to_small_only", "large->small"),
        ]:
            m = em[variant_key]
            print(
                f"{nu:<8} {label:<18} "
                f"{m['precision']:<12.6f} {m['recall']:<12.6f} {m['f1']:<12.6f} "
                f"{m['tp']:<8d} {m['fp']:<8d} {m['fn']:<8d}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Direct Estimation one-way directionality test for mesh overlap"
    )
    parser.add_argument("--runs", type=int, default=2, help="Runs per configuration")
    parser.add_argument("--grid-resolution", type=int, default=20, help="Preprocess grid resolution")
    parser.add_argument("--nu", type=int, nargs='+', help="Nu counts (e.g., 200 400 600 800)")
    args = parser.parse_args()

    nu_counts = args.nu if args.nu else DEFAULT_NU_COUNTS

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_direct_estimation_directionality")
    summary = run_experiment(args.runs, args.grid_resolution, nu_counts, run_layout)
    print_summary(summary)

    out_json = Path(run_layout["results_json"])
    write_json(out_json, summary)
    print(f"\nSaved summary to: {out_json}")


if __name__ == "__main__":
    main()
