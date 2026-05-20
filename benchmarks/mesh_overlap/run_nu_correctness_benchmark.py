#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import style_for
from benchmarks.common.adapters.tdbase_common import (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    TDBASE_TIMING_MODES,
)

sys.path.append(str(SCRIPT_DIR))
from adapters.raytracer_adapter import RaytracerAdapter
from adapters.tdbase_adapter import TDBaseAdapter

RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
TDBASE_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/tdbase"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

PROJECT_TMP_DIR = REPO_ROOT / ".tmp" / "mesh_overlap_correctness"
TIMEOUT_SECONDS = 120.0
DEFAULT_NU_COUNTS = [200, 400, 600, 800]


def find_dataset_files(nu: int) -> Tuple[Optional[Path], Optional[Path]]:
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
    pairs: Set[Tuple[int, int]] = set()
    if not path.exists():
        return pairs

    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
        has_header = "object_id_mesh1" in first
        if not has_header:
            parts = first.strip().split(",")
            if len(parts) == 2:
                try:
                    pairs.add((int(parts[0]), int(parts[1])))
                except ValueError:
                    pass

        for line in f:
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


def read_tdbase_pairs_from_log(path: Path) -> Set[Tuple[int, int]]:
    pairs: Set[Tuple[int, int]] = set()
    if not path.exists():
        return pairs

    pair_line = re.compile(r"^\s*(\d+)\s+(\d+)\s*$")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pair_line.match(line)
            if m:
                pairs.add((int(m.group(1)), int(m.group(2))))
    return pairs


def parse_tdbase_compute_ms(output: str) -> Optional[float]:
    match = re.search(r"compute:\s+([\d.]+)\s+(s|ms)", output)
    if match:
        val = float(match.group(1))
        unit = match.group(2)
        return val * 1000.0 if unit == "s" else val

    comp_matches = re.finditer(r"computation for checking intersection takes ([\d.]+) ms", output)
    values = [float(m.group(1)) for m in comp_matches]
    if values:
        return float(sum(values))

    return None


def run_tdbase_timing_within0(
    tdbase_exec: Path,
    file1: Path,
    file2: Path,
    num_runs: int,
    log_dir: Path,
    timeout: float,
) -> Dict[str, object]:
    lods = [20, 40, 60, 80, 100]
    runtimes: List[float] = []

    cmd = [
        str(tdbase_exec),
        "join",
        "-q", "within",
        "--tile1", str(file1),
        "--tile2", str(file2),
        "-w", "0",
    ]
    for lod in lods:
        cmd.extend(["-l", str(lod)])
    cmd.append("-g")

    for i in range(num_runs):
        run_log = log_dir / f"tdbase_within0_timing_run_{i:03d}.log"
        try:
            cp = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"Timeout reached ({timeout}s)"}
        except subprocess.CalledProcessError as e:
            return {"error": f"TDBase within(w=0) failed with exit code {e.returncode}: {e.stderr}"}

        out = cp.stdout + cp.stderr
        run_log.write_text(out, encoding="utf-8")
        compute_ms = parse_tdbase_compute_ms(out)
        if compute_ms is None:
            return {"error": "Computation timing not found for TDBase within(w=0)"}
        runtimes.append(compute_ms)

    arr = np.array(runtimes, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "raw_times": [float(x) for x in arr.tolist()],
        "lods": lods,
        "gpu": True,
    }


def run_exact_collect_pairs(
    exact_exec: Path,
    pre1: Path,
    pre2: Path,
    output_timing: Path,
    output_pairs_csv: Path,
    warmup_runs: int,
    timeout: float,
    work_dir: Path,
) -> Set[Tuple[int, int]]:
    if output_pairs_csv.exists():
        output_pairs_csv.unlink()

    default_csv = work_dir / "mesh_overlap_results.csv"
    if default_csv.exists():
        default_csv.unlink()

    cmd = [
        str(exact_exec),
        "--mesh1", str(pre1),
        "--mesh2", str(pre2),
        "--runs", "1",
        "--warmup-runs", str(warmup_runs),
        "--output", str(output_timing),
    ]

    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            cwd=str(work_dir),
            check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return set()

    run_log = output_timing.with_suffix(".log")
    run_log.write_text(cp.stdout + cp.stderr, encoding="utf-8")

    if not default_csv.exists():
        return set()

    default_csv.replace(output_pairs_csv)
    return read_pairs_csv(output_pairs_csv)


def run_tdbase_collect_pairs(
    tdbase_exec: Path,
    file1: Path,
    file2: Path,
    query: str,
    output_log: Path,
    timeout: float,
) -> Set[Tuple[int, int]]:
    cmd = [
        str(tdbase_exec),
        "join",
        "-q", query,
        "--tile1", str(file1),
        "--tile2", str(file2),
        "-l", "100",
        "-g",
        "-p",
    ]
    if query == "within":
        cmd.extend(["-w", "0"])

    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return set()

    output_log.write_text(cp.stdout + cp.stderr, encoding="utf-8")
    return read_tdbase_pairs_from_log(output_log)


def compare_sets(a: Set[Tuple[int, int]], b: Set[Tuple[int, int]]) -> Dict[str, float]:
    inter = len(a & b)
    only_a = len(a - b)
    only_b = len(b - a)
    union = len(a | b)
    jaccard = (inter / union) if union > 0 else 1.0
    return {
        "intersection": inter,
        "only_a": only_a,
        "only_b": only_b,
        "union": union,
        "jaccard": jaccard,
    }


def plot_runtime_lines(results: Dict[str, object], output_path: Path) -> None:
    counts = results["counts"]
    if not counts:
        return

    plt.figure(figsize=(10, 6))

    def plot_series(key: str, label: str, style: str, color: str) -> None:
        vals = results[key]["mean"]
        stds = results[key]["std"]
        valid_idx = [i for i, v in enumerate(vals) if v is not None]
        if not valid_idx:
            return
        xs = [counts[i] for i in valid_idx]
        ys = [vals[i] for i in valid_idx]
        es = [stds[i] for i in valid_idx]
        plt.errorbar(xs, ys, yerr=es, fmt=style, color=color, label=label, capsize=5)

    plot_series("exact", "RaySpace Exact (two-pass)", "-o", style_for("exact")["color"])
    plot_series("direct_estimation", "RaySpace Direct Estimation", "--s", style_for("direct_estimation")["color"])
    plot_series("estimated", "RaySpace Intersection Estimated", "-.^", style_for("estimated")["color"])
    plot_series("tdbase_intersect", "TDBase Intersect", "-.x", style_for("tdbase")["color"])
    plot_series("tdbase_within0", "TDBase Within(w=0)", ":d", "#9467bd")

    plt.xlabel("Nu")
    plt.ylabel("Query Time (ms)")
    plt.yscale("log")
    plt.title("Nu Correctness Benchmark: Runtime")
    plt.grid(False)
    plt.legend()
    plt.xticks(counts)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def run_experiment(
    runs: int,
    grid_cell_size: int,
    nu_counts: List[int],
    run_layout,
    tdbase_timing_mode: str = TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
) -> Dict[str, object]:
    PROJECT_TMP_DIR.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = run_layout["timestamp"]
    run_name = run_layout["run_name"]
    run_log_dir = Path(run_layout["logs_dir"])

    exact_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR),
        mode="exact",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_cell_size=grid_cell_size,
        warmup_runs=1,
    )
    direct_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR),
        mode="direct_estimation",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_cell_size=grid_cell_size,
        warmup_runs=1,
    )
    estimated_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR),
        mode="estimated",
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(TIMINGS_DIR),
        grid_cell_size=grid_cell_size,
        warmup_runs=1,
    )
    tdbase_adapter = TDBaseAdapter(
        str(TDBASE_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR),
        query_timing_mode=tdbase_timing_mode,
    )

    exact_exec = RAYSPACE_DIR / "query" / "build" / "bin" / "raytracer_mesh_overlap"

    results: Dict[str, object] = {
        "counts": [],
        "exact": {"mean": [], "std": []},
        "direct_estimation": {"mean": [], "std": []},
        "estimated": {"mean": [], "std": []},
        "tdbase_intersect": {"mean": [], "std": []},
        "tdbase_within0": {"mean": [], "std": []},
        "correctness": {},
        "run_name": run_name,
        "timestamp": timestamp,
    }

    for nu in nu_counts:
        f_v, f_n = find_dataset_files(nu)
        if not f_v or not f_n:
            print(f"[WARN] Missing dataset for nu={nu}, skipping")
            continue

        print(f"\n[nu={nu}] files: {f_v.name} vs {f_n.name}")

        if not exact_adapter.check_preprocessed(str(f_v)):
            exact_adapter.preprocess_from_source(str(f_v), str(f_v), log_dir=str(run_log_dir))
        else:
            print(f"[nu={nu}] preprocessed exists: {f_v.with_suffix('.pre').name}")

        if not exact_adapter.check_preprocessed(str(f_n)):
            exact_adapter.preprocess_from_source(str(f_n), str(f_n), log_dir=str(run_log_dir))
        else:
            print(f"[nu={nu}] preprocessed exists: {f_n.with_suffix('.pre').name}")

        pre_v = PREPROCESSED_DIR / f_v.with_suffix(".pre").name
        pre_n = PREPROCESSED_DIR / f_n.with_suffix(".pre").name
        # TDBase within can abort on this dataset order for (v,n); use (n,v) for TDBase.
        tdb_tile1 = f_n
        tdb_tile2 = f_v

        res_exact = exact_adapter.run_overlap(
            str(f_v),
            str(f_n),
            runs,
            timeout=TIMEOUT_SECONDS,
            log_dir=str(run_log_dir),
        )
        res_direct = direct_adapter.run_overlap(
            str(f_v),
            str(f_n),
            runs,
            timeout=TIMEOUT_SECONDS,
            log_dir=str(run_log_dir),
            query_direction="both",
            pairs_output=str(PROJECT_TMP_DIR / f"pairs_direct_nu{nu}.csv"),
        )
        res_estimated = estimated_adapter.run_overlap(
            str(f_v),
            str(f_n),
            runs,
            timeout=TIMEOUT_SECONDS,
            log_dir=str(run_log_dir),
            pairs_output=str(PROJECT_TMP_DIR / f"pairs_estimated_nu{nu}.csv"),
        )
        res_tdb_inter = tdbase_adapter.run_overlap(
            str(tdb_tile1),
            str(tdb_tile2),
            runs,
            timeout=TIMEOUT_SECONDS,
            log_dir=str(run_log_dir),
        )
        res_tdb_within = run_tdbase_timing_within0(
            tdbase_adapter.executable,
            tdb_tile1,
            tdb_tile2,
            runs,
            run_log_dir,
            TIMEOUT_SECONDS,
        )

        exact_pairs_csv = PROJECT_TMP_DIR / f"pairs_exact_nu{nu}.csv"
        exact_pairs = run_exact_collect_pairs(
            exact_exec,
            pre_v,
            pre_n,
            PROJECT_TMP_DIR / f"exact_pairs_timing_nu{nu}.json",
            exact_pairs_csv,
            warmup_runs=0,
            timeout=TIMEOUT_SECONDS,
            work_dir=PROJECT_TMP_DIR,
        )
        direct_pairs = read_pairs_csv(PROJECT_TMP_DIR / f"pairs_direct_nu{nu}.csv")
        estimated_pairs = read_pairs_csv(PROJECT_TMP_DIR / f"pairs_estimated_nu{nu}.csv")
        tdb_inter_pairs = run_tdbase_collect_pairs(
            tdbase_adapter.executable,
            tdb_tile1,
            tdb_tile2,
            query="intersect",
            output_log=PROJECT_TMP_DIR / f"tdbase_pairs_intersect_nu{nu}.log",
            timeout=TIMEOUT_SECONDS,
        )
        tdb_within_pairs = run_tdbase_collect_pairs(
            tdbase_adapter.executable,
            tdb_tile1,
            tdb_tile2,
            query="within",
            output_log=PROJECT_TMP_DIR / f"tdbase_pairs_within0_nu{nu}.log",
            timeout=TIMEOUT_SECONDS,
        )

        # Canonicalize TDBase orientation back to (v,n) to match RaySpace mesh1=v, mesh2=n.
        tdb_inter_pairs = {(b, a) for (a, b) in tdb_inter_pairs}
        tdb_within_pairs = {(b, a) for (a, b) in tdb_within_pairs}

        results["counts"].append(nu)
        results["exact"]["mean"].append(None if "error" in res_exact else float(res_exact["mean"]))
        results["exact"]["std"].append(None if "error" in res_exact else float(res_exact["std"]))
        results["direct_estimation"]["mean"].append(None if "error" in res_direct else float(res_direct["mean"]))
        results["direct_estimation"]["std"].append(None if "error" in res_direct else float(res_direct["std"]))
        results["estimated"]["mean"].append(None if "error" in res_estimated else float(res_estimated["mean"]))
        results["estimated"]["std"].append(None if "error" in res_estimated else float(res_estimated["std"]))
        results["tdbase_intersect"]["mean"].append(None if "error" in res_tdb_inter else float(res_tdb_inter["mean"]))
        results["tdbase_intersect"]["std"].append(None if "error" in res_tdb_inter else float(res_tdb_inter["std"]))
        results["tdbase_within0"]["mean"].append(None if "error" in res_tdb_within else float(res_tdb_within["mean"]))
        results["tdbase_within0"]["std"].append(None if "error" in res_tdb_within else float(res_tdb_within["std"]))

        approach_sets = {
            "exact": exact_pairs,
            "direct_estimation": direct_pairs,
            "estimated": estimated_pairs,
            "tdbase_intersect": tdb_inter_pairs,
            "tdbase_within0": tdb_within_pairs,
        }
        names = list(approach_sets.keys())

        pairwise: Dict[str, Dict[str, float]] = {}
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = names[i]
                b = names[j]
                pairwise[f"{a}__vs__{b}"] = compare_sets(approach_sets[a], approach_sets[b])

        all_intersection = set.intersection(*approach_sets.values()) if approach_sets else set()
        all_union = set.union(*approach_sets.values()) if approach_sets else set()

        results["correctness"][str(nu)] = {
            "counts": {k: len(v) for k, v in approach_sets.items()},
            "pairwise": pairwise,
            "all_intersection": len(all_intersection),
            "all_union": len(all_union),
        }

        print(
            f"[nu={nu}] counts exact={len(exact_pairs)} direct={len(direct_pairs)} "
            f"estimated={len(estimated_pairs)} "
            f"td_inter={len(tdb_inter_pairs)} td_within0={len(tdb_within_pairs)}"
        )
        for key, stats in pairwise.items():
            print(
                f"[nu={nu}] {key}: inter={stats['intersection']} "
                f"only_a={stats['only_a']} only_b={stats['only_b']} "
                f"jaccard={stats['jaccard']:.6f}"
            )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Nu correctness benchmark for overlap approaches")
    parser.add_argument("--runs", type=int, default=5, help="Number of timing runs per approach")
    parser.add_argument("--grid-cell-size", type=float, default=1500.0, help="Grid resolution for RaySpace preprocessing")
    parser.add_argument("--nu", type=int, nargs="+", default=DEFAULT_NU_COUNTS, help="Nu values to run")
    parser.add_argument(
        "--tdbase-timing-mode",
        type=str,
        default=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
        choices=TDBASE_TIMING_MODES,
        help="TDBase query-time definition. Default uses index+compute+evaluate; use compute_only to revert.",
    )
    args = parser.parse_args()

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_nu_correctness")
    results = run_experiment(
        args.runs,
        args.grid_cell_size,
        args.nu,
        run_layout,
        tdbase_timing_mode=args.tdbase_timing_mode,
    )
    if not results["counts"]:
        print("No successful datasets processed.")
        return

    out_json = Path(run_layout["results_json"])
    write_json(
        out_json,
        {
            "metadata": {
                "timestamp": run_layout["timestamp"],
                "run_name": run_layout["run_name"],
                "run_dir": str(run_layout["run_dir"]),
                "runs": args.runs,
                "grid_cell_size": args.grid_cell_size,
                "nu_counts": args.nu,
                "tdbase_timing_mode": args.tdbase_timing_mode,
            },
            "results": results,
        },
    )

    figures_dir = Path(run_layout["figures_dir"])
    fig_path = figures_dir / "nu_correctness_runtime.png"
    plot_runtime_lines(results, fig_path)

    print("\nSaved:")
    print(f"- Results JSON: {out_json}")
    print(f"- Runtime plot: {fig_path}")
    print(f"- Pair/log temp dir: {PROJECT_TMP_DIR}")

    print("\nRuntime summary (ms):")
    for i, nu in enumerate(results["counts"]):
        ex = results["exact"]["mean"][i]
        de = results["direct_estimation"]["mean"][i]
        es = results["estimated"]["mean"][i]
        ti = results["tdbase_intersect"]["mean"][i]
        tw = results["tdbase_within0"]["mean"][i]
        print(
            f"nu={nu}: exact={ex if ex is not None else 'N/A'} "
            f"direct={de if de is not None else 'N/A'} "
            f"inter_estimated={es if es is not None else 'N/A'} "
            f"td_intersect={ti if ti is not None else 'N/A'} "
            f"td_within0={tw if tw is not None else 'N/A'}"
        )


if __name__ == "__main__":
    main()
