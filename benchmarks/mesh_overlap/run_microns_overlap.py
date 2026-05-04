#!/usr/bin/env python3
import argparse
import sys
import subprocess
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to sys.path to allow imports from 'benchmarks'
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    RAYSPACE_DIR,
    canonical_microns_aggregated_paths,
    create_benchmark_run_layout,
    ensure_microns_splits,
    ensure_microns_aggregated_meshes,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.mesh_overlap.adapters.raytracer_adapter import RaytracerAdapter
from benchmarks.mesh_overlap.adapters.cgal_adapter import CGALAdapter
from benchmarks.mesh_overlap.adapters.touch_adapter import TOUCHAdapter

CGAL_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/CGAL"

def generate_microns_scalability_figure(results, approaches, figures_dir: Path, timestamp: str):
    """Generate a line plot for MICrONS overlap scalability from successful runs."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    style_map = {
        "direct_estimation": ("Direct Estimation", "#1f77b4", "o", "-"),
        "cgal": ("CGAL", "#ff7f0e", "s", "--"),
        "touch": ("TOUCH", "#2ca02c", "^", "-."),
    }

    plt.figure(figsize=(10, 6))
    has_any_series = False

    for approach in approaches:
        x_vals = []
        y_vals = []
        y_errs = []
        for row in results:
            res = row.get(approach)
            if not isinstance(res, dict) or "error" in res:
                continue
            mean = res.get("mean")
            std = res.get("std")
            if mean is None:
                continue
            x_vals.append(row.get("size_gb"))
            y_vals.append(mean)
            y_errs.append(0.0 if std is None else std)

        if not x_vals:
            continue

        has_any_series = True
        sorted_points = sorted(zip(x_vals, y_vals, y_errs), key=lambda t: t[0])
        xs = [p[0] for p in sorted_points]
        ys = [p[1] for p in sorted_points]
        es = [p[2] for p in sorted_points]

        label, color, marker, linestyle = style_map.get(
            approach, (approach, "#444444", "o", "-")
        )
        plt.errorbar(
            xs,
            ys,
            yerr=es,
            fmt=marker,
            linestyle=linestyle,
            color=color,
            capsize=4,
            linewidth=2,
            markersize=7,
            label=label,
        )

    if not has_any_series:
        print("No successful approach results available; skipping MICrONS scalability figure.")
        plt.close()
        return

    plt.yscale("log")
    plt.xlabel("MICrONS subset size (GB)")
    plt.ylabel("Overlap query time (ms) [log scale]")
    plt.title("MICrONS Overlap Scalability")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_png = figures_dir / f"microns_overlap_scalability_{timestamp}.png"
    output_pdf = figures_dir / f"microns_overlap_scalability_{timestamp}.pdf"
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {output_png}")
    print(f"Saved figure: {output_pdf}")

def main():
    parser = argparse.ArgumentParser(description="MICrONS subset benchmark for mesh overlap (Direct Estimation, CGAL, TOUCH)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8],
                        help="MICrONS subset sizes in GB to benchmark")
    parser.add_argument("--source-root", type=str, 
                        default=str(REPO_ROOT / "datasets_scripts" / "microns_data"),
                        help="Root directory containing MICrONS GLB subset folders")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--grid-cell-size", type=float, default=700.0)
    parser.add_argument("--overlap-max-iterations", type=int, default=100)
    parser.add_argument("--query-direction", type=str, default="both", choices=["both", "mesh1tomesh2", "mesh2tomesh1"])
    parser.add_argument("--approaches", type=str, nargs="+", default=["direct_estimation", "cgal", "touch"],
                        help="Approaches to run")
    args = parser.parse_args()

    dirs = get_shared_data_dirs("microns_overlap")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_microns")
    run_log_dir = Path(run_layout["logs_dir"])

    splits_dir = dirs["root"] / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_root)

    # Initialize Adapters
    adapters = {}
    if "direct_estimation" in args.approaches:
        adapters["direct_estimation"] = RaytracerAdapter(
            str(RAYSPACE_DIR), mode="direct_estimation", preprocessed_dir=str(dirs["preprocessed"]),
            timings_dir=str(dirs["timings"]), grid_cell_size=args.grid_cell_size, warmup_runs=args.warmup_runs,
        )
    if "cgal" in args.approaches:
        adapters["cgal"] = CGALAdapter(str(CGAL_DIR), preprocessed_dir=str(dirs["preprocessed"]))
    if "touch" in args.approaches:
        adapters["touch"] = TOUCHAdapter(str(CGAL_DIR), preprocessed_dir=str(dirs["preprocessed"]))

    results = []
    for size_gb in args.sizes:
        print(f"\n--- Preparing MICrONS {size_gb}GB dataset ---")
        split_a, split_b = ensure_microns_splits(size_gb, source_root, splits_dir)
        
        agg_a, agg_b = canonical_microns_aggregated_paths(dirs["raw"], size_gb)
        ensure_microns_aggregated_meshes(split_a, split_b, agg_a, agg_b)

        # Preprocessing for Raytracer (with grid)
        if "direct_estimation" in args.approaches:
             adapter = adapters["direct_estimation"]
             for file_path in (agg_a, agg_b):
                 if not adapter.check_preprocessed(str(file_path)):
                     adapter.preprocess_from_source(str(file_path), str(file_path), log_dir=str(run_log_dir))
        
        # Preprocessing for CGAL/TOUCH (no grid, .pre extension)
        if "cgal" in args.approaches or "touch" in args.approaches:
            print("Checking preprocessing for CGAL/TOUCH (no grid)...")
            preprocess_exec = RAYSPACE_DIR / "preprocess" / "build" / "bin" / "preprocess_dataset"
            for file_path in (agg_a, agg_b):
                out_pre = dirs["preprocessed"] / f"{file_path.stem}.pre"
                if not out_pre.exists():
                    print(f"Preprocessing {file_path.name} for CGAL/TOUCH...")
                    cmd = [
                        str(preprocess_exec),
                        "--mode", "mesh",
                        "--dataset", str(file_path),
                        "--output-geometry", str(out_pre),
                        "--output-timing", str(dirs["timings"] / f"{file_path.stem}_no_grid_timing.json")
                    ]
                    subprocess.run(cmd, check=True)

        entry = {
            "size_gb": size_gb,
            "size_bytes_a": agg_a.stat().st_size if agg_a.exists() else 0,
            "size_bytes_b": agg_b.stat().st_size if agg_b.exists() else 0,
        }

        # Run Benchmarks
        for approach_name in args.approaches:
            if approach_name not in adapters:
                continue
            
            print(f"Running {approach_name}...")
            adapter = adapters[approach_name]
            
            if approach_name == "direct_estimation":
                res = adapter.run_overlap(
                    str(agg_a), str(agg_b), args.runs, timeout=args.timeout, 
                    query_direction=args.query_direction,
                    overlap_max_iterations=args.overlap_max_iterations,
                    log_dir=str(run_log_dir)
                )
            else:
                # CGAL/TOUCH
                res = adapter.run_overlap(
                    str(agg_a), str(agg_b), args.runs, timeout=args.timeout,
                    log_dir=str(run_log_dir)
                )
            
            entry[approach_name] = res

        results.append(entry)
        print(f"size_gb={size_gb}: done")

    payload = {
        "metadata": {
            "scenario": "microns_overlap",
            "query_type": "overlap",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "sizes": args.sizes,
            "grid_cell_size": args.grid_cell_size,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "overlap_max_iterations": args.overlap_max_iterations,
            "query_direction": args.query_direction,
            "approaches": args.approaches,
            "shared_data_root": str(dirs["root"]),
        },
        "results": results,
    }

    out = run_layout["results_json"]
    write_json(out, payload)
    print(f"Saved: {out}")

    generate_microns_scalability_figure(
        results=results,
        approaches=args.approaches,
        figures_dir=Path(run_layout["figures_dir"]),
        timestamp=run_layout["timestamp"],
    )

if __name__ == "__main__":
    main()
