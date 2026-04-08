"""
Visualize grid-resolution sweep results for overlap direct estimation.

Expected input: JSON from run_grid_resolution_sweep.py

Figure panels:
1) Query time vs grid resolution
2) Pair counts vs grid resolution (replication-corrected and load-factor-applied + ground truth)
3) Hash table size vs grid resolution
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize grid-resolution sweep results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to JSON produced by run_grid_resolution_sweep.py")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR),
                        help="Directory for saved figures")
    parser.add_argument("--run-type", type=str, default=None,
                        help="Run type to visualize from merged JSON (e.g., cubes, nu)")
    parser.add_argument("--output-stem", type=str, default=None,
                        help="Optional output filename stem (without extension)")
    return parser.parse_args()


def safe_float(v, default=float("nan")):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def safe_int(v, default=0):
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def factor_text(value, ground_truth):
    if ground_truth <= 0:
        return "n/a"
    return f"{value / ground_truth:.2f}x"


def annotate_pair_series(ax, x_vals, y_vals, ground_truth, color, y_offset=0.0):
    for x, y in zip(x_vals, y_vals):
        if np.isnan(y):
            continue
        ax.annotate(
            factor_text(y, ground_truth),
            (x, y),
            textcoords="offset points",
            xytext=(0, y_offset),
            fontsize=7,
            color=color,
            alpha=0.95,
            ha="center",
        )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    run_label = None
    if "runs" in data:
        runs = data.get("runs", [])
        if args.run_type is None:
            if len(runs) != 1:
                raise ValueError("Merged input contains multiple run types. Pass --run-type.")
            selected = runs[0]
        else:
            selected = next((r for r in runs if r.get("run_type") == args.run_type), None)
            if selected is None:
                available = [r.get("run_type") for r in runs]
                raise ValueError(f"Run type '{args.run_type}' not found. Available: {available}")
        meta = selected["metadata"]
        results = sorted(selected["results"], key=lambda r: r["grid_resolution"])
        run_label = selected.get("display_name") or selected.get("run_type")
    else:
        meta = data["metadata"]
        results = sorted(data["results"], key=lambda r: r["grid_resolution"])
        run_label = meta.get("display_name") or meta.get("run_type")

    grid_res = np.array([safe_int(r.get("grid_resolution")) for r in results], dtype=float)

    mean_time = np.array([safe_float(r.get("timing", {}).get("mean_time_ms")) for r in results], dtype=float)
    std_time = np.array([safe_float(r.get("timing", {}).get("std_time_ms"), 0.0) for r in results], dtype=float)

    raw_estimated_pairs = np.array([
        safe_float(r.get("pairs", {}).get("raw_estimated_pairs_before_replication_correction")) for r in results
    ], dtype=float)
    replication_corrected_pairs = np.array([
        safe_float(r.get("pairs", {}).get("raw_estimated_pairs_after_replication_correction")) for r in results
    ], dtype=float)
    load_factor_applied_pairs = np.array([
        safe_float(r.get("pairs", {}).get("estimated_pairs_after_load_factor_application")) for r in results
    ], dtype=float)

    hash_size = np.array([
        safe_float(r.get("hash", {}).get("estimated_hash_table_size")) for r in results
    ], dtype=float)

    ground_truth = safe_int(meta.get("ground_truth_pairs_exact_overlap"), 0)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    if run_label is None:
        run_label = "dataset"

    fig.suptitle(
        (
            f"Estimated Overlap Grid Sweep ({run_label}) "
            f"| GT pairs={ground_truth:,}"
        ),
        fontsize=12,
        y=1.03,
    )

    # Panel 1: query timing vs grid resolution
    ax1 = axes[0]
    ax1.errorbar(
        grid_res,
        mean_time,
        yerr=std_time,
        fmt="o-",
        color="steelblue",
        ecolor="lightsteelblue",
        elinewidth=1.2,
        capsize=4,
        linewidth=1.8,
        markersize=5,
        label="Query time (mean ± std)",
    )
    ax1.set_xlabel("Grid resolution")
    ax1.set_ylabel("Query time (ms)")
    ax1.set_title("1. Performance")
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.legend(fontsize=8, loc="best")

    # Panel 2: pair counts vs grid resolution, with factors vs ground truth.
    ax2 = axes[1]
    ax2.plot(grid_res, raw_estimated_pairs, "d--", color="gray", linewidth=1.2, alpha=0.6, label="Raw Estimated (before alpha correction)")
    ax2.plot(grid_res, replication_corrected_pairs, "o-", color="#c0392b", linewidth=1.8, label="Replication Corrected (Final Estimate)")
    ax2.plot(grid_res, load_factor_applied_pairs, "s-", color="#1f8b4c", linewidth=1.8, label="Load Factor Applied (Hash Capacity)")

    if ground_truth > 0:
        ax2.axhline(
            y=ground_truth,
            color="black",
            linestyle="--",
            linewidth=1.3,
            label=f"Ground truth ({ground_truth:,})",
        )

    if ground_truth > 0:
        annotate_pair_series(ax2, grid_res, replication_corrected_pairs, ground_truth, color="#c0392b", y_offset=12.0)
        annotate_pair_series(ax2, grid_res, load_factor_applied_pairs, ground_truth, color="#1f8b4c", y_offset=12.0)
        annotate_pair_series(ax2, grid_res, raw_estimated_pairs, ground_truth, color="gray", y_offset=-15.0)

    ax2.set_xlabel("Grid resolution")
    ax2.set_ylabel("Pairs")
    ax2.set_title("2. Pair Estimates vs Ground Truth")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2.grid(True, linestyle="--", alpha=0.45)
    ax2.legend(fontsize=8, loc="best")

    # Panel 3: hash size vs grid resolution (space pressure proxy)
    ax3 = axes[2]
    ax3.plot(grid_res, hash_size, "o-", color="#6a4c93", linewidth=1.8, label="Estimated hash table size")
    ax3.set_xlabel("Grid resolution")
    ax3.set_ylabel("Hash table size (slots)")
    ax3.set_title("3. Hash Size Impact")
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax3.grid(True, linestyle="--", alpha=0.45)
    ax3.legend(fontsize=8, loc="best")

    plt.tight_layout()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.output_stem:
        stem = f"{args.output_stem}_{ts}"
    elif args.run_type:
        stem = f"grid_resolution_sweep_{args.run_type}_{ts}"
    else:
        stem = f"grid_resolution_sweep_{ts}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"

    fig.savefig(str(png_path), dpi=160, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved to:\n  {png_path}\n  {pdf_path}")


if __name__ == "__main__":
    main()
