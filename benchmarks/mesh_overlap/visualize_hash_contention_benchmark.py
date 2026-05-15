"""
Visualize the results of run_hash_contention_benchmark.py.

Produces a 3-panel figure:
    Panel 1: Query time (ms) ± std  vs hash table setting
    Panel 2: Hash accesses and contentions  vs hash table setting
    Panel 3: Pairs found vs hash table setting  (with reference line at true_result_count)
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# Repo-relative default output dir
SCRIPT_DIR  = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR / "figures"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize hash contention benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input",      type=str, required=True,
                        help="Path to the JSON file produced by run_hash_contention_benchmark.py")
    parser.add_argument("--output-dir", type=str, default=str(FIGURES_DIR),
                        help="Directory for saved figures")
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


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r") as f:
        data = json.load(f)

    meta    = data["metadata"]
    results = data["results"]

    true_count   = safe_int(meta.get("true_result_count"), 0)
    num_cubes    = meta.get("num_cubes", "?")
    selectivity  = meta.get("selectivity", "?")
    timestamp_in = meta.get("timestamp", "")

    mean_times   = [safe_float(r.get("mean_time_ms"))  for r in results]
    std_times    = [safe_float(r.get("std_time_ms"), 0.0) for r in results]
    accesses     = [safe_int(r.get("hash_accesses"))   for r in results]
    contentions  = [safe_int(r.get("hash_contentions")) for r in results]
    pairs_timing = [safe_int(r.get("pairs_found_timing")) for r in results]
    hash_sizes = []
    x_labels = []
    for r in results:
        multiplier = r.get("multiplier")
        size_kind = r.get("size_kind")
        effective_size = safe_int(
            r.get("actual_hash_table_size_timing") or r.get("actual_hash_table_size_contention") or r.get("hash_table_size"),
            0,
        )
        if size_kind == "gpu_auto" or multiplier is None:
            x_labels.append(f"GPU max\n({effective_size:,})")
            hash_sizes.append(effective_size)
        else:
            m = safe_float(multiplier, float("nan"))
            x_labels.append(f"{m:.1f}x\n({effective_size:,})")
            hash_sizes.append(effective_size)

    x = np.arange(len(results))

    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"Hash Table Contention Benchmark  |  {num_cubes:,} cubes per dataset"
        f",  selectivity≈{selectivity}  |  true pairs = {true_count:,}",
        fontsize=11, y=1.02,
    )

    # ---- Panel 1: Query time ----------------------------------------- #
    ax1 = axes[0]
    ax1.errorbar(
        x, mean_times, yerr=std_times,
        marker="o", linewidth=1.8, capsize=5, color="steelblue",
        label="Query time (mean ± std)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=7)
    ax1.set_xlabel("Hash table setting\n(slots)", fontsize=9)
    ax1.set_ylabel("Query time (ms, log-scale)", fontsize=9)
    ax1.set_yscale("log")
    ax1.set_title("1. Query Time (log scale)", fontsize=10)
    ax1.grid(False)
    ax1.legend(fontsize=8)

    # ---- Panel 2: Hash accesses and contentions ---------------------- #
    ax2 = axes[1]
    ax2.plot(x, accesses,   marker="s", linewidth=1.8, color="darkorange",
             label="Hash accesses")
    ax2.plot(x, contentions,marker="D", linewidth=1.8, color="crimson",
             label="Hash contentions")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=7)
    ax2.set_xlabel("Hash table setting\n(slots)", fontsize=9)
    ax2.set_ylabel("Count", fontsize=9)
    ax2.set_title("Hash Accesses & Contentions", fontsize=10)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"{int(val):,}" if val == int(val) else f"{val:,.0f}"
    ))
    ax2.grid(False)
    ax2.legend(fontsize=8)

    # Panel 2b: contention percentage on secondary y-axis
    contention_pcts = [safe_float(r.get("contention_pct")) for r in results]
    ax2b = ax2.twinx()
    ax2b.plot(x, contention_pcts, marker="^", linewidth=1.4, linestyle="--",
              color="purple", label="Contention %", alpha=0.7)
    ax2b.set_ylabel("Contention (%)", fontsize=9, color="purple")
    ax2b.tick_params(axis="y", labelcolor="purple")
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.get_legend().remove()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    # ---- Panel 3: Pairs found ---------------------------------------- #
    ax3 = axes[2]
    ax3.plot(x, pairs_timing, marker="o", linewidth=1.8, color="seagreen",
             label="Pairs found (timing run)")
    if true_count > 0:
        ax3.axhline(y=true_count, color="gray", linestyle="--", linewidth=1.4,
                    label=f"True result count ({true_count:,})")
    ax3.set_xticks(x)
    ax3.set_xticklabels(x_labels, fontsize=7)
    ax3.set_xlabel("Hash table setting\n(slots)", fontsize=9)
    ax3.set_ylabel("Pairs found", fontsize=9)
    ax3.set_title("Retrieved Pairs vs Hash Table Size", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda val, _: f"{int(val):,}" if val == int(val) else f"{val:,.0f}"
    ))
    ax3.grid(False)
    ax3.legend(fontsize=8)

    plt.tight_layout()

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"hash_contention_benchmark_{ts}"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    fig.savefig(str(pdf_path), bbox_inches="tight")
    print(f"Figure saved to:\n  {png_path}\n  {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
