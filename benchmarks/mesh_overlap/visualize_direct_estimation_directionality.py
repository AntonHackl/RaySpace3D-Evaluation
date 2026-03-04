#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    rows = data.get("results", [])
    if not rows:
        raise ValueError(f"No results in {path}")
    return data, rows


def plot_directionality(data, rows, output_path: Path):
    nu = [r["nu"] for r in rows]

    both_mean = [r["query_time_ms"]["ground_truth_both"]["mean"] for r in rows]
    both_std = [r["query_time_ms"]["ground_truth_both"]["std"] for r in rows]
    s2l_mean = [r["query_time_ms"]["small_to_large_only"]["mean"] for r in rows]
    s2l_std = [r["query_time_ms"]["small_to_large_only"]["std"] for r in rows]
    l2s_mean = [r["query_time_ms"]["large_to_small_only"]["mean"] for r in rows]
    l2s_std = [r["query_time_ms"]["large_to_small_only"]["std"] for r in rows]

    s2l_precision = [r["error_metrics"]["small_to_large_only"]["precision"] for r in rows]
    s2l_recall = [r["error_metrics"]["small_to_large_only"]["recall"] for r in rows]
    s2l_f1 = [r["error_metrics"]["small_to_large_only"]["f1"] for r in rows]

    l2s_precision = [r["error_metrics"]["large_to_small_only"]["precision"] for r in rows]
    l2s_recall = [r["error_metrics"]["large_to_small_only"]["recall"] for r in rows]
    l2s_f1 = [r["error_metrics"]["large_to_small_only"]["f1"] for r in rows]

    fig, (ax_time, ax_metrics) = plt.subplots(1, 2, figsize=(16, 6))

    ax_time.errorbar(nu, both_mean, yerr=both_std, fmt='-o', capsize=4, label='Both (Ground Truth)', color='#1f77b4')
    ax_time.errorbar(nu, s2l_mean, yerr=s2l_std, fmt='--s', capsize=4, label='Small → Large only', color='#ff7f0e')
    ax_time.errorbar(nu, l2s_mean, yerr=l2s_std, fmt='-.^', capsize=4, label='Large → Small only', color='#2ca02c')
    ax_time.set_title('Direct Estimation Query Time')
    ax_time.set_xlabel('Nu')
    ax_time.set_ylabel('Query Time (ms)')
    ax_time.grid(True, alpha=0.2)
    ax_time.legend()
    ax_time.set_xticks(nu)

    x = np.array(nu, dtype=float)
    ax_metrics.plot(x, s2l_precision, '--o', label='S→L Precision', color='#8c564b')
    ax_metrics.plot(x, s2l_recall, '--s', label='S→L Recall', color='#d62728')
    ax_metrics.plot(x, s2l_f1, '--^', label='S→L F1', color='#e377c2')

    ax_metrics.plot(x, l2s_precision, '-o', label='L→S Precision', color='#7f7f7f')
    ax_metrics.plot(x, l2s_recall, '-s', label='L→S Recall', color='#17becf')
    ax_metrics.plot(x, l2s_f1, '-^', label='L→S F1', color='#9467bd')

    ax_metrics.set_title('Error Metrics vs Both-Direction Ground Truth')
    ax_metrics.set_xlabel('Nu')
    ax_metrics.set_ylabel('Score')
    ax_metrics.set_ylim(0.88, 1.01)
    ax_metrics.grid(True, alpha=0.2)
    ax_metrics.legend(ncol=2, fontsize=9)
    ax_metrics.set_xticks(nu)

    fig.suptitle(
        f"Mesh Overlap Direct Estimation Directionality Test (runs={data.get('runs')})",
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight')


def main():
    parser = argparse.ArgumentParser(description='Visualize direct estimation directionality results')
    parser.add_argument('--input', required=True, help='Path to directionality JSON summary')
    parser.add_argument('--output', help='Output image path (.png); default in figures/')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    default_output = input_path.parent.parent / 'figures' / f"{input_path.stem}.png"
    output_path = Path(args.output) if args.output else default_output

    data, rows = load_results(input_path)
    plot_directionality(data, rows, output_path)
    print(f"Visualization saved to {output_path}")
    print(f"PDF saved to {output_path.with_suffix('.pdf')}")


if __name__ == '__main__':
    main()
