#!/usr/bin/env python3
"""Plot two experiment sequences.

This script compares two sequences of benchmark results (same experiment name,
multiple numbered runs, timestamps optional) in two ways:

1) Overlaid selectivity comparison (both sequences) scaled to a maximum of 1.0
2) Per-approach timing lines for each sequence (one subplot per sequence)

Usage examples:
    python plot_sequences.py cube sphere
    python plot_sequences.py cube sphere --results-dir ../results --metric total_query_ms

Positional args:
    exp_a  First experiment base name (e.g., cube)
    exp_b  Second experiment base name (e.g., sphere)

Options:
    --results-dir  Directory with `grid_benchmark_*.json` files (default: ../results)
    --output-dir   Directory to save generated figures (default: ./figures)
    --metric       Timing metric to plot (default: total_query_ms)

The script will look for files named like:
    grid_benchmark_<name>_<number>.json
    grid_benchmark_<name>_<number>_<timestamp>.json
    grid_benchmark_<name>_<number>_<timestamp>_<sub>.json

and will order runs by the numeric component.
"""

from pathlib import Path
import argparse
import json
import re
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def extract_experiment_info(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """Extract experiment base name and numeric extent from filename.

    Accepts files with or without timestamps at the end.
    Returns (name, number) or (None, None) if parsing fails.
    """
    s = str(filename)
    m = re.match(r'grid_benchmark_([^_]+)_(\d+)(?:_.*)?\.json$', s)
    if m:
        return m.group(1), int(m.group(2))
    m = re.search(r'grid_benchmark_(\w+)_(\d+)', s)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to load {path}: {e}")
        return None


def compute_selectivity_for_file(data: dict) -> Optional[float]:
    """Compute average selectivity for a benchmark file (average across approaches/results).

    Selectivity is defined as inside_count / total_points. We average all available
    selectivity measurements across all approaches and grid positions in the file.
    """
    if not data or 'results' not in data:
        return None

    vals = []
    for approach, results_list in data['results'].items():
        if not isinstance(results_list, list):
            continue
        for r in results_list:
            if not isinstance(r, dict):
                continue
            inside = r.get('inside_count')
            total = r.get('total_points')
            if inside is not None and total:
                try:
                    vals.append(float(inside) / float(total))
                except Exception:
                    continue
    if not vals:
        return None
    return float(np.mean(vals))


def compute_approach_means_for_file(data: dict, metric: str) -> Dict[str, Optional[float]]:
    """Return mean timing per approach from the given benchmark JSON data.

    For each approach, compute mean of `metric` across successful results.
    """
    out = {}
    if not data or 'results' not in data:
        return out

    for approach, results_list in data['results'].items():
        vals = []
        if not isinstance(results_list, list):
            continue
        for r in results_list:
            if not isinstance(r, dict):
                continue
            if not r.get('success'):
                continue
            v = None
            # try the requested metric first, then fallback to total_query_ms
            if metric in r and r.get(metric) is not None:
                v = r.get(metric)
            elif 'total_query_ms' in r and r.get('total_query_ms') is not None:
                v = r.get('total_query_ms')
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    continue
        out[approach] = float(np.mean(vals)) if vals else None
    return out


def collect_sequence(results_dir: Path, base_name: str) -> Tuple[List[int], Dict[int, Path]]:
    """Find JSON files for base_name and return sorted extents and mapping extent->path."""
    files = list(results_dir.glob(f'grid_benchmark_{base_name}_*.json'))
    mapping = {}
    for f in files:
        name, num = extract_experiment_info(f.name)
        if name == base_name and num is not None:
            mapping[num] = f
    extents = sorted(mapping.keys())
    return extents, mapping


def plot_selectivity_comparison(extents_a: List[int], mapping_a: Dict[int, Path],
                                extents_b: List[int], mapping_b: Dict[int, Path],
                                output_dir: Path, base_a: str, base_b: str):
    """Plot selectivities of two sequences against each other, scaled to ymax=1."""
    sel_a = []
    sel_b = []
    xs_a = []
    xs_b = []

    for e in extents_a:
        data = load_json(mapping_a[e])
        s = compute_selectivity_for_file(data)
        if s is not None:
            xs_a.append(e)
            sel_a.append(s)
    for e in extents_b:
        data = load_json(mapping_b[e])
        s = compute_selectivity_for_file(data)
        if s is not None:
            xs_b.append(e)
            sel_b.append(s)

    if not sel_a and not sel_b:
        print("No selectivity data found for either sequence; skipping selectivity plot.")
        return

    plt.figure(figsize=(10, 6))
    if sel_a:
        plt.plot(xs_a, sel_a, marker='o', label=base_a)
    if sel_b:
        plt.plot(xs_b, sel_b, marker='s', label=base_b)

    plt.xlabel('Extent (numeric part)')
    plt.ylabel('Average Selectivity')
    plt.title(f'Selectivity Comparison: {base_a} vs {base_b} (scaled to 1.0)')
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'{base_a}_vs_{base_b}_selectivities.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved selectivity comparison: {out_path}")


def plot_approach_times(extents: List[int], mapping: Dict[int, Path], output_dir: Path,
                        base_name: str, metric: str):
    """Plot timing lines for each approach across extents for one sequence."""
    # Collect per-extent approach means
    approach_series = defaultdict(lambda: [])  # approach -> list of (extent, mean)
    ext_sorted = sorted(extents)
    for e in ext_sorted:
        data = load_json(mapping[e])
        if not data:
            continue
        means = compute_approach_means_for_file(data, metric)
        for approach, val in means.items():
            approach_series[approach].append((e, val))

    if not approach_series:
        print(f"No timing data found for sequence {base_name}; skipping approach times plot.")
        return

    plt.figure(figsize=(12, 6))
    for approach, series in sorted(approach_series.items()):
        xs = [x for x, v in series]
        ys = [v for x, v in series]
        # Replace None or non-positive with np.nan so lines break and log scale is safe
        ys = [np.nan if (v is None or (isinstance(v, (int, float)) and v <= 0)) else v for v in ys]
        plt.plot(xs, ys, marker='o', label=approach)

    plt.xlabel('Extent (numeric part)')
    plt.ylabel(f'Mean {metric} (ms)')
    plt.yscale('log')
    plt.title(f'Approach Timing Across {base_name} Experiments (log scale)')
    plt.legend()
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()

    out_path = output_dir / f'{base_name}_approach_times_{metric}.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved approach times plot: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot two experiment sequences (selectivities + approach times)')
    parser.add_argument('exp_a', help='First experiment base name (e.g., cube)')
    parser.add_argument('exp_b', help='Second experiment base name (e.g., sphere)')
    parser.add_argument('--results-dir', default=None, help='Directory containing benchmark results (default: ../results)')
    parser.add_argument('--output-dir', default='figures', help='Directory to save figures')
    parser.add_argument('--metric', default='total_query_ms', help='Timing metric to use (default: total_query_ms)')

    args = parser.parse_args()

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = Path(__file__).parent.parent / 'results'

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    base_a = args.exp_a
    base_b = args.exp_b

    extents_a, mapping_a = collect_sequence(results_dir, base_a)
    extents_b, mapping_b = collect_sequence(results_dir, base_b)

    if not extents_a:
        print(f"No result files found for experiment '{base_a}' in {results_dir}")
    if not extents_b:
        print(f"No result files found for experiment '{base_b}' in {results_dir}")
    if not extents_a and not extents_b:
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Selectivity comparison
    plot_selectivity_comparison(extents_a, mapping_a, extents_b, mapping_b, output_dir, base_a, base_b)

    # 2) Per-approach timing lines: create one subplot per sequence
    # We'll create a combined figure with two subplots for side-by-side comparison.
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: approaches for exp_a
    plt.sca(axes[0])
    # reuse local plotting code but draw on current axes
    approach_series = defaultdict(lambda: [])
    for e in sorted(extents_a):
        data = load_json(mapping_a[e])
        if not data:
            continue
        means = compute_approach_means_for_file(data, args.metric)
        for approach, val in means.items():
            approach_series[approach].append((e, val))
    if approach_series:
        for approach, series in sorted(approach_series.items()):
            xs = [x for x, v in series]
            ys = [v for x, v in series]
            ys = [np.nan if (v is None or (isinstance(v, (int, float)) and v <= 0)) else v for v in ys]
            axes[0].plot(xs, ys, marker='o', label=approach)
    axes[0].set_title(f'{base_a} - per-approach mean {args.metric} (log scale)')
    axes[0].set_xlabel('Extent')
    axes[0].set_ylabel(f'Mean {args.metric} (ms)')
    axes[0].set_yscale('log')
    axes[0].grid(alpha=0.3, which='both')
    axes[0].legend(fontsize='small')

    # Right: approaches for exp_b
    plt.sca(axes[1])
    approach_series = defaultdict(lambda: [])
    for e in sorted(extents_b):
        data = load_json(mapping_b[e])
        if not data:
            continue
        means = compute_approach_means_for_file(data, args.metric)
        for approach, val in means.items():
            approach_series[approach].append((e, val))
    if approach_series:
        for approach, series in sorted(approach_series.items()):
            xs = [x for x, v in series]
            ys = [v for x, v in series]
            ys = [np.nan if (v is None or (isinstance(v, (int, float)) and v <= 0)) else v for v in ys]
            axes[1].plot(xs, ys, marker='o', label=approach)
    axes[1].set_title(f'{base_b} - per-approach mean {args.metric} (log scale)')
    axes[1].set_xlabel('Extent')
    axes[1].set_ylabel(f'Mean {args.metric} (ms)')
    axes[1].set_yscale('log')
    axes[1].grid(alpha=0.3, which='both')
    axes[1].legend(fontsize='small')

    plt.tight_layout()
    combined_out = output_dir / f'{base_a}_vs_{base_b}_approach_times_{args.metric}.png'
    plt.savefig(combined_out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved combined approach timing figure: {combined_out}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
