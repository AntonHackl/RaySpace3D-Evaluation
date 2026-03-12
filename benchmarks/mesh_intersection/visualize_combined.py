#!/usr/bin/env python3
"""
Updated visualization for mesh intersection benchmark results.
Generates:
- Left: Log-log line chart (runtime vs approach)
- Right: Bar chart with runtime breakdown for each approach
"""
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_results(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def visualize_combined(json_file, output_dir=None):
    data = load_results(json_file)
    metadata = data.get("metadata", {})
    results = data.get("results", {})
    valid_results = {name: res for name, res in results.items() if "error" not in res}
    if not valid_results:
        print("No valid results to visualize")
        return
    adapters = list(valid_results.keys())
    means = [valid_results[name]["mean"] for name in adapters]
    stds = [valid_results[name]["std"] for name in adapters]
    breakdowns = [valid_results[name].get("breakdown", {}) for name in adapters]
    dataset = metadata.get("dataset", "unknown")
    num_runs = metadata.get("num_runs", 0)
    timestamp = metadata.get("timestamp", "")

    # --- Figure Layout ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # --- Left: Log-log Line Chart ---
    x_pos = np.arange(len(adapters))
    ax1.plot(x_pos, means, marker='o', linestyle='-', color='#1f77b4', label='Mean Runtime')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(adapters, rotation=15, ha='right')
    ax1.set_xlabel('Adapter', fontsize=12)
    ax1.set_ylabel('Query Time (ms)', fontsize=12)
    ax1.set_title(f'Log-Log Runtime Comparison\n{dataset} ({num_runs} run)', fontsize=14, fontweight='bold')
    ax1.set_yscale('log')
    ax1.set_xscale('linear')
    ax1.grid(axis='y', which='both', alpha=0.3)
    for i, mean in enumerate(means):
        ax1.text(x_pos[i], mean * 1.05, f'{mean:.2f} ms', ha='center', va='bottom', fontsize=9)

    # --- Right: Breakdown Bar Chart ---
    # Collect all breakdown keys
    breakdown_keys = set()
    for b in breakdowns:
        breakdown_keys.update(b.keys())
    breakdown_keys = sorted(list(breakdown_keys))
    colors = plt.cm.tab10.colors
    color_map = {k: colors[i % len(colors)] for i, k in enumerate(breakdown_keys)}
    width = 0.6
    bottoms = np.zeros(len(adapters))
    for i, key in enumerate(breakdown_keys):
        vals = [b.get(key, 0.0) for b in breakdowns]
        ax2.bar(x_pos, vals, width, bottom=bottoms, color=color_map[key], label=key)
        bottoms += np.array(vals)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(adapters, rotation=15, ha='right')
    ax2.set_xlabel('Adapter', fontsize=12)
    ax2.set_ylabel('Query Time (ms)', fontsize=12)
    ax2.set_title(f'Runtime Breakdown per Adapter\n{dataset} ({num_runs} run)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', which='both', alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    if output_dir is None:
        json_path = Path(json_file)
        output_dir = json_path.parent.parent / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{dataset}_{num_runs}run_{timestamp}_combined.png"
    output_path = output_dir / output_name
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined figure saved to {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize mesh intersection benchmark results (combined)")
    parser.add_argument("json_file", type=str, help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()
    visualize_combined(args.json_file, args.output_dir)

if __name__ == "__main__":
    main()
