#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import apply_paper_style

QUERY_STYLES = {
    "overlap": {"label": "Overlap", "color": "#1f77b4"},
    "intersection": {"label": "Intersection", "color": "#ff7f0e"},
    "containment": {"label": "Containment", "color": "#2ca02c"},
}


@dataclass
class PointResult:
    group_name: str
    selector: str
    source_run: Path
    query_to_mean: Dict[str, float]


def _valid_mean(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0.0:
        return None
    return v


def _extract_from_row(row: Dict, *, group_name: str, selector: str, source_run: Path) -> Optional[PointResult]:
    query_to_mean: Dict[str, float] = {}
    for q in ["overlap", "intersection", "containment"]:
        cell = row.get(q)
        if not isinstance(cell, dict) or "error" in cell:
            continue
        m = _valid_mean(cell.get("mean"))
        if m is not None:
            query_to_mean[q] = m
    if not query_to_mean:
        return None
    return PointResult(
        group_name=group_name,
        selector=selector,
        source_run=source_run,
        query_to_mean=query_to_mean,
    )


def _pick_latest_point(
    runs_root: Path,
    run_prefix: str,
    *,
    row_matcher,
    group_name: str,
    selector: str,
) -> PointResult:
    run_dirs = sorted(runs_root.glob(f"{run_prefix}_*"), reverse=True)
    for run_dir in run_dirs:
        result_file = run_dir / "results.json"
        if not result_file.exists():
            continue
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            continue
        row = next((r for r in rows if row_matcher(r)), None)
        if row is None:
            continue
        point = _extract_from_row(row, group_name=group_name, selector=selector, source_run=run_dir)
        if point is not None:
            return point
    raise RuntimeError(f"No usable run found for {group_name} ({selector})")


def _plot(points: List[PointResult], output_base: Path) -> None:
    apply_paper_style()

    queries = [q for q in ["overlap", "intersection", "containment"] if any(q in p.query_to_mean for p in points)]
    x = np.arange(len(points), dtype=float)
    width = 0.8 / max(1, len(queries))

    fig, ax = plt.subplots(figsize=(12, 6.8))
    for i, q in enumerate(queries):
        vals = np.array([p.query_to_mean.get(q, np.nan) for p in points], dtype=float)
        mask = np.isfinite(vals)
        if not np.any(mask):
            continue
        offs = (i - (len(queries) - 1) / 2.0) * width
        st = QUERY_STYLES[q]
        ax.bar(x[mask] + offs, vals[mask], width=width, label=st["label"], color=st["color"], alpha=0.92)

    ax.set_yscale("log")
    ax.set_ylabel("Query time (ms) [log scale]")
    ax.set_xlabel("Dataset group")
    ax.set_xticks(x)
    ax.set_xticklabels([p.group_name for p in points])
    ax.grid(axis="y", which="both", linestyle="-", alpha=0.2)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped bar chart for mesh query comparison overall performance (nu600/nu800/microns4/microns8)."
    )
    parser.add_argument("--runs-root", type=Path, default=SCRIPT_DIR / "runs")
    args = parser.parse_args()

    runs_root = args.runs_root

    points = [
        _pick_latest_point(
            runs_root,
            "query_comparison_nu_scalability",
            row_matcher=lambda r: r.get("nu") == 600,
            group_name="NU (600)",
            selector="nu=600",
        ),
        _pick_latest_point(
            runs_root,
            "query_comparison_nu_scalability",
            row_matcher=lambda r: r.get("nu") == 800,
            group_name="NU (800)",
            selector="nu=800",
        ),
        _pick_latest_point(
            runs_root,
            "query_comparison_microns",
            row_matcher=lambda r: r.get("size_gb") == 4,
            group_name="MICrONS (4 GB)",
            selector="size_gb=4",
        ),
        _pick_latest_point(
            runs_root,
            "query_comparison_microns",
            row_matcher=lambda r: r.get("size_gb") == 8,
            group_name="MICrONS (8 GB)",
            selector="size_gb=8",
        ),
    ]

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "query_comparison_overall_performance")
    figures_dir = Path(run_layout["figures_dir"])
    output_base = figures_dir / f"mesh_query_comparison_overall_performance_{run_layout['timestamp']}"

    _plot(points, output_base)

    payload = {
        "metadata": {
            "scenario": "query_comparison_overall_performance",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
        },
        "groups": [
            {
                "group_name": p.group_name,
                "selector": p.selector,
                "source_run": str(p.source_run),
                "source_run_name": p.source_run.name,
                "results_ms": p.query_to_mean,
            }
            for p in points
        ],
    }
    write_json(Path(run_layout["results_json"]), payload)

    print(f"Saved figure: {output_base}.png")
    print(f"Saved figure: {output_base}.pdf")
    for p in points:
        print(f"{p.group_name}: {p.source_run.name} ({p.selector}), queries={sorted(p.query_to_mean.keys())}")


if __name__ == "__main__":
    main()
