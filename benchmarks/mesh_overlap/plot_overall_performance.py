#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import apply_paper_style, style_for


@dataclass
class GroupResult:
    group_name: str
    run_dir: Path
    run_timestamp: str
    selector_value: str
    approach_to_mean: Dict[str, float]


def _parse_ts_from_run_dir(run_dir: Path) -> str:
    parts = run_dir.name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return run_dir.name


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


def _approach_order_key(name: str) -> Tuple[int, str]:
    preferred = ["exact", "direct_estimation", "estimated", "estimated_mem10", "cgal", "touch", "tdbase"]
    if name in preferred:
        return (preferred.index(name), name)
    return (len(preferred), name)


def _extract_nu_large_400(path: Path) -> Optional[GroupResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    if md.get("dataset_profile") != "large_nu_v":
        return None

    results = payload.get("results", {})
    counts = results.get("counts", [])
    if 400 not in counts:
        return None
    idx = counts.index(400)

    approaches = results.get("enabled_approaches") or md.get("approaches") or []
    approach_to_mean: Dict[str, float] = {}
    for app in approaches:
        arr = (results.get(app) or {}).get("mean", [])
        if idx >= len(arr):
            continue
        mean = _valid_mean(arr[idx])
        if mean is not None:
            approach_to_mean[app] = mean

    if not approach_to_mean:
        return None

    run_dir = path.parent
    return GroupResult(
        group_name="Large nu (400)",
        run_dir=run_dir,
        run_timestamp=_parse_ts_from_run_dir(run_dir),
        selector_value="nu=400",
        approach_to_mean=approach_to_mean,
    )


def _extract_microns_4gb(path: Path) -> Optional[GroupResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    scenario = md.get("scenario")
    if scenario not in ("microns_overlap", None):
        return None

    results = payload.get("results", [])
    if not isinstance(results, list):
        return None

    row = next((r for r in results if r.get("size_gb") == 4), None)
    if row is None:
        return None

    approaches = md.get("approaches") or [k for k in row.keys() if k not in {"size_gb", "size_bytes_a", "size_bytes_b"}]
    approach_to_mean: Dict[str, float] = {}
    for app in approaches:
        res = row.get(app)
        if not isinstance(res, dict) or "error" in res:
            continue
        mean = _valid_mean(res.get("mean"))
        if mean is not None:
            approach_to_mean[app] = mean

    if not approach_to_mean:
        return None

    run_dir = path.parent
    return GroupResult(
        group_name="MICrONS (4 GB)",
        run_dir=run_dir,
        run_timestamp=_parse_ts_from_run_dir(run_dir),
        selector_value="size_gb=4",
        approach_to_mean=approach_to_mean,
    )


def _extract_cube_largest(path: Path) -> Optional[GroupResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return None
    counts = results.get("counts", [])
    if not counts:
        return None

    max_count = max(counts)
    idx = counts.index(max_count)
    candidate_approaches = [
        a for a in ["exact", "estimated", "cgal", "touch", "direct_estimation", "tdbase"]
        if a in results
    ]

    approach_to_mean: Dict[str, float] = {}
    for app in candidate_approaches:
        arr = (results.get(app) or {}).get("mean", [])
        if idx >= len(arr):
            continue
        mean = _valid_mean(arr[idx])
        if mean is not None:
            approach_to_mean[app] = mean

    if not approach_to_mean:
        return None

    run_dir = path.parent
    return GroupResult(
        group_name=f"Cubes ({max_count:,})",
        run_dir=run_dir,
        run_timestamp=_parse_ts_from_run_dir(run_dir),
        selector_value=f"count={max_count}",
        approach_to_mean=approach_to_mean,
    )


def _pick_latest_usable(runs_root: Path, prefix: str, extractor) -> GroupResult:
    run_dirs = sorted(runs_root.glob(f"{prefix}_*"), reverse=True)
    for run_dir in run_dirs:
        result_file = run_dir / "results.json"
        if not result_file.exists():
            continue
        try:
            extracted = extractor(result_file)
        except Exception as exc:
            print(f"Skipping {result_file}: failed to parse ({exc})")
            continue
        if extracted is not None:
            return extracted
    raise RuntimeError(f"No usable run found for {prefix}")


def _plot_grouped_bars(groups: List[GroupResult], output_base: Path) -> None:
    apply_paper_style()

    all_approaches = sorted({a for g in groups for a in g.approach_to_mean.keys()}, key=_approach_order_key)
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(1, len(all_approaches))

    fig, ax = plt.subplots(figsize=(12, 6.8))

    for i, app in enumerate(all_approaches):
        vals = []
        for g in groups:
            vals.append(g.approach_to_mean.get(app, np.nan))

        offset = (i - (len(all_approaches) - 1) / 2.0) * width
        st = style_for(app)
        heights = np.array(vals, dtype=float)
        mask = np.isfinite(heights)
        if not np.any(mask):
            continue

        ax.bar(
            x[mask] + offset,
            heights[mask],
            width=width,
            label=st["label"],
            color=st["color"],
            alpha=0.92,
        )

    ax.set_yscale("log")
    ax.set_ylabel("Overlap query time (ms) [log scale]")
    ax.set_xlabel("Dataset group")
    ax.set_xticks(x)
    ax.set_xticklabels([g.group_name for g in groups])
    ax.grid(axis="y", which="both", linestyle="-", alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    ax.legend(unique.values(), unique.keys(), loc="best")

    plt.tight_layout()
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped bar chart for overlap overall performance using latest usable runs: "
            "large nu (nu=400, large_nu_v), MICrONS (4GB), cube scalability (largest dataset)."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=SCRIPT_DIR / "runs",
        help="Directory that contains overlap run folders.",
    )
    args = parser.parse_args()

    runs_root = args.runs_root

    group_nu = _pick_latest_usable(runs_root, "overlap_nu_scalability", _extract_nu_large_400)
    group_microns = _pick_latest_usable(runs_root, "overlap_microns", _extract_microns_4gb)
    group_cube = _pick_latest_usable(runs_root, "overlap_cube_scalability", _extract_cube_largest)

    groups = [group_nu, group_microns, group_cube]

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_overall_performance")
    figures_dir = Path(run_layout["figures_dir"])
    output_base = figures_dir / f"mesh_overlap_overall_performance_{run_layout['timestamp']}"

    _plot_grouped_bars(groups, output_base)

    summary_payload = {
        "metadata": {
            "scenario": "overlap_overall_performance",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
        },
        "groups": [
            {
                "group_name": g.group_name,
                "selector": g.selector_value,
                "source_run": str(g.run_dir),
                "source_run_name": g.run_dir.name,
                "source_timestamp": g.run_timestamp,
                "results_ms": g.approach_to_mean,
            }
            for g in groups
        ],
    }
    write_json(Path(run_layout["results_json"]), summary_payload)

    print(f"Saved figure: {output_base}.png")
    print(f"Saved figure: {output_base}.pdf")
    for g in groups:
        print(f"{g.group_name}: {g.run_dir.name} ({g.selector_value}), approaches={sorted(g.approach_to_mean.keys(), key=_approach_order_key)}")


if __name__ == "__main__":
    main()
