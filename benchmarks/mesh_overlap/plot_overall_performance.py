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
from benchmarks.common.viz_utils import apply_paper_style, set_log_timing_axis_limits, style_for


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
    preferred = ["pierce", "cgal", "touch", "tdbase", "exact", "direct_estimation", "estimated", "estimated_mem10"]
    if name in preferred:
        return (preferred.index(name), name)
    return (len(preferred), name)

def _canonicalize_approaches(raw: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # Represent RaySpace with one canonical series to avoid duplicate legend entries
    # and empty slots when one internal mode is missing for a given dataset group.
    if raw.get("pierce") is not None:
        out["pierce"] = raw["pierce"]
    for k in ("direct_estimation", "estimated", "exact", "estimated_mem10"):
        v = raw.get(k)
        if v is not None:
            out["pierce"] = v
            break

    for src, dst in (("cgal", "cgal"), ("touch", "touch"), ("tdbase", "tdbase")):
        v = raw.get(src)
        if v is not None:
            out[dst] = v
    return out


def _extract_nu_profile_400(path: Path, *, dataset_profile: str, group_name: str) -> Optional[GroupResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    if md.get("dataset_profile") != dataset_profile:
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
        group_name=group_name,
        run_dir=run_dir,
        run_timestamp=_parse_ts_from_run_dir(run_dir),
        selector_value="nu=400",
        approach_to_mean=approach_to_mean,
    )


def _extract_nu_large_400(path: Path) -> Optional[GroupResult]:
    return _extract_nu_profile_400(path, dataset_profile="large_nu_v", group_name=r"Nuclei $\bowtie$ Vessel")


def _extract_nu_nn_large_400(path: Path) -> Optional[GroupResult]:
    return _extract_nu_profile_400(path, dataset_profile="large_nu_nn", group_name=r"Nuclei $\bowtie$ Nuclei")


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
        group_name=r"Neurons $\bowtie$ Neurons",
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
        group_name=r"Cubes $\bowtie$ Cubes",
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

def _pick_latest_usable_with_required_approach(
    runs_root: Path,
    prefix: str,
    extractor,
    required_approach: str,
) -> GroupResult:
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
        if extracted is None:
            continue
        canonical = _canonicalize_approaches(extracted.approach_to_mean)
        if required_approach in canonical:
            extracted.approach_to_mean = canonical
            return extracted
    raise RuntimeError(f"No usable run found for {prefix} with approach '{required_approach}'")


def _plot_grouped_bars(groups: List[GroupResult], output_base: Path) -> None:
    apply_paper_style()

    for g in groups:
        g.approach_to_mean = _canonicalize_approaches(g.approach_to_mean)

    all_approaches = {a for g in groups for a in g.approach_to_mean.keys()}
    # Ensure TDBase is included in the comparison even if missing from all selected runs,
    # so we can show it as 'missing/failed' rather than just absent.
    all_approaches.add("tdbase")
    all_approaches = sorted(all_approaches, key=_approach_order_key)

    num_apps = len(all_approaches)
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(1, num_apps)

    fig, ax = plt.subplots(figsize=(12, 6.8))

    seen_labels = set()
    for gi, g in enumerate(groups):
        for pi, app in enumerate(all_approaches):
            offset = (pi - (num_apps - 1) / 2.0) * width
            st = style_for(app)
            label = st["label"]
            
            if app in g.approach_to_mean:
                bar_label = label if label not in seen_labels else None
                if bar_label:
                    seen_labels.add(label)
                ax.bar(
                    x[gi] + offset,
                    g.approach_to_mean[app],
                    width=width,
                    label=bar_label,
                    color=st["color"],
                    alpha=0.92,
                    hatch=st.get("hatch", ""),
                    edgecolor="black",
                    linewidth=0.6,
                )
            elif app == "tdbase":
                # For TDBase, if it's missing, draw an 'X' to indicate it couldn't run
                # rather than just leaving an empty space or skipping it.
                ax.text(
                    x[gi] + offset,
                    1.2, # Slightly above the 1.0 floor for visibility
                    "X",
                    ha='center',
                    va='bottom',
                    color=st["color"],
                    fontsize=34,
                    fontweight='bold'
                )
                # Ensure it appears in the legend if this is the first time we encounter it
                if label not in seen_labels:
                    # Plot a dummy bar with 0 height to get a legend entry with the correct style
                    ax.bar(x[gi] + offset, 0, width=width, label=label, color=st["color"], 
                           alpha=0.92, hatch=st.get("hatch", ""), edgecolor="black", linewidth=0.6)
                    seen_labels.add(label)

    ax.set_yscale("log")
    all_vals = [v for g in groups for v in g.approach_to_mean.values()]
    set_log_timing_axis_limits(ax, all_vals, floor=1.0)
    ax.set_ylabel("Overlap query time (ms) [log scale]")
    ax.set_xticks(x)
    ax.set_xticklabels([g.group_name for g in groups])
    ax.grid(False)
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
            "large nu (nu=400, large_nu_v), large nu nn (nu=400, large_nu_nn), "
            "MICrONS (4GB), cube scalability (largest dataset)."
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
    group_nu_nn = _pick_latest_usable(runs_root, "overlap_nu_scalability", _extract_nu_nn_large_400)
    group_microns = _pick_latest_usable_with_required_approach(
        runs_root, "overlap_microns", _extract_microns_4gb, "pierce"
    )
    group_cube = _pick_latest_usable(runs_root, "overlap_cube_scalability", _extract_cube_largest)

    groups = [group_nu, group_nu_nn, group_microns, group_cube]

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
