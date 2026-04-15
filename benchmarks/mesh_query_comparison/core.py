from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.mesh_containment.adapters.raytracer_adapter import RaytracerContainmentAdapter
from benchmarks.mesh_intersection.adapters.raytracer_adapter import RaytracerIntersectionAdapter
from benchmarks.mesh_overlap.adapters.raytracer_adapter import RaytracerAdapter as RaytracerOverlapAdapter


QUERY_CHOICES = ["overlap", "intersection", "containment"]


def add_query_selection_arguments(parser) -> None:
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        choices=QUERY_CHOICES,
        default=None,
        help="Query types to compare. Default compares all three.",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        choices=QUERY_CHOICES,
        default=None,
        help="Alias for --queries for compatibility.",
    )


def resolve_queries(queries: Sequence[str] | None, approaches: Sequence[str] | None) -> list[str]:
    selected = list(queries) if queries else (list(approaches) if approaches else list(QUERY_CHOICES))
    ordered_unique = [q for q in QUERY_CHOICES if q in selected]
    if len(ordered_unique) < 2:
        raise ValueError("Select at least two query types with --queries/--approaches.")
    return ordered_unique


def build_raytracer_query_adapters(
    *,
    repo_root: Path,
    shared_dirs: Dict[str, Path],
    grid_resolution: int,
    warmup_runs: int,
    overlap_mode: str,
    intersection_mode: str,
    include_overlap_pairs: bool,
    use_anyhit_point_in_mesh: bool,
) -> Dict[str, Any]:
    rayspace_dir = repo_root / "src" / "RaySpace3D"

    overlap = RaytracerOverlapAdapter(
        str(rayspace_dir),
        mode=overlap_mode,
        preprocessed_dir=str(shared_dirs["preprocessed"]),
        timings_dir=str(shared_dirs["timings"]),
        grid_resolution=grid_resolution,
        warmup_runs=warmup_runs,
    )
    intersection = RaytracerIntersectionAdapter(
        str(rayspace_dir),
        mode=intersection_mode,
        preprocessed_dir=str(shared_dirs["preprocessed"]),
        timings_dir=str(shared_dirs["timings"]),
        grid_resolution=grid_resolution,
        warmup_runs=warmup_runs,
    )
    containment = RaytracerContainmentAdapter(
        str(rayspace_dir),
        preprocessed_dir=str(shared_dirs["preprocessed"]),
        timings_dir=str(shared_dirs["timings"]),
        grid_resolution=grid_resolution,
        warmup_runs=warmup_runs,
        use_anyhit_point_in_mesh=use_anyhit_point_in_mesh,
        include_overlap_pairs=include_overlap_pairs,
    )
    return {
        "overlap": overlap,
        "intersection": intersection,
        "containment": containment,
    }


def sanitize_case_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def ensure_preprocessed(adapters: Dict[str, Any], mesh_paths: Iterable[Path], log_dir: Path | None = None) -> None:
    # All query adapters read the same .pre output naming scheme.
    overlap_adapter = adapters["overlap"]
    for mesh_path in mesh_paths:
        if not overlap_adapter.check_preprocessed(str(mesh_path)):
            overlap_adapter.preprocess_from_source(
                str(mesh_path),
                str(mesh_path),
                log_dir=str(log_dir) if log_dir is not None else None,
            )


def run_selected_queries(
    *,
    adapters: Dict[str, Any],
    queries: Sequence[str],
    mesh1: Path,
    mesh2: Path,
    runs: int,
    timeout: float,
    overlap_query_direction: str,
    intersection_extra_args: list[str],
    log_dir: Path | None = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    log_dir_str = str(log_dir) if log_dir is not None else None

    if "overlap" in queries:
        results["overlap"] = adapters["overlap"].run_overlap(
            str(mesh1),
            str(mesh2),
            runs,
            timeout=timeout,
            log_dir=log_dir_str,
            query_direction=overlap_query_direction,
        )

    if "intersection" in queries:
        results["intersection"] = adapters["intersection"].run_intersection(
            str(mesh1),
            str(mesh2),
            runs,
            timeout=timeout,
            log_dir=log_dir_str,
            extra_args=intersection_extra_args,
        )

    if "containment" in queries:
        results["containment"] = adapters["containment"].run_containment(
            str(mesh1),
            str(mesh2),
            runs,
            timeout=timeout,
            log_dir=log_dir_str,
        )

    return results


def build_intersection_extra_args(
    *,
    overlap_max_iterations: int,
    containment_max_iterations: int,
    hash_load_factor: float,
    enable_profiling_stats: bool,
    intersection_query_direction: str,
) -> list[str]:
    args = [
        "--query-direction",
        intersection_query_direction,
        "--overlap-max-iterations",
        str(overlap_max_iterations),
        "--containment-max-iterations",
        str(containment_max_iterations),
        "--hash-load-factor",
        str(hash_load_factor),
    ]
    if enable_profiling_stats:
        args.append("--enable-profiling-stats")
    return args


def _normalize_breakdown_entry(entry: Any) -> Dict[str, float]:
    if not isinstance(entry, dict):
        return {}

    normalized: Dict[str, float] = {}
    for key, value in entry.items():
        if isinstance(value, (int, float)):
            normalized[str(key)] = float(value)
            continue
        if isinstance(value, dict):
            # Containment uses {'phase': {'mean': ..., 'min': ...}}.
            maybe_mean = value.get("mean")
            if isinstance(maybe_mean, (int, float)):
                normalized[str(key)] = float(maybe_mean)
    return normalized


def generate_query_comparison_figures(
    *,
    results_rows: Sequence[Dict[str, Any]],
    queries: Sequence[str],
    case_labels: Sequence[str],
    figures_dir: Path,
    title_prefix: str,
    x_axis_label: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not results_rows or not case_labels:
        return

    x = np.arange(len(case_labels))

    # Figure 1: Total query time comparison across all selected query types.
    width = 0.8 / max(1, len(queries))
    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(case_labels)), 6))
    for idx, query in enumerate(queries):
        means = []
        for row in results_rows:
            item = row.get(query, {})
            mean = item.get("mean") if isinstance(item, dict) else None
            means.append(float(mean) if isinstance(mean, (int, float)) else np.nan)
        offsets = x - (0.4 - width / 2.0) + idx * width
        ax.bar(offsets, means, width=width, label=query)

    ax.set_title(f"{title_prefix}: Total Query Time Comparison")
    ax.set_xlabel(x_axis_label)
    ax.set_ylabel("Query time (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=20, ha="right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "query_time_comparison.png", dpi=180)
    fig.savefig(figures_dir / "query_time_comparison.pdf")
    plt.close(fig)

    # Figure 2+: One stacked breakdown chart per query.
    for query in queries:
        query_breakdowns = []
        component_names: set[str] = set()
        for row in results_rows:
            item = row.get(query, {})
            normalized = _normalize_breakdown_entry(item.get("breakdown") if isinstance(item, dict) else {})
            query_breakdowns.append(normalized)
            component_names.update(normalized.keys())

        if not component_names:
            continue

        ordered_components = sorted(component_names)
        fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(case_labels)), 6))
        bottoms = np.zeros(len(case_labels), dtype=float)

        for component in ordered_components:
            vals = np.array([bd.get(component, 0.0) for bd in query_breakdowns], dtype=float)
            ax.bar(x, vals, bottom=bottoms, label=component)
            bottoms += vals

        ax.set_title(f"{title_prefix}: {query} breakdown")
        ax.set_xlabel(x_axis_label)
        ax.set_ylabel("Time (ms)")
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, rotation=20, ha="right")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
        fig.tight_layout()
        safe_query = sanitize_case_token(query)
        fig.savefig(figures_dir / f"breakdown_{safe_query}.png", dpi=180)
        fig.savefig(figures_dir / f"breakdown_{safe_query}.pdf")
        plt.close(fig)
