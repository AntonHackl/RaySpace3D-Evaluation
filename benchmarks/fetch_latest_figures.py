#!/usr/bin/env python3
from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR / "latest_figures"


@dataclass(frozen=True)
class FigureSpec:
    output_name: str
    runs_dir: Path
    run_prefix: str
    source_pattern: str


FIGURES = [
    FigureSpec(
        output_name="mesh_complexity_scalability.pdf",
        runs_dir=SCRIPT_DIR / "mesh_overlap" / "runs",
        run_prefix="overlap_mesh_complexity_",
        source_pattern="figures/mesh_complexity_scalability.pdf",
    ),
    FigureSpec(
        output_name="mesh_overlap_nu_scalability_scaling.pdf",
        runs_dir=SCRIPT_DIR / "mesh_overlap" / "runs",
        run_prefix="overlap_nu_scalability_",
        source_pattern="figures/mesh_overlap_nu_scalability_scaling.pdf",
    ),
    FigureSpec(
        output_name="mesh_overlap_overall_performance.pdf",
        runs_dir=SCRIPT_DIR / "mesh_overlap" / "runs",
        run_prefix="overlap_overall_performance_",
        source_pattern="figures/mesh_overlap_overall_performance_*.pdf",
    ),
    FigureSpec(
        output_name="mesh_query_comparison_overall_performance.pdf",
        runs_dir=SCRIPT_DIR / "mesh_query_comparison" / "runs",
        run_prefix="query_comparison_overall_performance_",
        source_pattern="figures/mesh_query_comparison_overall_performance_*.pdf",
    ),
    FigureSpec(
        output_name="nu_scalability_query_time_comparison.pdf",
        runs_dir=SCRIPT_DIR / "mesh_query_comparison" / "runs",
        run_prefix="query_comparison_nu_scalability_",
        source_pattern="figures/query_time_comparison.pdf",
    ),
    FigureSpec(
        output_name="overlap_selectivity_scaling.pdf",
        runs_dir=SCRIPT_DIR / "mesh_overlap" / "runs",
        run_prefix="overlap_selectivity_",
        source_pattern="figures/selectivity_scaling.pdf",
    ),
]


def newest_matching_pdf(spec: FigureSpec) -> Path:
    candidate_runs = sorted(
        (
            path
            for path in spec.runs_dir.iterdir()
            if path.is_dir() and path.name.startswith(spec.run_prefix)
        ),
        reverse=True,
    )

    for run_dir in candidate_runs:
        matches = sorted(run_dir.glob(spec.source_pattern))
        if matches:
            return matches[-1]

    raise FileNotFoundError(
        f"No PDF matching {spec.source_pattern!r} found in runs with prefix "
        f"{spec.run_prefix!r} under {spec.runs_dir}"
    )


def main() -> int:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    copied = []
    for spec in FIGURES:
        source = newest_matching_pdf(spec)
        target = EXPORT_DIR / spec.output_name
        shutil.copy2(source, target)
        copied.append((target, source))

    print(f"Copied {len(copied)} figures to {EXPORT_DIR}")
    for target, source in copied:
        print(f"{target.name} <- {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
