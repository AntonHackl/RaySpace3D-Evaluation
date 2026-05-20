#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess 
import json
import re

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import PAPER_FIGSIZE, apply_paper_style, make_legend_bold, style_for
from benchmarks.common.scenario_utils import canonical_nu_pair_paths, ensure_nu_pair_dataset, get_shared_data_dirs

sys.path.append(str(Path(__file__).parent))
from adapters.raytracer_adapter import RaytracerAdapter
from adapters.tdbase_adapter import TDBaseAdapter
from benchmarks.common.adapters.tdbase_common import (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    TDBASE_TIMING_MODES,
)

RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
TDBASE_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/tdbase"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

TIMEOUT_SECONDS = 120.0
DEFAULT_NU_COUNTS = [200, 400, 600]
LEGACY_RAW_DIR = RAW_DIR
SHARED_SCENARIO = "nu_scalability"
LARGE_NU_LEGACY_RAW_DIR = RAW_DIR / "large_nu_v"

def resolve_dataset_pair(raw_shared_dir, nu, nv=150, prefix="tdbase"):
    n_file, v_file = canonical_nu_pair_paths(raw_shared_dir, nu=nu, nv=nv, prefix=prefix)
    ensure_nu_pair_dataset(n_file, v_file, legacy_raw_dirs=[LEGACY_RAW_DIR])
    return v_file, n_file

def run_experiment(
    runs,
    grid_cell_size,
    nu_counts,
    run_log_dir,
    approaches=None,
    timeout=120.0,
    dataset_profile="large_nu_v",
    tdbase_timing_mode=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
):
    if approaches is None:
        approaches = ["direct_estimation", "tdbase"]
    
    print(f"--- Starting Query Preprocessing Time Experiment ({nu_counts}) ---")
    if dataset_profile == "large_nu_v":
        shared_dirs = get_shared_data_dirs("large_nu_nn_scalability")
        dataset_nv = 750
        dataset_prefix = "tdbase_large"
        legacy_raw_dirs = [LARGE_NU_LEGACY_RAW_DIR, LEGACY_RAW_DIR]
    else:
        shared_dirs = get_shared_data_dirs(SHARED_SCENARIO)
        dataset_nv = 150
        dataset_prefix = "tdbase"
        legacy_raw_dirs = [LEGACY_RAW_DIR]
    shared_raw_dir = shared_dirs["raw"]
    
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)

    direct_estimation_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), 
        mode="direct_estimation", 
        preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(TIMINGS_DIR),
        grid_cell_size=grid_cell_size,
        warmup_runs=1
    )
    tdbase_adapter = TDBaseAdapter(
        str(TDBASE_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR),
        query_timing_mode=tdbase_timing_mode,
    )
    
    results = {
        "counts": [],
        "enabled_approaches": approaches,
        "direct_estimation": {"query_prep_mean": []},
        "tdbase": {"query_prep_mean": []}
    }

    results_file = run_log_dir / "results.json"

    for nu in nu_counts:
        f_v_path, f_n_path = canonical_nu_pair_paths(
            shared_raw_dir, nu=nu, nv=dataset_nv, prefix=dataset_prefix
        )
        ensure_nu_pair_dataset(f_v_path, f_n_path, legacy_raw_dirs=legacy_raw_dirs)
        if not f_v_path or not f_n_path: continue
        print(f"\nProcessing nu={nu}: {f_v_path.name} vs {f_n_path.name}")

        direct_estimation_adapter.preprocess_from_source(str(f_v_path), str(f_v_path), log_dir=str(run_log_dir))
        direct_estimation_adapter.preprocess_from_source(str(f_n_path), str(f_n_path), log_dir=str(run_log_dir))
        tdbase_adapter.preprocess_from_source(str(f_v_path), str(f_v_path), log_dir=str(run_log_dir))
        tdbase_adapter.preprocess_from_source(str(f_n_path), str(f_n_path), log_dir=str(run_log_dir))

        results["counts"].append(nu)

        if "direct_estimation" in approaches:
            print(f"Running Direct Estimation ({runs} runs)...")
            res_de = direct_estimation_adapter.run_overlap(str(f_v_path), str(f_n_path), runs, log_dir=str(run_log_dir), timeout=timeout)
            if "error" in res_de:
                results["direct_estimation"]["query_prep_mean"].append(None)
            else:
                breakdown = res_de.get("breakdown", {})
                prep_time = 0.0
                query_components = ["selectivity estimation", "raytrace_hash", "raytrace_overlap_hash", "compact_hash_table_pairs", "download results", "query", "execute hash query"]
                for p, d in breakdown.items():
                    if not any(p.startswith(q) for q in query_components):
                        prep_time += d
                results["direct_estimation"]["query_prep_mean"].append(prep_time)

        if "tdbase" in approaches:
            print(f"Running TDBase ({runs} runs)...")
            res_td = tdbase_adapter.run_overlap(str(f_v_path), str(f_n_path), runs, log_dir=str(run_log_dir), timeout=timeout)
            if "error" in res_td:
                results["tdbase"]["query_prep_mean"].append(None)
            else:
                results["tdbase"]["query_prep_mean"].append(res_td.get("mean_preprocessing", 0.0))

        write_json(results_file, results)
    return results_file

def create_plot(results_file, figures_dir, default_approaches=None, title_suffix=""):
    results_path = Path(results_file)
    with open(results_path, 'r') as f:
        results = json.load(f)
    counts = results['counts']
    enabled_approaches = results.get('enabled_approaches', default_approaches or ["direct_estimation", "tdbase"])

    apply_paper_style()
    bar_width = 0.35
    index = np.arange(len(counts))
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    
    if "tdbase" in enabled_approaches and any(v is not None for v in results["tdbase"]["query_prep_mean"]):
        ax.bar(index - bar_width/2, results["tdbase"]["query_prep_mean"], bar_width, label='TDBase', color=style_for("tdbase")['color'])
    if "direct_estimation" in enabled_approaches and any(v is not None for v in results["direct_estimation"]["query_prep_mean"]):
        ax.bar(index + bar_width/2, results["direct_estimation"]["query_prep_mean"], bar_width, label='RaySpace3D', color=style_for("raytracer_estimated")['color'])

    ax.set_xlabel(r'Number of Objects $|U|$')
    ax.set_ylabel('Query Setup Time (ms)')
    ax.set_title('Query Preprocessing Overheads' + title_suffix)
    ax.set_xticks(index)
    ax.set_xticklabels(counts)
    make_legend_bold(ax)
    fig.tight_layout()
    output_pdf = figures_dir / f"query_preprocessing{title_suffix.lower().replace(' ', '_')}.pdf"
    output_png = figures_dir / f"query_preprocessing{title_suffix.lower().replace(' ', '_')}.png"
    plt.savefig(output_pdf)
    try:
        plt.savefig(output_png, dpi=300)
    except Exception:
        # Fallback: save without dpi if PNG writer fails for some reason
        plt.savefig(output_png)

def main():
    parser = argparse.ArgumentParser(description="Query Preprocessing Overlap Benchmark")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--grid-cell-size", type=float, default=200.0)
    parser.add_argument("--nu-counts", type=int, nargs="+", default=DEFAULT_NU_COUNTS)
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--results-file", type=str)
    parser.add_argument("--approaches", type=str, nargs="+", choices=["direct_estimation", "tdbase"], default=["direct_estimation", "tdbase"])
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--tdbase-timing-mode",
        type=str,
        default=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
        choices=TDBASE_TIMING_MODES,
        help="TDBase query-time definition. Default uses index+compute+evaluate; use compute_only to revert.",
    )
    parser.add_argument(
        "--dataset-profile",
        type=str,
        choices=["standard", "large_nu_v"],
        default="large_nu_v",
        help="Dataset source profile for NU pair inputs.",
    )
    args = parser.parse_args()

    title_suffix = ""
    
    if args.skip_runs and args.results_file:
        figures_dir = Path(args.results_file).parent / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        create_plot(args.results_file, figures_dir, args.approaches, title_suffix)
    else:
        run_name = f"query_preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        layout = create_benchmark_run_layout(SCRIPT_DIR, run_name)
        res_file = run_experiment(
            args.runs,
            args.grid_cell_size,
            args.nu_counts,
            layout["logs_dir"],
            args.approaches,
            args.timeout,
            args.dataset_profile,
            args.tdbase_timing_mode,
        )
        import shutil
        dest_res = layout["results_json"]
        shutil.copy(res_file, dest_res)
        create_plot(dest_res, layout["figures_dir"], args.approaches, title_suffix)

if __name__ == "__main__":
    main()
