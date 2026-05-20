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
# Add project root to sys.path to allow imports from 'benchmarks'
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import PAPER_FIGSIZE, PAPER_WIDE_FIGSIZE, apply_paper_style, make_legend_bold, set_log_timing_axis_limits, style_for
from benchmarks.common.scenario_utils import (
    canonical_nn_pair_paths,
    ensure_nn_pair_dataset,
    get_shared_data_dirs,
)

# Add current directory to path to import adapters
sys.path.append(str(Path(__file__).parent))
from adapters.raytracer_adapter import RaytracerAdapter
from adapters.cgal_adapter import CGALAdapter
from adapters.touch_adapter import TOUCHAdapter
from adapters.tdbase_adapter import TDBaseAdapter
from benchmarks.common.adapters.tdbase_common import (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    TDBASE_TIMING_MODES,
)

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
CGAL_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/CGAL"
TDBASE_DIR = REPO_ROOT / "baselines/RaySpace3DBaselines/tdbase"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

TIMEOUT_SECONDS = 120.0

# Nu Counts for Dataset B (Dataset A is fixed at corresponding vessel count)
DEFAULT_NU_COUNTS = [200, 400, 600, 800]
LEGACY_RAW_DIR = RAW_DIR
SHARED_SCENARIO = "nn_scalability"

def resolve_nn_dataset_pair(raw_shared_dir: Path, nu: int):
    n_file1, n_file2 = canonical_nn_pair_paths(raw_shared_dir, nu=nu)
    ensure_nn_pair_dataset(
        n_file1,
        n_file2,
        legacy_raw_dirs=[LEGACY_RAW_DIR],
    )
    return n_file1, n_file2

def run_experiment(
    runs,
    grid_cell_size,
    nu_counts,
    run_log_dir,
    approaches=None,
    track_hash_contention=False,
    timeout=120.0,
    tdbase_timing_mode=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
):
    if approaches is None:
        approaches = ["exact", "direct_estimation", "cgal", "touch", "tdbase"]
    
    print(f"--- Starting NN Scalability Experiment ({nu_counts}) ---")
    print(f"Approaches: {approaches}")
    if track_hash_contention:
        print("Direct estimation hash contention tracking: enabled")
    
    # Ensure directories exist
    shared_dirs = get_shared_data_dirs(SHARED_SCENARIO)
    shared_raw_dir = shared_dirs["raw"]
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Logging runs to: {run_log_dir}")
    print(f"RaySpace Dir: {RAYSPACE_DIR}")

    # Initialize Adapters
    print("Initializing adapters...")
    exact_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), 
        mode="exact", 
        preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(TIMINGS_DIR),
        grid_cell_size=grid_cell_size,
        warmup_runs=1
    )
    
    direct_estimation_adapter = RaytracerAdapter(
        str(RAYSPACE_DIR), 
        mode="direct_estimation", 
        preprocessed_dir=str(PREPROCESSED_DIR), 
        timings_dir=str(TIMINGS_DIR),
        grid_cell_size=grid_cell_size,
        warmup_runs=1,
        track_hash_contention=track_hash_contention,
    )

    cgal_adapter = CGALAdapter(
        str(CGAL_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR),
        grid_cell_size=grid_cell_size
    )
    
    touch_adapter = TOUCHAdapter(
        str(CGAL_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR),
        grid_cell_size=grid_cell_size
    )

    tdbase_adapter = TDBaseAdapter(
        str(TDBASE_DIR),
        preprocessed_dir=str(PREPROCESSED_DIR),
        query_timing_mode=tdbase_timing_mode,
    )
    
    results = {
        "counts": [],
        "enabled_approaches": approaches,
        "exact": {"mean": [], "std": [], "breakdown": []},
        "direct_estimation": {"mean": [], "std": [], "breakdown": []},
        "cgal": {"mean": [], "std": []},
        "touch": {"mean": [], "std": []},
        "tdbase": {"mean": [], "std": []}
    }

    for nu in nu_counts:
        f_v_path, f_n_path = resolve_nn_dataset_pair(shared_raw_dir, nu)
        
        if not f_v_path or not f_n_path:
            print(f"Error: Datasets for nu={nu} not found in {shared_raw_dir}! Skipping.")
            continue
        
        print(f"\nProcessing nu={nu}: {f_v_path.name} vs {f_n_path.name}")

        # Check/Run Preprocessing for Raytracer (also used by Face/TOUCH adapters)
        needs_preprocessing = any(a in approaches for a in ["exact", "direct_estimation", "cgal", "touch"])
        if needs_preprocessing:
            print("Checking preprocessing...")
            exact_adapter.preprocess_from_source(str(f_v_path), str(f_v_path), log_dir=str(run_log_dir))
            exact_adapter.preprocess_from_source(str(f_n_path), str(f_n_path), log_dir=str(run_log_dir))

        # Run Exact Benchmark
        res_exact = {"mean": None, "std": None, "breakdown": {}}
        if "exact" in approaches:
            print(f"Running Exact Mode ({runs} runs)...")
            res_exact = exact_adapter.run_overlap(
                str(f_v_path), 
                str(f_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_exact:
                print(f"Error in exact run: {res_exact['error']}")
                res_exact = {"mean": None, "std": None, "breakdown": {}}
            
        # Run Selectivity Estimation Benchmark
        res_direct = {"mean": None, "std": None, "breakdown": {}}
        if "direct_estimation" in approaches:
            print(f"Running Selectivity Estimation Mode ({runs} runs)...")
            res_direct = direct_estimation_adapter.run_overlap(
                str(f_v_path), 
                str(f_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_direct:
                print(f"Error in selectivity estimation run: {res_direct['error']}")
                res_direct = {"mean": None, "std": None, "breakdown": {}}

        # Run Face Benchmark
        res_cgal = {"mean": None, "std": None}
        if "cgal" in approaches:
            print(f"Running Face Mode ({runs} runs)...")
            res_cgal = cgal_adapter.run_overlap(
                str(f_v_path), 
                str(f_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_cgal:
                print(f"Error in Face run: {res_cgal['error']}")
                res_cgal = {"mean": None, "std": None}

        # Run TOUCH Benchmark
        res_touch = {"mean": None, "std": None}
        if "touch" in approaches:
            print(f"Running TOUCH Mode ({runs} runs)...")
            res_touch = touch_adapter.run_overlap(
                str(f_v_path), 
                str(f_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_touch:
                print(f"Error in TOUCH run: {res_touch['error']}")
                res_touch = {"mean": None, "std": None}

        # Run TDBase Benchmark
        res_td = {"mean": None, "std": None}
        if "tdbase" in approaches:
            print(f"Running TDBase Mode ({runs} runs)...")
            res_td = tdbase_adapter.run_overlap(
                str(f_v_path), 
                str(f_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_td:
                print(f"Error in TDBase run: {res_td['error']}")
                res_td = {"mean": None, "std": None}

        results["counts"].append(nu)
        
        results["exact"]["mean"].append(res_exact["mean"])
        results["exact"]["std"].append(res_exact["std"])
        results["exact"]["breakdown"].append(res_exact.get("breakdown", {}))
        
        results["direct_estimation"]["mean"].append(res_direct["mean"])
        results["direct_estimation"]["std"].append(res_direct["std"])
        results["direct_estimation"]["breakdown"].append(res_direct.get("breakdown", {}))
        
        results["cgal"]["mean"].append(res_cgal["mean"])
        results["cgal"]["std"].append(res_cgal["std"])
        
        results["touch"]["mean"].append(res_touch["mean"])
        results["touch"]["std"].append(res_touch["std"])

        results["tdbase"]["mean"].append(res_td["mean"])
        results["tdbase"]["std"].append(res_td["std"])
        
        # Add dataset sizes
        if "num_obj1" not in results:
             results["num_obj1"] = []
             results["num_obj2"] = []
             results["num_intersections"] = []
             results["size_bytes1"] = []
             results["size_bytes2"] = []
             results["universe_extents1"] = []
             results["universe_extents2"] = []
        
        # Use first available result for counts
        found_counts = False
        for res in [res_exact, res_direct]:
            if "num_obj1" in res and res["num_obj1"] > 0:
                results["num_obj1"].append(int(res["num_obj1"]))
                results["num_obj2"].append(int(res["num_obj2"]))
                results["num_intersections"].append(int(res.get("num_intersections", 0)))
                results["universe_extents1"].append(res.get("universe_extents1", [0.0, 0.0, 0.0]))
                results["universe_extents2"].append(res.get("universe_extents2", [0.0, 0.0, 0.0]))
                found_counts = True
                break
        
        if not found_counts:
            results["num_obj1"].append(0)
            results["num_obj2"].append(0)
            results["num_intersections"].append(0)
            results["universe_extents1"].append([0.0, 0.0, 0.0])
            results["universe_extents2"].append([0.0, 0.0, 0.0])

        results["size_bytes1"].append(f_v_path.stat().st_size if f_v_path.exists() else 0)
        results["size_bytes2"].append(f_n_path.stat().st_size if f_n_path.exists() else 0)
        
        exact_str = f"{res_exact['mean']:.2f}ms" if res_exact['mean'] is not None else "N/A"
        direct_str = f"{res_direct['mean']:.2f}ms" if res_direct['mean'] is not None else "N/A"
        cgal_str = f"{res_cgal['mean']:.2f}ms" if res_cgal['mean'] is not None else "N/A"
        touch_str = f"{res_touch['mean']:.2f}ms" if res_touch['mean'] is not None else "N/A"
        td_str = f"{res_td['mean']:.2f}ms" if res_td['mean'] is not None else "N/A"
        print(f"Done nu={nu}: Exact={exact_str}, Selectivity Estimation={direct_str}, Face={cgal_str}, TOUCH={touch_str}, TDBase={td_str}")

    return results

def plot_results(results, figures_dir):
    print("\nPlotting results...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    counts = results["counts"]
    if not counts:
        print("No results to plot.")
        return

    # --- Plot 1: Line Chart (Scaling) ---
    def generate_scaling_plot(ax, results, counts):
        enabled = set(results.get("enabled_approaches", ["exact", "direct_estimation", "cgal", "touch", "tdbase"]))
        all_y_vals = []

        x_labels = []
        for i, count in enumerate(counts):
            if "num_intersections" in results and i < len(results["num_intersections"]) and results.get("num_obj1", [])[i] > 0 and results.get("num_obj2", [])[i] > 0:
                sel = results["num_intersections"][i] / (results["num_obj1"][i] * results["num_obj2"][i])
                x_labels.append(f"{count}\n(sel={sel*100:.2e}%)")
            else:
                x_labels.append(str(count))

        if "exact" in enabled:
            exact_valid_indices = [i for i, m in enumerate(results["exact"]["mean"]) if m is not None]
            if exact_valid_indices:
                exact_counts = [counts[i] for i in exact_valid_indices]
                exact_means = [results["exact"]["mean"][i] for i in exact_valid_indices]
                exact_stds = [results["exact"]["std"][i] for i in exact_valid_indices]
                all_y_vals.extend(exact_means)
                st = style_for("exact")
                ax.errorbar(exact_counts, exact_means, yerr=exact_stds, linestyle=st.get("linestyle", "-"), marker=st.get("marker"), capsize=5, color=st["color"])
                ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        if "direct_estimation" in enabled:
            direct_valid_indices = [i for i, m in enumerate(results["direct_estimation"]["mean"]) if m is not None]
            if direct_valid_indices:
                direct_counts = [counts[i] for i in direct_valid_indices]
                direct_means = [results["direct_estimation"]["mean"][i] for i in direct_valid_indices]
                direct_stds = [results["direct_estimation"]["std"][i] for i in direct_valid_indices]
                all_y_vals.extend(direct_means)
                st = style_for("direct_estimation")
                ax.errorbar(direct_counts, direct_means, yerr=direct_stds, linestyle=st.get("linestyle", "-"), marker=st.get("marker"), capsize=5, color=st["color"])
                ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])
        
        # Filter valid Face points
        cgal_valid_indices = [i for i, m in enumerate(results["cgal"]["mean"]) if m is not None] if "cgal" in enabled else []
        if cgal_valid_indices:
            cgal_counts = [counts[i] for i in cgal_valid_indices]
            cgal_means = [results["cgal"]["mean"][i] for i in cgal_valid_indices]
            cgal_stds = [results["cgal"]["std"][i] for i in cgal_valid_indices]
            all_y_vals.extend(cgal_means)
            st = style_for("cgal")
            ax.errorbar(cgal_counts, cgal_means, yerr=cgal_stds, linestyle=st.get("linestyle", "-"), marker=st.get("marker"), capsize=5, color=st["color"])
            ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        # Filter valid TOUCH points
        touch_valid_indices = [i for i, m in enumerate(results["touch"]["mean"]) if m is not None] if "touch" in enabled else []
        if touch_valid_indices:
            touch_counts = [counts[i] for i in touch_valid_indices]
            touch_means = [results["touch"]["mean"][i] for i in touch_valid_indices]
            touch_stds = [results["touch"]["std"][i] for i in touch_valid_indices]
            all_y_vals.extend(touch_means)
            st = style_for("touch")
            ax.errorbar(touch_counts, touch_means, yerr=touch_stds, linestyle=st.get("linestyle", "-"), marker=st.get("marker"), capsize=5, color=st["color"])
            ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        # Filter valid TDBase points
        td_valid_indices = [i for i, m in enumerate(results["tdbase"]["mean"]) if m is not None] if "tdbase" in enabled else []
        if td_valid_indices:
            td_counts = [counts[i] for i in td_valid_indices]
            td_means = [results["tdbase"]["mean"][i] for i in td_valid_indices]
            td_stds = [results["tdbase"]["std"][i] for i in td_valid_indices]
            all_y_vals.extend(td_means)
            st = style_for("tdbase")
            ax.errorbar(td_counts, td_means, yerr=td_stds, linestyle=st.get("linestyle", "-"), marker=st.get("marker"), capsize=5, color=st["color"])
            ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        ax.set_xlabel('Nuclei per Vessel (Total objects ≃ Nu * 300)')
        ax.set_ylabel('Execution Time (ms) [Log Scale]')
        ax.set_yscale('log')
        set_log_timing_axis_limits(ax, all_y_vals)
        make_legend_bold(ax)
        ax.grid(False)
        ax.set_xticks(counts)
        ax.set_xticklabels(x_labels)

    # --- Plot 2: Breakdown Bar Chart ---
    def generate_breakdown_plot(ax, results, counts):
        enabled = set(results.get("enabled_approaches", ["exact", "direct_estimation", "cgal", "touch", "tdbase"]))
        
        x_labels = []
        for i, count in enumerate(counts):
            if "num_intersections" in results and i < len(results["num_intersections"]) and results.get("num_obj1", [])[i] > 0 and results.get("num_obj2", [])[i] > 0:
                sel = results["num_intersections"][i] / (results["num_obj1"][i] * results["num_obj2"][i])
                x_labels.append(f"{count}\n(sel={sel*100:.2e}%)")
            else:
                x_labels.append(str(count))

        def normalize_phase_key(phase: str) -> str:
            key = re.sub(r"_\d+$", "", phase.lower())
            key = re.sub(r"_+$", "", key)
            return key

        phase_labels = {
            "selectivity estimation": "Selectivity Est.",
            "query": "Ray Query",
            "execute hash query": "Hash Query",
            "gpu deduplication": "Deduplication",
            "download results": "Download",
            "raytrace_mesh1tomesh2_pass1": "Raytrace M1→M2 (Pass 1)",
            "raytrace_mesh2tomesh1_pass1": "Raytrace M2→M1 (Pass 1)",
            "raytrace_mesh2tomesh1_pass2": "Raytrace M2→M1 (Pass 2)",
            "raytrace_mesh1tomesh2_pass2": "Raytrace M1→M2 (Pass 2)",
            "raytrace_hash_mesh1tomesh2": "Hash Raytrace M1→M2",
            "raytrace_hash_mesh2tomesh1": "Hash Raytrace M2→M1",
        }
        phase_order = [
            "selectivity estimation",
            "raytrace_mesh1tomesh2_pass1",
            "raytrace_mesh2tomesh1_pass1",
            "raytrace_mesh1tomesh2_pass2",
            "raytrace_mesh2tomesh1_pass2",
            "raytrace_hash_mesh1tomesh2",
            "raytrace_hash_mesh2tomesh1",
            "query",
            "execute hash query",
            "gpu deduplication",
            "download results",
        ]
        phase_colors = {
            "selectivity estimation": "#ff9896",
            "raytrace_mesh1tomesh2_pass1": "#1f77b4",
            "raytrace_mesh2tomesh1_pass1": "#17becf",
            "raytrace_mesh2tomesh1_pass2": "#2ca02c",
            "raytrace_mesh1tomesh2_pass2": "#9467bd",
            "raytrace_hash_mesh1tomesh2": "#bcbd22",
            "raytrace_hash_mesh2tomesh1": "#8c564b",
            "query": "#aec7e8",
            "execute hash query": "#7f7f7f",
            "gpu deduplication": "#98df8a",
            "download results": "#ffbb78",
        }

        modes_to_plot = [
            mode for mode in ["exact", "direct_estimation"]
            if mode in enabled
            if any((m is not None and m > 0) for m in results[mode]["mean"])
        ]
        if not modes_to_plot:
            return

        normalized_breakdowns = {mode: [] for mode in modes_to_plot}
        all_active_phases = set()
        for mode in modes_to_plot:
            for bd in results[mode]["breakdown"]:
                merged = {}
                for key, value in bd.items():
                    nk = normalize_phase_key(key)
                    merged[nk] = merged.get(nk, 0.0) + value
                normalized_breakdowns[mode].append(merged)
                all_active_phases.update(merged.keys())

        active_phases_ordered = [p for p in phase_order if p in all_active_phases]
        for p in sorted(all_active_phases):
            if p not in active_phases_ordered:
                active_phases_ordered.append(p)

        legend_handles = []
        legend_labels = []
        for phase in active_phases_ordered:
            label = phase_labels.get(phase, phase.replace("_", " ").title())
            color = phase_colors.get(phase, "#cccccc")
            patch = plt.Rectangle((0, 0), 1, 1, fc=color, ec='white')
            legend_handles.append(patch)
            legend_labels.append(label)

        num_modes = len(modes_to_plot)
        group_width = 0.8
        mode_width = group_width / num_modes if num_modes > 0 else group_width
        
        x_indices = np.arange(len(counts))

        for i, count_idx in enumerate(x_indices):
            for j, mode in enumerate(modes_to_plot):
                x_pos = i - group_width/2 + (j + 0.5) * mode_width
                
                # Get breakdown for this run
                breakdown = normalized_breakdowns[mode][i]
                mean_time = results[mode]["mean"][i]
                
                if not breakdown or mean_time == 0:
                    pass
                else:
                    bottom = 0
                    for phase in active_phases_ordered:
                        val = breakdown.get(phase, 0.0)
                        if val > 0:
                            ax.bar(x_pos, val, mode_width, bottom=bottom, 
                                             color=phase_colors.get(phase, "#cccccc"), edgecolor='white')
                            bottom += val

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Nuclei per Vessel')
        ax.set_ylabel('Query Time (ms)')
        ax.grid(False)
        for j, mode in enumerate(modes_to_plot):
            x_annot = 0 - group_width/2 + (j + 0.5) * mode_width
            ax.text(
                x_annot,
                -0.08,
                "Exact" if mode == "exact" else "Direct",
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=10,
                fontweight='bold',
                color="#444444",
            )
        
        # Legend
        if legend_handles:
            make_legend_bold(
                ax,
                legend_handles,
                legend_labels,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                fontsize=9,
                ncol=1,
                frameon=True,
            )

    # 1. Generate Combined Figure
    apply_paper_style()
    fig, (ax_main, ax_breakdown) = plt.subplots(1, 2, figsize=PAPER_FIGSIZE)
    generate_scaling_plot(ax_main, results, counts)
    generate_breakdown_plot(ax_breakdown, results, counts)
    plt.tight_layout()
    combined_path = figures_dir / "mesh_overlap_nn_scalability.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(combined_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Combined visualization saved to {combined_path}")
    plt.close(fig)

    # 2. Generate Separate Scaling Figure
    apply_paper_style()
    fig_scaling, ax_scaling = plt.subplots(figsize=PAPER_FIGSIZE)
    generate_scaling_plot(ax_scaling, results, counts)
    plt.tight_layout()
    scaling_path = figures_dir / "mesh_overlap_nn_scalability_scaling.png"
    plt.savefig(scaling_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(scaling_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Scaling visualization saved to {scaling_path}")
    plt.close(fig_scaling)

    # 3. Generate Separate Breakdown Figure
    apply_paper_style()
    fig_breakdown, ax_breakdown_sep = plt.subplots(figsize=PAPER_WIDE_FIGSIZE)
    generate_breakdown_plot(ax_breakdown_sep, results, counts)
    plt.tight_layout()
    breakdown_path = figures_dir / "mesh_overlap_nn_scalability_breakdown.png"
    plt.savefig(breakdown_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(breakdown_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Breakdown visualization saved to {breakdown_path}")
    plt.close(fig_breakdown)

def main():
    parser = argparse.ArgumentParser(description="Mesh Overlap NN Scalability Experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per method")
    parser.add_argument("--grid-cell-size", type=float, default=200.0, help="Grid resolution for RaySpace")
    parser.add_argument("--nu", type=int, nargs='+', help="Nu counts to test (e.g. 200 400 600 800)")
    parser.add_argument("--approaches", type=str, nargs='+', choices=["exact", "direct_estimation", "cgal", "touch", "tdbase"], help="Approaches to run")
    parser.add_argument("--track-hash-contention", action="store_true", help="Enable direct-estimation hash contention tracking")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout in seconds per run")
    parser.add_argument("--revisualize", type=str, help="Path to results.json to re-generate plots from")
    parser.add_argument(
        "--tdbase-timing-mode",
        type=str,
        default=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
        choices=TDBASE_TIMING_MODES,
        help="TDBase query-time definition. Default uses index+compute+evaluate; use compute_only to revert.",
    )
    args = parser.parse_args()
    
    if args.revisualize:
        print(f"Re-visualizing results from {args.revisualize}...")
        with open(args.revisualize, 'r') as f:
            data = json.load(f)
        results = data["results"]
        figures_dir = Path(args.revisualize).parent / "figures"
        plot_results(results, figures_dir)
        return

    nu_counts = args.nu if args.nu else DEFAULT_NU_COUNTS
    
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_nn_scalability")
    run_log_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])
    results = run_experiment(
        args.runs,
        args.grid_cell_size,
        nu_counts,
        run_log_dir,
        approaches=args.approaches,
        track_hash_contention=args.track_hash_contention,
        timeout=args.timeout,
        tdbase_timing_mode=args.tdbase_timing_mode,
    )
    
    if results and results["counts"]:
        print("\nResults Summary:")
        header = f"{'Nu':<10} {'Exact (ms)':<15} {'Selectivity Estimation (ms)':<15} {'Face (ms)':<15} {'TOUCH (ms)':<15} {'TDBase (ms)':<15}"
        print(header)
        print("-" * len(header))
        for i, n in enumerate(results["counts"]):
            ex = results['exact']['mean'][i]
            direct = results['direct_estimation']['mean'][i]
            cg = results['cgal']['mean'][i]
            to = results['touch']['mean'][i]
            td = results['tdbase']['mean'][i]
            
            ex_str = f"{ex:.2f}" if ex is not None else "N/A"
            direct_str = f"{direct:.2f}" if direct is not None else "N/A"
            cg_str = f"{cg:.2f}" if cg is not None else "N/A"
            to_str = f"{to:.2f}" if to is not None else "N/A"
            td_str = f"{td:.2f}" if td is not None else "N/A"
            
            print(f"{n:<10} {ex_str:<15} {direct_str:<15} {cg_str:<15} {to_str:<15} {td_str:<15}")
                
        plot_results(results, figures_dir)
        
        # Save summary to canonical run results path
        out_json = Path(run_layout["results_json"])
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean_results[k] = {ki: (vi.tolist() if isinstance(vi, np.ndarray) else vi) for ki, vi in v.items()}
            elif isinstance(v, list):
                clean_results[k] = v
            else:
                clean_results[k] = v
        write_json(
            out_json,
            {
                "metadata": {
                    "timestamp": run_layout["timestamp"],
                    "run_name": run_layout["run_name"],
                    "run_dir": str(run_layout["run_dir"]),
                    "runs": args.runs,
                    "grid_cell_size": args.grid_cell_size,
                    "nu_counts": nu_counts,
                    "timeout": args.timeout,
                    "tdbase_timing_mode": args.tdbase_timing_mode,
                },
                "results": clean_results,
            },
        )
        print(f"Raw results saved to {out_json}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
