#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime
from adapters.comparison_adapter import ComparisonAdapter

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAYSPACE_DIR = REPO_ROOT / "src/RaySpace3D"
DATA_DIR = REPO_ROOT / "benchmarks/mesh_overlap/data/raw" # reuse existing data
PREPROCESSED_DIR = SCRIPT_DIR / "data/preprocessed"
TIMINGS_DIR = SCRIPT_DIR / "data/timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RESULTS_DIR = SCRIPT_DIR / "results"

NU_DATASETS = [
    ("tdbase_n_nv150_nu200_n_nv150_nu200_vs100_r30.dt", "tdbase_n_nv150_nu200_v_nv150_nu200_vs100_r30.dt", "nu200"),
    ("tdbase_n_nv150_nu400_n_nv150_nu400_vs100_r30.dt", "tdbase_n_nv150_nu400_v_nv150_nu400_vs100_r30.dt", "nu400"),
    ("tdbase_n_nv150_nu600_n_nv150_nu600_vs100_r30.dt", "tdbase_n_nv150_nu600_v_nv150_nu600_vs100_r30.dt", "nu600"),
    ("tdbase_n_nv150_nu800_n_nv150_nu800_vs100_r30.dt", "tdbase_n_nv150_nu800_v_nv150_nu800_vs100_r30.dt", "nu800"),
]

CUBE_DATASETS = [
    ("cubes_200k_a.obj", "cubes_200k_b.obj", "cube200k"),
    ("cubes_200k_a.obj", "cubes_400k_b.obj", "cube400k"),
    ("cubes_200k_a.obj", "cubes_600k_b.obj", "cube600k"),
    ("cubes_200k_a.obj", "cubes_1000k_b.obj", "cube1000k"),
]


def get_datasets(dataset_set):
    if dataset_set == "cube":
        return CUBE_DATASETS
    return NU_DATASETS

def _serialize_result(result):
    if isinstance(result, dict):
        return {k: _serialize_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_serialize_result(v) for v in result]
    if isinstance(result, (np.floating, np.integer)):
        return result.item()
    return result


def run_benchmark(
    runs,
    grid_res,
    dataset_set,
    intersection_query_direction,
    overlap_max_iterations,
    containment_max_iterations,
    hash_load_factor,
    enable_profiling_stats,
):
    print("--- Starting Intersection vs Overlap Benchmark ---")
    print(f"Dataset set: {dataset_set}")
    print(f"Intersection query direction: {intersection_query_direction}")
    print(f"Overlap max iterations: {overlap_max_iterations}")
    print(f"Containment max iterations: {containment_max_iterations}")
    print(f"Hash load factor: {hash_load_factor}")
    print(f"Profiling stats: {'enabled' if enable_profiling_stats else 'disabled'}")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_timings_dir = TIMINGS_DIR / run_id
    run_results_dir = RESULTS_DIR / run_id
    run_figures_dir = FIGURES_DIR / run_id
    run_timings_dir.mkdir(parents=True, exist_ok=True)
    run_results_dir.mkdir(parents=True, exist_ok=True)
    run_figures_dir.mkdir(parents=True, exist_ok=True)
    
    inter_adapter = ComparisonAdapter(
        str(RAYSPACE_DIR), 
        query_type="intersection", 
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(run_timings_dir),
        grid_resolution=grid_res,
        warmup_runs=2,
        intersection_extra_args=[
            "--query-direction", intersection_query_direction,
            "--overlap-max-iterations", str(overlap_max_iterations),
            "--containment-max-iterations", str(containment_max_iterations),
            "--hash-load-factor", str(hash_load_factor),
        ] + (["--enable-profiling-stats"] if enable_profiling_stats else [])
    )
    
    over_adapter = ComparisonAdapter(
        str(RAYSPACE_DIR), 
        query_type="overlap", 
        preprocessed_dir=str(PREPROCESSED_DIR),
        timings_dir=str(run_timings_dir),
        grid_resolution=grid_res,
        warmup_runs=2
    )

    datasets = []

    for f1_name, f2_name, label in get_datasets(dataset_set):
        f1_path = DATA_DIR / f1_name
        f2_path = DATA_DIR / f2_name
        
        if not f1_path.exists() or not f2_path.exists():
            print(f"Skipping {label}: file not found.")
            continue
            
        print(f"\nProcessing {label}...")
        
        # Preprocess
        for f_path in [f1_path, f2_path]:
            if not inter_adapter.check_preprocessed(str(f_path)):
                inter_adapter.preprocess_from_source(str(f_path), str(f_path))

        # Run Intersection
        print(f"Running Intersection Query...")
        try:
            res_inter = inter_adapter.run_query(str(f1_path), str(f2_path), runs, run_id=label)
        except subprocess.CalledProcessError as exc:
            print(f"Skipping {label}: intersection query failed with exit code {exc.returncode}.")
            continue
        
        # Run Overlap
        print(f"Running Overlap Query (Direct Est.)...")
        try:
            res_over = over_adapter.run_query(str(f1_path), str(f2_path), runs, run_id=label)
        except subprocess.CalledProcessError as exc:
            print(f"Skipping {label}: overlap query failed with exit code {exc.returncode}.")
            continue
        
        datasets.append({
            "label": label,
            "mesh1": str(f1_path),
            "mesh2": str(f2_path),
            "intersection": res_inter,
            "overlap": res_over
        })

    output = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(),
        "runs_per_dataset": runs,
        "grid_resolution": grid_res,
        "dataset_set": dataset_set,
        "intersection_query_direction": intersection_query_direction,
        "overlap_max_iterations": overlap_max_iterations,
        "containment_max_iterations": containment_max_iterations,
        "hash_load_factor": hash_load_factor,
        "enable_profiling_stats": enable_profiling_stats,
        "datasets": datasets,
    }

    latest_path = RESULTS_DIR / "comparison_results.json"
    run_path = run_results_dir / "comparison_results.json"

    with open(run_path, "w") as f:
        json.dump(_serialize_result(output), f, indent=4)

    with open(latest_path, "w") as f:
        json.dump(_serialize_result(output), f, indent=4)
        
    return output, run_figures_dir, run_results_dir

def plot_line_graph(result_obj, output_dir):
    datasets = result_obj["datasets"]
    labels = [r["label"] for r in datasets]
    inter_times = [r["intersection"]["avg_time_ms"] for r in datasets]
    over_times = [r["overlap"]["avg_time_ms"] for r in datasets]
    
    plt.figure(figsize=(10, 6))
    plt.plot(labels, inter_times, marker='o', label='Intersection')
    plt.plot(labels, over_times, marker='s', label='Overlap (Direct Est.)')
    plt.xlabel('Dataset')
    plt.ylabel('Avg Time (ms)')
    plt.title('Intersection vs Overlap Query Performance')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "performance_line_graph.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Line graph saved to {out}")

def plot_breakdown_chart_for_dataset(data, output_dir):
    label = data["label"]

    methods = ["Intersection", "Overlap"]
    
    # Define phases for alignment
    # Phase groups:
    intersection_breakdown = data["intersection"].get("breakdown_avg_ms", {})
    overlap_breakdown = data["overlap"].get("breakdown_avg_ms", {})
    
    # Intersection phases:
    # Raytrace_Overlap_Mesh1ToMesh2_Pass1/2
    # Raytrace_Containment_Mesh1ToMesh2_Pass1/2
    # GPU deduplication_ (or Deduplication_)
    
    # Let's simplify and aggregate for the chart
    
    def get_inter_groups(br):
        estimation = br.get("selectivity_estimation_ms", 0.0)
        overlap_m1_m2 = br.get("raytrace_overlap_hash_mesh1_to_mesh2_ms", 0.0)
        overlap_m2_m1 = br.get("raytrace_overlap_hash_mesh2_to_mesh1_ms", 0.0)
        containment_m1_m2 = br.get("raytrace_containment_hash_mesh1_to_mesh2_ms", 0.0)
        containment_m2_m1 = br.get("raytrace_containment_hash_mesh2_to_mesh1_ms", 0.0)

        # Backward compatibility with legacy output schema.
        if overlap_m1_m2 == 0.0 and overlap_m2_m1 == 0.0:
            overlap_m1_m2 = br.get("raytrace_hash_mesh1_to_mesh2_ms", 0.0)
            overlap_m2_m1 = br.get("raytrace_hash_mesh2_to_mesh1_ms", 0.0)

        dedup = br.get("deduplication_ms", 0.0)
        return {
            "Selectivity Est.": estimation,
            "Overlap Raytrace M1->M2": overlap_m1_m2,
            "Overlap Raytrace M2->M1": overlap_m2_m1,
            "Containment Raytrace M1->M2": containment_m1_m2,
            "Containment Raytrace M2->M1": containment_m2_m1,
            "Deduplication": dedup,
        }
        
    def get_over_groups(br):
        estimation = br.get("selectivity_estimation_ms", 0.0)
        raytrace_m1_m2 = br.get("raytrace_hash_mesh1_to_mesh2_ms", 0.0)
        raytrace_m2_m1 = br.get("raytrace_hash_mesh2_to_mesh1_ms", 0.0)
        dedup = br.get("deduplication_ms", 0.0)
        return {
            "Selectivity Est.": estimation,
            "Raytrace Hash M1->M2": raytrace_m1_m2,
            "Raytrace Hash M2->M1": raytrace_m2_m1,
            "Deduplication": dedup,
        }

    inter_agg = get_inter_groups(intersection_breakdown)
    over_agg = get_over_groups(overlap_breakdown)
    
    # Union of all category names
    categories = sorted(list(set(inter_agg.keys()) | set(over_agg.keys())))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.5
    indices = np.arange(len(methods))
    bottoms = np.zeros(len(methods))
    
    for cat in categories:
        vals = [inter_agg.get(cat, 0), over_agg.get(cat, 0)]
        ax.bar(indices, vals, bar_width, bottom=bottoms, label=cat)
        bottoms += np.array(vals)
        
    ax.set_xticks(indices)
    ax.set_xticklabels(methods)
    ax.set_ylabel('Time (ms)')
    ax.set_title(f'Breakdown: Intersection (Estimated) vs Overlap (Direct) [{label}]')
    ax.legend(loc='upper right')

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"breakdown_{label}.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Breakdown chart saved to {out}")


def plot_breakdown_charts(result_obj, output_dir):
    datasets = result_obj["datasets"]
    if not datasets:
        return
    for data in datasets:
        plot_breakdown_chart_for_dataset(data, output_dir)


def write_readme_summary(result_obj, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "README.txt"
    lines = []
    lines.append(f"Run ID: {result_obj['run_id']}")
    lines.append(f"Created At: {result_obj['created_at']}")
    lines.append(f"Runs per Dataset: {result_obj['runs_per_dataset']}")
    lines.append(f"Dataset Set: {result_obj.get('dataset_set', 'nu')}")
    lines.append(f"Intersection Query Direction: {result_obj.get('intersection_query_direction', 'both')}")
    lines.append(f"Overlap Max Iterations: {result_obj.get('overlap_max_iterations', 100)}")
    lines.append(f"Containment Max Iterations: {result_obj.get('containment_max_iterations', 2048)}")
    lines.append(f"Hash Load Factor: {result_obj.get('hash_load_factor', 0.5)}")
    lines.append(f"Profiling Stats: {result_obj.get('enable_profiling_stats', False)}")
    lines.append("")
    lines.append("Notes:")
    lines.append("- Intersection uses raytracer_intersection_estimated (hash-based estimated pipeline).")
    lines.append("- Overlap uses raytracer_overlap_direct_estimation.")
    lines.append("- Estimated intersection runs overlap and containment raytracing (both directions), then deduplicates pairs.")
    lines.append("")
    lines.append("Datasets:")
    for d in result_obj["datasets"]:
        lines.append(
            f"- {d['label']}: intersection_avg_ms={d['intersection']['avg_time_ms']:.4f}, overlap_avg_ms={d['overlap']['avg_time_ms']:.4f}"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Summary written to {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--grid-res", type=int, default=10)
    parser.add_argument("--dataset-set", choices=["nu", "cube"], default="nu")
    parser.add_argument("--intersection-query-direction", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"], default="both")
    parser.add_argument("--overlap-max-iterations", type=int, default=100)
    parser.add_argument("--containment-max-iterations", type=int, default=512)
    parser.add_argument("--hash-load-factor", type=float, default=0.5)
    parser.add_argument("--enable-profiling-stats", action="store_true")
    args = parser.parse_args()
    
    result_obj, run_figures_dir, run_results_dir = run_benchmark(
        args.runs,
        args.grid_res,
        args.dataset_set,
        args.intersection_query_direction,

        args.overlap_max_iterations,
        args.containment_max_iterations,
        args.hash_load_factor,
        args.enable_profiling_stats,
    )
    if result_obj["datasets"]:
        plot_line_graph(result_obj, run_figures_dir)
        plot_breakdown_charts(result_obj, run_figures_dir)
        write_readme_summary(result_obj, run_results_dir)

        # Keep latest figures at root for quick inspection
        plot_line_graph(result_obj, FIGURES_DIR)
        plot_breakdown_charts(result_obj, FIGURES_DIR)

if __name__ == "__main__":
    main()
