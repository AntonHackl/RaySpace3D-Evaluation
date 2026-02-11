#!/usr/bin/env python3
"""3D Spatial Query Grid Benchmark.

Compares CGAL, SQL (PostgreSQL/PostGIS), RaySpace3D raytracer, and raytracer_filter_refine
approaches using a 3x3x3 grid translation pattern.

Example:
    python grid_benchmark.py \\
        --query-obj ../datasets/range_query/ranges/HighRes_sphere.obj \\
        --points ../datasets/range_query/points/uniform_points_10000000.wkt \\
        --approaches cgal,sql,raytracer,raytracer_filter_refine \\
        --output results/benchmark.json
"""

import argparse
import json
import os
import re
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import adapters from subdirectory
from adapters import (
    CGALAdapter,
    SQLAdapter,
    RaytracerAdapter,
    FilterRefineAdapter,
    CUDAAdapter
)
from adapters.utils import run_subprocess_streaming, compute_obj_bbox


# ============================================================================
# Grid Position Generation
# ============================================================================

def compute_bbox_from_wkt(wkt_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box from WKT points file."""
    points = []
    with open(wkt_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('POINT'):
                continue
            # Parse "POINT Z (x y z)" or "POINT(x y z)"
            match = re.search(r'\(([^)]+)\)', line)
            if match:
                coords = match.group(1).split()
                if len(coords) >= 3:
                    points.append([float(coords[0]), float(coords[1]), float(coords[2])])
    
    if not points:
        raise ValueError(f"No valid points found in {wkt_file}")
    
    points_array = np.array(points)
    return points_array.min(axis=0), points_array.max(axis=0)


def generate_grid_positions(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    mesh_center: np.ndarray,
    grid_size: Tuple[int, int, int] = (3, 3, 3)
) -> List[Tuple[int, int, int, np.ndarray]]:
    """Generate grid positions for geometry translation.
    
    Args:
        bbox_min: Minimum corner of point cloud bounding box
        bbox_max: Maximum corner of point cloud bounding box
        mesh_center: Center of the mesh to be positioned
        grid_size: Grid dimensions (nx, ny, nz)
    
    Returns:
        List of (ix, iy, iz, translation_vector) tuples where translation_vector
        will position the mesh center at the grid cell center
    """
    positions = []
    bbox_range = bbox_max - bbox_min
    cell_size = bbox_range / np.array(grid_size)
    
    for ix in range(grid_size[0]):
        for iy in range(grid_size[1]):
            for iz in range(grid_size[2]):
                # Center of grid cell (where we want the mesh center to be)
                grid_center = bbox_min + (np.array([ix, iy, iz]) + 0.5) * cell_size
                # Translation needed to move mesh center to grid center
                translation = grid_center - mesh_center
                positions.append((ix, iy, iz, translation))
    
    return positions


def generate_centered_positions(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    mesh_center: np.ndarray
) -> List[Tuple[int, int, int, np.ndarray]]:
    """Generate centered positions (same position twice) for geometry translation.
    
    Returns the center of the bounding box twice to run the experiment twice
    and average the results, similar to how grid results are averaged.
    
    Args:
        bbox_min: Minimum corner of point cloud bounding box
        bbox_max: Maximum corner of point cloud bounding box
        mesh_center: Center of the mesh to be positioned
    
    Returns:
        List of (run_id, 0, 0, translation_vector) tuples (2 entries) where
        translation_vector will position the mesh center at the bbox center
    """
    bbox_center = (bbox_min + bbox_max) / 2.0
    # Translation needed to move mesh center to bbox center
    translation = bbox_center - mesh_center
    # Return two identical positions for two runs
    return [
        (0, 0, 0, translation),
        (1, 0, 0, translation)
    ]


# ============================================================================
# Visualization
# ============================================================================

def plot_results(results: Dict[str, List[Dict]], output_dir: str, metric: str = 'total_query_ms', 
                 benchmark_name: Optional[str] = None, grid_size: Tuple[int, int, int] = (3, 3, 3),
                 centered: bool = False):
    """Generate visualizations of benchmark results.
    
    Args:
        results: Dictionary of results by approach
        output_dir: Directory to save visualizations
        metric: Metric to visualize
        benchmark_name: Optional benchmark name
        grid_size: Grid size tuple
        centered: If True, skip grid visualizations (only 2 runs at center)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip visualizations for centered mode (only 2 identical runs)
    if centered:
        print(f"\nSkipping visualizations for centered mode (no grid to visualize)")
        return
    
    # Extract data for each approach
    approaches = list(results.keys())
    
    # Build title prefix with benchmark name if provided
    title_prefix = ""
    if benchmark_name:
        title_prefix = f"{benchmark_name} - "
    
    grid_str = f"{grid_size[0]}x{grid_size[1]}x{grid_size[2]}"
    
    # 3D scatter plot of query times
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {'CGAL': 'blue', 'SQL': 'green', 'Raytracer': 'red', 'FilterRefine': 'orange'}
    markers = {'CGAL': 'o', 'SQL': 's', 'Raytracer': '^', 'FilterRefine': 'd'}
    
    # Collect all query times to determine log scale range
    all_query_times = []
    for approach in approaches:
        for result in results[approach]:
            val = result.get(metric) if result.get(metric) is not None else result.get('total_query_ms')
            if result.get('success') and val is not None and val > 0:
                all_query_times.append(val)
    
    # Set up logarithmic normalization if we have positive values
    from matplotlib.colors import LogNorm
    use_log_norm = len(all_query_times) > 0 and min(all_query_times) > 0
    if use_log_norm:
        norm = LogNorm(vmin=min(all_query_times), vmax=max(all_query_times))
    else:
        norm = None
    
    scatter_plots = []
    for approach in approaches:
        positions = []
        query_times = []

        for result in results[approach]:
            # Use selected metric if available, otherwise fall back to 'total_query_ms'
            val = result.get(metric) if result.get(metric) is not None else result.get('total_query_ms')
            if result.get('success') and val is not None:
                trans = result['translation']
                positions.append(trans)
                query_times.append(val)
        
        if positions:
            positions = np.array(positions)
            query_times = np.array(query_times)
            
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=query_times,
                s=100,
                alpha=0.7,
                marker=markers.get(approach, 'o'),
                label=approach,
                cmap='viridis',
                norm=norm
            )
            scatter_plots.append(scatter)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'{title_prefix}Query Performance Across {grid_str} Grid (Log Scale)')
    ax.legend()
    
    # Add colorbar if we have scatter plots
    if scatter_plots:
        cbar = plt.colorbar(scatter_plots[0], ax=ax, pad=0.1)
        cbar.set_label('Query Time (ms)', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'query_times_3d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar chart comparing approaches
    fig, ax = plt.subplots(figsize=(12, 6))
    
    approach_stats = {}
    for approach in approaches:
        times = []
        for r in results[approach]:
            if not r.get('success'):
                continue
            val = r.get(metric) if r.get(metric) is not None else r.get('total_query_ms')
            if val is not None:
                times.append(val)
        if times:
            approach_stats[approach] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times)
            }
    
    x = np.arange(len(approach_stats))
    width = 0.6
    
    means = [approach_stats[a]['mean'] for a in approach_stats]
    stds = [approach_stats[a]['std'] for a in approach_stats]
    
    bars = ax.bar(x, means, width, yerr=stds, capsize=5, alpha=0.7)
    
    ax.set_xlabel('Approach')
    ax.set_ylabel('Query Time (ms)')
    ax.set_title(f'{title_prefix}Average Query Time by Approach (Log Scale)')
    ax.set_xticks(x)
    ax.set_xticklabels(approach_stats.keys())
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'approach_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualizations saved to {output_dir}/")


# ============================================================================
# Main Benchmark
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3D Spatial Query Grid Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--query-obj', required=True,
                        help='Path to query OBJ file')
    parser.add_argument('--points', required=True,
                        help='Path to WKT points file')
    parser.add_argument('--approaches', default='cgal,sql,raytracer,raytracer_filter_refine,cuda',
                        help='Comma-separated list of approaches to test (cgal,sql,raytracer,raytracer_filter_refine,cuda)')
    parser.add_argument('--output', default='results/grid_benchmark.json',
                        help='Output JSON file')
    parser.add_argument('--workspace', default='workspace',
                        help='Workspace directory for intermediate files')
    parser.add_argument('--grid-size', nargs=3, type=int, default=None,
                        help='Grid size (e.g., 3 3 3). Mutually exclusive with --centered')
    parser.add_argument('--centered', action='store_true',
                        help='Run experiment twice at bbox center instead of grid. Mutually exclusive with --grid-size')
    parser.add_argument('--cgal-dir', 
                        default='../../baselines/RaySpace3DBaselines/CGAL',
                        help='CGAL baseline directory')
    parser.add_argument('--sql-dir',
                        default='../../baselines/RaySpace3DBaselines/SQL',
                        help='SQL baseline directory')
    parser.add_argument('--rayspace-dir',
                        default='../../src/RaySpace3D',
                        help='RaySpace3D directory')
    parser.add_argument('--cuda-dir',
                        default='../../baselines/RaySpace3DBaselines/CUDA',
                        help='CUDA baseline directory')
    parser.add_argument('--evaluate', action='store_true',
                        help='Run evaluation script after benchmark completes')
    parser.add_argument('--eval-metric', default='total_query_ms',
                        choices=['query_ms', 'total_query_ms', 'rayspace_total_ms'],
                        help='Timing metric for evaluation (default: total_query_ms)')
    parser.add_argument('--name', type=str, default=None,
                        help='Optional benchmark name to include in output filename')
    
    args = parser.parse_args()
    
    # Validate grid-size and centered are mutually exclusive
    if args.grid_size is not None and args.centered:
        parser.error("--grid-size and --centered are mutually exclusive. Please specify only one.")
    if args.grid_size is None and not args.centered:
        parser.error("Either --grid-size or --centered must be specified.")
    
    # Set default grid_size if not using centered mode
    if args.grid_size is None:
        args.grid_size = [3, 3, 3]  # Default for display purposes only
    
    def start_postgres(sql_dir: str) -> bool:
        """Start a local PostgreSQL server for the SQL baseline.

        Uses a pgdata directory under the SQL base dir and pg_ctl to start the server if it's not running.
        Returns True if the server is running (either already or after start), False on failure.
        """
        try:
            cmd = f'''
if ! pg_isready -q; then
  if [ ! -d "{sql_dir}/pgdata" ]; then
    initdb -D "{sql_dir}/pgdata"
  fi
  pg_ctl -D "{sql_dir}/pgdata" -l "{sql_dir}/postgres.log" start
  sleep 2
fi
'''
            subprocess.run(["bash", "-c", cmd], check=True, capture_output=True, text=True, timeout=120)
            # verify
            res = subprocess.run(["bash", "-c", "pg_isready -q"], check=False)
            return res.returncode == 0
        except Exception as e:
            print(f"[SQL] Failed to start PostgreSQL: {e}")
            return False

    def stop_postgres(sql_dir: str) -> None:
        """Stop the local PostgreSQL server started in `sql_dir` pgdata directory."""
        try:
            cmd = f'pg_ctl -D "{sql_dir}/pgdata" -m fast stop'
            subprocess.run(["bash", "-c", cmd], check=False, capture_output=True, text=True, timeout=30)
        except Exception:
            pass
    def reset_sql_database(sql_dir: str) -> bool:
        """Destroy and re-initialize the SQL baseline database non-interactively."""
        try:
            script_path = os.path.join(sql_dir, "scripts", "destroy_and_init.sh")
            if not os.path.exists(script_path):
                print(f"[SQL] destroy_and_init.sh not found at {script_path}")
                return False
            cmd = f'bash "{script_path}" --yes'
            run_subprocess_streaming(
                ["bash", "-c", cmd],
                prefix="[SQL]",
                check=True,
                timeout=600,
            )
            print("[SQL] Database destroyed and re-initialized")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[SQL] destroy_and_init.sh failed: {e.stderr}")
            return False
        except Exception as e:
            print(f"[SQL] Error while resetting SQL database: {e}")
            return False

    # Parse approaches
    approaches_list = [a.strip() for a in args.approaches.split(',')]
    
    print("="*70)
    print("3D SPATIAL QUERY GRID BENCHMARK")
    print("="*70)
    print(f"Query OBJ:      {args.query_obj}")
    print(f"Points:        {args.points}")
    print(f"Approaches:    {', '.join(approaches_list)}")
    if args.centered:
        print(f"Mode:          Centered (2 runs at bbox center)")
    else:
        print(f"Grid size:     {args.grid_size[0]}x{args.grid_size[1]}x{args.grid_size[2]}")
    print(f"Output:        {args.output}")
    print("="*70)
    
    # Compute bounding box from points
    print("\nComputing point cloud bounding box...")
    bbox_min, bbox_max = compute_bbox_from_wkt(args.points)
    print(f"  Min: {bbox_min}")
    print(f"  Max: {bbox_max}")
    
    # Compute mesh bounding box and center
    print("\nComputing mesh bounding box...")
    mesh_bbox_min, mesh_bbox_max = compute_obj_bbox(args.query_obj)
    mesh_center = (mesh_bbox_min + mesh_bbox_max) / 2.0
    print(f"  Mesh Min: {mesh_bbox_min}")
    print(f"  Mesh Max: {mesh_bbox_max}")
    print(f"  Mesh Center: {mesh_center}")
    
    # Generate positions (either grid or centered)
    if args.centered:
        print(f"\nGenerating centered positions (2 runs at bbox center)...")
        grid_positions = generate_centered_positions(bbox_min, bbox_max, mesh_center)
        print(f"  Total runs: {len(grid_positions)}")
        print(f"  Target position (bbox center): {(bbox_min + bbox_max) / 2.0}")
    else:
        print(f"\nGenerating {args.grid_size[0]}x{args.grid_size[1]}x{args.grid_size[2]} grid positions...")
        grid_positions = generate_grid_positions(bbox_min, bbox_max, mesh_center, tuple(args.grid_size))
        print(f"  Total positions: {len(grid_positions)}")
    
    # Create workspace
    os.makedirs(args.workspace, exist_ok=True)
    
    # Initialize adapters
    adapters = {}
    
    if 'cgal' in approaches_list:
        adapters['CGAL'] = CGALAdapter(
            os.path.join(args.workspace, 'cgal'),
            args.cgal_dir
        )
    
    if 'sql' in approaches_list:
        adapters['SQL'] = SQLAdapter(
            os.path.join(args.workspace, 'sql'),
            args.sql_dir
        )
    
    if 'raytracer' in approaches_list:
        adapters['Raytracer'] = RaytracerAdapter(
            os.path.join(args.workspace, 'raytracer'),
            args.rayspace_dir
        )
    
    if 'raytracer_filter_refine' in approaches_list:
        adapters['FilterRefine'] = FilterRefineAdapter(
            os.path.join(args.workspace, 'filter_refine'),
            args.rayspace_dir
        )
    
    if 'cuda' in approaches_list:
        adapters['CUDA'] = CUDAAdapter(
            os.path.join(args.workspace, 'cuda'),
            args.cuda_dir
        )
    
    # If SQL adapter present, reset DB and start a local postgres server before setup
    postgres_started = False
    if 'SQL' in adapters:
        print('\n[SQL] Resetting database (destroy + init)...')
        if not reset_sql_database(args.sql_dir):
            print('[SQL] Error: failed to destroy/init database; SQL adapter setup may fail')
        print('\n[SQL] Ensuring PostgreSQL server is running...')
        postgres_started = start_postgres(args.sql_dir)
        if not postgres_started:
            print('[SQL] Warning: could not start PostgreSQL server; SQL adapter setup may fail')

    # Setup all adapters
    print("\n" + "="*70)
    print("SETUP PHASE")
    print("="*70)
    
    for name, adapter in list(adapters.items()):
        print(f"\n[{name}] Setting up...")
        if not adapter.setup(points_path=args.points):
            print(f"[{name}] Setup failed!")
            del adapters[name]
        else:
            print(f"[{name}] Setup complete")
    
    if not adapters:
        print("\nNo adapters successfully initialized. Exiting.")
        return 1
    
    # Run benchmark
    print("\n" + "="*70)
    print("BENCHMARK PHASE")
    print("="*70)
    
    all_results = {name: [] for name in adapters.keys()}
    
    for idx, (gx, gy, gz, translation) in enumerate(grid_positions, 1):
        if args.centered:
            print(f"\n--- Run {idx}/{len(grid_positions)} (Centered) ---")
        else:
            print(f"\n--- Grid Position {idx}/{len(grid_positions)}: ({gx}, {gy}, {gz}) ---")
        print(f"    Translation: {translation}")
        
        for name, adapter in adapters.items():
            print(f"  [{name}] Running query...")
            start_time = time.time()
            
            result = adapter.execute_query(
                args.query_obj,
                args.points,
                (gx, gy, gz),
                translation
            )
            
            elapsed = time.time() - start_time
            
            result['grid_position'] = [gx, gy, gz]
            result['translation'] = translation.tolist()
            result['wall_time_s'] = elapsed
            
            # Check that total_query_ms is set by the adapter
            if result.get('success'):
                if 'total_query_ms' not in result or result.get('total_query_ms') is None:
                    print(f"  [{name}] ERROR: total_query_ms not set by adapter!")
                else:
                    print(f"  [{name}] Success! Query time: {result.get('total_query_ms', 'N/A')} ms")
            else:
                print(f"  [{name}] Failed: {result.get('error', 'Unknown error')}")
            
            all_results[name].append(result)
    
    # Cleanup
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)
    
    for name, adapter in adapters.items():
        print(f"[{name}] Cleaning up...")
        adapter.cleanup()

    # Stop postgres if we started it here
    if postgres_started:
        print('\n[SQL] Stopping PostgreSQL server started by benchmark...')
        stop_postgres(args.sql_dir)
    
    # Save results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    # Add timestamp to output filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(args.output)
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    output_dir = output_path.parent
    
    if args.name:
        timestamped_output = output_dir / f"{output_stem}_{args.name}_{timestamp}{output_suffix}"
    else:
        timestamped_output = output_dir / f"{output_stem}_{timestamp}{output_suffix}"
    
    output_data = {
        'timestamp': timestamp,
        'configuration': {
            'query_obj': args.query_obj,
            'points_file': args.points,
            'mode': 'centered' if args.centered else 'grid',
            'grid_size': None if args.centered else args.grid_size,
            'approaches': list(adapters.keys()),
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist()
        },
        'results': all_results
    }
    
    # Compute statistics
    for name in adapters.keys():
        times = [r['total_query_ms'] for r in all_results[name] 
                 if r.get('success') and r.get('total_query_ms') is not None]
        
        if times:
            stats = {
                'mean': float(np.mean(times)),
                'std': float(np.std(times)),
                'min': float(np.min(times)),
                'max': float(np.max(times)),
                'count': len(times)
            }
            
            print(f"\n{name} Statistics:")
            print(f"  Successful queries: {stats['count']}/{len(all_results[name])}")
            print(f"  Mean query time:    {stats['mean']:.2f} ms")
            print(f"  Std deviation:      {stats['std']:.2f} ms")
            print(f"  Min:                {stats['min']:.2f} ms")
            print(f"  Max:                {stats['max']:.2f} ms")
            
            output_data[f'{name}_statistics'] = stats
    
    # Save JSON
    os.makedirs(output_dir or '.', exist_ok=True)
    with open(timestamped_output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {timestamped_output}")
    
    # Generate visualizations
    viz_dir = output_dir / 'visualizations' / timestamp
    plot_results(all_results, str(viz_dir), metric=args.eval_metric, 
                 benchmark_name=args.name, grid_size=tuple(args.grid_size),
                 centered=args.centered)
    
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE!")
    print("="*70)
    
    # Run evaluation if requested
    if args.evaluate:
        print("\n" + "="*70)
        print("RUNNING EVALUATION")
        print("="*70)
        
        eval_script = Path(__file__).parent / 'evaluation' / 'evaluate_benchmarks.py'
        if eval_script.exists():
            eval_output = output_dir / 'evaluation_figures' / timestamp
            cmd = [
                sys.executable,
                str(eval_script),
                '--results-dir', str(output_dir),
                '--output-dir', str(eval_output),
                '--metric', args.eval_metric
            ]
            
            try:
                subprocess.run(cmd, check=True)
                print(f"\nEvaluation complete! Figures saved to: {eval_output}/")
            except subprocess.CalledProcessError as e:
                print(f"Warning: Evaluation failed: {e}")
        else:
            print(f"Warning: Evaluation script not found at {eval_script}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
