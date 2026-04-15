# Benchmarks Overview

This file gives a quick map of benchmark entrypoints in this repository and what each one is for.

## Mesh Overlap Benchmarks

- benchmarks/mesh_overlap/benchmark.py: Main overlap benchmark runner. Compares RaySpace overlap modes with CGAL, TOUCH, and TDBase on selected datasets.
- benchmarks/mesh_overlap/run_cube_scalability.py: Standard cube scalability benchmark. Measures how overlap runtime changes as cube dataset size increases.
- benchmarks/mesh_overlap/run_nu_scalability.py: Standard NU scalability benchmark. Measures overlap runtime as nuclei-per-vessel complexity grows.
- benchmarks/mesh_overlap/selectivity_test.py: Standard selectivity benchmark. Tests overlap behavior and runtime across different target selectivities.
- benchmarks/mesh_overlap/run_mesh_complexity_benchmark.py: Standard mesh complexity benchmark. Measures overlap performance as per-object geometric complexity increases.
- benchmarks/mesh_overlap/run_breakdown_benchmark.py: Phase breakdown benchmark. Splits overlap time into major internal phases (estimation, query, dedup, output).
- benchmarks/mesh_overlap/run_grid_resolution_sweep.py: Grid-resolution sensitivity benchmark for direct estimation overlap. Studies quality and runtime trade-offs across grid sizes.
- benchmarks/mesh_overlap/run_hash_contention_benchmark.py: Hash-table pressure benchmark for direct estimation overlap. Measures performance and contention under different hash capacities.
- benchmarks/mesh_overlap/run_direct_estimation_directionality_test.py: Directionality benchmark for direct estimation overlap. Compares both-direction vs one-way query strategies.
- benchmarks/mesh_overlap/run_nu_correctness_benchmark.py: NU correctness benchmark. Compares predicted overlap pairs against exact and TDBase references.

## Mesh Intersection Benchmarks

- benchmarks/mesh_intersection/benchmark.py: Main intersection benchmark runner (cube datasets). Compares RaySpace intersection modes with CGAL.
- benchmarks/mesh_intersection/run_cube_scalability.py: Standard cube scalability benchmark for intersection.
- benchmarks/mesh_intersection/run_nu_scalability.py: Standard NU scalability benchmark for intersection.
- benchmarks/mesh_intersection/selectivity_test.py: Standard selectivity benchmark for intersection.
- benchmarks/mesh_intersection/run_mesh_complexity_benchmark.py: Standard mesh complexity benchmark for intersection.
- benchmarks/mesh_intersection/compare_closesthit_vs_anyhit.py: RaySpace internal algorithm comparison for intersection. Compares closest-hit containment path vs any-hit containment path.

## Mesh Containment Benchmarks

- benchmarks/mesh_containment/benchmark.py: Main containment benchmark runner. Compares RaySpace containment with CGAL.
- benchmarks/mesh_containment/run_cube_scalability.py: Standard cube scalability benchmark for containment.
- benchmarks/mesh_containment/run_nu_scalability.py: Standard NU scalability benchmark for containment.
- benchmarks/mesh_containment/selectivity_test.py: Standard selectivity benchmark for containment.
- benchmarks/mesh_containment/run_mesh_complexity_benchmark.py: Standard mesh complexity benchmark for containment.

## Cross-Query Comparison Benchmark

- benchmarks/mesh_query_comparison/run_nu_scalability.py: Standard NU scalability benchmark that compares overlap, intersection, and containment in one run (default all three, configurable subset via flags).
- benchmarks/mesh_query_comparison/run_cube_scalability.py: Standard cube scalability benchmark for multi-query comparison.
- benchmarks/mesh_query_comparison/selectivity_test.py: Standard selectivity benchmark for multi-query comparison.
- benchmarks/mesh_query_comparison/run_mesh_complexity_benchmark.py: Standard mesh complexity benchmark for multi-query comparison.
- benchmarks/intersection_vs_overlap/run_benchmark.py: Deprecated wrapper that forwards nu/cube runs to mesh_query_comparison.

## Correctness and Disagreement Benchmarks

- benchmarks/correctness_tests/run_correctness.py: Multi-operation correctness suite for overlap, intersection, and containment against expected results.
- benchmarks/correctness_tests/run_intersection_disagreement_analysis.py: Pair-level disagreement analysis between RaySpace and CGAL for intersection, with sampled adjudication.
- benchmarks/correctness_tests/run_containment_disagreement_analysis.py: Pair-level disagreement analysis between RaySpace and CGAL for containment, with sampled adjudication.

## Point-in-Mesh (PIP) Benchmarks

- benchmarks/pip/grid_benchmark.py: Main PIP benchmark. Runs point-in-mesh queries over translated mesh positions on a 3D grid and compares approaches.
- benchmarks/pip/run_benchmark.sh: Wrapper script for the default PIP benchmark flow.
- benchmarks/pip/cubes_benchmark.sh: PIP benchmark variant focused on cube mesh workloads.
- benchmarks/pip/cubes_large_benchmark.sh: Large-scale cube PIP benchmark variant.
- benchmarks/pip/spheres_benchmark.sh: PIP benchmark variant focused on sphere mesh workloads.
- benchmarks/pip/spheres_large_benchmark.sh: Large-scale sphere PIP benchmark variant.
- benchmarks/pip/spheres_complexity_benchmark.sh: PIP complexity benchmark where sphere mesh detail/complexity changes.
- benchmarks/pip/spheres_selectivity_benchmark.sh: PIP selectivity benchmark where overlap/selectivity conditions are swept.
- benchmarks/pip/spheres_selectivity_benchmark_200M.sh: High-scale PIP selectivity run (200M configuration).
- benchmarks/pip/spheres_selectivity_benchmark_500M.sh: High-scale PIP selectivity run (500M configuration).

## Notes

- The standard mesh benchmark set used for comparability across overlap, intersection, and containment is: cube scalability, nu scalability, selectivity, mesh complexity.
- Most benchmark entrypoints write per-run artifacts under each benchmark folder in runs/<benchmark_name>_<timestamp>/ with results, logs, and figures.
