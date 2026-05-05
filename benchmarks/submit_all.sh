#!/bin/bash

# Submit all top-level benchmark SLURM jobs
echo "Submitting all benchmarks..."

sbatch benchmarks/slurm_intersection_breakdown.sh
sbatch benchmarks/slurm_intersection_mesh_complexity.sh
sbatch benchmarks/slurm_overlap_breakdown.sh
sbatch benchmarks/slurm_overlap_cube_scalability.sh
sbatch benchmarks/slurm_overlap_mesh_complexity.sh
sbatch benchmarks/slurm_overlap_microns.sh
sbatch benchmarks/slurm_overlap_large_nu_nn_scalability.sh
sbatch benchmarks/slurm_overlap_nn_scalability.sh
sbatch benchmarks/slurm_overlap_nu_scalability.sh
sbatch benchmarks/slurm_overlap_selectivity.sh
sbatch benchmarks/slurm_query_comparison_microns.sh
sbatch benchmarks/slurm_query_comparison_nu.sh

echo "All benchmarks submitted."
