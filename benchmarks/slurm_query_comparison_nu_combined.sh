#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=5
#SBATCH --job-name=query_cmp_nu_v_combined
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_query_comparison_nu_v_combined_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_query_comparison_nu_v_combined_%j.err

# Combined query comparison NU benchmarks (vessel and nuclei joins).
# This remains a convenience wrapper for sequential execution.

srun --cpu-bind=none --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "source /sc/home/anton.hackl/conda3/etc/profile.d/conda.sh && conda activate spatial_benchmark && export PYTHONPATH=\$PYTHONPATH:. && \
     echo '--- Starting Query Comparison NU Benchmark (large_nu_v) ---' && \
     python benchmarks/mesh_query_comparison/run_nu_scalability.py --timeout 1200 && \
     echo '--- Starting Query Comparison NU Benchmark (large_nu_nn) ---' && \
     python benchmarks/mesh_query_comparison/run_nu_scalability.py --dataset-profile large_nu_nn --nu 800 --timeout 1200"
