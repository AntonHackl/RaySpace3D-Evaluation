#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=20
#SBATCH --job-name=intersect_mesh_complexity
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_intersection_mesh_complexity_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_intersection_mesh_complexity_%j.err

srun --cpu-bind=none --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "source /sc/home/anton.hackl/conda3/etc/profile.d/conda.sh && conda activate spatial_benchmark && export PYTHONPATH=\$PYTHONPATH:. && python benchmarks/mesh_intersection/run_mesh_complexity_benchmark.py --runs 3 --timeout 1200"
