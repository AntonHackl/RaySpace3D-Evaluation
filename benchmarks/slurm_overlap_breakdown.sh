#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=50G
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=20
#SBATCH --job-name=overlap_bd
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_overlap_breakdown_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_overlap_breakdown_%j.err

srun --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "export PYTHONPATH=\$PYTHONPATH:. && python benchmarks/mesh_overlap/run_breakdown_benchmark.py --approaches direct_estimation"
