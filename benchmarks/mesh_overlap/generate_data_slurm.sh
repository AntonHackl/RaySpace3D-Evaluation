#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --mem=256G
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --job-name=generate_data
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=20
#SBATCH --container-writable
#SBATCH --output=/sc/home/anton.hackl/Spatial_Data_Management/RaySpace3D-Evaluation/benchmarks/mesh_overlap/logs/generate_data_%j.out
#SBATCH --error=/sc/home/anton.hackl/Spatial_Data_Management/RaySpace3D-Evaluation/benchmarks/mesh_overlap/logs/generate_data_%j.err

# Ensure SLURM variables are propagated
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "Submitting job for data generation..."
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"

srun \
    --container-name rayspace \
    --container-workdir /sc/home/anton.hackl/Spatial_Data_Management/RaySpace3D-Evaluation/benchmarks/mesh_overlap \
    --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/ \
    bash -c 'bash ../../build_all.sh --only tdbase --clean --jobs $SLURM_CPUS_PER_TASK && bash ../data_generation/generate_data.sh'
