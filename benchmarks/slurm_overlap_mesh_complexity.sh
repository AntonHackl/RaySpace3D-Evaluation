#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=20
#SBATCH --job-name=overlap_mesh_complexity
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_overlap_mesh_complexity_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_overlap_mesh_complexity_%j.err

srun --cpu-bind=none --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "source /sc/home/anton.hackl/conda3/etc/profile.d/conda.sh && conda activate spatial_benchmark && export PYTHONPATH=\$PYTHONPATH:. && export CPU_THREADS=\${SLURM_CPUS_PER_TASK:-20} && export OMP_NUM_THREADS=\$CPU_THREADS && export OMP_DISPLAY_ENV=VERBOSE && echo \"[CPU Baselines] SLURM_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK:-unset} OMP_NUM_THREADS=\$OMP_NUM_THREADS TDBase_threads=\$CPU_THREADS TDBase_compute_threads=1\" && python benchmarks/mesh_overlap/run_mesh_complexity_benchmark.py --runs 3 --approaches direct_estimation cgal touch tdbase --timeout 300 --threads \$CPU_THREADS --tdbase-threads \$CPU_THREADS --tdbase-compute-threads 1"
