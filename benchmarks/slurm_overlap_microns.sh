#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=20
#SBATCH --job-name=overlap_microns
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_overlap_microns_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_overlap_microns_%j.err

srun --cpu-bind=none --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "source /sc/home/anton.hackl/conda3/etc/profile.d/conda.sh && conda activate spatial_benchmark && export PYTHONPATH=\$PYTHONPATH:. && export CGAL_THREADS=\${SLURM_CPUS_PER_TASK:-20} && export OMP_NUM_THREADS=\$CGAL_THREADS && export OMP_DISPLAY_ENV=VERBOSE && echo \"[OpenMP] SLURM_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK:-unset} OMP_NUM_THREADS=\$OMP_NUM_THREADS\" && python benchmarks/mesh_overlap/run_microns_overlap.py --timeout 1200 --sizes 4 --approaches direct_estimation cgal touch --threads \$CGAL_THREADS"
