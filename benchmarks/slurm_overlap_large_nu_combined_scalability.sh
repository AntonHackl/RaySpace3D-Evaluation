#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --mem=128G
#SBATCH --time=14:00:00
#SBATCH --cpus-per-task=20
#SBATCH --job-name=overlap_large_nu_combined
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_overlap_large_nu_combined_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/slurm_overlap_large_nu_combined_%j.err

# Combined Large Nu Scalability Benchmarks (Vessel and Nuclei joins)
# These are run sequentially in a single job to avoid race conditions in the shared preprocessed directory.

srun --cpu-bind=none --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "source /sc/home/anton.hackl/conda3/etc/profile.d/conda.sh && conda activate spatial_benchmark && export PYTHONPATH=\$PYTHONPATH:. && \
     export CPU_THREADS=\${SLURM_CPUS_PER_TASK:-20} && export OMP_NUM_THREADS=\$CPU_THREADS && export OMP_DISPLAY_ENV=VERBOSE && \
     echo \"[CPU Baselines] SLURM_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK:-unset} OMP_NUM_THREADS=\$OMP_NUM_THREADS TDBase_threads=\$CPU_THREADS TDBase_compute_threads=1\" && \
     echo '--- Starting Large Nu Vessel Join Benchmark ---' && \
     python benchmarks/mesh_overlap/run_nu_scalability.py --dataset-profile large_nu_v --nu 200 400 600 800 --approaches direct_estimation cgal tdbase touch --timeout 1200 --threads \$CPU_THREADS --tdbase-threads \$CPU_THREADS --tdbase-compute-threads 1 && \
     echo '--- Starting Large Nu Nuclei Join Benchmark ---' && \
     python benchmarks/mesh_overlap/run_nu_scalability.py --dataset-profile large_nu_nn --nu 200 400 600 800 --approaches direct_estimation cgal tdbase touch --timeout 1200 --threads \$CPU_THREADS --tdbase-threads \$CPU_THREADS --tdbase-compute-threads 1"
