#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=20
#SBATCH --job-name=generate_nn
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/generate_nn_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/generate_nn_%j.err

srun --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "
BASE_DIR='/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation'
TDBASE_BUILD_DIR=\"\$BASE_DIR/baselines/RaySpace3DBaselines/tdbase_patch/build_agent\"
OUTPUT_DIR=\"\$BASE_DIR/benchmarks/mesh_overlap/data/raw\"
THREADS=\$SLURM_CPUS_PER_TASK
NV=\${NV:-750}
PREFIX=\${PREFIX:-tdbase_large}

mkdir -p \"\$OUTPUT_DIR\"
cd \"\$TDBASE_BUILD_DIR\"

for nu in 200 400 600 800; do
    echo \"Generating SECOND nuclei dataset for nv=\$NV, nu=\$nu...\"
    ./tdbase simulator \
        -n \"\$BASE_DIR/baselines/RaySpace3DBaselines/tdbase/data/nuclei.pt\" \
        -v \"\$BASE_DIR/baselines/RaySpace3DBaselines/tdbase/data/vessel.pt\" \
        -o \"\$OUTPUT_DIR/\${PREFIX}_nn_nv\${NV}_nu\${nu}\" \
        --hausdorff \
        --nv \$NV \
        --nu \$nu \
        -r 30 \
        -i \
        -t \$THREADS
done
"
