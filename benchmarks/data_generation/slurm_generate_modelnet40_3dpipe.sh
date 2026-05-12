#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --mem=300G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --job-name=gen_modelnet40_3dpipe
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/gen_modelnet40_3dpipe_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/gen_modelnet40_3dpipe_%j.err

srun --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "
set -euo pipefail

BASE_DIR='/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation'
MODELNET_VERSION=\${MODELNET_VERSION:-1}
OUTPUT_DIR=\${OUTPUT_DIR:-\$BASE_DIR/datasets_scripts/modelnet_data/generated_3dpipe}

cd \"\$BASE_DIR\"

echo 'Building 3DPipe (simulator_modelnet)...'
./build_all.sh --only 3dpipe --jobs \"\${SLURM_CPUS_PER_TASK}\"

echo 'Generating two ModelNet40 train-based replicated datasets (~5GB target each)...'
MODELNET_VERSION=\"\$MODELNET_VERSION\" \
OUTPUT_DIR=\"\$OUTPUT_DIR\" \
TARGET_TRAIN_GB=\"\${TARGET_TRAIN_GB:-5}\" \
./datasets_scripts/generate_modelnet40_3dpipe.sh
"
