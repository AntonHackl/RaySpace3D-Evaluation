#!/bin/bash
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --mem=500G
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=10
#SBATCH --job-name=generate_large_nu_nn
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/generate_large_nu_nn_%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/slurm_logs/generate_large_nu_nn_%j.err

srun --cpu-bind=none --container-name RaySpace \
     --container-workdir /sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation \
     --container-mounts /sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     bash -c "
set -euo pipefail

BASE_DIR='/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation'
TDBASE_BUILD_DIR=\"\$BASE_DIR/baselines/RaySpace3DBaselines/tdbase_patch/build\"
OUTPUT_DIR=\"\$BASE_DIR/benchmarks/mesh_overlap/data/raw\"
THREADS=\$SLURM_CPUS_PER_TASK
NV=\${NV:-750}
PREFIX=\${PREFIX:-tdbase_large}
NU_VALUES=\${NU_VALUES:-\"200 400 600 800\"}
DATASET_KINDS=\${DATASET_KINDS:-\"n nn\"}

mkdir -p \"\$OUTPUT_DIR\"
cd \"\$TDBASE_BUILD_DIR\"

stage_complete() {
    local base=\"\$1\"
    local nuclei_file=\"\${base}_n_nv\${NV}_nu\${nu}_vs100_r30.dt\"
    local vessel_file=\"\${base}_v_nv\${NV}_nu\${nu}_vs100_r30.dt\"
    [[ -s \"\$nuclei_file\" && -s \"\$vessel_file\" ]]
}

run_stage() {
    local dataset_kind=\"\$1\"
    local base
    base=\"\$OUTPUT_DIR/\${PREFIX}_\${dataset_kind}_nv\${NV}_nu\${nu}\"

    if stage_complete \"\$base\"; then
        echo \"Skipping LARGE \${dataset_kind} dataset for nu=\$nu because outputs already exist.\"
        return
    fi

    echo \"Generating LARGE \${dataset_kind} dataset with prefix=\$PREFIX, nv=\$NV, nu=\$nu...\"
    ./tdbase simulator \
        -n \"\$BASE_DIR/baselines/RaySpace3DBaselines/tdbase/data/nuclei.pt\" \
        -v \"\$BASE_DIR/baselines/RaySpace3DBaselines/tdbase/data/vessel.pt\" \
        -o \"\$base\" \
        --hausdorff \
        --nv \$NV \
        --nu \$nu \
        -r 30 \
        -i \
        -t \$THREADS
}

for nu in \$NU_VALUES; do
    for dataset_kind in \$DATASET_KINDS; do
        run_stage \"\$dataset_kind\"
    done
done
"
