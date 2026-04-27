#!/bin/bash
#SBATCH --job-name=microns_intersection_est
#SBATCH --account=sci-zacharatou
#SBATCH --partition=gpu-batch
#SBATCH --gpus=rtx_pro_6000:1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
#SBATCH --cpus-per-task=20
#SBATCH --output=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/mesh_intersection/runs/slurm-%j.out
#SBATCH --error=/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation/benchmarks/mesh_intersection/runs/slurm-%j.err

set -e

PROJECT_ROOT="/sc/projects/sci-zacharatou/chair/RaySpace/RaySpace3D-Evaluation"

echo "Starting MICrONS Intersection Estimated Benchmark"
echo "Date: $(date)"

# Run everything inside the enroot container using srun.
# The container provides the necessary build dependencies (OptiX, nlohmann-json, etc.)
srun --container-image=anthac/rayspace:latest \
     --container-name=RaySpace \
     --container-mounts=/sc/home/anton.hackl/:/sc/home/anton.hackl/,/sc/projects/sci-zacharatou/chair/RaySpace/:/sc/projects/sci-zacharatou/chair/RaySpace/ \
     --container-workdir="$PROJECT_ROOT" \
     --container-writable \
     bash -c "
    set -e
    
    # Robust conda activation
    source /sc/home/anton.hackl/conda3/etc/profile.d/conda.sh
    conda activate spatial_benchmark
    
    echo \"Building RaySpace3D components (preprocess and query)...\"
    ./build_all.sh --only preprocess
    ./build_all.sh --only query
    
    echo \"Preflight check for MICrONS GLB subset folders...\"
    ROOT_DIR=\"$PROJECT_ROOT/datasets_scripts/microns_data\"
    for size in 4 8 16; do
        DIR=\"\$ROOT_DIR/microns_region_\${size}gb_glb\"
        if [ ! -d \"\$DIR\" ]; then
            echo \"Error: MICrONS dataset directory not found at \$DIR\"
            exit 1
        fi
        echo \"Found \$DIR\"
    done
    
    mkdir -p benchmarks/data_shared/microns_intersection_estimated/splits
    mkdir -p benchmarks/data_shared/microns_intersection_estimated/raw
    
    echo \"Running benchmark with explicit parameters...\"
    python benchmarks/mesh_intersection/run_microns_intersection_estimated.py \
        --sizes 4 8 16 \
        --runs 5 \
        --warmup-runs 1 \
        --timeout 36000.0 \
        --grid-cell-size 700.0
"

echo "Benchmark completed successfully"
echo "Date: $(date)"
