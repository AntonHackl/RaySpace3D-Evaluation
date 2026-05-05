#!/bin/bash
# Initialize conda for bash
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_DEFAULT_ENV" != "tdbase_env" ]]; then
    # Try to find conda and source it (fallback logic from run_benchmark.sh)
    CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    # If conda command failed, try known specific path if available or guess
    if [ ! -d "$CONDA_PATH" ]; then
        CONDA_PATH="/sc/home/anton.hackl/conda3" # explicit fallback based on user envs
    fi
    
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        source "$CONDA_PATH/etc/profile.d/conda.sh"
        conda activate tdbase_env
    else
        echo "Warning: Could not find conda.sh at $CONDA_PATH. Attempting to run without explicit activation."
    fi
fi

export USE_GPU=TRUE
THREADS=${SLURM_CPUS_PER_TASK:-20}
PREFIX=${PREFIX:-tdbase_large}

BASE_DIR="/sc/home/anton.hackl/Spatial_Data_Management/RaySpace3D-Evaluation"
TDBASE_BUILD_DIR="$BASE_DIR/baselines/RaySpace3DBaselines/tdbase_patch/build_agent"
OUTPUT_DIR="$BASE_DIR/benchmarks/mesh_overlap/data/raw"
NV=${NV:-750}

mkdir -p $OUTPUT_DIR

cd $TDBASE_BUILD_DIR

for nu in 200 400 600 800; do
    echo "Generating dataset with nv=$NV, nu=$nu using $THREADS threads..."
    # Note: Simulator does not support -g flag in help, but we rely on threads.
    ./tdbase simulator \
        -n ../../data/nuclei.pt \
        -v ../../data/vessel.pt \
        -o "$OUTPUT_DIR/${PREFIX}_n_nv${NV}_nu${nu}" \
        --hausdorff \
        --nv $NV \
        --nu $nu \
        -r 30 \
        -i \
        -t $THREADS

    echo "Generating SECOND nuclei dataset for nn benchmark with nv=$NV, nu=$nu using $THREADS threads..."
    ./tdbase simulator \
        -n ../../data/nuclei.pt \
        -v ../../data/vessel.pt \
        -o "$OUTPUT_DIR/${PREFIX}_nn_nv${NV}_nu${nu}" \
        --hausdorff \
        --nv $NV \
        --nu $nu \
        -r 30 \
        -i \
        -t $THREADS
done
