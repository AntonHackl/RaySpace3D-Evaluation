#!/bin/bash
# Spheres Large Benchmark Script
# Runs grid_benchmark.py with different sphere sizes (1x1x1, 5x5x5, 10x10x10, 20x20x20, 40x40x40, 50x50x50)
# Uses 100 million uniform points and grid-size 1 2 2
#
# Usage:
#   ./spheres_large_benchmark.sh                          # Run all approaches
#   ./spheres_large_benchmark.sh cgal,raytracer           # Run only CGAL and Raytracer
#   ./spheres_large_benchmark.sh sql                      # Run only SQL

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
NUM_POINTS=100000000
GRID_SIZE="1 2 2"
APPROACHES="${1:-cgal,sql,raytracer,raytracer_filter_refine,cuda}"
SPHERE_SIZES=(1 5 10 20 40 50)

# Paths
ORIGINAL_SPHERE="../datasets/range_query/ranges/HighRes_sphere.obj"
POINTS_FILE="workspace/uniform_points_${NUM_POINTS}.wkt"
WORKSPACE_DIR="workspace"
RESULTS_DIR="results"

# Check if original sphere exists
if [ ! -f "$ORIGINAL_SPHERE" ]; then
    echo "Error: Original sphere file not found at $ORIGINAL_SPHERE"
    exit 1
fi

# Create necessary directories
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Spheres Large Benchmark Configuration"
echo "========================================"
echo "Approaches:   $APPROACHES"
echo "Grid size:    $GRID_SIZE"
echo "Sphere sizes: ${SPHERE_SIZES[*]}"
echo "Points:       $NUM_POINTS"
echo ""

# Generate 100 million uniform points if they don't exist
if [ ! -f "$POINTS_FILE" ]; then
    echo "========================================"
    echo "Generating $NUM_POINTS uniform points..."
    echo "========================================"
    
    python3 ../RaySpace3D/scripts/generate_points_in_bbox.py \
        --min 0 0 0 \
        --max 50 50 50 \
        --num_points "$NUM_POINTS" \
        --output "$POINTS_FILE" \
        --chunk-size 1000000
    
    echo "Points generated: $POINTS_FILE"
    echo ""
else
    echo "Using existing points file: $POINTS_FILE"
    echo ""
fi

# Run benchmark for each sphere size
for SIZE in "${SPHERE_SIZES[@]}"; do
    echo "========================================"
    echo "Running benchmark for sphere size: ${SIZE}x${SIZE}x${SIZE}"
    echo "========================================"
    
    # Rescale sphere
    RESCALED_SPHERE="${WORKSPACE_DIR}/sphere_${SIZE}.obj"
    
    echo "Rescaling sphere to ${SIZE}x${SIZE}x${SIZE}..."
    python3 rescale_obj.py \
        "$ORIGINAL_SPHERE" \
        "$RESCALED_SPHERE" \
        --extent-x "$SIZE" \
        --extent-y "$SIZE" \
        --extent-z "$SIZE"
    
    echo ""
    
    # Run benchmark
    BENCHMARK_NAME="sphere_${SIZE}"
    OUTPUT_FILE="${RESULTS_DIR}/grid_benchmark.json"
    
    echo "Running benchmark with name: $BENCHMARK_NAME"
    python3 grid_benchmark.py \
        --query-obj "$RESCALED_SPHERE" \
        --points "$POINTS_FILE" \
        --approaches "$APPROACHES" \
        --output "$OUTPUT_FILE" \
        --workspace "${WORKSPACE_DIR}/${BENCHMARK_NAME}" \
        --grid-size $GRID_SIZE \
        --name "$BENCHMARK_NAME"
    
    echo ""
    echo "Benchmark complete for sphere size ${SIZE}x${SIZE}x${SIZE}"
    echo ""
done

echo "========================================"
echo "All sphere benchmarks complete!"
echo "========================================"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "Look for files named: grid_benchmark_sphere_<size>_<timestamp>.json"

