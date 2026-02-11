#!/bin/bash
# Cubes Benchmark Script
# Runs grid_benchmark.py with different cube sizes (1x1x1, 5x5x5, 10x10x10, 25x25x25, 50x50x50)
# Uses 10 million uniform points and grid-size 1 2 2
#
# Usage:
#   ./cubes_benchmark.sh                          # Run all approaches
#   ./cubes_benchmark.sh cgal,raytracer           # Run only CGAL and Raytracer
#   ./cubes_benchmark.sh sql                      # Run only SQL

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
NUM_POINTS=10000000
GRID_SIZE="1 2 2"
APPROACHES="${1:-cgal,sql,raytracer,raytracer_filter_refine,cuda}"
# CUBE_SIZES=(1 5 10 25 50)
CUBE_SIZES=(50)

# Paths
ORIGINAL_CUBE="../RaySpace3DBaslines/CGAL/data/cube.obj"
POINTS_FILE="workspace/uniform_points_${NUM_POINTS}.wkt"
WORKSPACE_DIR="workspace"
RESULTS_DIR="results"

# Check if original cube exists
if [ ! -f "$ORIGINAL_CUBE" ]; then
    echo "Error: Original cube file not found at $ORIGINAL_CUBE"
    exit 1
fi

# Create necessary directories
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Cubes Benchmark Configuration"
echo "========================================"
echo "Approaches:   $APPROACHES"
echo "Grid size:    $GRID_SIZE"
echo "Cube sizes:   ${CUBE_SIZES[*]}"
echo ""

# Generate 10 million uniform points if they don't exist
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

# Run benchmark for each cube size
for SIZE in "${CUBE_SIZES[@]}"; do
    echo "========================================"
    echo "Running benchmark for cube size: ${SIZE}x${SIZE}x${SIZE}"
    echo "========================================"
    
    # Rescale cube
    RESCALED_CUBE="${WORKSPACE_DIR}/cube_${SIZE}.obj"
    
    echo "Rescaling cube to ${SIZE}x${SIZE}x${SIZE}..."
    python3 rescale_obj.py \
        "$ORIGINAL_CUBE" \
        "$RESCALED_CUBE" \
        --extent-x "$SIZE" \
        --extent-y "$SIZE" \
        --extent-z "$SIZE"
    
    echo ""
    
    # Run benchmark
    BENCHMARK_NAME="cube_${SIZE}"
    OUTPUT_FILE="${RESULTS_DIR}/grid_benchmark.json"
    
    echo "Running benchmark with name: $BENCHMARK_NAME"
    python3 grid_benchmark.py \
        --query-obj "$RESCALED_CUBE" \
        --points "$POINTS_FILE" \
        --approaches "$APPROACHES" \
        --output "$OUTPUT_FILE" \
        --workspace "${WORKSPACE_DIR}/${BENCHMARK_NAME}" \
        --grid-size $GRID_SIZE \
        --name "$BENCHMARK_NAME"
    
    echo ""
    echo "Benchmark complete for cube size ${SIZE}x${SIZE}x${SIZE}"
    echo ""
done

echo "========================================"
echo "All cube benchmarks complete!"
echo "========================================"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "Look for files named: grid_benchmark_cube_<size>_<timestamp>.json"

