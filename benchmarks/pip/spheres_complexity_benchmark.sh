#!/bin/bash
# Spheres Complexity Benchmark Script
# Tests query performance with spheres of varying mesh complexity (Stage 1-10)
# All spheres have the same selectivity (1%) but different mesh detail levels
# Uses centered approach (2 runs at bbox center) with 100 million points
#
# Mesh Complexity (vertices/faces):
#   Stage  1:     26 vertices,     48 faces (4.0K)
#   Stage  2:   2966 vertices,   5928 faces (486K)
#   Stage  3:  10806 vertices,  21608 faces (1.8M)
#   Stage  4:  23546 vertices,  47088 faces (4.2M)
#   Stage  5:  41186 vertices,  82368 faces (7.4M)
#   Stage  6:  63904 vertices, 127804 faces (12M)
#   Stage  7:  91379 vertices, 182754 faces (17M)
#   Stage  8: 123754 vertices, 247504 faces (24M)
#   Stage  9: 161029 vertices, 322054 faces (31M)
#   Stage 10: 204162 vertices, 408320 faces (40M)
#
# Usage:
#   ./spheres_complexity_benchmark.sh                          # Run all approaches
#   ./spheres_complexity_benchmark.sh cgal,raytracer           # Run only CGAL and Raytracer
#   ./spheres_complexity_benchmark.sh cuda                     # Run only CUDA

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
NUM_POINTS=100000000  # 100 million points
APPROACHES="${1:-cgal,sql,raytracer,raytracer_filter_refine,cuda}"
SELECTIVITY=0.01  # 1% selectivity

# Sphere stages to test (1-10)
STAGES=(1 2 3 4 5 6 7 8 9 10)

# Paths
SPHERE_DIR="../datasets/range_query/ranges"
POINTS_FILE="workspace/uniform_points_${NUM_POINTS}.wkt"
WORKSPACE_DIR="workspace"
RESULTS_DIR="results"

# Bounding box dimensions (points are in 50x50x50 box)
BBOX_SIZE=50
BBOX_VOLUME=125000  # 50^3

# Calculate required diameter for 1% selectivity
# r = ((3 * selectivity * bbox_volume) / (4 * π))^(1/3)
# For 1% selectivity: diameter = 13.365046
REQUIRED_DIAMETER=$(python3 -c "import math; r = ((3 * $SELECTIVITY * $BBOX_VOLUME) / (4 * math.pi)) ** (1/3); print(f'{2*r:.6f}')")

echo "========================================"
echo "Spheres Complexity Benchmark Configuration"
echo "========================================"
echo "Approaches:     $APPROACHES"
echo "Mode:           Centered (2 runs at bbox center)"
echo "Num points:     $NUM_POINTS (100M)"
echo "Selectivity:    $SELECTIVITY (1%)"
echo "Sphere stages:  ${STAGES[*]}"
echo "Target diameter: $REQUIRED_DIAMETER"
echo "Bbox size:      ${BBOX_SIZE}x${BBOX_SIZE}x${BBOX_SIZE}"
echo ""

# Create necessary directories
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$RESULTS_DIR"

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

# Run benchmark for each sphere stage
for STAGE in "${STAGES[@]}"; do
    echo "========================================"
    echo "Running benchmark for Sphere Stage: $STAGE"
    echo "========================================"
    
    # Original sphere file
    ORIGINAL_SPHERE="${SPHERE_DIR}/Sphere_Stage_${STAGE}.obj"
    
    if [ ! -f "$ORIGINAL_SPHERE" ]; then
        echo "Error: Sphere file not found at $ORIGINAL_SPHERE"
        echo "Skipping stage $STAGE"
        echo ""
        continue
    fi
    
    # Get mesh statistics
    VERTICES=$(grep -c "^v " "$ORIGINAL_SPHERE" || echo 0)
    FACES=$(grep -c "^f " "$ORIGINAL_SPHERE" || echo 0)
    FILE_SIZE=$(du -h "$ORIGINAL_SPHERE" | cut -f1)
    
    echo "Mesh complexity:"
    echo "  Vertices: $VERTICES"
    echo "  Faces:    $FACES"
    echo "  File size: $FILE_SIZE"
    echo ""
    
    # Rescale sphere to achieve 1% selectivity
    RESCALED_SPHERE="${WORKSPACE_DIR}/sphere_stage_${STAGE}_1pct.obj"
    
    echo "Rescaling sphere to diameter ${REQUIRED_DIAMETER} (1% selectivity)..."
    python3 rescale_obj.py \
        "$ORIGINAL_SPHERE" \
        "$RESCALED_SPHERE" \
        --extent-x "$REQUIRED_DIAMETER" \
        --extent-y "$REQUIRED_DIAMETER" \
        --extent-z "$REQUIRED_DIAMETER"
    
    echo ""
    
    # Run benchmark with centered approach
    BENCHMARK_NAME="sphere_complexity_stage_${STAGE}"
    OUTPUT_FILE="${RESULTS_DIR}/grid_benchmark.json"
    
    echo "Running benchmark with name: $BENCHMARK_NAME"
    python3 grid_benchmark.py \
        --query-obj "$RESCALED_SPHERE" \
        --points "$POINTS_FILE" \
        --approaches "$APPROACHES" \
        --output "$OUTPUT_FILE" \
        --workspace "${WORKSPACE_DIR}/${BENCHMARK_NAME}" \
        --centered \
        --name "$BENCHMARK_NAME"
    
    echo ""
    echo "Benchmark complete for Sphere Stage $STAGE ($VERTICES vertices, $FACES faces)"
    echo ""
done

echo "========================================"
echo "All sphere complexity benchmarks complete!"
echo "========================================"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "Look for files named: grid_benchmark_sphere_complexity_stage_<stage>_<timestamp>.json"

