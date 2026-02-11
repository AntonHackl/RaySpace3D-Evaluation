#!/bin/bash
# Spheres Selectivity Benchmark Script
# Runs grid_benchmark.py with different sphere sizes to achieve specific selectivities
# Uses centered approach (2 runs at bbox center) instead of grid
# Selectivities: 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.8
#
# For a sphere with radius r in a 50x50x50 bounding box:
#   Volume of sphere: (4/3) * π * r³
#   Volume of bbox: 50³ = 125,000
#   Selectivity: (4/3) * π * r³ / 125,000
#   Therefore: r = ((3 * selectivity * 125,000) / (4 * π))^(1/3)
#
# Usage:
#   ./spheres_selectivity_benchmark.sh [approaches] [num_points] [name_suffix]
#   ./spheres_selectivity_benchmark.sh                                    # Run all approaches, 10M points
#   ./spheres_selectivity_benchmark.sh cgal,raytracer                     # Run only CGAL and Raytracer, 10M points
#   ./spheres_selectivity_benchmark.sh cgal,raytracer 200000000 200M      # Run with 200M points

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
NUM_POINTS="${2:-10000000}"
APPROACHES="${1:-cgal,sql,raytracer,raytracer_filter_refine,cuda}"
NAME_SUFFIX="${3:-}"

# Selectivities to test
SELECTIVITIES=(0.0001 0.001 0.01 0.05 0.1 0.2 0.4 0.8)

# Paths
ORIGINAL_SPHERE="../datasets/range_query/ranges/HighRes_sphere.obj"
POINTS_FILE="workspace/uniform_points_${NUM_POINTS}.wkt"
WORKSPACE_DIR="workspace"
RESULTS_DIR="results"

# Bounding box dimensions (points are in 50x50x50 box)
BBOX_SIZE=50
BBOX_VOLUME=125000  # 50^3

# Check if original sphere exists
if [ ! -f "$ORIGINAL_SPHERE" ]; then
    echo "Error: Original sphere file not found at $ORIGINAL_SPHERE"
    exit 1
fi

# Create necessary directories
mkdir -p "$WORKSPACE_DIR"
mkdir -p "$RESULTS_DIR"

echo "========================================"
echo "Spheres Selectivity Benchmark Configuration"
echo "========================================"
echo "Approaches:     $APPROACHES"
echo "Mode:           Centered (2 runs at bbox center)"
echo "Num points:     $NUM_POINTS"
echo "Selectivities:  ${SELECTIVITIES[*]}"
echo "Bbox size:      ${BBOX_SIZE}x${BBOX_SIZE}x${BBOX_SIZE}"
if [ -n "$NAME_SUFFIX" ]; then
    echo "Name suffix:    $NAME_SUFFIX"
fi
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

# Function to calculate sphere radius for given selectivity
# r = ((3 * selectivity * bbox_volume) / (4 * π))^(1/3)
calculate_radius() {
    local selectivity=$1
    python3 -c "import math; r = ((3 * $selectivity * $BBOX_VOLUME) / (4 * math.pi)) ** (1/3); print(f'{r:.6f}')"
}

# Function to calculate sphere diameter (2*r) for extent parameter
calculate_diameter() {
    local selectivity=$1
    python3 -c "import math; r = ((3 * $selectivity * $BBOX_VOLUME) / (4 * math.pi)) ** (1/3); print(f'{2*r:.6f}')"
}

# Run benchmark for each selectivity
for SELECTIVITY in "${SELECTIVITIES[@]}"; do
    echo "========================================"
    echo "Running benchmark for selectivity: $SELECTIVITY"
    echo "========================================"
    
    # Calculate required sphere diameter (extent)
    DIAMETER=$(calculate_diameter $SELECTIVITY)
    RADIUS=$(calculate_radius $SELECTIVITY)
    
    echo "Calculated radius:   $RADIUS"
    echo "Calculated diameter: $DIAMETER"
    
    # Rescale sphere to achieve desired selectivity
    # The sphere is rescaled to have diameter = DIAMETER (extent in all dimensions)
    RESCALED_SPHERE="${WORKSPACE_DIR}/sphere_sel_${SELECTIVITY}.obj"
    
    echo "Rescaling sphere to diameter ${DIAMETER}..."
    python3 rescale_obj.py \
        "$ORIGINAL_SPHERE" \
        "$RESCALED_SPHERE" \
        --extent-x "$DIAMETER" \
        --extent-y "$DIAMETER" \
        --extent-z "$DIAMETER"
    
    echo ""
    
    # Run benchmark with centered approach
    if [ -n "$NAME_SUFFIX" ]; then
        BENCHMARK_NAME="sphere_selectivity_${SELECTIVITY}_${NAME_SUFFIX}"
    else
        BENCHMARK_NAME="sphere_selectivity_${SELECTIVITY}"
    fi
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
    echo "Benchmark complete for selectivity $SELECTIVITY (radius: $RADIUS)"
    echo ""
done

echo "========================================"
echo "All sphere selectivity benchmarks complete!"
echo "========================================"
echo ""
echo "Results saved in: $RESULTS_DIR"
echo "Look for files named: grid_benchmark_sphere_sel_<selectivity>_<timestamp>.json"

