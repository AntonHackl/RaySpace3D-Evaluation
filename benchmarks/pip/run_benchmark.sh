#!/bin/bash
# Grid Benchmark Runner
# 
# Runs the comprehensive 3D spatial query benchmark comparing:
# - CGAL (AABB tree baseline)
# - SQL (PostgreSQL/PostGIS)
# - RaySpace3D Raytracer
# - RaySpace3D Filter-Refine
# - CUDA (Bounding box filter + ray-triangle intersection)
#
# Usage:
#   ./run_benchmark.sh [approaches] [--evaluate]
#
# Example:
#   ./run_benchmark.sh                          # Run all approaches
#   ./run_benchmark.sh cgal,raytracer           # Run only CGAL and Raytracer
#   ./run_benchmark.sh sql                      # Run only SQL
#   ./run_benchmark.sh raytracer --evaluate     # Run raytracer and generate evaluation figures

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default paths (relative to first_benchmark directory)
QUERY_OBJ="../datasets/range_query/ranges/Cube_large.obj"
POINTS_WKT="../datasets/range_query/points/uniform_points_10000000.wkt"
OUTPUT_JSON="results/grid_benchmark.json"
WORKSPACE="workspace"

# Approaches to test (default: all)
APPROACHES="${1:-cgal,sql,raytracer,raytracer_filter_refine,cuda}"

# Check for --evaluate flag
EVALUATE_FLAG=""
if [[ "$2" == "--evaluate" ]] || [[ "$1" == "--evaluate" ]]; then
    EVALUATE_FLAG="--evaluate"
    if [[ "$1" == "--evaluate" ]]; then
        APPROACHES="cgal,sql,raytracer,raytracer_filter_refine,cuda"
    fi
fi

echo "========================================"
echo "  3D Spatial Query Grid Benchmark"
echo "========================================"
echo ""
echo "Query OBJ:     $QUERY_OBJ"
echo "Points:       $POINTS_WKT"
echo "Approaches:   $APPROACHES"
echo "Output:       $OUTPUT_JSON"
echo ""

# Check if conda environment exists
if ! conda env list | grep -q "spatial_benchmark"; then
    echo "Creating conda environment 'spatial_benchmark'..."
    conda env create -f environment.yml
    echo ""
fi

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate spatial_benchmark

# Verify Python dependencies
echo "Checking Python dependencies..."
python -c "import numpy, matplotlib; print('✓ Dependencies OK')"
echo ""

# Create output directory
mkdir -p results
mkdir -p "$WORKSPACE"

# Make Python scripts executable
chmod +x grid_benchmark.py
chmod +x rescale_obj.py

echo "========================================"
echo "  Starting Benchmark"
echo "========================================"
echo ""

# Run the benchmark
python grid_benchmark.py \
    --query-obj "$QUERY_OBJ" \
    --points "$POINTS_WKT" \
    --approaches "$APPROACHES" \
    --output "$OUTPUT_JSON" \
    --workspace "$WORKSPACE" \
    --grid-size 1 1 1 \
    --cgal-dir ../RaySpace3DBaslines/CGAL \
    --sql-dir ../RaySpace3DBaslines/SQL \
    --rayspace-dir ../RaySpace3D \
    $EVALUATE_FLAG

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "  Benchmark Complete!"
    echo "========================================"
    echo ""
    echo "Results saved to:"
    echo "  - JSON:          $OUTPUT_JSON"
    echo "  - Visualizations: results/visualizations/"
    echo ""
else
    echo ""
    echo "Benchmark failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi
