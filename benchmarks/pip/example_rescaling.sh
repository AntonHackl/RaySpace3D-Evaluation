#!/bin/bash
# Example: Rescale cube to different sizes and run benchmarks

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  OBJ Rescaling Examples"
echo "========================================"
echo ""

# Original cube (2x2x2, centered at origin)
ORIGINAL_CUBE="../RaySpace3DBaslines/CGAL/data/cube.obj"

# Create rescaled versions
echo "Creating rescaled cube meshes..."

# Small cube: 10x10x10
python rescale_obj.py \
    "$ORIGINAL_CUBE" \
    workspace/cube_10.obj \
    --extent-x 10 --extent-y 10 --extent-z 10

echo ""

# Medium cube: 50x50x50
python rescale_obj.py \
    "$ORIGINAL_CUBE" \
    workspace/cube_50.obj \
    --extent-x 50 --extent-y 50 --extent-z 50

echo ""

# Large cube: 100x100x100
python rescale_obj.py \
    "$ORIGINAL_CUBE" \
    workspace/cube_100.obj \
    --extent-x 100 --extent-y 100 --extent-z 100

echo ""

# Rectangular prism: 100x50x25
python rescale_obj.py \
    "$ORIGINAL_CUBE" \
    workspace/cube_100x50x25.obj \
    --extent-x 100 --extent-y 50 --extent-z 25

echo ""
echo "========================================"
echo "  Rescaled meshes created in workspace/"
echo "========================================"
echo ""
echo "Now you can run benchmarks with these:"
echo ""
echo "  python grid_benchmark.py \\
echo "      --query-obj workspace/cube_50.obj \\
echo "      --points ../datasets/range_query/points/uniform_points_10000000.wkt \\
echo "      --approaches cgal,raytracer \\
echo "      --output results/cube_50_benchmark.json"
echo ""
