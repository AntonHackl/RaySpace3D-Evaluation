#!/bin/bash
# Spheres Selectivity Benchmark Script - 500 Million Points
# Wrapper script that runs spheres_selectivity_benchmark.sh with 500M points
#
# Usage:
#   ./spheres_selectivity_benchmark_500M.sh                    # Run all approaches
#   ./spheres_selectivity_benchmark_500M.sh cgal,raytracer     # Run only CGAL and Raytracer

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
NUM_POINTS=500000000
NAME_SUFFIX="500M"
APPROACHES="${1:-cgal,sql,raytracer,raytracer_filter_refine,cuda}"

echo "========================================"
echo "Running Spheres Selectivity Benchmark"
echo "with 500 Million Points"
echo "========================================"
echo ""

# Call the main benchmark script with 500M points
./spheres_selectivity_benchmark.sh "$APPROACHES" "$NUM_POINTS" "$NAME_SUFFIX"
