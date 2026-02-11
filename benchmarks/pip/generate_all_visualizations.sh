#!/bin/bash
# Quick script to run both visualization scripts

echo "=========================================="
echo "  BENCHMARK VISUALIZATION GENERATOR"
echo "=========================================="
echo ""
echo "This script generates all benchmark visualizations:"
echo "  1. Complexity comparison (all approaches, 10 stages)"
echo "  2. Selectivity plots (10M, 200M, 500M points)"
echo ""
echo "Environment: spatial_benchmark"
echo ""
echo "=========================================="
echo ""

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please ensure conda is installed and in PATH."
    exit 1
fi

# Activate environment
echo "Activating spatial_benchmark environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate spatial_benchmark

echo ""
echo "=========================================="
echo "  STEP 1: Complexity Visualization"
echo "=========================================="
python visualize_complexity.py

echo ""
echo "=========================================="
echo "  STEP 2: Selectivity Visualizations"
echo "=========================================="
python visualize_selectivity.py

echo ""
echo "=========================================="
echo "  ALL VISUALIZATIONS COMPLETE!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - results/visualizations/sphere_complexity_comparison.png"
echo "  - results/visualizations/sphere_selectivity_10M.png"
echo "  - results/visualizations/sphere_selectivity_200M.png"
echo "  - results/visualizations/sphere_selectivity_500M.png"
echo ""
