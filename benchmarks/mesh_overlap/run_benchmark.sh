#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Initialize conda for bash
if [ -z "$CONDA_PREFIX" ] || [[ "$CONDA_DEFAULT_ENV" != "mesh_overlap_benchmark" ]]; then
    # Try to find conda and source it
    CONDA_PATH=$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")
    if [ -f "$CONDA_PATH/etc/profile.d/conda.sh" ]; then
        source "$CONDA_PATH/etc/profile.d/conda.sh"
        conda activate mesh_overlap_benchmark
    else
        echo "Warning: Could not find conda.sh at $CONDA_PATH. Attempting to run with system python."
    fi
fi

# Run the python benchmark script
python benchmark.py "$@"
