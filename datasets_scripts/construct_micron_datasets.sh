#!/bin/bash
set -e

# This script constructs the MICrONS datasets of 8GB, 16GB, and 32GB.
# Coordinates are calculated to provide "Dense" spatial subsets (cubes).

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

echo "--------------------------------------------------"
echo "Constructing 8GB Dataset (194um cube)..."
python ./download_microns_region_by_mesh_bbox.py \
    --target-gb 8.0 --max-gb 9.0 \
    --x-min-nm 800688 --x-max-nm 994688 \
    --y-min-nm 611628 --y-max-nm 805628 \
    --z-min-nm 757160 --z-max-nm 951160 \
    --shuffle --seed 42

echo "--------------------------------------------------"
echo "Constructing 16GB Dataset (244um cube)..."
python ./download_microns_region_by_mesh_bbox.py \
    --target-gb 16.0 --max-gb 18.0 \
    --x-min-nm 775688 --x-max-nm 1019688 \
    --y-min-nm 586628 --y-max-nm 830628 \
    --z-min-nm 732160 --z-max-nm 976160 \
    --shuffle --seed 42

echo "--------------------------------------------------"
echo "Constructing 32GB Dataset (308um cube)..."
python ./download_microns_region_by_mesh_bbox.py \
    --target-gb 32.0 --max-gb 35.0 \
    --x-min-nm 743688 --x-max-nm 1051688 \
    --y-min-nm 554628 --y-max-nm 862628 \
    --z-min-nm 700160 --z-max-nm 1008160 \
    --shuffle --seed 42

echo "--------------------------------------------------"
echo "All requested datasets have been processed."
echo "Output directories:"
echo "  - ./microns_data/microns_region_8gb_glb"
echo "  - ./microns_data/microns_region_16gb_glb"
echo "  - ./microns_data/microns_region_32gb_glb"
