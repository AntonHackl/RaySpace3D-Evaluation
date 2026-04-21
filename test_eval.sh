#!/bin/bash
set -e
mkdir -p test_dir
# create mock meshes
cat << 'MOCK1' > test_dir/mesh1.obj
v -1.0 -1.0 -1.0
v 1.0 -1.0 -1.0
v 0.0 1.0 -1.0
f 1 2 3
MOCK1

cat << 'MOCK2' > test_dir/mesh2.obj
v -0.5 -0.5 -1.0
v 1.5 -0.5 -1.0
v 0.5 1.5 -1.0
f 1 2 3
MOCK2

# activate conda
source /sc/projects/sci-zacharatou/chair/RaySpace/conda3/etc/profile.d/conda.sh

# run preprocess
conda activate rayspace3d_preprocess
echo "Preprocessing mesh1..."
./src/RaySpace3D/preprocess/build/bin/preprocess_dataset --input test_dir/mesh1.obj --output-geometry test_dir/mesh1.ray.bin --generate-grid --grid-cell-size 0.5
echo "Preprocessing mesh2..."
./src/RaySpace3D/preprocess/build/bin/preprocess_dataset --input test_dir/mesh2.obj --output-geometry test_dir/mesh2.ray.bin --generate-grid --grid-cell-size 0.5

# run query
conda activate rayspace3d_query_linux
echo "Running intersection estimated query..."
./src/RaySpace3D/query/build/bin/raytracer_intersection_estimated --mesh1 test_dir/mesh1.ray.bin --mesh2 test_dir/mesh2.ray.bin --runs 1

echo "Done!"
