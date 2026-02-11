# Available Datasets

This benchmark includes four dataset configurations for testing mesh overlap join performance:

## 1. Small Dataset (`--dataset small`)
- **Files**: `test_small_n_nv15_nu30_vs100_r30.dt`, `test_small_v_nv15_nu30_vs100_r30.dt`
- **Configuration**: 15 vessels, 30 nuclei per vessel
- **Total Objects**: ~450 nuclei, 15 vessels
- **Size**: ~3.5 MB (nuclei), ~12 MB (vessels)
- **Use Case**: Quick tests and development
- **Default**: Yes

## 2. Medium Dataset (`--dataset medium`)
- **Files**: `medium_n_nv150_nu200_vs100_r30.dt`, `medium_v_nv150_nu200_vs100_r30.dt`
- **Configuration**: 150 vessels, 200 nuclei per vessel (with `-i` allowing intersections)
- **Total Objects**: ~30,000 nuclei, 150 vessels
- **Use Case**: Standard benchmarking with realistic complexity
- **Note**: Generated with TDBase README parameters

## 3. Nuclei-200 Dataset (`--dataset nu200`)
- **Files**: `nu200_n_nv150_nu200_vs100_r30.dt`, `nu200_v_nv150_nu200_vs100_r30.dt`
- **Configuration**: 150 vessels, 200 nuclei per vessel (with `-i` allowing intersections)
- **Total Objects**: ~30,000 nuclei, 150 vessels
- **Use Case**: Scaling tests

## 4. Nuclei-400 Dataset (`--dataset nu400`)
- **Files**: `nu400_n_nv150_nu400_vs100_r30.dt`, `nu400_v_nv150_nu400_vs100_r30.dt`
- **Configuration**: 150 vessels, 400 nuclei per vessel (with `-i` allowing intersections)
- **Total Objects**: ~60,000 nuclei, 150 vessels
- **Use Case**: Scaling tests

## 5. Nuclei-to-Nuclei Join (`--dataset nuclei_join`)
- **Files**: `medium_n_nv150_nu200_vs100_r30.dt`, `medium_n2_n_nv150_nu200_vs100_r30.dt`
- **Configuration**: Two nuclei datasets, each with 150 vessels × 200 nuclei
- **Total Objects**: ~30,000 nuclei vs ~30,000 nuclei
- **Use Case**: Testing performance on nuclei-nuclei overlap detection

## 6. Small Cubes Dataset (`--dataset cubes_100k`)
- **Files**: `cubes_100k.dt`, `cubes_100k_v2.dt`
- **Configuration**: Two sets of 100,000 cubes in 50×50×50 space

## 6b. Cubes Dataset (~0.2% Selectivity) (`--dataset cubes_100k_sel02`)
- **Files**: `cubes_100k_s002_real.obj`, `cubes_100k_s002_real_v2.obj`
- **Generation**: `RaySpace3D/scripts/generate_cubes_by_selectivity.py` with `--num-cubes-a 100000 --num-cubes-b 100000 --min-size 1 --max-size 5 --selectivity 0.002`
- **Note**: The script now verifies selectivity by sampling directly on the generated cubes (before writing to disk), so the printed verified selectivity should closely match what the raytracer reports.

## 7. Large Cubes Dataset (`--dataset cubes_1m`)
- **Files**: `cubes_1m.obj`, `cubes_1m_v2.obj`
- **Configuration**: Two sets of 1,000,000 cubes in 100×100×100 space
- **Total Objects**: 1,000,000 cubes each
- **Triangles**: 12,000,000 per file
- **Size**: ~550 MB each (.obj format)
- **Use Case**: Stress testing and extreme scale performance benchmark

## 7b. Large Cubes Dataset (~0.5% Selectivity) (`--dataset cubes_1m_sel05`)
- **Files**: `cubes_1m_s005.obj`, `cubes_1m_s005_v2.obj`
- **Generation**: `RaySpace3D/scripts/generate_cubes_by_selectivity.py` with `--num-cubes-a 1000000 --num-cubes-b 1000000 --min-size 1 --max-size 5 --selectivity 0.005`
- **Note**: For 1M cubes, the generator uses streaming mode and verifies selectivity from a reservoir sample of the generated cubes.

## Usage

```bash
# Run with default (small) dataset
./run_benchmark.sh --runs 10

# Run with medium dataset
./run_benchmark.sh --dataset medium --runs 10

# Run with nuclei-to-nuclei join
./run_benchmark.sh --dataset nuclei_join --runs 10

# Run with cubes dataset
./run_benchmark.sh --dataset cubes --runs 10

# Custom files (overrides --dataset)
./run_benchmark.sh --file1 custom1.dt --file2 custom2.dt --runs 10
```

## Preprocessing

All datasets are automatically preprocessed for RaySpace3D on first use. Preprocessing generates:
- `.pre` files (binary geometry for RaySpace3D and CGAL)
- `_timing.json` files (preprocessing timing statistics)

These are stored in `mesh_overlap_benchmark/data/` alongside the original `.dt` files.
