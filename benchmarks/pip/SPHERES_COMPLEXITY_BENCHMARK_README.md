# Spheres Complexity Benchmark

## Overview

This benchmark script (`spheres_complexity_benchmark.sh`) tests spatial query performance with spheres of **varying mesh complexity** while keeping selectivity constant. Unlike the selectivity benchmarks that vary the sphere size, this benchmark uses spheres with different levels of geometric detail (Stage 1-10) to measure the impact of query object complexity on performance.

## Key Characteristics

- **Constant Selectivity**: All spheres are scaled to achieve exactly **1% selectivity**
- **Constant Point Count**: Always uses **100 million points**
- **Variable Complexity**: Tests 10 different mesh detail levels (Stage 1-10)
- **Centered Approach**: 2 runs at bbox center for averaging
- **Bounding Box**: 50×50×50 (125,000 cubic units)

## Mesh Complexity Levels

The benchmark uses pre-generated sphere meshes with increasing geometric detail:

| Stage | Vertices | Faces | File Size | Complexity Level |
|-------|----------|-------|-----------|------------------|
| 1 | 26 | 48 | 4.0K | Very Low (icosahedron) |
| 2 | 2,966 | 5,928 | 486K | Low |
| 3 | 10,806 | 21,608 | 1.8M | Medium-Low |
| 4 | 23,546 | 47,088 | 4.2M | Medium |
| 5 | 41,186 | 82,368 | 7.4M | Medium-High |
| 6 | 63,904 | 127,804 | 12M | High |
| 7 | 91,379 | 182,754 | 17M | Very High |
| 8 | 123,754 | 247,504 | 24M | Ultra High |
| 9 | 161,029 | 322,054 | 31M | Extreme |
| 10 | 204,162 | 408,320 | 40M | Maximum |

### Complexity Scaling

- **Stage 1 → Stage 10**: ~7,850x more vertices, ~8,500x more faces
- **File Size Growth**: 4KB → 40MB (10,000x increase)
- **Geometric Detail**: From basic icosahedron to highly detailed sphere

## Selectivity Calculation

All spheres are rescaled to achieve **1% selectivity**:

- **Target selectivity**: 0.01 (1%)
- **Bounding box volume**: 125,000 (50³)
- **Required sphere volume**: 1,250 cubic units
- **Required radius**: 6.682523
- **Required diameter**: 13.365046

Formula: `r = ((3 × 0.01 × 125,000) / (4 × π))^(1/3) ≈ 6.68`

## Usage

### Run all approaches:
```bash
./spheres_complexity_benchmark.sh
```

### Run specific approaches:
```bash
./spheres_complexity_benchmark.sh cgal,raytracer
./spheres_complexity_benchmark.sh cuda
./spheres_complexity_benchmark.sh sql
```

## Benchmark Configuration

- **Points**: 100 million uniform points in 50×50×50 bounding box
- **Mode**: Centered (2 runs at bbox center for averaging)
- **Selectivity**: 1% (constant across all stages)
- **Approaches**: CGAL, SQL, Raytracer, FilterRefine, CUDA (configurable)
- **Source spheres**: `../datasets/range_query/ranges/Sphere_Stage_{1-10}.obj`

## Output

Results are saved in the `results/` directory with filenames like:
```
grid_benchmark_sphere_complexity_stage_<stage>_<timestamp>.json
```

Examples:
- `grid_benchmark_sphere_complexity_stage_1_20251211_025530.json`
- `grid_benchmark_sphere_complexity_stage_5_20251211_025845.json`
- `grid_benchmark_sphere_complexity_stage_10_20251211_030230.json`

## How It Works

1. **Point Generation**: Generates 100M uniform points in 50×50×50 bbox (if not already present)
2. **Sphere Processing**: For each stage (1-10):
   - Loads the original `Sphere_Stage_N.obj` file
   - Counts vertices and faces for statistics
   - Rescales the sphere to diameter 13.365046 (1% selectivity)
   - Saves rescaled sphere to `workspace/sphere_stage_N_1pct.obj`
3. **Benchmark Execution**: Runs grid_benchmark.py with:
   - `--centered` flag (positions sphere at bbox center)
   - The rescaled sphere as query object
   - All specified approaches
4. **Results**: Saves timing and performance metrics to JSON

## Research Questions

This benchmark helps answer:

1. **How does mesh complexity affect query performance?**
   - Does performance scale linearly with vertex/face count?
   - Are there complexity thresholds where performance degrades significantly?

2. **Which approaches handle complex geometries better?**
   - Does CGAL's exact predicates scale well with complexity?
   - How does raytracing performance change with mesh detail?
   - Does CUDA's parallelism help with complex meshes?

3. **What's the practical complexity limit?**
   - At what stage does query time become impractical?
   - Is there a sweet spot for mesh detail vs. performance?

## Comparison with Other Benchmarks

| Aspect | Selectivity Benchmark | Complexity Benchmark |
|--------|----------------------|---------------------|
| **Variable** | Sphere size (selectivity) | Mesh detail (vertices/faces) |
| **Constant** | Mesh complexity | Selectivity (1%) |
| **Points** | 10M, 200M, 500M | 100M |
| **Purpose** | Test selectivity impact | Test complexity impact |
| **Stages** | 8 selectivities | 10 complexity levels |

## Expected Performance Patterns

### CGAL
- May show linear or super-linear scaling with face count
- Exact predicates require more computation per face

### Raytracer
- Should scale roughly linearly with face count
- BVH construction time increases with complexity
- Ray intersection tests increase with mesh detail

### CUDA
- Parallel processing may mitigate complexity impact
- Memory bandwidth could become bottleneck for large meshes

### SQL (PostGIS)
- Mesh complexity affects spatial index size
- May show non-linear scaling due to index overhead

## Notes

- All spheres have radius 1 in their original form
- Rescaling maintains the relative mesh detail
- The centered approach eliminates spatial variation effects
- Each stage is run twice at the same position for averaging
- File sizes range from 4KB to 40MB, testing I/O impact as well
