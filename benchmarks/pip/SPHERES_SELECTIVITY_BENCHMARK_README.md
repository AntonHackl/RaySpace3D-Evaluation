# Spheres Selectivity Benchmark

## Overview

This benchmark script (`spheres_selectivity_benchmark.sh`) tests spatial query performance with spheres of different sizes to achieve specific selectivities. Unlike the grid-based benchmarks, this uses the **centered approach** where the sphere is positioned at the center of the bounding box and the query is run twice (for averaging).

## Selectivity Calculation

For a sphere with radius `r` in a 50×50×50 bounding box:

- **Volume of sphere**: `V_sphere = (4/3) × π × r³`
- **Volume of bounding box**: `V_bbox = 50³ = 125,000`
- **Selectivity**: `S = V_sphere / V_bbox`

To achieve a target selectivity, we solve for radius:

```
r = ((3 × S × V_bbox) / (4 × π))^(1/3)
```

## Tested Selectivities

The benchmark tests the following selectivities:

| Selectivity | Radius (r) | Diameter (2r) | Sphere Volume | Notes |
|-------------|------------|---------------|---------------|-------|
| 0.0001      | 1.439706   | 2.879412      | 12.50         | Very small sphere (0.01% of bbox) |
| 0.001       | 3.101752   | 6.203505      | 125.00        | Small sphere (0.1% of bbox) |
| 0.01        | 6.682523   | 13.365046     | 1,250.00      | 1% of bbox |
| 0.05        | 11.426954  | 22.853907     | 6,250.00      | 5% of bbox |
| 0.1         | 14.397060  | 28.794119     | 12,500.00     | 10% of bbox |
| 0.2         | 18.139158  | 36.278317     | 25,000.00     | 20% of bbox |
| 0.4         | 22.853907  | 45.707815     | 50,000.00     | 40% of bbox |
| 0.8         | 28.794119  | 57.588238     | 100,000.00    | 80% of bbox |

## Usage

### Main Script (10M points by default)

Run with 10 million points (default):
```bash
./spheres_selectivity_benchmark.sh
```

Run specific approaches:
```bash
./spheres_selectivity_benchmark.sh cgal,raytracer
./spheres_selectivity_benchmark.sh sql
./spheres_selectivity_benchmark.sh cuda
```

Run with custom number of points and name suffix:
```bash
# Syntax: ./spheres_selectivity_benchmark.sh [approaches] [num_points] [name_suffix]
./spheres_selectivity_benchmark.sh cgal,raytracer 200000000 200M
./spheres_selectivity_benchmark.sh all 500000000 500M
```

### 200 Million Points Benchmark

```bash
# Run all approaches with 200M points
./spheres_selectivity_benchmark_200M.sh

# Run specific approaches with 200M points
./spheres_selectivity_benchmark_200M.sh cgal,raytracer
```

### 500 Million Points Benchmark

```bash
# Run all approaches with 500M points
./spheres_selectivity_benchmark_500M.sh

# Run specific approaches with 500M points
./spheres_selectivity_benchmark_500M.sh cgal,cuda
```

## Benchmark Configuration

- **Points**: Configurable (default: 10M, also available: 200M, 500M) uniform points in 50×50×50 bounding box
- **Mode**: Centered (2 runs at bbox center for averaging)
- **Approaches**: CGAL, SQL, Raytracer, FilterRefine, CUDA (configurable)
- **Source sphere**: `../datasets/range_query/ranges/HighRes_sphere.obj`

### Available Configurations

| Script | Points | Name Suffix | Use Case |
|--------|--------|-------------|----------|
| `spheres_selectivity_benchmark.sh` | 10M (default) | None | Standard benchmark |
| `spheres_selectivity_benchmark_200M.sh` | 200M | `200M` | Large-scale benchmark |
| `spheres_selectivity_benchmark_500M.sh` | 500M | `500M` | Very large-scale benchmark |

## Output

Results are saved in the `results/` directory with filenames like:
```
grid_benchmark_sphere_selectivity_<selectivity>_<timestamp>.json
grid_benchmark_sphere_selectivity_<selectivity>_<suffix>_<timestamp>.json
```

Examples:
- **10M points**: `grid_benchmark_sphere_selectivity_0.0001_20251211_021559.json`
- **10M points**: `grid_benchmark_sphere_selectivity_0.01_20251211_021612.json`
- **200M points**: `grid_benchmark_sphere_selectivity_0.1_200M_20251211_031845.json`
- **500M points**: `grid_benchmark_sphere_selectivity_0.8_500M_20251211_041230.json`

## How It Works

1. **Point Generation**: Generates 10M uniform points in 50×50×50 bbox (if not already present)
2. **Sphere Rescaling**: For each selectivity:
   - Calculates required sphere diameter using the formula above
   - Rescales the high-resolution sphere to that diameter
   - Saves rescaled sphere to `workspace/sphere_sel_<selectivity>.obj`
3. **Benchmark Execution**: Runs the grid_benchmark.py with:
   - `--centered` flag (positions sphere at bbox center)
   - The rescaled sphere as query object
   - All specified approaches
4. **Results**: Saves timing and performance metrics to JSON

## Comparison with Grid Benchmarks

| Aspect | Grid Benchmarks | Selectivity Benchmark |
|--------|----------------|----------------------|
| **Positioning** | Grid of positions (e.g., 3×3×3 = 27 runs) | Centered (2 runs at center) |
| **Variable** | Sphere size (1, 5, 10, 20, 40, 50) | Selectivity (0.0001 to 0.8) |
| **Purpose** | Test spatial variation | Test selectivity impact |
| **Grid size** | `--grid-size 1 2 2` | `--centered` |

## Notes

- The sphere is rescaled to have the same extent in all dimensions (x, y, z)
- The centered approach is more appropriate for selectivity studies as it eliminates spatial variation
- Each selectivity is run twice at the same position to allow for averaging (similar to grid approach)
