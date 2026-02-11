# 3D Spatial Query Benchmark Suite

Comprehensive benchmark system comparing CGAL, PostgreSQL/PostGIS, and RaySpace3D approaches for 3D point-in-mesh queries.

## Overview

This benchmark translates a cube mesh to 27 positions in a 3x3x3 grid (quarters 1-3 in each dimension) and executes queries at each position, measuring:
- Query execution time
- Points inside/outside counts
- Upload/download times (for GPU approaches)

## Approaches Tested

1. **CGAL** - AABB tree with Side_of_triangle_mesh (CPU baseline)
2. **SQL** - PostgreSQL/PostGIS with spatial indexing and ray casting
3. **Raytracer** - RaySpace3D OptiX-based GPU raytracer
4. **FilterRefine** - RaySpace3D two-phase filter-refine approach

## Quick Start

```bash
cd first_benchmark

# Run all approaches
./run_benchmark.sh

# Run specific approaches
./run_benchmark.sh cgal,raytracer
./run_benchmark.sh sql
```

## File Structure

```
first_benchmark/
├── run_benchmark.sh           # Main runner script
├── grid_benchmark.py          # Benchmark orchestrator
├── rescale_obj.py             # OBJ rescaling utility
├── environment.yml            # Conda environment
├── README.md                  # This file
├── workspace/                 # Temporary files (created at runtime)
└── results/                   # Benchmark results
    ├── grid_benchmark.json    # Raw results
    └── visualizations/        # Generated plots
        ├── query_times_3d.png
        └── approach_comparison.png
```

## Detailed Usage

### Prerequisites

- Conda
- Built CGAL baseline (`RaySpace3DBaslines/CGAL/build/cgal_query`)
- Built SQL baseline (`RaySpace3DBaslines/SQL/build/spatial_query`)
- Built RaySpace3D (`RaySpace3D/build/bin/raytracer` and `raytracer_filter_refine`)

### Setup Environment

```bash
# Create conda environment (first time only)
conda env create -f environment.yml
conda activate spatial_benchmark
```

### Run Benchmark

**Full benchmark (all approaches):**
```bash
python grid_benchmark.py \
  --query-obj ../RaySpace3DBaslines/CGAL/data/cube.obj \
  --points ../datasets/range_query/points/uniform_points_10000000.wkt \
  --approaches cgal,sql,raytracer,raytracer_filter_refine \
  --output results/grid_benchmark.json
```

**Selective approaches:**
```bash
# Only CGAL and Raytracer
python grid_benchmark.py \
  --query-obj ../RaySpace3DBaslines/CGAL/data/cube.obj \
  --points ../datasets/range_query/points/uniform_points_10000000.wkt \
  --approaches cgal,raytracer \
  --output results/cgal_raytracer_comparison.json
```

**Custom grid size:**
```bash
python grid_benchmark.py \
  --query-obj ../RaySpace3DBaslines/CGAL/data/cube.obj \
  --points ../datasets/range_query/points/uniform_points_10000000.wkt \
  --approaches cgal \
  --grid-size 5 5 5 \
  --output results/fine_grid.json
```

## Rescaling Utility

Rescale a mesh to specific bounding box extents:

```bash
python rescale_obj.py input.obj output.obj \
    --extent-x 100 \
    --extent-y 100 \
    --extent-z 100
```

Example - create a 50x50x50 cube:
```bash
python rescale_obj.py \
    ../RaySpace3DBaslines/CGAL/data/cube.obj \
    cube_50.obj \
    --extent-x 50 --extent-y 50 --extent-z 50
```

## Output Format

### JSON Results

```json
{
  "configuration": {
    "query_obj": "path/to/cube.obj",
    "points_file": "path/to/points.wkt",
    "grid_size": [3, 3, 3],
    "approaches": ["CGAL", "SQL", "Raytracer", "FilterRefine"],
    "bbox_min": [x, y, z],
    "bbox_max": [x, y, z]
  },
  "results": {
    "CGAL": [
      {
        "grid_position": [0, 0, 0],
        "translation": [x, y, z],
        "query_ms": 123.45,
        "inside_count": 5000,
        "total_points": 10000000,
        "success": true,
        "wall_time_s": 1.5
      },
      ...
    ],
    ...
  },
  "CGAL_statistics": {
    "mean": 120.5,
    "std": 5.2,
    "min": 110.3,
    "max": 130.8,
    "count": 27
  },
  ...
}
```

### Visualizations

1. **query_times_3d.png** - 3D scatter plot showing query times across grid positions
2. **approach_comparison.png** - Bar chart comparing average query times with error bars

## Implementation Details

### Grid Translation

The benchmark computes the bounding box of the point cloud and divides it into a 3x3x3 grid:
- Each cell represents 1/3 of the range in each dimension
- Geometry is translated to the center of each cell
- This tests different spatial distributions and query complexity

### Timing Extraction

**CGAL & SQL:**
- Parse stdout for timing patterns:
  - CGAL: `CONTAINMENT QUERY TIME: XXX ms`
  - SQL: `QUERY TIME: XXX ms`
- Also extract `Points inside mesh: XXX` and `Total points: XXX`

**RaySpace3D (Raytracer):**
- Read timing JSON (`phases` section)
- Extract: `upload_1`, `query_1`, `output_1` durations
- Parse stdout for point counts

**RaySpace3D (FilterRefine):**
- Read timing JSON
- Extract: `Upload BBox Geometry`, `BBox Filter Query`, `Query`, `Output` durations
- Parse stdout for final results

### Database Reuse (SQL)

The SQL adapter loads points to PostgreSQL once during setup and reuses the database for all queries. Only the mesh geometry changes per grid position.

## Customization

### Use Different Datasets

```bash
python grid_benchmark.py \
  --query-obj path/to/your/mesh.obj \
  --points path/to/your/points.wkt \
  --approaches cgal,sql,raytracer,raytracer_filter_refine \
  --output results/custom_benchmark.json
```

### Modify Grid Resolution

For finer spatial sampling:
```bash
python grid_benchmark.py \
    ... \
    --grid-size 5 5 5  # 125 positions
```

### Test Single Approach

```bash
./run_benchmark.sh cgal      # Only CGAL
./run_benchmark.sh sql       # Only SQL
./run_benchmark.sh raytracer # Only Raytracer
```

## Troubleshooting

### SQL Database Issues

If SQL queries fail:
```bash
cd ../RaySpace3DBaslines/SQL
./scripts/init_db.sh
```

### Build Errors

Ensure all baselines are built:
```bash
# CGAL
cd ../RaySpace3DBaslines/CGAL
./scripts/build.sh

# SQL
cd ../RaySpace3DBaslines/SQL
./scripts/build.sh

# RaySpace3D
cd ../RaySpace3D
mkdir -p build && cd build
cmake .. && make
```

### Missing Dependencies

```bash
conda activate spatial_benchmark
conda install numpy matplotlib scipy pandas plotly
```

## Performance Notes

- **CGAL**: CPU-bound, benefits from OpenMP parallelization
- **SQL**: Database I/O overhead, excellent for persistent datasets
- **Raytracer**: GPU-accelerated, fast for complex geometries
- **FilterRefine**: Two-phase approach reduces candidate set

Expected runtime: 10-30 minutes for all 4 approaches × 27 positions with 10M points.

## Citation

If you use this benchmark suite in your research, please cite the RaySpace3D paper.
