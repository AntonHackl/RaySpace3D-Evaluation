# First Benchmark - Implementation Summary

## What Was Created

A comprehensive benchmark suite in `first_benchmark/` for comparing 4 spatial query approaches across a 3x3x3 grid of geometry positions.

## Files Created

### Core Scripts
1. **`grid_benchmark.py`** (main orchestrator)
   - Generates 27 grid positions (3×3×3) based on point cloud bounding box
   - Implements 4 adapters: CGAL, SQL, Raytracer, FilterRefine
   - Parses stdout (CGAL/SQL) and JSON (RaySpace) for timing extraction
   - Sequential execution (no parallelization)
   - Reuses SQL database (loads points once)
   - Generates JSON results and visualizations

2. **`rescale_obj.py`** (OBJ rescaling utility)
   - Reads OBJ file and computes bounding box
   - Rescales to target extents (--extent-x, --extent-y, --extent-z)
   - Preserves mesh center position
   - Useful for testing different geometry sizes

3. **`run_benchmark.sh`** (easy runner)
   - Sets up conda environment
   - Runs benchmark with sensible defaults
   - Usage: `./run_benchmark.sh [approaches]`

4. **`environment.yml`** (conda environment)
   - Merges dependencies from CGAL, SQL, RaySpace3D
   - Adds numpy, matplotlib, scipy, pandas, plotly

5. **`README.md`** (documentation)
   - Quick start guide
   - Detailed usage examples
   - Output format specification
   - Troubleshooting tips

6. **`example_rescaling.sh`** (rescaling examples)
   - Shows how to create different cube sizes
   - Demonstrates rectangular prism rescaling

## Key Features

### Timing Extraction
- **CGAL**: Parses `CONTAINMENT QUERY TIME: XXX ms` from stdout
- **SQL**: Parses `QUERY TIME: XXX ms` from stdout
- **Raytracer**: Reads `upload_1`, `query_1`, `output_1` from JSON phases
- **FilterRefine**: Reads `Upload BBox Geometry`, `BBox Filter Query`, `Query`, `Output` from JSON phases

### Grid Translation
- Computes point cloud bounding box
- Divides into 3×3×3 grid (quarters 1-3 per dimension)
- Translates geometry to center of each cell
- Tests 27 spatial positions sequentially

### Database Handling (SQL)
- Loads points to PostgreSQL **once** during setup
- Reuses database for all 27 queries
- Only updates mesh geometry per position
- No database recreation between runs

### Visualization
- 3D scatter plot: Query times across grid positions
- Bar chart: Average query time per approach with error bars
- Statistics: Mean, std, min, max for each approach

## Usage Examples

### Basic Usage
```bash
cd first_benchmark
./run_benchmark.sh
```

### Test Specific Approaches
```bash
./run_benchmark.sh cgal,raytracer
./run_benchmark.sh sql
```

### Full Python Command
```bash
python grid_benchmark.py \
  --query-obj ../RaySpace3DBaslines/CGAL/data/cube.obj \
  --points ../datasets/range_query/points/uniform_points_10000000.wkt \
  --approaches cgal,sql,raytracer,raytracer_filter_refine \
  --output results/grid_benchmark.json \
  --grid-size 3 3 3
```

### Rescale Cube
```bash
python rescale_obj.py \
    ../RaySpace3DBaslines/CGAL/data/cube.obj \
    cube_50.obj \
    --extent-x 50 --extent-y 50 --extent-z 50
```

## Output Structure

```
first_benchmark/
├── results/
│   ├── grid_benchmark.json       # Raw results with timing data
│   └── visualizations/
│       ├── query_times_3d.png    # 3D scatter plot
│       └── approach_comparison.png # Bar chart
└── workspace/                    # Temporary files (auto-cleaned)
    ├── cgal/
    ├── sql/
    ├── raytracer/
    └── filter_refine/
```

## JSON Output Format

```json
{
  "configuration": {
    "query_obj": "...",
    "points_file": "...",
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
        "success": true
      },
      // ... 26 more positions
    ],
    "SQL": [...],
    "Raytracer": [
      {
        "query_ms": 45.67,
        "upload_ms": 10.5,
        "download_ms": 2.3,
        // ...
      }
    ],
    "FilterRefine": [
      {
        "query_ms": 30.2,
        "bbox_query_ms": 5.1,
        "upload_ms": 8.2,
        // ...
      }
    ]
  },
  "CGAL_statistics": {
    "mean": 120.5,
    "std": 5.2,
    "min": 110.3,
    "max": 130.8,
    "count": 27
  }
  // ... statistics for each approach
}
```

## Design Decisions

### Python + Shell (Hybrid)
- Python orchestrator for complex logic and parsing
- Shell subprocess calls for executables
- Best of both worlds: flexibility + easy integration

### Sequential Execution
- No parallelization (as requested)
- Easier debugging and resource management
- Clear timing measurements

### Database Reuse
- SQL adapter loads points once
- Significant time savings (no recreation)
- Only mesh geometry changes per query

### Stdout vs JSON Parsing
- CGAL/SQL: Parse stdout with regex patterns
- RaySpace: Read structured JSON timing files
- Handles both output formats seamlessly

## Prerequisites

Before running:
1. Build CGAL baseline: `cd RaySpace3DBaslines/CGAL && ./scripts/build.sh`
2. Build SQL baseline: `cd RaySpace3DBaslines/SQL && ./scripts/build.sh`
3. Initialize SQL database: `cd RaySpace3DBaslines/SQL && ./scripts/init_db.sh`
4. Build RaySpace3D: `cd RaySpace3D/build && cmake .. && make`

## Next Steps

To run the benchmark:
```bash
cd /sc/home/anton.hackl/Spatial_Data_Management/first_benchmark
./run_benchmark.sh
```

Or test individual components:
```bash
# Test rescaling
./example_rescaling.sh

# Test specific approach
./run_benchmark.sh cgal
```
