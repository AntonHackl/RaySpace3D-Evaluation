# Benchmark Visualization Scripts

This document describes the visualization scripts created for the sphere benchmark results.

## Scripts

### 1. `visualize_complexity.py`
**Purpose**: Visualizes how query runtime scales with mesh complexity (number of faces)

**What it does**:
- Loads all complexity benchmark results (`grid_benchmark_sphere_complexity_stage_*.json`)
- Automatically selects the newest timestamp when multiple results exist for the same stage
- Generates a single plot comparing all approaches (CGAL, CUDA, FilterRefine, Raytracer) across 10 complexity stages
- X-axis: Number of faces (48 to 408,320)
- Y-axis: Runtime in milliseconds (log scale)
- Prints a summary table with all data

**Output**: `results/visualizations/sphere_complexity_comparison.png`

**Usage**:
```bash
conda activate spatial_benchmark
python visualize_complexity.py
```

### 2. `visualize_selectivity.py`
**Purpose**: Visualizes how query runtime scales with selectivity for different point counts

**What it does**:
- Loads all selectivity benchmark results (`grid_benchmark_sphere_selectivity_*.json`)
- Automatically selects the newest timestamp when multiple results exist
- Generates **separate plots** for each point count (10M, 200M, 500M)
- X-axis: Selectivity (0.0001 to 0.8, log scale)
- Y-axis: Runtime in milliseconds (log scale)
- Prints summary tables for each point count

**Outputs**:
- `results/visualizations/sphere_selectivity_10M.png`
- `results/visualizations/sphere_selectivity_200M.png`
- `results/visualizations/sphere_selectivity_500M.png`

**Usage**:
```bash
conda activate spatial_benchmark
python visualize_selectivity.py
```

## Complexity Stages Reference

| Stage | Vertices | Faces   | Complexity Level |
|-------|----------|---------|------------------|
| 1     | 26       | 48      | Very Low         |
| 2     | 2,966    | 5,928   | Low              |
| 3     | 10,806   | 21,608  | Medium-Low       |
| 4     | 23,546   | 47,088  | Medium           |
| 5     | 41,186   | 82,368  | Medium-High      |
| 6     | 63,904   | 127,804 | High             |
| 7     | 91,379   | 182,754 | Very High        |
| 8     | 123,754  | 247,504 | Ultra High       |
| 9     | 161,029  | 322,054 | Extreme          |
| 10    | 204,162  | 408,320 | Maximum          |

## Key Features

1. **Automatic timestamp selection**: Both scripts automatically find and use the most recent results when multiple runs exist
2. **Consistent styling**: Both scripts use consistent colors and markers for each approach
3. **High-quality output**: Plots are saved at 300 DPI for publication quality
4. **Summary tables**: Both scripts print detailed summary tables to the console
5. **Log scales**: Appropriate log scales are used to handle the wide range of values

## Environment

Both scripts should be run in the `spatial_benchmark` conda environment:
```bash
conda activate spatial_benchmark
```

Required packages:
- matplotlib
- numpy
- json (standard library)
- glob (standard library)

## Results Summary

### Complexity Results (100M points, ~1% selectivity)
- **Raytracer**: Fastest, scales very well with complexity (~11-15 ms across all stages)
- **FilterRefine**: Second fastest, very consistent (~32-38 ms across all stages)
- **CUDA**: Good for simple meshes, degrades with complexity (95 ms → 9,326 ms)
- **CGAL**: Slowest but most consistent behavior (916 ms → 2,354 ms)

### Selectivity Results
Different approaches show different scaling characteristics with selectivity and point count. The visualizations clearly show:
- **Small selectivity** (<1%): Raytracer and FilterRefine perform best
- **Large selectivity** (>10%): Performance differences amplify with point count
- **200M and 500M points**: The gap between approaches widens significantly
