# Mesh Overlap Benchmark - Results and Visualization

## Directory Structure

```
mesh_overlap_benchmark/
├── runs/           # Timestamped benchmark results (JSON)
├── figures/        # Generated visualizations (PNG, PDF)
├── timings/        # Temporary timing files
└── data/           # Dataset files (.dt, .pre)
```

## Running Benchmarks

Results are automatically saved to `runs/` directory with metadata:

```bash
# Run with default 10 runs
./run_benchmark.sh --dataset nuclei_join

# Run with custom number of runs
./run_benchmark.sh --dataset medium --runs 20
```

### Result File Format

Each run generates a timestamped JSON file in `runs/`:
- Filename: `{dataset}_{num_runs}runs_{timestamp}.json`
- Example: `nuclei_join_20runs_20260129_011630.json`

The JSON contains:
```json
{
  "metadata": {
    "timestamp": "20260129_011630",
    "dataset": "nuclei_join",
    "file1": "medium_n_nv50_nu200_vs100_r30.dt",
    "file2": "medium_n2_n_nv50_nu200_vs100_r30.dt",
    "num_runs": 20
  },
  "results": {
    "CGAL": {
      "mean": 2187.8,
      "min": 2180.4,
      "max": 2195.2,
      "std": 4.3,
      "raw_times": [...]
    },
    ...
  }
}
```

## Visualizing Results

Generate publication-quality figures from benchmark results:

```bash
# Visualize a specific run
python visualize_results.py runs/nuclei_join_20runs_20260129_011630.json

# Specify custom output directory
python visualize_results.py runs/medium_10runs_20260129_010000.json --output-dir custom_figures/
```

### Generated Visualizations

The script creates two plots:

1. **Bar Chart with Error Bars**: Shows mean query time with standard deviation as error bars
2. **Min/Max/Mean Comparison**: Shows the range (min-max) with markers for min, max, and mean values

Output formats:
- High-resolution PNG (300 DPI)
- Vector PDF (for publications)

### Output Statistics

The visualization script also prints comprehensive statistics:
- Mean, Min, Max, Std Dev for each adapter
- Coefficient of Variation (CV%) - measure of relative variability

## Example Workflow

```bash
# 1. Run benchmark
./run_benchmark.sh --dataset nuclei_join --runs 20

# 2. Visualize results (use the most recent JSON file)
python visualize_results.py runs/nuclei_join_20runs_*.json

# 3. View figures
ls figures/
# Output: nuclei_join_20runs_20260129_011630.png
#         nuclei_join_20runs_20260129_011630.pdf
```

## Comparing Multiple Runs

All results are preserved in the `runs/` directory, allowing you to:
- Compare different datasets
- Track performance changes over time
- Analyze different configurations

Use standard JSON tools or custom scripts to compare multiple result files.
