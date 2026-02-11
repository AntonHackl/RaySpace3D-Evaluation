# Benchmark Evaluation

This directory contains tools for evaluating and comparing benchmark results.

## Usage

### Evaluate all benchmarks in results directory

```bash
python evaluate_benchmarks.py --results-dir ../results --output-dir figures
```

### Evaluate specific benchmark files

```bash
python evaluate_benchmarks.py --results ../results/grid_benchmark_*.json --output-dir figures
```

### Select timing metric

For RaySpace approaches (includes upload + query + download):
```bash
python evaluate_benchmarks.py --metric rayspace_total_ms
```

For query-only timing:
```bash
python evaluate_benchmarks.py --metric query_ms
```

## Output

The evaluation script generates:

1. **Statistics Table** - Printed to console with mean, median, std dev, min, max for each approach
2. **mean_comparison.png** - Bar chart comparing mean query times across all benchmarks
3. **median_comparison.png** - Bar chart comparing median query times across all benchmarks
4. **latest_benchmark_detailed.png** - Detailed view of most recent benchmark with labeled bars

## Timing Metrics

- `query_ms`: Raw query execution time only
- `rayspace_total_ms`: Total time for RaySpace approaches (upload + query + download)
  - For Raytracer: upload_points + upload_geometry + query + output
  - For FilterRefine: upload_bbox + upload_geometry + upload_points + bbox_query + query + output

The `rayspace_total_ms` metric provides a fair comparison by excluding preprocessing time (which runs once) and including only the per-query costs.
