# Correctness Test Suite

This directory contains correctness-oriented benchmark scripts and datasets for:
- overlap
- intersection
- containment

Included approaches:
- RaySpace
- CGAL
- TOUCH (overlap only)

TDBase is intentionally not part of this suite yet.

## Datasets

Two dataset families are used for each operation:
1. `cubes_20k_sel_0_001`: generated cubes with 20k objects in A, 20k objects in B, selectivity 0.001.
2. `manual`: handcrafted non-touching scenarios.

Manual scenarios are operation-specific:
- Overlap manual: A has 1 object, B has 10 objects.
- Expected overlap pairs: 5.
- One additional B object is strictly contained in A and is intentionally excluded from overlap expectation.
- No touching-only configurations are included.

- Intersection manual: A has 1 object, B has 10 objects.
- Expected intersection pairs: 5.
- This includes 4 partial overlaps and 1 strict containment.
- No touching-only configurations are included.

- Containment manual: A has 1 object, B has 10 objects.
- Expected containment pairs (B in A): 5.
- One additional B object partially overlaps A but is not contained and is intentionally excluded.
- No touching-only configurations are included.

All generated expectations are written to:
- `data/raw/manual_expected_results.json`

## Scripts

- `generate_datasets.py`
- Generates the 20k/20k/0.001 dataset and all manual datasets.

- `snapshot_rayspace_ground_truth.py`
- Runs RaySpace now on the 20k dataset and writes a snapshot JSON to:
  - `ground_truth/rayspace_20k_sel0_001_current.json`

- `run_correctness.py`
- Runs correctness checks and writes report JSON to:
  - `runs/correctness_<timestamp>.json`
  - `runs/correctness_latest.json`

- `run_overlap_disagreement_analysis.py`
- Runs RaySpace direct-estimation overlap and CGAL overlap on the cubes_20k dataset,
  computes pair disagreements, samples up to N mismatches (default 10,000), and
  adjudicates sampled pairs using a cube AABB truth predicate with strict containment
  excluded from overlap semantics.

- `run_intersection_disagreement_analysis.py`
- Runs RaySpace two-pass intersection and CGAL intersection on the cubes_20k dataset,
  computes pair disagreements, samples up to N mismatches (default 10,000), and
  adjudicates sampled pairs using a cube AABB intersection predicate.

## Usage

Activate your conda environment first, then run:

```bash
conda activate spatial_benchmark
python benchmarks/correctness_tests/generate_datasets.py
python benchmarks/correctness_tests/snapshot_rayspace_ground_truth.py
```

After snapshotting (or to use the currently frozen values), run:

```bash
conda activate spatial_benchmark
python benchmarks/correctness_tests/run_correctness.py --operations overlap intersection containment --approaches rayspace cgal touch
```

To inspect overlap disagreements and estimate which engine is correct on sampled mismatches:

```bash
conda activate spatial_benchmark
python benchmarks/correctness_tests/run_overlap_disagreement_analysis.py --max-eval-pairs 10000 --seed 42
```

To inspect intersection disagreements and estimate which engine is correct on sampled mismatches:

```bash
conda activate spatial_benchmark
python benchmarks/correctness_tests/run_intersection_disagreement_analysis.py --max-eval-pairs 10000 --seed 42
```

Artifacts for the disagreement analysis are written to:
- `runs/overlap_disagreement_<timestamp>/summary.json`
- `runs/overlap_disagreement_<timestamp>/sampled_pair_decisions.csv`
- `runs/overlap_disagreement_latest.json`
- `runs/intersection_disagreement_<timestamp>/summary.json`
- `runs/intersection_disagreement_<timestamp>/sampled_pair_decisions.csv`
- `runs/intersection_disagreement_latest.json`

Important:
- The script requires CGAL overlap binary support for `--output-csv`.
- If you changed `baselines/RaySpace3DBaselines/CGAL/src/overlap_query.cpp`, rebuild via `./build_all.sh`.
- Intersection disagreement analysis requires CGAL intersection binary support for `--output-csv`.
- If you changed `baselines/RaySpace3DBaselines/CGAL/src/intersection_query.cpp`, rebuild via `./build_all.sh --only cgal`.

Notes:
- Current RaySpace 20k constants are already hardcoded in `run_correctness.py`.
- If `touch` is requested for intersection or containment, it is ignored for those operations.
