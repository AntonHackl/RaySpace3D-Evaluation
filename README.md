# RaySpace3D-Evaluation

Evaluation framework for RaySpace3D, benchmarking GPU-accelerated spatial joins against CPU and database baselines (CGAL, CUDA, PostGIS) on point-in-mesh and mesh overlap workloads.

## Repository Structure

```
RaySpace3D-Evaluation/
├── src/RaySpace3D/              # Core RaySpace3D implementation
│   ├── preprocess/              #   Dataset preprocessing tool
│   ├── query/                   #   OptiX-based query executables
│   └── common/                  #   Shared library (geometry types, I/O, timers)
├── baselines/RaySpace3DBaselines/
│   ├── CGAL/                    # CPU baseline: CGAL AABB tree + Side_of_triangle_mesh
│   ├── CUDA/                    # GPU baseline: brute-force Möller-Trumbore ray-triangle
│   ├── SQL/                     # Database baseline: PostgreSQL + PostGIS
│   └── tdbase/                  # TDBase: .dt file format support for mesh overlap
├── benchmarks/
│   ├── pip/                     # Point-in-polygon benchmark (27-position grid test)
│   ├── mesh_overlap/            # Mesh overlap join benchmark
│   ├── mesh_containment/        # Mesh containment validation
│   └── mesh_query_comparison/   # Unified overlap/intersection/containment comparison
├── datasets/                    # Shared test datasets
├── build_all.sh                 # Build all components
└── test_all.sh                  # Smoke-test all built executables
```

## Prerequisites

- **Linux** with NVIDIA GPU (compute capability >= 7.5)
- **NVIDIA OptiX SDK 7.5+** — set `OptiX_INSTALL_DIR` to the SDK root
- **CUDA Toolkit 12.x** with compatible driver
- **Conda** (Miniforge or Miniconda)
- **PostgreSQL 16 + PostGIS 3.4** (only for the SQL baseline)

## Building

### Quick Start — Build Everything

```bash
./build_all.sh
```

This builds all five components (preprocess, query, CGAL, CUDA, SQL) using their respective conda environments. Each component's conda environment is created from its `environment.yml` if not already present.

**Options:**

```bash
./build_all.sh --clean                # Clean rebuild (removes build directories first)
./build_all.sh --only preprocess      # Build only one component
./build_all.sh --only query           # Components: preprocess, query, cgal, cuda, sql, tdbase
./build_all.sh --jobs 16              # Parallel compilation jobs (default: nproc)
```

### Building Individual Components

Each component can also be built manually. All components use CMake:

**RaySpace3D Preprocess:**
```bash
cd src/RaySpace3D/preprocess
conda env create -f environment-linux.yml
conda activate rayspace3d_preprocess_linux
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

**RaySpace3D Query** (requires OptiX SDK):
```bash
cd src/RaySpace3D/query
conda env create -f environment-linux.yml
conda activate rayspace3d_query_linux
export OptiX_INSTALL_DIR=/path/to/NVIDIA-OptiX-SDK
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

**CGAL Baseline:**
```bash
cd baselines/RaySpace3DBaselines/CGAL
conda env create -f environment.yml
conda activate cgal_spatial
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

**CUDA Baseline:**
```bash
cd baselines/RaySpace3DBaselines/CUDA
conda env create -f environment.yml
conda activate cuda_baseline
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

**SQL Baseline** (requires PostgreSQL):
```bash
cd baselines/RaySpace3DBaselines/SQL
conda env create -f environment.yml
conda activate spatial3d
./scripts/init_db.sh   # Initialize the PostGIS database
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
```

### Verifying the Build

```bash
./test_all.sh                    # Test all components
./test_all.sh --only query       # Test a specific component
```

## Running Benchmarks

### Mesh Overlap Benchmark

Compares mesh overlap join performance across approaches. Located in `benchmarks/mesh_overlap/`.

```bash
cd benchmarks/mesh_overlap
conda activate mesh_overlap_benchmark   # or create from environment.yml
./run_benchmark.sh
```

**Direct invocation with options:**
```bash
python benchmark.py \
    --dataset small \
    --approaches raytracer cgal \
    --runs 5 \
    --timeout 300 \
    --grid-resolution 128
```

**Available datasets:** `small`, `medium`, `nu200`, `nu400`, `nuclei_join`, `cubes_100k`, `cubes_1m`, etc. See `DATASETS.md` for full descriptions.

**Available approaches:** `raytracer` (modes: exact, estimated, estimate_only, direct_estimation), `cgal`, `touch`, `tdbase`.

**Output:** JSON results in `runs/`, logs in `runs/logs/`.

**Additional experiments:**
- `run_scalability.py` — vary dataset size to measure scaling
- `run_nu_scalability.py` — vary number of objects
- `run_cube_scalability.py` — synthetic cube datasets
- `run_breakdown_benchmark.py` — per-phase timing breakdown
- `selectivity_test.py` — selectivity estimation accuracy
- `visualize_results.py` — generate plots from result JSON files

### Mesh Query Comparison Benchmark

Compares mesh overlap, mesh intersection, and mesh containment in a single run and writes isolated run artifacts under `benchmarks/mesh_query_comparison/runs/<benchmark_name>_<timestamp>/results.json`.

```bash
cd benchmarks/mesh_query_comparison

# Standard defaults compare all three query types
python run_nu_scalability.py
python run_cube_scalability.py
python selectivity_test.py
python run_mesh_complexity_benchmark.py

# Optional subset selection
python run_nu_scalability.py --queries overlap intersection
python run_nu_scalability.py --queries intersection containment
```

Key options shared by the runners:
- `--queries` (or `--approaches` alias): choose which query types to compare
- `--runs`, `--warmup-runs`, `--timeout`, `--grid-resolution`
- `--overlap-mode`, `--intersection-mode` for RaySpace implementation selection

### Point-in-Polygon (PIP) Benchmark

Measures point-in-mesh containment query performance using a 3x3x3 spatial grid (27 query positions). Located in `benchmarks/pip/`.

```bash
cd benchmarks/pip
conda activate spatial_benchmark   # or create from environment.yml
./run_benchmark.sh
```

**Direct invocation:**
```bash
python grid_benchmark.py \
    --approaches raytracer filter_refine cgal cuda \
    --runs 5 \
    --mesh workspace/Cube_large.obj \
    --points workspace/uniform_points_10000000.wkt
```

**Available approaches:** `raytracer`, `filter_refine`, `cgal`, `cuda`, `sql`.

**Output:** `results/grid_benchmark.json` + optional PNG visualizations.

**Additional experiments:**
- `spheres_benchmark.sh` / `cubes_benchmark.sh` — standard benchmarks with sphere/cube meshes
- `spheres_complexity_benchmark.sh` — vary mesh complexity (triangle count)
- `spheres_selectivity_benchmark.sh` — vary spatial selectivity
- `visualize_complexity.py` / `visualize_selectivity.py` — generate plots

## Components

### RaySpace3D (`src/RaySpace3D/`)

The core system. See [src/RaySpace3D/README.md](src/RaySpace3D/README.md) for detailed documentation of:
- All query executables and their algorithms
- Preprocessing options and binary file format
- Selectivity estimation methodology

**Key executables:**
| Executable | Purpose |
|---|---|
| `preprocess_dataset` | Convert .obj/.dt/.wkt to binary format |
| `raytracer` | Point-in-mesh containment |
| `raytracer_filter_refine` | Two-phase filter-refine containment |
| `raytracer_mesh_overlap` | Mesh overlap join |
| `raytracer_mesh_intersection` | Full mesh intersection join |
| `raytracer_containment` | Mesh containment join |
| `raytracer_overlap_estimated` | Overlap with selectivity estimation |
| `raytracer_intersection_estimated` | Intersection with selectivity estimation |

### Baselines (`baselines/RaySpace3DBaselines/`)

| Baseline | Technology | Supported Queries | Executable |
|---|---|---|---|
| **CGAL** | CGAL AABB tree, C++17, OpenMP | Point-in-mesh, mesh overlap | `cgal_query`, `cgal_overlap` |
| **CUDA** | Pure CUDA, Möller-Trumbore, BVH filter | Point-in-mesh | `cuda_query` |
| **SQL** | PostgreSQL 16 + PostGIS 3.4 | Point-in-mesh | `spatial_query` |
| **TDBase** | .dt spatial file format | Mesh overlap | (library, used via adapter) |

All baselines accept `.obj` mesh files and `.wkt` point files (SQL loads points into a database first).

### Benchmark Framework (`benchmarks/`)

The benchmarks use a Python adapter pattern: each system implements a common interface (`SpatialQueryAdapter` for PIP, `OverlapBenchmarkAdapter` for mesh overlap), enabling uniform measurement and comparison. Adapters handle conda environment activation, executable invocation, and result parsing.

**Adapter files:**
- `benchmarks/pip/adapters/` — `raytracer_adapter.py`, `filter_refine_adapter.py`, `cgal_adapter.py`, `cuda_adapter.py`, `sql_adapter.py`
- `benchmarks/mesh_overlap/adapters/` — `raytracer_adapter.py`, `cgal_adapter.py`, `touch_adapter.py`, `tdbase_adapter.py`

## Typical End-to-End Workflow

1. **Build** all components: `./build_all.sh`
2. **Verify** the build: `./test_all.sh`
3. **Prepare data**: place `.obj` or `.dt` files in `benchmarks/*/data/` or `datasets/`
4. **Run benchmarks**: `cd benchmarks/mesh_overlap && ./run_benchmark.sh`
5. **Analyze results**: `python visualize_results.py` or inspect JSON in `runs/`
