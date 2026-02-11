# RaySpace3D-Evaluation

Monorepo for evaluating RaySpace3D against baselines (CGAL, CUDA, PostGIS) on various benchmarks.

## Repository Structure

- **`src/RaySpace3D`**: Core RaySpace3D implementation.
  - `preprocess`: Data preprocessing tools.
  - `query`: Ray tracing query engine (Requires OptiX).
- **`baselines/RaySpace3DBaselines`**: Baseline implementations.
  - `CGAL`: CPU-based geometry processing.
  - `CUDA`: Naive CUDA implementation.
  - `SQL`: PostGIS-based implementation.
- **`benchmarks`**: Benchmark scripts and data.
  - `pip`: Point-in-Polygon benchmark.
  - `mesh_overlap`: Mesh overlap benchmark.
- **`datasets`**: Shared datasets (e.g., `range_query`).

## Prerequisites

- **Linux** environment.
- **Conda** for environment management.
- **NVIDIA Driver & CUDA** (for GPU components).
- **OptiX SDK** (Required for `RaySpace3D/query`).

## Setup

1. **Clone the repository:**
   ```bash
   git clone --recursive <repo_url>
   cd RaySpace3D-Evaluation
   ```

2. **Copy Data:**
   Use the `copy_data.sh` script to populate the monorepo with necessary data files.
   ```bash
   ./copy_data.sh
   ```

3. **Build:**
   Use the `build_all.sh` script to build all components.
   ```bash
   ./build_all.sh
   # Or build specific components:
   ./build_all.sh --only preprocess
   ./build_all.sh --only query
   ```

4. **Test:**
   Run `test_all.sh` to verify built executables and data presence.
   ```bash
   ./test_all.sh
   ```

## Running Benchmarks

### Mesh Overlap
Located in `benchmarks/mesh_overlap`.
```bash
cd benchmarks/mesh_overlap
./run_benchmark.sh
```

### Point-in-Polygon (PIP)
Located in `benchmarks/pip`.
```bash
cd benchmarks/pip
./run_benchmark.sh
```
