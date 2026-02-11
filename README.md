# RaySpace3D-Evaluation (Monorepo)

This repository serves as the central hub for the RaySpace3D project, its baselines, and evaluation benchmarks. It consolidates the core library, alternative implementations, and experimental setups into a single structure.

## Structure

- **src/RaySpace3D/**: The core RaySpace3D library (Git Submodule). Contains the primary implementation using OptiX for hardware-accelerated queries.
- **baselines/RaySpace3DBaselines/**: Alternative implementations (Git Submodule). Includes CPU-based (CGAL), SQL, and other baseline methods for comparison.
- **benchmarks/**: Evaluation scripts and datasets.
  - **pip/**: Point-in-Polygon (PIP) benchmarks (formerly `first_benchmark`).
  - **mesh_overlap/**: Mesh Overlap benchmarks (formerly `mesh_overlap_benchmark`).

## Usage

### Prerequisites

Since this repository uses Git submodules, ensure you initialize them after cloning:

```bash
git clone <this-repo-url>
cd RaySpace3D-Evaluation
git submodule update --init --recursive
```

### Running Benchmarks

Navigate to the respective benchmark directory and follow the instructions in their local READMEs.

**Example (PIP Benchmark):**
```bash
cd benchmarks/pip
./run_benchmark.sh
```

**Example (Mesh Overlap Benchmark):**
```bash
cd benchmarks/mesh_overlap
./run_benchmark.sh
```

### Developing Core Library

The `src/RaySpace3D` directory is a submodule pointing to the `RaySpace3D` repository. Changes made here can be committed and pushed to the upstream `RaySpace3D` repository.

```bash
cd src/RaySpace3D
# Make changes
git add .
git commit -m "Update core feature"
git push
```
