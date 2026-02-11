#!/bin/bash
# =============================================================================
# build_all.sh — Build all C/C++/CUDA applications in RaySpace3D-Evaluation
# =============================================================================
# Builds every CMake project:
#   1. src/RaySpace3D/preprocess  (preprocess_dataset)
#   2. src/RaySpace3D/query       (raytracer, raytracer_filter_refine,
#                                  raytracer_mesh_overlap, raytracer_mesh_intersection,
#                                  raytracer_intersection_estimated, raytracer_overlap_estimated)
#   3. baselines/RaySpace3DBaselines/CGAL  (cgal_query, cgal_overlap)
#   4. baselines/RaySpace3DBaselines/CUDA  (cuda_query)
#   5. baselines/RaySpace3DBaselines/SQL   (spatial_query)
#
# Usage:
#   ./build_all.sh              # Build everything
#   ./build_all.sh --clean      # Clean build (remove build dirs first)
#   ./build_all.sh --only X     # Build only component X
#                               #   X = preprocess | query | cgal | cuda | sql
#   ./build_all.sh --jobs N     # Use N parallel jobs (default: $(nproc))
# =============================================================================

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Defaults
CLEAN=false
ONLY=""
JOBS=$(nproc 2>/dev/null || echo 4)
EXTRA_ARGS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --clean)   CLEAN=true; shift ;;
        --only)    ONLY="$2"; shift 2 ;;
        --jobs)    JOBS="$2"; shift 2 ;;
        --)        shift; EXTRA_ARGS="$*"; break ;;
        -h|--help)
            echo "Usage: $0 [--clean] [--only COMPONENT] [--jobs N] [-- EXTRA_CMAKE_ARGS]"
            echo ""
            echo "Components: preprocess, query, cgal, cuda, sql"
            exit 0
            ;;
        *) 
            if [[ "$1" == -D* ]]; then
                EXTRA_ARGS="${EXTRA_ARGS} $1"
                shift
            else
                echo "Unknown option: $1"
                exit 1
            fi
            ;;
    esac
done

SUCCESS=0
FAIL=0
SKIP=0
BUILT_COMPONENTS=()
FAILED_COMPONENTS=()

# -----------------------------------------------------------------------
# Build helper
# -----------------------------------------------------------------------
build_cmake_project() {
    local name="$1"
    local src_dir="$2"
    local build_dir="${src_dir}/build"
    local conda_env="${3:-}"
    local extra_cmake_args="${4:-}"
    
    echo ""
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  Building: ${CYAN}${name}${NC}"
    echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  Source:  ${src_dir}"
    echo -e "  Build:   ${build_dir}"
    if [[ -n "$conda_env" ]]; then
        echo -e "  Conda:   ${conda_env}"
    fi
    echo ""

    # Check if source dir exists
    if [[ ! -f "${src_dir}/CMakeLists.txt" ]]; then
        echo -e "  ${RED}ERROR${NC}: CMakeLists.txt not found in ${src_dir}"
        ((FAIL++)) || true
        FAILED_COMPONENTS+=("$name")
        return 1
    fi

    # Activate conda environment if specified
    if [[ -n "$conda_env" ]] && [[ -z "${NO_CONDA:-}" ]]; then
        # Try to find conda if not in PATH
        if ! command -v conda &> /dev/null; then
            if [[ -f "/sc/home/anton.hackl/conda3/bin/conda" ]]; then
                export PATH="/sc/home/anton.hackl/conda3/bin:$PATH"
            fi
        fi

        if ! conda env list 2>/dev/null | grep -q "^${conda_env} "; then
            echo -e "  ${YELLOW}WARNING${NC}: Conda environment '${conda_env}' not found."
            echo -e "  Attempting build with current environment..."
        else
            echo -e "  Activating conda environment: ${conda_env}"
            # Initialize conda for the current shell
            eval "$(conda shell.bash hook)"
            conda activate "$conda_env" || {
                echo -e "  ${RED}ERROR${NC}: Failed to activate conda env '${conda_env}'"
                return 1
            }
            echo -e "  ${GREEN}✓${NC} Environment activated: $(which g++)"

            # --- Workaround: conda cross-compiler vs enroot/container sysroot ---
            # Conda environments with compiler packages (e.g. gxx_linux-64) ship a
            # cross-compiler toolchain whose linker references its own sysroot
            # (x86_64-conda-linux-gnu/sysroot).  Inside an enroot container the
            # system glibc differs, causing GLIBC_PRIVATE link errors.
            # Fix: if the active g++ is conda's cross-compiler, tell CMake to use
            # the system compiler/linker instead while keeping conda for library
            # discovery (CGAL, Boost, tinyobjloader, etc.).
            local active_gxx
            active_gxx="$(which g++ 2>/dev/null || true)"
            if [[ "$active_gxx" == *conda* ]]; then
                if [[ -x /usr/bin/g++ ]]; then
                    echo -e "  ${YELLOW}NOTE${NC}: Detected conda cross-compiler; overriding with system compiler"
                    extra_cmake_args="${extra_cmake_args} -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_C_COMPILER=/usr/bin/gcc"
                fi
            fi
        fi
    elif [[ -n "${NO_CONDA:-}" ]]; then
        echo -e "  ${CYAN}INFO${NC}: Skipping conda activation (NO_CONDA set)"
    fi

    # Clean if requested
    if $CLEAN && [[ -d "$build_dir" ]]; then
        echo -e "  ${YELLOW}Cleaning${NC} build directory..."
        rm -rf "$build_dir"
    fi

    mkdir -p "$build_dir"

    # CMake configure
    echo -e "  [1/2] ${CYAN}Configuring${NC} with CMake..."
    if cmake -S "$src_dir" -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        $extra_cmake_args \
        2>&1 | while IFS= read -r line; do echo "        $line"; done; then
        echo -e "  ${GREEN}✓${NC} Configure succeeded"
    else
        echo -e "  ${RED}✗${NC} Configure FAILED"
        ((FAIL++)) || true
        FAILED_COMPONENTS+=("$name")
        return 1
    fi

    # CMake build
    echo -e "  [2/2] ${CYAN}Building${NC} with ${JOBS} parallel jobs..."
    if cmake --build "$build_dir" --config Release -j "$JOBS" \
        2>&1 | while IFS= read -r line; do echo "        $line"; done; then
        echo -e "  ${GREEN}✓${NC} Build succeeded"
        ((SUCCESS++)) || true
        BUILT_COMPONENTS+=("$name")
    else
        echo -e "  ${RED}✗${NC} Build FAILED"
        ((FAIL++)) || true
        FAILED_COMPONENTS+=("$name")
        return 1
    fi

    # List built executables
    echo ""
    echo -e "  ${GREEN}Built executables:${NC}"
    if [[ -d "${build_dir}/bin" ]]; then
        find "${build_dir}/bin" -maxdepth 1 -type f -executable 2>/dev/null | sort | while read -r exe; do
            echo -e "    • $(basename "$exe")"
        done
    else
        find "${build_dir}" -maxdepth 1 -type f -executable 2>/dev/null | sort | while read -r exe; do
            echo -e "    • $(basename "$exe")"
        done
    fi
    echo ""
}

should_build() {
    local component="$1"
    [[ -z "$ONLY" ]] || [[ "$ONLY" == "$component" ]]
}

echo "=============================================="
echo "  Build All — RaySpace3D-Evaluation"
echo "=============================================="
echo ""
echo "Root:     $SCRIPT_DIR"
echo "Jobs:     $JOBS"
echo "Clean:    $CLEAN"
if [[ -n "$ONLY" ]]; then
    echo "Only:     $ONLY"
fi

# -----------------------------------------------------------------------
# 1. RaySpace3D Preprocess
# -----------------------------------------------------------------------
if should_build "preprocess"; then
    build_cmake_project \
        "RaySpace3D Preprocess" \
        "$SCRIPT_DIR/src/RaySpace3D/preprocess" \
        "rayspace3d_preprocess" \
        "$EXTRA_ARGS" \
        || true
fi

# -----------------------------------------------------------------------
# 2. RaySpace3D Query (requires OptiX SDK)
# -----------------------------------------------------------------------
if should_build "query"; then
    # Check for OptiX SDK — FindOptiX.cmake reads $ENV{OptiX_INSTALL_DIR} on Linux
    if [[ -z "${OptiX_INSTALL_DIR:-}" ]]; then
        if [[ -d "/opt/optix" ]]; then
            export OptiX_INSTALL_DIR="/opt/optix"
        elif [[ -d "$HOME/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64" ]]; then
            export OptiX_INSTALL_DIR="$HOME/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64"
        fi
    fi

    if [[ -n "${OptiX_INSTALL_DIR:-}" ]]; then
        echo -e "\n  ${GREEN}Using OptiX SDK:${NC} $OptiX_INSTALL_DIR"
    else
        echo -e "\n  ${YELLOW}WARNING${NC}: OptiX SDK not found. Set OptiX_INSTALL_DIR env var."
        echo -e "  The query build may fail without it."
    fi

    build_cmake_project \
        "RaySpace3D Query" \
        "$SCRIPT_DIR/src/RaySpace3D/query" \
        "rayspace3d_query_linux" \
        "$EXTRA_ARGS" \
        || true
fi

# -----------------------------------------------------------------------
# 3. CGAL Baseline
# -----------------------------------------------------------------------
if should_build "cgal"; then
    build_cmake_project \
        "CGAL Baseline" \
        "$SCRIPT_DIR/baselines/RaySpace3DBaselines/CGAL" \
        "cgal_spatial" \
        "$EXTRA_ARGS" \
        || true
fi

# -----------------------------------------------------------------------
# 4. CUDA Baseline
# -----------------------------------------------------------------------
if should_build "cuda"; then
    build_cmake_project \
        "CUDA Baseline" \
        "$SCRIPT_DIR/baselines/RaySpace3DBaselines/CUDA" \
        "cuda_baseline" \
        "$EXTRA_ARGS" \
        || true
fi

# -----------------------------------------------------------------------
# 5. SQL Baseline
# -----------------------------------------------------------------------
if should_build "sql"; then
    build_cmake_project \
        "SQL Baseline" \
        "$SCRIPT_DIR/baselines/RaySpace3DBaselines/SQL" \
        "spatial3d" \
        "$EXTRA_ARGS" \
        || true
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
echo ""
echo "=============================================="
echo "  Build Summary"
echo "=============================================="
echo ""
echo -e "  ${GREEN}Succeeded${NC}: $SUCCESS"
echo -e "  ${RED}Failed${NC}:    $FAIL"
echo ""

if [[ ${#BUILT_COMPONENTS[@]} -gt 0 ]]; then
    echo -e "  ${GREEN}Built:${NC}"
    for comp in "${BUILT_COMPONENTS[@]}"; do
        echo -e "    ✓ $comp"
    done
fi

if [[ ${#FAILED_COMPONENTS[@]} -gt 0 ]]; then
    echo -e "  ${RED}Failed:${NC}"
    for comp in "${FAILED_COMPONENTS[@]}"; do
        echo -e "    ✗ $comp"
    done
fi

echo ""
if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}Some builds failed! Check output above for details.${NC}"
    exit 1
else
    echo -e "${GREEN}All builds succeeded!${NC}"
fi
