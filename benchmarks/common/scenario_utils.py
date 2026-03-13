import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARKS_DIR.parent
RAYSPACE_DIR = REPO_ROOT / "src" / "RaySpace3D"
SHARED_DATA_ROOT = BENCHMARKS_DIR / "data_shared"
GENERATE_CUBES_SCRIPT = RAYSPACE_DIR / "scripts" / "generate_cubes_by_selectivity.py"
GENERATE_SPHERES_BIN = RAYSPACE_DIR / "scripts" / "cpp_generator" / "generate_spheres"


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sanitize_float_token(value: float) -> str:
    return str(value).replace(".", "_")


def canonical_cube_pair_paths(
    raw_dir: Path,
    *,
    num_cubes_a: int,
    num_cubes_b: int,
    min_size: float,
    max_size: float,
    selectivity: float,
    seed: int,
    grid_resolution: int | None = None,
) -> Tuple[Path, Path]:
    min_tok = sanitize_float_token(min_size)
    max_tok = sanitize_float_token(max_size)
    sel_tok = sanitize_float_token(selectivity)
    stem = (
        f"cubes_na{num_cubes_a}_nb{num_cubes_b}_"
        f"min{min_tok}_max{max_tok}_sel{sel_tok}_seed{seed}"
    )
    if grid_resolution is not None:
        stem += f"_g{grid_resolution}"
    return raw_dir / f"{stem}_a.obj", raw_dir / f"{stem}_b.obj"


def canonical_sphere_pair_paths(
    raw_dir: Path,
    *,
    template_name: str,
    num_objects: int,
    min_size: float,
    max_size: float,
    selectivity: float,
    seed: int,
    grid_resolution: int | None = None,
) -> Tuple[Path, Path]:
    min_tok = sanitize_float_token(min_size)
    max_tok = sanitize_float_token(max_size)
    sel_tok = sanitize_float_token(selectivity)
    template_token = template_name.replace(".obj", "").replace(" ", "_")
    stem = (
        f"spheres_tpl{template_token}_n{num_objects}_"
        f"min{min_tok}_max{max_tok}_sel{sel_tok}_seed{seed}"
    )
    if grid_resolution is not None:
        stem += f"_g{grid_resolution}"
    return raw_dir / f"{stem}_a.obj", raw_dir / f"{stem}_b.obj"


def compute_universe_for_selectivity(target_selectivity: float, min_size: float, max_size: float) -> float:
    if target_selectivity <= 0:
        raise ValueError("Target selectivity must be positive")
    avg_size = (min_size + max_size) / 2.0
    return (2.0 * avg_size) / (target_selectivity ** (1.0 / 3.0))


def get_shared_data_dirs(scenario_name: str) -> Dict[str, Path]:
    scenario_root = SHARED_DATA_ROOT / scenario_name
    raw_dir = scenario_root / "raw"
    preprocessed_dir = scenario_root / "preprocessed"
    timings_dir = scenario_root / "timings"
    for d in (raw_dir, preprocessed_dir, timings_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "root": scenario_root,
        "raw": raw_dir,
        "preprocessed": preprocessed_dir,
        "timings": timings_dir,
    }


def run_cmd(cmd, desc: str):
    print(f"\n>>> {desc}")
    print("    " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True)


def ensure_cube_pair_dataset(
    output_a: Path,
    output_b: Path,
    *,
    num_cubes_a: int,
    num_cubes_b: int,
    min_size: float,
    max_size: float,
    selectivity: float,
    seed: int,
    python_executable: str = sys.executable,
) -> Tuple[Path, Path]:
    if output_a.exists() and output_b.exists():
        return output_a, output_b

    cmd = [
        python_executable,
        str(GENERATE_CUBES_SCRIPT),
        "--num-cubes-a", str(num_cubes_a),
        "--num-cubes-b", str(num_cubes_b),
        "--min-size", str(min_size),
        "--max-size", str(max_size),
        "--selectivity", str(selectivity),
        "--output-a", str(output_a),
        "--output-b", str(output_b),
        "--seed", str(seed),
    ]
    run_cmd(cmd, f"Generating cubes (nA={num_cubes_a}, nB={num_cubes_b}, sel={selectivity})")
    return output_a, output_b


def ensure_sphere_pair_dataset(
    output_a: Path,
    output_b: Path,
    *,
    template_obj: Path,
    num_objects: int,
    min_size: float,
    max_size: float,
    selectivity: float,
    seed: int,
) -> Tuple[Path, Path]:
    if output_a.exists() and output_b.exists():
        return output_a, output_b

    cmd = [
        str(GENERATE_SPHERES_BIN),
        "--template-obj", str(template_obj),
        "--num-objs-a", str(num_objects),
        "--num-objs-b", str(num_objects),
        "--min-size", str(min_size),
        "--max-size", str(max_size),
        "--selectivity", str(selectivity),
        "-oa", str(output_a),
        "-ob", str(output_b),
        "--seed", str(seed),
    ]
    run_cmd(cmd, f"Generating spheres from {template_obj.name} (n={num_objects}, sel={selectivity})")
    return output_a, output_b


def count_vertices(obj_path: Path) -> int:
    count = 0
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                count += 1
    return count


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
