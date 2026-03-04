import subprocess
import sys
import time

def run_cmd(cmd):
    start = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    end = time.perf_counter()
    if res.returncode != 0:
        print(f"Error executing {' '.join(cmd)}")
        print(res.stderr)
        sys.exit(1)
    return res, end - start

m1 = "benchmarks/mesh_overlap/data/raw/tdbase_n_nv150_nu200_v_nv150_nu200_vs100_r30.dt"
m2 = "benchmarks/mesh_overlap/data/raw/tdbase_n_nv150_nu200_n_nv150_nu200_vs100_r30.dt"

m1_pre = "benchmarks/mesh_overlap/data/preprocessed/tdbase_n_nv150_nu200_v_nv150_nu200_vs100_r30.pre"
m2_pre = "benchmarks/mesh_overlap/data/preprocessed/tdbase_n_nv150_nu200_n_nv150_nu200_vs100_r30.pre"

def get_pairs(stdout):
    pairs = set()
    for line in stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            v1, v2 = int(parts[0]), int(parts[1])
            if v1 > v2: v1, v2 = v2, v1
            pairs.add(f"{v1},{v2}")
    return pairs

print("Running TDBase (Fast Mode)...")
cmd_fast = [
    "baselines/RaySpace3DBaselines/tdbase_patch/build/tdbase", "join", "-q", "intersect",
    "--tile1", m1, "--tile2", m2, "-p"
]
res_fast, time_fast = run_cmd(cmd_fast)
pairs_fast = get_pairs(res_fast.stdout)

print(f"TDBase Fast Mode: {len(pairs_fast)} pairs, {time_fast:.4f}s")

print("\nRunning TDBase (Exact Mode)...")
cmd_exact = [
    "baselines/RaySpace3DBaselines/tdbase_patch/build/tdbase", "join", "-q", "intersect",
    "--tile1", m1, "--tile2", m2, "-p", "-e"
]
res_exact, time_exact = run_cmd(cmd_exact)
pairs_exact = get_pairs(res_exact.stdout)

print(f"TDBase Exact Mode: {len(pairs_exact)} pairs, {time_exact:.4f}s")

print("\nRunning RaySpace Exact Intersection...")
cmd_rs = [
    "src/RaySpace3D/query/build/bin/raytracer_mesh_intersection",
    "--mesh1", m1_pre, "--mesh2", m2_pre,
    "--runs", "1", "--warmup-runs", "0"
]
res_rs, time_rs = run_cmd(cmd_rs)
pairs_rs = set()
with open("mesh_intersection_results.csv", "r") as f:
    lines = f.readlines()
    for line in lines[1:]: # skip header
        parts = line.strip().split(',')
        if len(parts) == 2:
            v1, v2 = int(parts[0]), int(parts[1])
            if v1 > v2: v1, v2 = v2, v1
            pairs_rs.add(f"{v1},{v2}")

print(f"RaySpace Intersection: {len(pairs_rs)} pairs, {time_rs:.4f}s")

print("\n" + "="*50)
print(f"{'Method':<25} | {'Pairs':<8} | {'Time (s)':<10}")
print("-" * 50)
print(f"{'TDBase (Fast)':<25} | {len(pairs_fast):<8} | {time_fast:.4f}")
print(f"{'TDBase (Exact)':<25} | {len(pairs_exact):<8} | {time_exact:.4f}")
print(f"{'RaySpace (Exact)':<25} | {len(pairs_rs):<8} | {time_rs:.4f}")
print("="*50)

print(f"\nDiscrepancy (Exact Mode vs RaySpace):")
print(f"In RaySpace but NOT TDBase Exact: {len(pairs_rs - pairs_exact)}")
print(f"In TDBase Exact but NOT RaySpace: {len(pairs_exact - pairs_rs)}")
