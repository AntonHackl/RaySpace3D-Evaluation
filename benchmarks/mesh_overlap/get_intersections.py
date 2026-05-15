import json
import subprocess
from pathlib import Path

DATA_DIR = Path("data/preprocessed")
BIN = Path("../../src/RaySpace3D/query/build/bin/raytracer_overlap_direct_estimation")

def get_intersections(f1, f2):
    cmd = [str(BIN), "--mesh1", str(f1), "--mesh2", str(f2), "--runs", "1", "--warmup-runs", "0", "--estimate-only"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if "Unique object pairs:" in line:
            return int(line.split(":")[1].strip())
    return 0

with open("runs/overlap_nu_scalability_20260513_183146/results.json", "r") as f:
    data = json.load(f)

for i, count in enumerate(data["metadata"]["nu_counts"]):
    # For large_nu_v, f1 is v_file, f2 is n_file.
    # We can just look at the datasets used... Wait, I know the datasets!
    pass
