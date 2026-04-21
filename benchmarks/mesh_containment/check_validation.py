
import subprocess
import os
import csv
import json
from pathlib import Path

def main():
    # Detect paths relative to this script
    current_dir = Path(__file__).parent.resolve()
    base_dir = current_dir.parent.parent
    
    data_dir = current_dir / "data" / "raw"
    results_dir = current_dir / "results" / "validation"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    preprocess_bin = base_dir / "src" / "RaySpace3D" / "preprocess" / "build" / "bin" / "preprocess_dataset"
    containment_bin = base_dir / "src" / "RaySpace3D" / "query" / "build" / "bin" / "raytracer_containment"
    
    mesh_a = data_dir / "validation_a.obj"
    mesh_b = data_dir / "validation_b.obj"
    pre_a = data_dir / "validation_a.pre"
    pre_b = data_dir / "validation_b.pre"
    
    # 1. Preprocess
    print("Preprocessing Mesh A...")
    subprocess.run([
        str(preprocess_bin), "--mode", "mesh", "--dataset", str(mesh_a),
        "--output-geometry", str(pre_a), "--generate-grid", "--grid-cell-size", "128"
    ], check=True)
    
    print("Preprocessing Mesh B...")
    subprocess.run([
        str(preprocess_bin), "--mode", "mesh", "--dataset", str(mesh_b),
        "--output-geometry", str(pre_b), "--generate-grid", "--grid-cell-size", "128"
    ], check=True)
    
    # 2. Run Containment
    print("Running RaySpace Containment...")
    # Change to results dir so output is saved there
    old_cwd = os.getcwd()
    os.chdir(results_dir)
    
    subprocess.run([
        str(containment_bin), "--mesh1", str(pre_a), "--mesh2", str(pre_b),
        "--output", "validation_timing.json"
    ], check=True)
    
    os.chdir(old_cwd)
    
    # 3. Validate
    ground_truth = { (0, 0), (0, 4), (1, 1) }
    found_pairs = set()
    
    csv_path = results_dir / "mesh_containment_results.csv"
    if not csv_path.exists():
        print("Error: mesh_containment_results.csv not found!")
        return
        
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader) # a_object_id, b_object_id
        for row in reader:
            if row:
                found_pairs.add((int(row[0]), int(row[1])))
                
    print(f"Found pairs: {found_pairs}")
    print(f"Ground truth: {ground_truth}")
    
    if found_pairs == ground_truth:
        print("\nSUCCESS: RaySpace containment query is CORRECT on the validation set.")
    else:
        print("\nFAILURE: Mismatch in results!")
        only_in_gt = ground_truth - found_pairs
        only_in_rs = found_pairs - ground_truth
        if only_in_gt:
            print(f"  Missing in RaySpace: {only_in_gt}")
        if only_in_rs:
            print(f"  Extra in RaySpace: {only_in_rs}")

if __name__ == "__main__":
    main()
