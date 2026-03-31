#!/usr/bin/env python3
"""
Analyze intersection disagreement pairs to understand causes of false positives.
Focuses on identifying patterns in remaining 40 false positives after nextafterf patch.
"""

import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Set, Tuple, List
import numpy as np


@dataclass
class ObjectBounds:
    obj_id: int
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float
    
    def intersects(self, other: 'ObjectBounds') -> bool:
        """Check if two bounding boxes intersect."""
        return (self.min_x <= other.max_x and self.max_x >= other.min_x and
                self.min_y <= other.max_y and self.max_y >= other.min_y and
                self.min_z <= other.max_z and self.max_z >= other.min_z)
    
    def contains(self, other: 'ObjectBounds') -> bool:
        """Check if other is strictly contained within self."""
        return (self.min_x < other.min_x and self.max_x > other.max_x and
                self.min_y < other.min_y and self.max_y > other.max_y and
                self.min_z < other.min_z and self.max_z > other.max_z)
    
    def center(self) -> Tuple[float, float, float]:
        """Return center of bounding box."""
        return ((self.min_x + self.max_x) / 2,
                (self.min_y + self.max_y) / 2,
                (self.min_z + self.max_z) / 2)
    
    def size(self) -> Tuple[float, float, float]:
        """Return size of bounding box."""
        return (self.max_x - self.min_x,
                self.max_y - self.min_y,
                self.max_z - self.min_z)


def parse_obj_file(obj_path: Path) -> dict:
    """Parse OBJ file and extract per-object bounding boxes."""
    bounds_map = {}
    current_obj_id = -1
    current_vertices = []
    
    with open(obj_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('o '):
                # Save previous object if any
                if current_obj_id >= 0 and current_vertices:
                    bounds_map[current_obj_id] = compute_bounds(current_vertices)
                
                # Parse new object
                parts = line.split()
                current_obj_id = int(parts[1])
                current_vertices = []
            
            elif line.startswith('v '):
                parts = line.split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    current_vertices.append((x, y, z))
        
        # Save last object
        if current_obj_id >= 0 and current_vertices:
            bounds_map[current_obj_id] = compute_bounds(current_vertices)
    
    return bounds_map


def compute_bounds(vertices: List[Tuple[float, float, float]]) -> ObjectBounds:
    """Compute bounding box for a set of vertices."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    
    return ObjectBounds(
        obj_id=-1,  # Will be set by caller
        min_x=min(xs), max_x=max(xs),
        min_y=min(ys), max_y=max(ys),
        min_z=min(zs), max_z=max(zs)
    )


def analyze_disagreement_run(results_dir: Path):
    """Analyze intersection disagreement run to understand false positives."""
    
    # Load summary
    summary_path = results_dir / 'summary.json'
    if not summary_path.exists():
        print(f"ERROR: No summary found at {summary_path}")
        return
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print(f"\n=== INTERSECTION DISAGREEMENT ANALYSIS ===")
    print(f"RaySpace pairs: {summary.get('rayspace_pairs', 0)}")
    print(f"CGAL pairs: {summary.get('cgal_pairs', 0)}")
    print(f"Agreed pairs: {summary.get('agreed_pairs', 0)}")
    print(f"Only RaySpace: {summary.get('only_rayspace', 0)}")
    print(f"Only CGAL: {summary.get('only_cgal', 0)}\n")
    
    # Load disagreement pairs
    disagreements_csv = results_dir / 'disagreements.csv'
    if not disagreements_csv.exists():
        print(f"WARNING: No disagreements CSV at {disagreements_csv}")
        return
    
    disagreement_pairs = []
    with open(disagreements_csv, 'r') as f:
        header = f.readline().strip()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                try:
                    obj_a, obj_b = int(parts[0]), int(parts[1])
                    source = parts[2] if len(parts) > 2 else 'unknown'
                    disagreement_pairs.append((obj_a, obj_b, source))
                except ValueError:
                    continue
    
    print(f"Loaded {len(disagreement_pairs)} disagreement pairs\n")
    
    # Load dataset OBJ file
    data_dir = Path('/sc/home/anton.hackl/Spatial_Data_Management/RaySpace3D-Evaluation/benchmarks/correctness_tests/data')
    mesh_a_path = data_dir / 'cubes_20k_a.obj'
    mesh_b_path = data_dir / 'cubes_20k_b.obj'
    
    if not mesh_a_path.exists() or not mesh_b_path.exists():
        print(f"ERROR: Mesh files not found at {mesh_a_path} or {mesh_b_path}")
        return
    
    print(f"Loading mesh bounds from OBJ files...")
    bounds_a = parse_obj_file(mesh_a_path)
    bounds_b = parse_obj_file(mesh_b_path)
    
    # Add object IDs to bounds
    for obj_id, bounds in bounds_a.items():
        bounds.obj_id = obj_id
    for obj_id, bounds in bounds_b.items():
        bounds.obj_id = obj_id
    
    print(f"Mesh A: {len(bounds_a)} objects")
    print(f"Mesh B: {len(bounds_b)} objects\n")
    
    # Analyze each disagreement
    false_positive_reasons = {
        'non_intersecting': [],
        'one_contains_other': [],
        'touching_only': [],
        'close_but_separate': [],
    }
    
    for obj_a, obj_b, source in disagreement_pairs:
        if obj_a not in bounds_a or obj_b not in bounds_b:
            continue
        
        bbox_a = bounds_a[obj_a]
        bbox_b = bounds_b[obj_b]
        
        # Check AABB relationship
        if not bbox_a.intersects(bbox_b):
            print(f"PAIR ({obj_a}, {obj_b}): Non-intersecting AABBs (FALSE POSITIVE)")
            false_positive_reasons['non_intersecting'].append((obj_a, obj_b))
        elif bbox_a.contains(bbox_b) or bbox_b.contains(bbox_a):
            print(f"PAIR ({obj_a}, {obj_b}): One contains the other (containment not intersection?)")
            false_positive_reasons['one_contains_other'].append((obj_a, obj_b))
        else:
            # Check if they're just touching (axis-aligned touching)
            eps = 1e-6
            touching = (
                (abs(bbox_a.max_x - bbox_b.min_x) < eps or abs(bbox_a.min_x - bbox_b.max_x) < eps or
                 abs(bbox_a.max_y - bbox_b.min_y) < eps or abs(bbox_a.min_y - bbox_b.max_y) < eps or
                 abs(bbox_a.max_z - bbox_b.min_z) < eps or abs(bbox_a.min_z - bbox_b.max_z) < eps)
            )
            if touching:
                print(f"PAIR ({obj_a}, {obj_b}): Touching at boundary (numerical precision issue?)")
                false_positive_reasons['touching_only'].append((obj_a, obj_b))
            else:
                # True intersection AABB
                print(f"PAIR ({obj_a}, {obj_b}): True AABB intersection (geometry issue?)")
                false_positive_reasons['close_but_separate'].append((obj_a, obj_b))
    
    # Summary
    print(f"\n=== FALSE POSITIVE CATEGORIZATION ===")
    print(f"Non-intersecting AABBs: {len(false_positive_reasons['non_intersecting'])}")
    print(f"Containment pairs: {len(false_positive_reasons['one_contains_other'])}")
    print(f"Touching at boundary: {len(false_positive_reasons['touching_only'])}")
    print(f"True AABB overlap: {len(false_positive_reasons['close_but_separate'])}")
    
    # Print examples from each category
    for category, pairs in false_positive_reasons.items():
        if pairs:
            print(f"\n{category.upper()} examples (first 5):")
            for obj_a, obj_b in pairs[:5]:
                if obj_a in bounds_a and obj_b in bounds_b:
                    ba = bounds_a[obj_a]
                    bb = bounds_b[obj_b]
                    print(f"  ({obj_a}, {obj_b})")
                    print(f"    A: [{ba.min_x:.6f}, {ba.max_x:.6f}] x [{ba.min_y:.6f}, {ba.max_y:.6f}] x [{ba.min_z:.6f}, {ba.max_z:.6f}]")
                    print(f"    B: [{bb.min_x:.6f}, {bb.max_x:.6f}] x [{bb.min_y:.6f}, {bb.max_y:.6f}] x [{bb.min_z:.6f}, {bb.max_z:.6f}]")


if __name__ == '__main__':
    # Find latest results directory
    base_dir = Path('/sc/home/anton.hackl/Spatial_Data_Management/RaySpace3D-Evaluation/benchmarks/correctness_tests/runs')
    
    # Get latest intersection_disagreement_* directory
    dirs = sorted([d for d in base_dir.glob('intersection_disagreement_*') if d.is_dir()])
    if not dirs:
        print(f"No disagreement runs found in {base_dir}")
        sys.exit(1)
    
    latest_dir = dirs[-1]
    print(f"Analyzing latest run: {latest_dir}")
    analyze_disagreement_run(latest_dir)
