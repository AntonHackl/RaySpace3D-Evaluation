#!/usr/bin/env python3
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import canonical_cube_pair_paths, ensure_cube_pair_dataset

RAW_DIR = SCRIPT_DIR / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manual_expected_results.json"


def write_cube(handle, obj_id, x_min, y_min, z_min, size, vertex_offset):
    handle.write(f"o Cube_{obj_id}\n")

    vertices = [
        (x_min, y_min, z_min),
        (x_min + size, y_min, z_min),
        (x_min + size, y_min + size, z_min),
        (x_min, y_min + size, z_min),
        (x_min, y_min, z_min + size),
        (x_min + size, y_min, z_min + size),
        (x_min + size, y_min + size, z_min + size),
        (x_min, y_min + size, z_min + size),
    ]
    for vx, vy, vz in vertices:
        handle.write(f"v {vx} {vy} {vz}\n")

    faces = [
        (1, 2, 3), (1, 3, 4),
        (5, 6, 7), (5, 7, 8),
        (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6),
        (3, 4, 8), (3, 8, 7),
        (4, 1, 5), (4, 5, 8),
    ]
    for a, b, c in faces:
        handle.write(f"f {a + vertex_offset} {b + vertex_offset} {c + vertex_offset}\n")

    return vertex_offset + 8


def write_dataset(path: Path, cubes):
    with open(path, "w", encoding="utf-8") as handle:
        vertex_offset = 0
        for obj_id, x_min, y_min, z_min, size in cubes:
            vertex_offset = write_cube(handle, obj_id, x_min, y_min, z_min, size, vertex_offset)


def generate_20k_cube_dataset():
    obj_a, obj_b = canonical_cube_pair_paths(
        RAW_DIR,
        num_cubes_a=20000,
        num_cubes_b=20000,
        min_size=1,
        max_size=4,
        selectivity=0.001,
        seed=42,
        grid_cell_size=None,
    )
    ensure_cube_pair_dataset(
        obj_a,
        obj_b,
        num_cubes_a=20000,
        num_cubes_b=20000,
        min_size=1,
        max_size=4,
        selectivity=0.001,
        seed=42,
    )
    return obj_a, obj_b


def generate_overlap_manual_dataset():
    file_a = RAW_DIR / "overlap_manual_a.obj"
    file_b = RAW_DIR / "overlap_manual_b.obj"

    # A has one object with object id 0 spanning [0,10]^3.
    cubes_a = [
        (0, 0.0, 0.0, 0.0, 10.0),
    ]

    # B has 10 objects.
    # IDs 0-4: strict overlaps with A (not contained).
    # ID 5: strict containment inside A (must NOT count as overlap in this suite).
    # IDs 6-9: disjoint.
    # All coordinates avoid touching-only contacts.
    cubes_b = [
        (0, -2.0, 1.0, 1.0, 4.0),
        (1, 8.5, 1.0, 1.0, 3.5),
        (2, 1.0, -2.0, 1.0, 4.0),
        (3, 1.0, 8.5, 1.0, 3.5),
        (4, 1.0, 1.0, 8.5, 3.5),
        (5, 2.0, 2.0, 2.0, 1.0),
        (6, 12.5, 12.5, 12.5, 2.0),
        (7, -6.0, -6.0, -6.0, 2.0),
        (8, 12.5, 1.0, 1.0, 2.0),
        (9, 1.0, 12.5, 1.0, 2.0),
    ]

    write_dataset(file_a, cubes_a)
    write_dataset(file_b, cubes_b)

    return file_a, file_b, {
        "expected_overlap_pairs": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
        "contained_b_ids_excluded_from_overlap": [5],
    }


def generate_intersection_manual_dataset():
    file_a = RAW_DIR / "intersection_manual_a.obj"
    file_b = RAW_DIR / "intersection_manual_b.obj"

    cubes_a = [
        (0, 0.0, 0.0, 0.0, 10.0),
    ]

    # IDs 0-3: partial intersections.
    # ID 4: strict containment, still intersection by geometric definition.
    # IDs 5-9: disjoint.
    cubes_b = [
        (0, -2.0, 2.0, 2.0, 4.0),
        (1, 8.5, 2.0, 2.0, 3.5),
        (2, 2.0, -2.0, 2.0, 4.0),
        (3, 2.0, 2.0, 8.5, 3.5),
        (4, 3.0, 3.0, 3.0, 1.0),
        (5, 12.5, 12.5, 12.5, 2.0),
        (6, -6.0, -6.0, -6.0, 2.0),
        (7, 12.5, 1.0, 1.0, 2.0),
        (8, 1.0, 12.5, 1.0, 2.0),
        (9, 1.0, 1.0, 12.5, 2.0),
    ]

    write_dataset(file_a, cubes_a)
    write_dataset(file_b, cubes_b)

    return file_a, file_b, {
        "expected_intersection_pairs": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
        "contained_b_ids_included_in_intersection": [4],
    }


def generate_containment_manual_dataset():
    file_a = RAW_DIR / "containment_manual_a.obj"
    file_b = RAW_DIR / "containment_manual_b.obj"

    cubes_a = [
        (0, 0.0, 0.0, 0.0, 10.0),
    ]

    # IDs 0-4: strict containment B in A.
    # ID 5: overlap but not contained (must NOT count as containment).
    # IDs 6-9: disjoint.
    cubes_b = [
        (0, 1.0, 1.0, 1.0, 1.0),
        (1, 2.5, 2.5, 2.5, 1.0),
        (2, 4.0, 4.0, 4.0, 1.5),
        (3, 6.0, 6.0, 6.0, 1.0),
        (4, 8.0, 1.0, 1.0, 1.0),
        (5, -1.5, 2.0, 2.0, 4.0),
        (6, 12.5, 12.5, 12.5, 2.0),
        (7, -6.0, -6.0, -6.0, 2.0),
        (8, 12.5, 1.0, 1.0, 2.0),
        (9, 1.0, 12.5, 1.0, 2.0),
    ]

    write_dataset(file_a, cubes_a)
    write_dataset(file_b, cubes_b)

    return file_a, file_b, {
        "expected_containment_pairs": [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
        "overlap_b_ids_excluded_from_containment": [5],
    }


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cubes20k_a, cubes20k_b = generate_20k_cube_dataset()
    overlap_a, overlap_b, overlap_meta = generate_overlap_manual_dataset()
    intersection_a, intersection_b, intersection_meta = generate_intersection_manual_dataset()
    containment_a, containment_b, containment_meta = generate_containment_manual_dataset()

    manifest = {
        "cubes_20k_sel_0_001": {
            "mesh_a": str(cubes20k_a),
            "mesh_b": str(cubes20k_b),
            "num_cubes_a": 20000,
            "num_cubes_b": 20000,
            "selectivity": 0.001,
            "seed": 42,
        },
        "manual": {
            "overlap": {
                "mesh_a": str(overlap_a),
                "mesh_b": str(overlap_b),
                **overlap_meta,
            },
            "intersection": {
                "mesh_a": str(intersection_a),
                "mesh_b": str(intersection_b),
                **intersection_meta,
            },
            "containment": {
                "mesh_a": str(containment_a),
                "mesh_b": str(containment_b),
                **containment_meta,
            },
        },
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote manifest: {MANIFEST_PATH}")
    print("Generated datasets:")
    for key, value in manifest["manual"].items():
        print(f"  {key}: {value['mesh_a']} | {value['mesh_b']}")
    print(f"  cubes_20k_sel_0_001: {manifest['cubes_20k_sel_0_001']['mesh_a']} | {manifest['cubes_20k_sel_0_001']['mesh_b']}")


if __name__ == "__main__":
    main()
