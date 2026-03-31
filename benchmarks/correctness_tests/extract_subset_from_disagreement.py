#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


def parse_ids(summary_path: Path) -> Tuple[List[int], List[int], Dict[str, str]]:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    pairs = summary.get("inspected_only_rayspace_pairs", [])
    if not pairs:
        raise ValueError("No inspected_only_rayspace_pairs found in summary JSON")

    ids_a = sorted({int(p["a_object_id"]) for p in pairs})
    ids_b = sorted({int(p["b_object_id"]) for p in pairs})
    metadata = summary.get("metadata", {})
    return ids_a, ids_b, metadata


def extract_obj_subset(input_obj: Path, keep_ids: Set[int], output_obj: Path) -> int:
    obj_re = re.compile(r"^[oO]\s+[^0-9]*([0-9]+)\s*$")

    global_vertices: List[Tuple[float, float, float]] = []
    out_lines: List[str] = [f"# Subset extracted from {input_obj.name}\n"]

    keep_block = False
    block_id = None
    block_vertices: List[Tuple[float, float, float]] = []
    block_faces: List[List[int]] = []
    block_vertex_index_map: Dict[int, int] = {}
    global_vertex_counter = 0
    kept_count = 0

    def flush_block() -> None:
        nonlocal kept_count
        if not keep_block or block_id is None:
            return
        if not block_vertices or not block_faces:
            return

        start_idx = len(global_vertices) + 1
        out_lines.append(f"o cube_{block_id}\n")
        for v in block_vertices:
            global_vertices.append(v)
            out_lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in block_faces:
            remapped = [str(start_idx + (idx - 1)) for idx in face]
            out_lines.append("f " + " ".join(remapped) + "\n")
        out_lines.append("\n")
        kept_count += 1

    with open(input_obj, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("o ") or line.startswith("O "):
                flush_block()
                block_vertices = []
                block_faces = []
                block_vertex_index_map = {}
                m = obj_re.match(line)
                if m:
                    block_id = int(m.group(1))
                    keep_block = block_id in keep_ids
                else:
                    block_id = None
                    keep_block = False
                continue

            if not keep_block:
                if line.startswith("v "):
                    global_vertex_counter += 1
                continue

            if line.startswith("v "):
                global_vertex_counter += 1
                _, x, y, z = line.split()
                block_vertices.append((float(x), float(y), float(z)))
                block_vertex_index_map[global_vertex_counter] = len(block_vertices)
            elif line.startswith("f "):
                parts = line.split()[1:]
                face = []
                for p in parts:
                    idx_str = p.split("/")[0]
                    global_idx = int(idx_str)
                    local_idx = block_vertex_index_map.get(global_idx)
                    if local_idx is None:
                        raise ValueError(
                            f"Face references vertex {global_idx} not in current kept object {block_id}"
                        )
                    face.append(local_idx)
                block_faces.append(face)

    flush_block()

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_obj, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    return kept_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract subset OBJ meshes from disagreement summary IDs")
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--mesh-a", default=None)
    parser.add_argument("--mesh-b", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    ids_a, ids_b, metadata = parse_ids(summary_path)

    mesh_a = Path(args.mesh_a or metadata.get("mesh_a", ""))
    mesh_b = Path(args.mesh_b or metadata.get("mesh_b", ""))
    if not mesh_a.exists() or not mesh_b.exists():
        raise FileNotFoundError("Could not resolve mesh_a or mesh_b from args/summary metadata")

    out_dir = Path(args.output_dir)
    subset_a = out_dir / "subset_a.obj"
    subset_b = out_dir / "subset_b.obj"

    kept_a = extract_obj_subset(mesh_a, set(ids_a), subset_a)
    kept_b = extract_obj_subset(mesh_b, set(ids_b), subset_b)

    manifest = {
        "source_summary": str(summary_path),
        "source_mesh_a": str(mesh_a),
        "source_mesh_b": str(mesh_b),
        "subset_mesh_a": str(subset_a),
        "subset_mesh_b": str(subset_b),
        "ids_a": ids_a,
        "ids_b": ids_b,
        "kept_objects_a": kept_a,
        "kept_objects_b": kept_b,
    }
    with open(out_dir / "subset_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Created subset meshes:\n  A: {subset_a} ({kept_a} objects)\n  B: {subset_b} ({kept_b} objects)")


if __name__ == "__main__":
    main()
