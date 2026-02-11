#!/usr/bin/env python3
"""OBJ mesh rescaling utility.

Reads an OBJ file, computes its bounding box, and rescales vertices
to match target extents in x, y, z dimensions while preserving the center position.

Example:
    python rescale_obj.py cube.obj cube_rescaled.obj --extent-x 10 --extent-y 10 --extent-z 10
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple


class OBJMesh:
    """Simple OBJ mesh representation."""
    
    def __init__(self):
        self.vertices: List[Tuple[float, float, float]] = []
        self.faces: List[str] = []  # Store as strings to preserve format
        self.other_lines: List[str] = []  # Comments, normals, etc.
    
    def compute_bbox(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Compute bounding box (min, max) of the mesh."""
        if not self.vertices:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))
    
    def compute_center(self) -> Tuple[float, float, float]:
        """Compute center of bounding box."""
        bbox_min, bbox_max = self.compute_bbox()
        return (
            (bbox_min[0] + bbox_max[0]) / 2.0,
            (bbox_min[1] + bbox_max[1]) / 2.0,
            (bbox_min[2] + bbox_max[2]) / 2.0
        )
    
    def rescale(self, target_extent_x: float, target_extent_y: float, target_extent_z: float):
        """Rescale mesh to target extents while preserving center.
        
        Args:
            target_extent_x: Target extent in x dimension
            target_extent_y: Target extent in y dimension
            target_extent_z: Target extent in z dimension
        """
        if not self.vertices:
            return
        
        # Compute current bounding box and center
        bbox_min, bbox_max = self.compute_bbox()
        center = self.compute_center()
        
        current_extent_x = bbox_max[0] - bbox_min[0]
        current_extent_y = bbox_max[1] - bbox_min[1]
        current_extent_z = bbox_max[2] - bbox_min[2]
        
        # Compute scale factors (handle zero extents)
        scale_x = target_extent_x / current_extent_x if current_extent_x > 0 else 1.0
        scale_y = target_extent_y / current_extent_y if current_extent_y > 0 else 1.0
        scale_z = target_extent_z / current_extent_z if current_extent_z > 0 else 1.0
        
        # Rescale vertices: translate to origin, scale, translate back
        new_vertices = []
        for x, y, z in self.vertices:
            # Translate to origin
            x_centered = x - center[0]
            y_centered = y - center[1]
            z_centered = z - center[2]
            
            # Scale
            x_scaled = x_centered * scale_x
            y_scaled = y_centered * scale_y
            z_scaled = z_centered * scale_z
            
            # Translate back
            x_new = x_scaled + center[0]
            y_new = y_scaled + center[1]
            z_new = z_scaled + center[2]
            
            new_vertices.append((x_new, y_new, z_new))
        
        self.vertices = new_vertices


def load_obj(filepath: str) -> OBJMesh:
    """Load OBJ file."""
    mesh = OBJMesh()
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line or line.startswith('#'):
                mesh.other_lines.append(line)
                continue
            
            parts = line.split()
            if not parts:
                mesh.other_lines.append(line)
                continue
            
            if parts[0] == 'v':
                # Vertex
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    mesh.vertices.append((x, y, z))
                else:
                    mesh.other_lines.append(line)
            elif parts[0] == 'f':
                # Face
                mesh.faces.append(line)
            else:
                # Other lines (vn, vt, etc.)
                mesh.other_lines.append(line)
    
    return mesh


def save_obj(mesh: OBJMesh, filepath: str):
    """Save OBJ file."""
    with open(filepath, 'w') as f:
        # Write header comments
        f.write("# Rescaled OBJ mesh\n")
        
        # Write original comments/headers
        for line in mesh.other_lines:
            if line.startswith('#'):
                f.write(line + '\n')
        
        f.write(f"# {len(mesh.vertices)} vertices\n")
        
        # Write vertices
        for x, y, z in mesh.vertices:
            f.write(f"v {x} {y} {z}\n")
        
        f.write(f"\n# {len(mesh.faces)} faces\n")
        
        # Write faces
        for face_line in mesh.faces:
            f.write(face_line + '\n')
        
        # Write other non-comment lines (normals, textures, etc.)
        for line in mesh.other_lines:
            if not line.startswith('#') and line:
                f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser(
        description="Rescale OBJ mesh to target bounding box extents",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('input', help='Input OBJ file')
    parser.add_argument('output', help='Output OBJ file')
    parser.add_argument('--extent-x', type=float, required=True,
                        help='Target extent in x dimension')
    parser.add_argument('--extent-y', type=float, required=True,
                        help='Target extent in y dimension')
    parser.add_argument('--extent-z', type=float, required=True,
                        help='Target extent in z dimension')
    
    args = parser.parse_args()
    
    # Validate input
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    if args.extent_x <= 0 or args.extent_y <= 0 or args.extent_z <= 0:
        print("Error: Target extents must be positive", file=sys.stderr)
        return 1
    
    # Load mesh
    print(f"Loading OBJ file: {args.input}")
    mesh = load_obj(args.input)
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces: {len(mesh.faces)}")
    
    # Compute original bounding box
    bbox_min, bbox_max = mesh.compute_bbox()
    center = mesh.compute_center()
    print(f"\nOriginal bounding box:")
    print(f"  Min: ({bbox_min[0]:.6f}, {bbox_min[1]:.6f}, {bbox_min[2]:.6f})")
    print(f"  Max: ({bbox_max[0]:.6f}, {bbox_max[1]:.6f}, {bbox_max[2]:.6f})")
    print(f"  Center: ({center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f})")
    print(f"  Extents: ({bbox_max[0]-bbox_min[0]:.6f}, "
          f"{bbox_max[1]-bbox_min[1]:.6f}, {bbox_max[2]-bbox_min[2]:.6f})")
    
    # Rescale
    print(f"\nRescaling to target extents: "
          f"({args.extent_x}, {args.extent_y}, {args.extent_z})")
    mesh.rescale(args.extent_x, args.extent_y, args.extent_z)
    
    # Verify new bounding box
    bbox_min_new, bbox_max_new = mesh.compute_bbox()
    center_new = mesh.compute_center()
    print(f"\nNew bounding box:")
    print(f"  Min: ({bbox_min_new[0]:.6f}, {bbox_min_new[1]:.6f}, {bbox_min_new[2]:.6f})")
    print(f"  Max: ({bbox_max_new[0]:.6f}, {bbox_max_new[1]:.6f}, {bbox_max_new[2]:.6f})")
    print(f"  Center: ({center_new[0]:.6f}, {center_new[1]:.6f}, {center_new[2]:.6f})")
    print(f"  Extents: ({bbox_max_new[0]-bbox_min_new[0]:.6f}, "
          f"{bbox_max_new[1]-bbox_min_new[1]:.6f}, {bbox_max_new[2]-bbox_min_new[2]:.6f})")
    
    # Save
    print(f"\nSaving rescaled mesh to: {args.output}")
    save_obj(mesh, args.output)
    print("Done!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
