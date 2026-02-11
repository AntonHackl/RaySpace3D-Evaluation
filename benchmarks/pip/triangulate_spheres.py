#!/usr/bin/env python3
"""
Triangulate all sphere OBJ files in the workspace directory.
Converts any polygonal faces (quads, n-gons) to triangles using fan triangulation.
"""

import os
import glob

def triangulate_face(face_indices):
    """
    Convert a face with N vertices to (N-2) triangles using fan triangulation.
    
    Args:
        face_indices: List of vertex indices forming the face
    
    Returns:
        List of triangular faces
    """
    if len(face_indices) == 3:
        return [face_indices]
    
    # Fan triangulation from first vertex
    triangles = []
    for i in range(1, len(face_indices) - 1):
        triangles.append([face_indices[0], face_indices[i], face_indices[i + 1]])
    
    return triangles

def parse_face_index(index_str):
    """Parse a face index which may be v, v/vt, or v/vt/vn format"""
    return index_str.split('/')[0]

def triangulate_obj(input_path, output_path):
    """
    Read an OBJ file and write a triangulated version.
    
    Args:
        input_path: Path to input OBJ file
        output_path: Path to output OBJ file
    """
    vertices = []
    faces = []
    other_lines = []
    
    # Read the OBJ file
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                other_lines.append(line)
            elif line.startswith('v '):
                vertices.append(line)
            elif line.startswith('f '):
                # Parse face indices
                parts = line.split()[1:]
                indices = [parse_face_index(p) for p in parts]
                
                # Triangulate if needed
                triangulated = triangulate_face(indices)
                faces.extend(triangulated)
            else:
                other_lines.append(line)
    
    # Write the triangulated OBJ file
    with open(output_path, 'w') as f:
        # Write header
        f.write("# Triangulated OBJ file\n")
        
        # Write vertices
        for v in vertices:
            f.write(v + '\n')
        
        # Write triangulated faces
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"Triangulated {input_path} -> {output_path}")
    print(f"  Vertices: {len(vertices)}, Triangles: {len(faces)}")

def main():
    workspace_dir = "/sc/home/anton.hackl/Spatial_Data_Management/first_benchmark/workspace"
    
    # Find all sphere OBJ files
    sphere_files = glob.glob(os.path.join(workspace_dir, "sphere_*.obj"))
    
    if not sphere_files:
        print("No sphere OBJ files found!")
        return
    
    print(f"Found {len(sphere_files)} sphere files to triangulate\n")
    
    for sphere_file in sorted(sphere_files):
        # Create backup
        backup_file = sphere_file + ".backup"
        if not os.path.exists(backup_file):
            os.rename(sphere_file, backup_file)
            print(f"Created backup: {backup_file}")
            input_file = backup_file
        else:
            print(f"Using existing backup: {backup_file}")
            input_file = backup_file
        
        # Triangulate
        output_file = input_file.replace(".backup", "")
        triangulate_obj(input_file, output_file)
        print()

if __name__ == "__main__":
    main()
