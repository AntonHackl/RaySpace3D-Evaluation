
import os
from pathlib import Path

def write_cube(f, obj_id, x_min, y_min, z_min, size, v_offset):
    f.write(f"o Cube_{obj_id}\n")
    # 8 vertices
    verts = [
        (x_min, y_min, z_min),
        (x_min + size, y_min, z_min),
        (x_min + size, y_min + size, z_min),
        (x_min, y_min + size, z_min),
        (x_min, y_min, z_min + size),
        (x_min + size, y_min, z_min + size),
        (x_min + size, y_min + size, z_min + size),
        (x_min, y_min + size, z_min + size)
    ]
    for v in verts:
        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
    
    # 12 faces (triangles)
    faces = [
        (1, 2, 3), (1, 3, 4), # bottom
        (5, 6, 7), (5, 7, 8), # top
        (1, 2, 6), (1, 6, 5), # side 1
        (2, 3, 7), (2, 7, 6), # side 2
        (3, 4, 8), (3, 8, 7), # side 3
        (4, 1, 5), (4, 5, 8)  # side 4
    ]
    for face in faces:
        f.write(f"f {face[0]+v_offset} {face[1]+v_offset} {face[2]+v_offset}\n")
    
    return v_offset + 8

def main():
    # Use relative path to this script
    current_dir = Path(__file__).parent
    data_dir = current_dir / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    file_a = data_dir / "validation_a.obj"
    file_b = data_dir / "validation_b.obj"
    
    # Dataset A: Containers
    with open(file_a, 'w') as f:
        v_off = 0
        v_off = write_cube(f, 0, 1.0, 1.0, 1.0, 4.0, v_off) # [1,5] x [1,5] x [1,5]
        v_off = write_cube(f, 1, 6.0, 6.0, 6.0, 3.0, v_off) # [6,9] x [6,9] x [6,9]
        
    # Dataset B: Objects
    with open(file_b, 'w') as f:
        v_off = 0
        v_off = write_cube(f, 0, 2.0, 2.0, 2.0, 1.0, v_off) # In A0
        v_off = write_cube(f, 1, 7.0, 7.0, 7.0, 1.0, v_off) # In A1
        v_off = write_cube(f, 2, 4.0, 4.0, 4.0, 2.0, v_off) # [4,6]^3 - Intersects A0/A1? No, B2 is [4,6]. Not contained.
        v_off = write_cube(f, 3, 0.0, 0.0, 0.0, 1.0, v_off) # Outside
        v_off = write_cube(f, 4, 1.5, 1.5, 1.5, 3.0, v_off) # In A0 ([1.5, 4.5] < [1, 5])
        
    print(f"Created {file_a} and {file_b}")
    print("Ground Truth Containment (A_id, B_id):")
    print("0, 0")
    print("0, 4")
    print("1, 1")

if __name__ == "__main__":
    main()
