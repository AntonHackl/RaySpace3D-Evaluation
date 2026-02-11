
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from adapters import RaytracerAdapter

def create_cube_obj(path, center, size):
    half = size / 2
    cx, cy, cz = center
    verts = [
        (cx-half, cy-half, cz-half), (cx+half, cy-half, cz-half),
        (cx+half, cy+half, cz-half), (cx-half, cy+half, cz-half),
        (cx-half, cy-half, cz+half), (cx+half, cy-half, cz+half),
        (cx+half, cy+half, cz+half), (cx-half, cy+half, cz+half)
    ]
    faces = [
        (1,3,2), (1,4,3), (5,6,7), (5,7,8),
        (1,2,6), (1,6,5), (2,3,7), (2,7,6),
        (3,4,8), (3,8,7), (4,1,5), (4,5,8)
    ]
    with open(path, 'w') as f:
        f.write("o cube\n")
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

TEST_DIR = Path("test_debug")
TEST_DIR.mkdir(exist_ok=True)
RAYSPACE_DIR = Path("../RaySpace3D").resolve()
PREPROCESSED_DIR = Path("preprocessed")
TIMINGS_DIR = Path("timings")
PREPROCESSED_DIR.mkdir(exist_ok=True)
TIMINGS_DIR.mkdir(exist_ok=True)

adapter = RaytracerAdapter(str(RAYSPACE_DIR), mode="exact", preprocessed_dir=str(PREPROCESSED_DIR), timings_dir=str(TIMINGS_DIR))

# Case 1: Intersecting
f1 = TEST_DIR / "cube1.obj"
f2 = TEST_DIR / "cube2.obj"
create_cube_obj(f1, (10, 10, 10), 4) # [8, 12]
create_cube_obj(f2, (11, 11, 11), 4) # [9, 13] -> Overlap in 8-12 and 9-13 is 9-12 (Length 3)

# Preprocess
adapter.preprocess_from_source(str(f1), str(f1))
adapter.preprocess_from_source(str(f2), str(f2))

res = adapter.run_overlap(str(f1), str(f2), 1)
print(f"Case 1 (Overlap): Intersections = {res.get('num_intersections')}")

# Case 2: No Overlap
f3 = TEST_DIR / "cube3.obj"
create_cube_obj(f3, (20, 20, 20), 4) # [18, 22]
adapter.preprocess_from_source(str(f3), str(f3))
res = adapter.run_overlap(str(f1), str(f3), 1)
print(f"Case 2 (Separate): Intersections = {res.get('num_intersections')}")

# Case 3: Containment (Cube 4 inside Cube 1)
f4 = TEST_DIR / "cube4.obj"
create_cube_obj(f4, (10, 10, 10), 2) # [9, 11] subset of [8, 12]
adapter.preprocess_from_source(str(f4), str(f4))
res = adapter.run_overlap(str(f1), str(f4), 1)
print(f"Case 3 (Containment): Intersections = {res.get('num_intersections')}")
