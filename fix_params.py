import re
from pathlib import Path

# Paths
dir_cuda = Path('src/RaySpace3D/query/src/cuda')
dir_apps = Path('src/RaySpace3D/query/src/applications')
dir_raytracing = Path('src/RaySpace3D/query/src/raytracing')

# Update mesh_intersection.h
f_mi = dir_cuda / 'mesh_intersection.h'
text = f_mi.read_text()
text = re.sub(r'float3\*\s*mesh1_vertices;\s*uint3\*\s*mesh1_indices;\s*int\*\s*mesh1_triangle_to_object;\s*int\s*mesh1_num_triangles;', '', text)
text = re.sub(r'float3\*\s*mesh2_vertices;\s*uint3\*\s*mesh2_indices;', '', text)
text = re.sub(r'int\*\s*first_triangle_index_per_object;', 'float3* first_vertex_per_object;', text)
f_mi.write_text(text)

# Update mesh_overlap.h
f_mo = dir_cuda / 'mesh_overlap.h'
text = f_mo.read_text()
text = re.sub(r'float3\*\s*mesh1_vertices;\s*uint3\*\s*mesh1_indices;\s*int\*\s*mesh1_triangle_to_object;\s*int\s*mesh1_num_triangles;', '', text)
text = re.sub(r'float3\*\s*mesh2_vertices;\s*uint3\*\s*mesh2_indices;', '', text)
f_mo.write_text(text)

# Update MeshOverlapEdgesLauncher.h
f_moel = dir_raytracing / 'MeshOverlapEdgesLauncher.h'
text = f_moel.read_text()
text = re.sub(r'float3\*\s*mesh2_vertices;\s*uint3\*\s*mesh2_indices;', '', text)
f_moel.write_text(text)

# Update mesh_intersection.cu
f_cu_mi = dir_cuda / 'mesh_intersection.cu'
text = f_cu_mi.read_text()
text = re.sub(r'const int firstTri = mesh_intersection_params\.first_triangle_index_per_object\[sourceObjectId\];\s*if \(firstTri < 0 \|\| firstTri >= mesh_intersection_params\.mesh1_num_triangles\) \{\s*return;\s*\}\s*const uint3 triIndices = mesh_intersection_params\.mesh1_indices\[firstTri\];\s*const float3 queryPoint = mesh_intersection_params\.mesh1_vertices\[triIndices\.x\];', 
              r'const float3 queryPoint = mesh_intersection_params.first_vertex_per_object[sourceObjectId];', text)
f_cu_mi.write_text(text)

print("Headers and CU updated")
