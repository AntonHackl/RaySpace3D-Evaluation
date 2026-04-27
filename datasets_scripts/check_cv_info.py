from cloudvolume import CloudVolume
import numpy as np

cv = CloudVolume("precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300")
print("Info:")
print(cv.info)
print("\nResolution (mip 0):", cv.resolution)
print("Bounds (voxels, mip 0):", cv.bounds)

# Resolution at mip 0 is usually [4, 4, 40] nm
# Bounds are in voxels
res = np.array(cv.resolution)
bounds_min = np.array(cv.bounds.minpt)
bounds_max = np.array(cv.bounds.maxpt)

size_nm = (bounds_max - bounds_min) * res
size_um = size_nm / 1000.0

print(f"\nSize (nm): {size_nm}")
print(f"Size (um): {size_um}")
print(f"Size (mm): {size_um / 1000.0}")
