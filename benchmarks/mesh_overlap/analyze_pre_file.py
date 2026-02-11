#!/usr/bin/env python3
"""
Analyze .pre files to debug underestimation in selectivity estimation.

The .pre file format (binary):
- FileHeader (56 bytes):
  - uint32_t magic (0x52334442 = "R3DB")
  - uint32_t version
  - uint64_t numVertices
  - uint64_t numIndices
  - uint64_t numMappings
  - uint64_t totalTriangles
  - uint8_t hasGrid
  - uint8_t padding[7]
- Vertices: numVertices * 12 bytes (float3: x, y, z)
- Indices: numIndices * 12 bytes (uint3: i0, i1, i2)
- TriangleToObject: numMappings * 4 bytes (int32)
- If hasGrid:
  - GridParams (40 bytes):
    - float minBound[3]
    - float maxBound[3]
    - uint32_t resolution[3]
    - uint32_t padding
  - GridCells: resolution[0] * resolution[1] * resolution[2] * 16 bytes
    - uint32_t CenterCount
    - uint32_t TouchCount
    - float AvgSizeMean
    - float VolRatio
"""

import struct
import numpy as np
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt


BINARY_FILE_MAGIC = 0x52334442  # "R3DB"


def read_pre_file(filepath: str) -> Dict[str, Any]:
    """Read a .pre file and return parsed data."""
    data = {}
    
    with open(filepath, 'rb') as f:
        # Read header (48 bytes)
        # uint32_t magic, version
        # uint64_t numVertices, numIndices, numMappings, totalTriangles
        # uint8_t hasGrid + 7 bytes padding
        header_bytes = f.read(48)
        magic, version = struct.unpack('<II', header_bytes[:8])
        numVertices, numIndices, numMappings, totalTriangles = struct.unpack('<QQQQ', header_bytes[8:40])
        hasGrid = struct.unpack('<B', header_bytes[40:41])[0]
        
        if magic != BINARY_FILE_MAGIC:
            raise ValueError(f"Invalid magic number: {hex(magic)} (expected {hex(BINARY_FILE_MAGIC)})")
        
        data['version'] = version
        data['numVertices'] = numVertices
        data['numIndices'] = numIndices
        data['numMappings'] = numMappings
        data['totalTriangles'] = totalTriangles
        data['hasGrid'] = bool(hasGrid)
        
        # Skip main data arrays to read grid
        f.seek(48 + numVertices * 12 + numIndices * 12 + numMappings * 4)
        
        if hasGrid:
            # Read GridParams (40 bytes)
            grid_params = f.read(40)
            minBound = struct.unpack('<3f', grid_params[:12])
            maxBound = struct.unpack('<3f', grid_params[12:24])
            resolution = struct.unpack('<3I', grid_params[24:36])
            
            grid_dict: Dict[str, Any] = {
                'minBound': minBound,
                'maxBound': maxBound,
                'resolution': resolution,
            }
            data['grid'] = grid_dict
            
            numCells = resolution[0] * resolution[1] * resolution[2]
            
            # Read GridCells
            cells_data = f.read(numCells * 16)
            cells = np.frombuffer(cells_data, dtype=[
                ('CenterCount', np.uint32),
                ('TouchCount', np.uint32),
                ('AvgSizeMean', np.float32),
                ('VolRatio', np.float32),
            ])
            
            grid_dict['cells'] = cells
        
    return data


def analyze_grid(data: Dict[str, Any], name: str):
    """Analyze grid statistics and print summary."""
    if not data['hasGrid']:
        print(f"No grid data in {name}")
        return
    
    grid = data['grid']
    cells = grid['cells']
    resolution = grid['resolution']
    minBound = grid['minBound']
    maxBound = grid['maxBound']
    
    # Calculate world and cell size
    worldSize = tuple(maxBound[i] - minBound[i] for i in range(3))
    cellSize = tuple(worldSize[i] / resolution[i] for i in range(3))
    cellVolume = cellSize[0] * cellSize[1] * cellSize[2]
    
    numCells = resolution[0] * resolution[1] * resolution[2]
    nonEmptyCells = np.sum(cells['TouchCount'] > 0)
    
    print(f"\n{'='*60}")
    print(f"Analysis of {name}")
    print(f"{'='*60}")
    print(f"\nBasic Info:")
    print(f"  Vertices:    {data['numVertices']:,}")
    print(f"  Triangles:   {data['numIndices']:,}")
    print(f"  Objects:     {data['totalTriangles']:,}")
    
    print(f"\nGrid Bounds:")
    print(f"  Min: ({minBound[0]:.4f}, {minBound[1]:.4f}, {minBound[2]:.4f})")
    print(f"  Max: ({maxBound[0]:.4f}, {maxBound[1]:.4f}, {maxBound[2]:.4f})")
    print(f"  World Size: ({worldSize[0]:.4f}, {worldSize[1]:.4f}, {worldSize[2]:.4f})")
    
    print(f"\nGrid Resolution: {resolution[0]} x {resolution[1]} x {resolution[2]} = {numCells:,} cells")
    print(f"Cell Size: ({cellSize[0]:.6f}, {cellSize[1]:.6f}, {cellSize[2]:.6f})")
    print(f"Cell Volume: {cellVolume:.10f}")
    
    print(f"\nCell Statistics:")
    print(f"  Non-empty cells: {nonEmptyCells:,} ({100*nonEmptyCells/numCells:.2f}%)")
    
    # TouchCount statistics
    touch_counts = cells['TouchCount']
    nonzero_touch = touch_counts[touch_counts > 0]
    print(f"\n  TouchCount (non-zero cells):")
    print(f"    Min:    {np.min(nonzero_touch):,}")
    print(f"    Max:    {np.max(nonzero_touch):,}")
    print(f"    Mean:   {np.mean(nonzero_touch):.2f}")
    print(f"    Median: {np.median(nonzero_touch):.2f}")
    print(f"    Sum:    {np.sum(touch_counts):,}")
    
    # CenterCount statistics
    center_counts = cells['CenterCount']
    nonzero_center = center_counts[center_counts > 0]
    if len(nonzero_center) > 0:
        print(f"\n  CenterCount (non-zero cells):")
        print(f"    Non-zero count: {len(nonzero_center):,}")
        print(f"    Max:    {np.max(nonzero_center):,}")
        print(f"    Mean:   {np.mean(nonzero_center):.2f}")
        print(f"    Sum:    {np.sum(center_counts):,}")
    
    # AvgSizeMean statistics
    avg_sizes = cells['AvgSizeMean'][cells['TouchCount'] > 0]
    print(f"\n  AvgSizeMean (non-zero cells):")
    print(f"    Min:    {np.min(avg_sizes):.6f}")
    print(f"    Max:    {np.max(avg_sizes):.6f}")
    print(f"    Mean:   {np.mean(avg_sizes):.6f}")
    print(f"    Median: {np.median(avg_sizes):.6f}")
    
    # Compare to cell size
    avg_obj_size = np.mean(avg_sizes)
    print(f"\n  Object Size vs Cell Size:")
    print(f"    Avg object size: {avg_obj_size:.6f}")
    print(f"    Cell diagonal:   {np.sqrt(sum(c**2 for c in cellSize)):.6f}")
    print(f"    Ratio (obj/cell diagonal): {avg_obj_size / np.sqrt(sum(c**2 for c in cellSize)):.4f}")
    
    # VolRatio statistics
    vol_ratios = cells['VolRatio'][cells['TouchCount'] > 0]
    print(f"\n  VolRatio (non-zero cells):")
    print(f"    Min:    {np.min(vol_ratios):.6f}")
    print(f"    Max:    {np.max(vol_ratios):.6f}")
    print(f"    Mean:   {np.mean(vol_ratios):.6f}")
    print(f"    Median: {np.median(vol_ratios):.6f}")
    
    return {
        'cellVolume': cellVolume,
        'cellSize': cellSize,
        'cells': cells,
        'avgSize': avg_obj_size,
    }


def estimate_overlap(data1: Dict[str, Any], data2: Dict[str, Any], gamma: float = 0.8, epsilon: float = 0.001) -> Dict[str, Any]:
    """
    Reproduce the overlap estimation algorithm.
    Returns: dictionary with estimation results
    """
    grid1 = data1['grid']
    grid2 = data2['grid']
    cells1 = grid1['cells']
    cells2 = grid2['cells']
    
    res = grid1['resolution']
    minBound = grid1['minBound']
    maxBound = grid1['maxBound']
    
    worldSize = tuple(maxBound[i] - minBound[i] for i in range(3))
    cellSize = tuple(worldSize[i] / res[i] for i in range(3))
    cellVolume = cellSize[0] * cellSize[1] * cellSize[2]
    
    # Per-cell estimation
    raw_estimate = 0.0
    
    for i in range(len(cells1)):
        tc1 = cells1['TouchCount'][i]
        tc2 = cells2['TouchCount'][i]
        
        if tc1 == 0 or tc2 == 0:
            continue
        
        avg1 = cells1['AvgSizeMean'][i]
        avg2 = cells2['AvgSizeMean'][i]
        vr1 = cells1['VolRatio'][i]
        vr2 = cells2['VolRatio'][i]
        
        combined_size = avg1 + avg2 + epsilon
        minkowski_vol = combined_size ** 3
        prob = minkowski_vol / cellVolume
        
        combined_ratio = np.sqrt(vr1 * vr2)
        shape_correction = combined_ratio ** gamma
        
        prob *= shape_correction
        prob = min(prob, 1.0)
        
        raw_estimate += tc1 * tc2 * prob
    
    # Calculate global average sizes and volume ratios
    def calc_global_avg_size(cells):
        total_size = 0.0
        total_count = 0
        for c in cells:
            if c['TouchCount'] > 0:
                total_size += c['AvgSizeMean'] * c['TouchCount']
                total_count += c['TouchCount']
        return total_size / total_count if total_count > 0 else 0.0
    
    def calc_global_avg_vol_ratio(cells):
        total_ratio = 0.0
        total_count = 0
        for c in cells:
            if c['TouchCount'] > 0:
                total_ratio += c['VolRatio'] * c['TouchCount']
                total_count += c['TouchCount']
        return total_ratio / total_count if total_count > 0 else 1.0
    
    avg_size1 = calc_global_avg_size(cells1)
    avg_size2 = calc_global_avg_size(cells2)
    avg_vol_ratio1 = calc_global_avg_vol_ratio(cells1)
    avg_vol_ratio2 = calc_global_avg_vol_ratio(cells2)
    
    # Shape-corrected effective sizes
    effective_size1 = avg_size1 * np.cbrt(avg_vol_ratio1)
    effective_size2 = avg_size2 * np.cbrt(avg_vol_ratio2)
    
    combined_size = effective_size1 + effective_size2
    minkowski_vol = combined_size ** 3
    
    alpha = max(minkowski_vol / cellVolume, 1.0)
    
    final_estimate = raw_estimate / alpha
    
    return {
        'raw_estimate': raw_estimate,
        'alpha': alpha, 
        'final_estimate': final_estimate,
        'avg_size1': avg_size1,
        'avg_size2': avg_size2,
        'avg_vol_ratio1': avg_vol_ratio1,
        'avg_vol_ratio2': avg_vol_ratio2,
        'effective_size1': effective_size1,
        'effective_size2': effective_size2,
        'cellVolume': cellVolume,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze .pre files for debugging selectivity estimation")
    parser.add_argument("files", nargs='+', help=".pre files to analyze")
    parser.add_argument("--gamma", type=float, default=0.8, help="Gamma parameter for estimation")
    parser.add_argument("--epsilon", type=float, default=0.001, help="Epsilon parameter for estimation")
    parser.add_argument("--compare", action="store_true", help="Compare first two files for overlap estimation")
    args = parser.parse_args()
    
    all_data = {}
    
    for filepath in args.files:
        path = Path(filepath)
        print(f"\nLoading {path.name}...")
        data = read_pre_file(filepath)
        all_data[path.name] = analyze_grid(data, path.name)
        all_data[path.name]['data'] = data
    
    if args.compare and len(args.files) >= 2:
        file1, file2 = args.files[:2]
        data1 = read_pre_file(file1)
        data2 = read_pre_file(file2)
        
        print(f"\n{'='*60}")
        print(f"Overlap Estimation: {Path(file1).name} x {Path(file2).name}")
        print(f"{'='*60}")
        
        result = estimate_overlap(data1, data2, args.gamma, args.epsilon)
        
        print(f"\nEstimation Parameters:")
        print(f"  Gamma:   {args.gamma}")
        print(f"  Epsilon: {args.epsilon}")
        
        print(f"\nResults:")
        print(f"  Cell Volume:          {result['cellVolume']:.10f}")
        print(f"  Avg Object Size 1:    {result['avg_size1']:.6f}")
        print(f"  Avg Object Size 2:    {result['avg_size2']:.6f}")
        print(f"  Avg VolRatio 1:       {result['avg_vol_ratio1']:.6f}")
        print(f"  Avg VolRatio 2:       {result['avg_vol_ratio2']:.6f}")
        print(f"  Effective Size 1:     {result['effective_size1']:.6f}")
        print(f"  Effective Size 2:     {result['effective_size2']:.6f}")
        print(f"  Combined Eff Size:    {result['effective_size1'] + result['effective_size2']:.6f}")
        print(f"  Alpha (replication):  {result['alpha']:.4f}")
        print(f"  Raw Estimate:         {result['raw_estimate']:.2f}")
        print(f"  Final Estimate:       {result['final_estimate']:.2f}")
        
        # Diagnose potential issues
        print(f"\n{'='*60}")
        print("Diagnostic Analysis")
        print(f"{'='*60}")
        
        # Check if alpha is too high
        if result['alpha'] > 50:
            print(f"⚠️  High alpha ({result['alpha']:.2f}): May still be over-correcting")
        elif result['alpha'] < 1.5:
            print(f"ℹ️  Low alpha ({result['alpha']:.2f}): Objects are smaller than cells")
        
        # Check cell coverage
        cells1 = data1['grid']['cells']
        cells2 = data2['grid']['cells']
        
        both_occupied = np.sum((cells1['TouchCount'] > 0) & (cells2['TouchCount'] > 0))
        only1 = np.sum((cells1['TouchCount'] > 0) & (cells2['TouchCount'] == 0))
        only2 = np.sum((cells1['TouchCount'] == 0) & (cells2['TouchCount'] > 0))
        
        print(f"\nCell Overlap:")
        print(f"  Cells with both datasets: {both_occupied:,}")
        print(f"  Cells with only dataset1: {only1:,}")
        print(f"  Cells with only dataset2: {only2:,}")
        
        # Analyze where pairs could be missed
        # Check probability values in overlapping cells
        prob_values = []
        cellVol = result['cellVolume']
        for i in range(len(cells1)):
            tc1 = cells1['TouchCount'][i]
            tc2 = cells2['TouchCount'][i]
            
            if tc1 == 0 or tc2 == 0:
                continue
            
            avg1_cell = cells1['AvgSizeMean'][i]
            avg2_cell = cells2['AvgSizeMean'][i]
            vr1 = cells1['VolRatio'][i]
            vr2 = cells2['VolRatio'][i]
            
            combined_size = avg1_cell + avg2_cell + args.epsilon
            minkowski_vol = combined_size ** 3
            prob = minkowski_vol / cellVol
            
            combined_ratio = np.sqrt(vr1 * vr2)
            shape_correction = combined_ratio ** args.gamma
            
            prob *= shape_correction
            prob = min(prob, 1.0)
            prob_values.append(prob)
        
        prob_values = np.array(prob_values)
        print(f"\nProbability Distribution in Overlapping Cells:")
        print(f"  Min:    {np.min(prob_values):.6f}")
        print(f"  Max:    {np.max(prob_values):.6f}")
        print(f"  Mean:   {np.mean(prob_values):.6f}")
        print(f"  Median: {np.median(prob_values):.6f}")
        print(f"  Cells at prob=1.0: {np.sum(prob_values >= 1.0):,}")
        
        if np.mean(prob_values) < 0.5:
            print(f"\n⚠️  Low average probability ({np.mean(prob_values):.4f}): May be underestimating")
            print(f"    Check if objects are very sparse or if cell volume is too large")


if __name__ == "__main__":
    main()