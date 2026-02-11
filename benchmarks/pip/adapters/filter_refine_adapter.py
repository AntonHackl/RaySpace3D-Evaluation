"""RaySpace3D Filter-Refine adapter."""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from .base import SpatialQueryAdapter
from .utils import translate_obj, run_subprocess_streaming


class FilterRefineAdapter(SpatialQueryAdapter):
    """Adapter for RaySpace3D raytracer_filter_refine."""
    
    def __init__(self, workspace: str, rayspace_dir: str):
        super().__init__("FilterRefine", workspace)
        self.rayspace_dir = Path(rayspace_dir)
        self.executable = self.rayspace_dir / "build" / "bin" / "raytracer_filter_refine"
        self.preprocess_exec = self.rayspace_dir / "build" / "bin" / "preprocess_dataset"
    
    def setup(self, **kwargs) -> bool:
        """Check executable exists."""
        if not self.executable.exists():
            print(f"[FilterRefine] Executable not found: {self.executable}")
            return False
        
        if not self.preprocess_exec.exists():
            print(f"[FilterRefine] Preprocess executable not found: {self.preprocess_exec}")
            return False
        
        return True
    
    def execute_query(
        self,
        geometry_path: str,
        points_path: str,
        grid_pos: Tuple[int, int, int],
        translation: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute filter-refine query."""
        gx, gy, gz = grid_pos
        
        # Translate OBJ and place in its own subdirectory (required by preprocess_dataset)
        obj_dir = os.path.join(self.workspace, f"mesh_{gx}_{gy}_{gz}")
        os.makedirs(obj_dir, exist_ok=True)
        translated_obj = os.path.join(obj_dir, "mesh.obj")
        translate_obj(geometry_path, translated_obj, translation)
        
        # Preprocess
        preprocessed_geom = os.path.join(self.workspace, f"geom_{gx}_{gy}_{gz}.txt")
        preprocess_timing = os.path.join(self.workspace, f"preprocess_timing_{gx}_{gy}_{gz}.json")
        
        cmd_preprocess = [
            str(self.preprocess_exec),
            "--mode", "mesh",
            "--dataset", obj_dir,
            "--output-geometry", preprocessed_geom,
            "--output-timing", preprocess_timing
        ]
        
        try:
            run_subprocess_streaming(
                cmd_preprocess,
                prefix="[FilterRefine]",
                timeout=600,
                check=True
            )
        except Exception as e:
            return {'success': False, 'error': f'Preprocessing failed: {str(e)}'}
        
        # Run filter-refine
        timing_json = os.path.join(self.workspace, f"timing_fr_{gx}_{gy}_{gz}.json")
        
        cmd_raytrace = [
            str(self.executable),
            "--geometry", preprocessed_geom,
            "--points", points_path,
            "--output", timing_json,
            "--no-export"
        ]
        
        try:
            result, stdout, stderr = run_subprocess_streaming(
                cmd_raytrace,
                prefix="[FilterRefine]",
                timeout=3600,
                check=True
            )
            
            # Read timing JSON
            with open(timing_json, 'r') as f:
                timing_data = json.load(f)
            
            # Extract timings robustly (keys may be lowercase and have suffixes like "_1")
            phases = timing_data.get('phases', {})

            # Initialize accumulators
            upload_bbox = 0.0
            upload_geom = 0.0
            upload_points = 0.0
            query_ms = None
            download_ms = None
            filter_ms = None

            for key, val in phases.items():
                nk = key.lower()
                dur = val.get('duration_ms') if isinstance(val, dict) else None
                if dur is None:
                    # try alternate field
                    dur = val.get('duration_us', 0) / 1000.0 if isinstance(val, dict) else None
                if dur is None:
                    continue

                # Upload components
                if 'upload' in nk and 'bbox' in nk:
                    upload_bbox += float(dur)
                elif 'upload' in nk and 'geometry' in nk and 'bbox' not in nk:
                    upload_geom += float(dur)
                elif 'upload' in nk and 'points' in nk:
                    upload_points += float(dur)

                # Filter phase (filter_1)
                if 'filter' in nk:
                    filter_ms = float(dur)

                # Main query phase (exclude warmup and bbox-related queries)
                if 'query' in nk and 'warmup' not in nk:
                    # prefer a plain 'query' phase (may include suffix), but if multiple
                    # exist, choose the first non-bbox, non-warmup occurrence
                    if query_ms is None:
                        query_ms = float(dur)

                # Download / output phases
                if 'download' in nk or 'output' in nk:
                    # prefer download results, fallback to output
                    if download_ms is None:
                        download_ms = float(dur)

            upload_ms = upload_bbox + upload_geom + upload_points

            # Compute total RaySpace query time if available
            rayspace_total_ms = None

            # If timing JSON contains the explicit keys we expect, prefer the simple sum
            # Filter-Refine expected keys (common): 'upload points_1', 'upload query geometry_1', 'query_1', 'download results_1'
            def _dur(k):
                v = phases.get(k)
                if isinstance(v, dict):
                    return v.get('duration_ms') or (v.get('duration_us', 0) / 1000.0)
                return None

            explicit_keys = [
                'upload points_1',
                'upload query geometry_1',
                'filter_1',
                'query_1',
                'download results_1'
            ]

            explicit_vals = [(_dur(k) or 0.0) for k in explicit_keys]
            if any(phases.get(k) is not None for k in explicit_keys):
                # Use sum of available explicit keys (missing ones are treated as 0)
                rayspace_total_ms = sum(explicit_vals)
            else:
                # Compute total RaySpace query time (upload + filter + query + download) if available
                # Include filter_1 if available
                filter_val = filter_ms if filter_ms is not None else 0.0
                if all(x is not None for x in [upload_ms, query_ms, download_ms]):
                    rayspace_total_ms = float(upload_ms or 0.0) + float(filter_val) + float(query_ms or 0.0) + float(download_ms or 0.0)
            
            # Compute total_query_ms: upload geometry_1 + build index_1 + query_1 + download results_1 + filter_1
            total_query_ms = None
            # For filter_refine, geometry upload might be "upload query geometry_1" or "upload geometry_1"
            upload_geom_dur = _dur('upload query geometry_1') or _dur('upload geometry_1') or 0.0
            # Build index might be "build query index_1" or "build index_1"
            build_index_dur = _dur('build query index_1') or _dur('build index_1') or 0.0
            query_dur = _dur('query_1') or 0.0
            download_dur = _dur('download results_1') or 0.0
            # Filter phase is "filter_1"
            filter_dur = _dur('filter_1') or 0.0
            
            total_query_ms = upload_geom_dur + build_index_dur + query_dur + download_dur + filter_dur
            if total_query_ms == 0.0:
                total_query_ms = None
            
            # Parse stdout
            inside_count = None
            total_points = None
            
            match = re.search(r'Points INSIDE polygons:\s*(\d+)', stdout)
            if match:
                inside_count = int(match.group(1))
            
            match = re.search(r'Total points:\s*(\d+)', stdout)
            if match:
                total_points = int(match.group(1))
            
            return {
                'query_ms': query_ms,
                'filter_ms': filter_ms,
                'upload_ms': upload_ms,
                'download_ms': download_ms,
                'total_query_ms': total_query_ms,
                'inside_count': inside_count,
                'total_points': total_points,
                'success': query_ms is not None,
                'timing_data': timing_data
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Timeout'}
        except subprocess.CalledProcessError as e:
            return {'success': False, 'error': stderr if 'stderr' in locals() else str(e)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            # Cleanup
            import shutil
            for f in [preprocessed_geom, preprocess_timing, timing_json]:
                if os.path.exists(f):
                    os.remove(f)
            # Remove the mesh directory
            if os.path.exists(obj_dir):
                shutil.rmtree(obj_dir)
    
    def cleanup(self):
        """Cleanup workspace."""
        pass
