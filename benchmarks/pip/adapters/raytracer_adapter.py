"""RaySpace3D Raytracer adapter."""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from .base import SpatialQueryAdapter
from .utils import translate_obj, run_subprocess_streaming


class RaytracerAdapter(SpatialQueryAdapter):
    """Adapter for RaySpace3D raytracer."""
    
    def __init__(self, workspace: str, rayspace_dir: str):
        super().__init__("Raytracer", workspace)
        self.rayspace_dir = Path(rayspace_dir)
        self.executable = self.rayspace_dir / "build" / "bin" / "raytracer"
        self.preprocess_exec = self.rayspace_dir / "build" / "bin" / "preprocess_dataset"
    
    def setup(self, **kwargs) -> bool:
        """Check raytracer executable exists."""
        if not self.executable.exists():
            print(f"[Raytracer] Executable not found: {self.executable}")
            print(f"[Raytracer] Please build RaySpace3D first")
            return False
        
        if not self.preprocess_exec.exists():
            print(f"[Raytracer] Preprocess executable not found: {self.preprocess_exec}")
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
        """Execute raytracer query."""
        gx, gy, gz = grid_pos
        
        # Translate OBJ and place in its own subdirectory (required by preprocess_dataset)
        obj_dir = os.path.join(self.workspace, f"mesh_{gx}_{gy}_{gz}")
        os.makedirs(obj_dir, exist_ok=True)
        translated_obj = os.path.join(obj_dir, "mesh.obj")
        translate_obj(geometry_path, translated_obj, translation)
        
        # Preprocess to get geometry file
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
                prefix="[Raytracer]",
                timeout=600,
                check=True
            )
        except Exception as e:
            return {'success': False, 'error': f'Preprocessing failed: {str(e)}'}
        
        # Run raytracer
        timing_json = os.path.join(self.workspace, f"timing_{gx}_{gy}_{gz}.json")
        
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
                prefix="[Raytracer]",
                timeout=3600,
                check=True
            )
            
            # Read timing JSON
            with open(timing_json, 'r') as f:
                timing_data = json.load(f)
            
            # Extract relevant timings robustly (keys may vary/case and include suffixes)
            phases = timing_data.get('phases', {})

            upload_points = 0.0
            upload_geom = 0.0
            query_ms = None
            download_ms = None

            for key, val in phases.items():
                nk = key.lower()
                dur = None
                if isinstance(val, dict):
                    dur = val.get('duration_ms')
                    if dur is None and 'duration_us' in val:
                        dur = val.get('duration_us', 0) / 1000.0
                if dur is None:
                    continue

                # Upload parts
                if 'upload' in nk and 'points' in nk:
                    upload_points += float(dur)
                elif 'upload' in nk and 'geometry' in nk:
                    upload_geom += float(dur)

                # Query phase (avoid warmup or bbox labels)
                if 'query' in nk and 'warmup' not in nk and 'bbox' not in nk:
                    if query_ms is None:
                        query_ms = float(dur)

                # Download / output
                if 'download' in nk or 'output' in nk:
                    if download_ms is None:
                        download_ms = float(dur)

            upload_ms = upload_points + upload_geom

            # If timing JSON contains the explicit keys we expect, prefer the simple sum
            # Raytracer expected keys: 'upload points_1', 'upload query geometry_1' or 'upload geometry_1', 'query_1', 'download results_1'
            def _dur(k):
                v = phases.get(k)
                if isinstance(v, dict):
                    return v.get('duration_ms') or (v.get('duration_us', 0) / 1000.0)
                return None

            explicit_keys = [
                'upload geometry_1',
                'build index_1',
                'query_1',
                'download results_1'
            ]

            # Sum available explicit keys (if any present)
            if any(k in phases for k in explicit_keys):
                explicit_sum = 0.0
                # upload parts
                explicit_sum += (_dur('upload geometry_1') or 0.0)
                explicit_sum += (_dur('build index_1') or 0.0)
                # query and download
                explicit_sum += (_dur('query_1') or 0.0)
                explicit_sum += (_dur('download results_1') or 0.0)
                rayspace_total_ms = explicit_sum
            else:
                # Compute total RaySpace query time (upload + query + download) if available
                rayspace_total_ms = None
                if all(x is not None for x in [upload_ms, query_ms, download_ms]):
                    rayspace_total_ms = float(upload_ms or 0.0) + float(query_ms or 0.0) + float(download_ms or 0.0)
            
            # Compute total_query_ms: upload geometry_1 + build index_1 + query_1 + download results_1
            total_query_ms = None
            upload_geom_dur = _dur('upload geometry_1') or 0.0
            build_index_dur = _dur('build index_1') or 0.0
            query_dur = _dur('query_1') or 0.0
            download_dur = _dur('download results_1') or 0.0
            
            total_query_ms = upload_geom_dur + build_index_dur + query_dur + download_dur
            if total_query_ms == 0.0:
                total_query_ms = None
            
            # Parse stdout for results
            inside_count = None
            total_points = None
            
            match = re.search(r'Points INSIDE polygons:\s*(\d+)', stdout)
            if match:
                inside_count = int(match.group(1))
            
            match = re.search(r'Total rays:\s*(\d+)', stdout)
            if match:
                total_points = int(match.group(1))
            
            return {
                'query_ms': query_ms,
                'upload_geometry_ms': upload_geom_dur,
                'build_index_ms': build_index_dur,
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
        # finally:
        #     # Cleanup
        #     import shutil
        #     for f in [preprocessed_geom, preprocess_timing, timing_json]:
        #         if os.path.exists(f):
        #             os.remove(f)
        #     # Remove the mesh directory
        #     if os.path.exists(obj_dir):
        #         shutil.rmtree(obj_dir)
    
    def cleanup(self):
        """Cleanup workspace."""
        pass
