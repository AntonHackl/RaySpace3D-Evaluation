"""CUDA baseline adapter."""

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from .base import SpatialQueryAdapter
from .utils import translate_obj, run_subprocess_streaming


class CUDAAdapter(SpatialQueryAdapter):
    """Adapter for pure CUDA baseline (bbox filter + ray-triangle intersection)."""
    
    def __init__(self, workspace: str, cuda_dir: str):
        super().__init__("CUDA", workspace)
        self.cuda_dir = Path(cuda_dir)
        self.executable = self.cuda_dir / "build" / "cuda_query"
        self.build_script = self.cuda_dir / "scripts" / "build.sh"
    
    def setup(self, **kwargs) -> bool:
        """Build CUDA executable if needed."""
        # Check if executable exists
        if self.executable.exists():
            print(f"[CUDA] Executable already exists: {self.executable}")
            return True
        
        # Check if build script exists
        if not self.build_script.exists():
            print(f"[CUDA] Build script not found: {self.build_script}")
            return False
        
        # Build the executable
        print(f"[CUDA] Building executable...")
        try:
            run_subprocess_streaming(
                ["bash", str(self.build_script)],
                prefix="[CUDA]",
                timeout=600,
                check=True
            )
            
            if not self.executable.exists():
                print(f"[CUDA] Build completed but executable not found: {self.executable}")
                return False
            
            print(f"[CUDA] Build successful: {self.executable}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"[CUDA] Build failed: {e}")
            return False
        except Exception as e:
            print(f"[CUDA] Build error: {e}")
            return False
    
    def execute_query(
        self,
        geometry_path: str,
        points_path: str,
        grid_pos: Tuple[int, int, int],
        translation: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute CUDA query."""
        gx, gy, gz = grid_pos
        
        # Translate OBJ
        translated_obj = os.path.join(self.workspace, f"mesh_{gx}_{gy}_{gz}.obj")
        translate_obj(geometry_path, translated_obj, translation)
        
        # Output timing JSON
        timing_json = os.path.join(self.workspace, f"timing_{gx}_{gy}_{gz}.json")
        
        # Run CUDA query executable
        cmd = [
            str(self.executable),
            translated_obj,
            points_path,
            timing_json
        ]
        
        try:
            result, stdout, stderr = run_subprocess_streaming(
                cmd,
                prefix="[CUDA]",
                timeout=3600,
                check=True
            )
            
            # Read timing JSON
            with open(timing_json, 'r') as f:
                timing_data = json.load(f)
            
            # Extract timings
            phases = timing_data.get('phases', {})
            
            def _dur(k):
                v = phases.get(k)
                if isinstance(v, dict):
                    return v.get('duration_ms', 0.0)
                return 0.0
            
            upload_geom_ms = _dur('upload_geometry_1')
            upload_points_ms = _dur('upload_points_1')
            filter_ms = _dur('filter_1')
            query_ms = _dur('query_1')
            download_ms = _dur('download_results_1')
            
            # Total query time = upload_geometry + filter + query + download
            # (upload_points is excluded as it's a one-time cost, not per-query)
            # (filter is the bbox filtering phase, query is the ray-triangle intersection)
            total_query_ms = upload_geom_ms + filter_ms + query_ms + download_ms
            
            # Extract result counts
            num_inside = timing_data.get('num_inside', 0)
            num_points = timing_data.get('num_points', 0)
            
            return {
                'query_ms': query_ms,
                'filter_ms': filter_ms,
                'upload_geometry_ms': upload_geom_ms,
                'upload_points_ms': upload_points_ms,
                'download_ms': download_ms,
                'total_query_ms': total_query_ms,
                'inside_count': num_inside,
                'total_points': num_points,
                'success': True,
                'timing_data': timing_data
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Timeout'}
        except subprocess.CalledProcessError as e:
            return {'success': False, 'error': stderr if 'stderr' in locals() else str(e)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def cleanup(self):
        """Cleanup workspace."""
        pass
