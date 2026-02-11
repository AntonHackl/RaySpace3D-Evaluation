"""CGAL baseline adapter."""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from .base import SpatialQueryAdapter
from .utils import translate_obj, run_subprocess_streaming


class CGALAdapter(SpatialQueryAdapter):
    """Adapter for CGAL baseline."""
    
    def __init__(self, workspace: str, cgal_basedir: str):
        super().__init__("CGAL", workspace)
        self.cgal_basedir = Path(cgal_basedir)
        self.executable = self.cgal_basedir / "build" / "cgal_query"
    
    def setup(self, **kwargs) -> bool:
        """Build CGAL executable if needed."""
        if self.executable.exists():
            print(f"[CGAL] Executable already exists: {self.executable}")
            return True
        
        print(f"[CGAL] Building...")
        build_script = self.cgal_basedir / "scripts" / "build.sh"
        if not build_script.exists():
            print(f"[CGAL] Build script not found: {build_script}")
            return False
        
        try:
            # Activate conda environment and build
            run_subprocess_streaming(
                ["bash", "-c", 
                 f"source $(conda info --base)/etc/profile.d/conda.sh && "
                 f"conda activate cgal_spatial && "
                 f"cd {self.cgal_basedir} && bash {build_script}"],
                prefix="[CGAL]",
                check=True
            )
            print(f"[CGAL] Build successful")
            return self.executable.exists()
        except subprocess.CalledProcessError as e:
            print(f"[CGAL] Build failed: {e.stderr}")
            return False
    
    def execute_query(
        self,
        geometry_path: str,
        points_path: str,
        grid_pos: Tuple[int, int, int],
        translation: np.ndarray,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute CGAL query."""
        # Translate geometry
        gx, gy, gz = grid_pos
        translated_obj = os.path.join(self.workspace, f"translated_{gx}_{gy}_{gz}.obj")
        translate_obj(geometry_path, translated_obj, translation)
        
        # Run CGAL query with conda environment
        cmd = f"""
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cgal_spatial
{self.executable} {translated_obj} {points_path}
"""
        
        try:
            result, stdout, stderr = run_subprocess_streaming(
                ["bash", "-c", cmd],
                prefix="[CGAL]",
                timeout=3600,
                check=True
            )
            
            # Parse stdout for timing and results
            
            query_time_ms = None
            inside_count = None
            total_points = None
            
            # Look for "CONTAINMENT QUERY TIME: XXX ms"
            match = re.search(r'CONTAINMENT QUERY TIME:\s*([0-9.]+)\s*ms', stdout)
            if match:
                query_time_ms = float(match.group(1))
            
            # Look for "Points inside mesh: XXX"
            match = re.search(r'Points inside mesh:\s*(\d+)', stdout)
            if match:
                inside_count = int(match.group(1))
            
            # Look for "Total points: XXX"
            match = re.search(r'Total points:\s*(\d+)', stdout)
            if match:
                total_points = int(match.group(1))
            
            return {
                'query_ms': query_time_ms,
                'total_query_ms': query_time_ms,
                'inside_count': inside_count,
                'total_points': total_points,
                'success': query_time_ms is not None,
                'stdout': stdout
            }
            
        except subprocess.TimeoutExpired:
            return {'success': False, 'error': 'Timeout'}
        except subprocess.CalledProcessError as e:
            return {'success': False, 'error': stderr if 'stderr' in locals() else str(e)}
        except Exception as e:
            return {'success': False, 'error': str(e)}
        finally:
            # Cleanup translated file
            if os.path.exists(translated_obj):
                os.remove(translated_obj)
    
    def cleanup(self):
        """Cleanup workspace."""
        pass
