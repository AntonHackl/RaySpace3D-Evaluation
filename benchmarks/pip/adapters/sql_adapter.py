"""PostgreSQL/PostGIS baseline adapter."""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np

from .base import SpatialQueryAdapter
from .utils import translate_obj, wkt_points_to_csv, run_subprocess_streaming


class SQLAdapter(SpatialQueryAdapter):
    """Adapter for PostgreSQL/PostGIS baseline."""
    
    def __init__(self, workspace: str, sql_basedir: str, db_name: str = "spatial3d"):
        super().__init__("SQL", workspace)
        # Resolve the SQL base directory to an absolute path to avoid
        # relative-path confusion when running subprocesses that `cd`.
        self.sql_basedir = Path(sql_basedir).resolve()
        self.executable = self.sql_basedir / "build" / "spatial_query"
        self.db_name = db_name
        self.points_loaded = False
    
    def setup(self, points_path: str, **kwargs) -> bool:
        """Build SQL executable and load points once."""
        # Build if needed
        if not self.executable.exists():
            print(f"[SQL] Building...")
            # Run the build script from within the SQL base dir. Use a
            # path relative to that directory (scripts/build.sh) so the
            # `cd` and script invocation align correctly.
            build_script_rel = "scripts/build.sh"
            if not (self.sql_basedir / build_script_rel).exists():
                print(f"[SQL] Build script not found: {self.sql_basedir / build_script_rel}")
                return False

            try:
                run_subprocess_streaming(
                    ["bash", "-c",
                     f"source $(conda info --base)/etc/profile.d/conda.sh && "
                     f"conda activate spatial3d && "
                     f"cd {self.sql_basedir} && bash {build_script_rel}"],
                    prefix="[SQL]",
                    check=True
                )
                print(f"[SQL] Build successful")
            except subprocess.CalledProcessError as e:
                print(f"[SQL] Build failed: {e.stderr}")
                return False
        
        if not self.executable.exists():
            return False
        
        # Initialize database if needed
        print(f"[SQL] Checking database...")
        # Run the initialization script from within the SQL base dir. Use
        # a relative path (scripts/init_db.sh) after `cd` to avoid the
        # common problem where a path like "../RaySpace3DBaslines/..." is
        # no longer correct once we've `cd`'d into the directory.
        # Pass --yes to avoid interactive prompts that would hang the process
        init_script_rel = "scripts/init_db.sh"
        if (self.sql_basedir / init_script_rel).exists():
            try:
                run_subprocess_streaming(
                    ["bash", "-c",
                     f"source $(conda info --base)/etc/profile.d/conda.sh && "
                     f"conda activate spatial3d && "
                     f"cd {self.sql_basedir} && bash {init_script_rel} --yes"],
                    prefix="[SQL]",
                    check=True
                )
                print(f"[SQL] Database initialized")
            except subprocess.CalledProcessError as e:
                print(f"[SQL] Database init warning: {e.stderr}")
        
        # Load points once (reuse database)
        if not self.points_loaded:
            print(f"[SQL] Loading points to database (one-time per benchmark)...")
            csv_file = os.path.join(self.workspace, "points.csv")
            wkt_points_to_csv(points_path, csv_file)
            
            cmd = f"""
source $(conda info --base)/etc/profile.d/conda.sh
conda activate spatial3d
{self.executable} load_points {csv_file}
"""
            try:
                result, stdout, stderr = run_subprocess_streaming(
                    ["bash", "-c", cmd],
                    prefix="[SQL]",
                    timeout=3600,
                    check=True
                )
                print(f"[SQL] Points loaded successfully")
                self.points_loaded = True
            except subprocess.CalledProcessError as e:
                print(f"[SQL] Failed to load points: {e.stderr}")
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
        """Execute SQL query."""
        # Translate geometry
        gx, gy, gz = grid_pos
        translated_obj = os.path.join(self.workspace, f"translated_{gx}_{gy}_{gz}.obj")
        translate_obj(geometry_path, translated_obj, translation)
        
        # Run SQL query
        cmd = f"""
source $(conda info --base)/etc/profile.d/conda.sh
conda activate spatial3d
{self.executable} query {translated_obj}
"""
        
        try:
            result, stdout, stderr = run_subprocess_streaming(
                ["bash", "-c", cmd],
                prefix="[SQL]",
                timeout=3600,
                check=True
            )
            
            # Parse stdout for timing and results
            
            query_time_ms = None
            inside_count = None
            total_points = None
            
            # Look for "QUERY TIME: XXX ms"
            match = re.search(r'QUERY TIME:\s*([0-9.]+)\s*ms', stdout)
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
