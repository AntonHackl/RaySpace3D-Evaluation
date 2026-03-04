import subprocess
import time
import json
import re
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import OverlapBenchmarkAdapter, run_command_streaming

class RaytracerAdapter(OverlapBenchmarkAdapter):
    def __init__(
        self,
        rayspace_dir: str,
        mode: str = "exact",
        preprocessed_dir: str = "preprocessed",
        timings_dir: str = "timings",
        grid_resolution: int = 10,
        warmup_runs: int = 10,
    ):
        """
        mode: 'exact' or 'estimated'
        grid_resolution: resolution for grid generation (default: 10)
        """
        super().__init__(f"Raytracer_{mode}")
        self.rayspace_dir = Path(rayspace_dir)
        self.mode = mode
        self.preprocessed_dir = Path(preprocessed_dir)
        self.timings_dir = Path(timings_dir)
        self.grid_resolution = grid_resolution
        self.warmup_runs = warmup_runs
        # Ensure directories exist
        self.timings_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine executable based on mode
        # Binaries are in query/build/bin
        query_bin_dir = self.rayspace_dir / "query" / "build" / "bin"
        if self.mode == "exact":
            self.executable = query_bin_dir / "raytracer_mesh_overlap"
        elif self.mode in ("estimated", "estimate_only"):
            self.executable = query_bin_dir / "raytracer_overlap_estimated"
        elif self.mode == "direct_estimation":
            self.executable = query_bin_dir / "raytracer_overlap_direct_estimation"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Preprocess binary is in preprocess/build/bin
        self.preprocess_exec = self.rayspace_dir / "preprocess" / "build" / "bin" / "preprocess_dataset"
        
    def check_preprocessed(self, file_path: str) -> bool:
        """Check if .pre file exists for the given .dt or .obj file in preprocessed dir."""
        input_path = Path(file_path)
        pre_file = self.preprocessed_dir / input_path.with_suffix('.pre').name
        return pre_file.exists()

    def preprocess(self, file_path: str):
        """Run the RaySpace3D preprocessing tool including grid generation."""
        self.preprocess_from_source(file_path, file_path)
    
    def preprocess_from_source(self, source_file: str, dt_file: str, log_dir: Optional[str] = None):
        """Run preprocessing using a source file (.obj) but naming outputs based on dt_file."""
        source_path = Path(source_file)
        dt_path = Path(dt_file)
        
        # Output files are named based on dt_file for consistency, stored in PREPROCESSED_DIR
        output_geometry = self.preprocessed_dir / dt_path.with_suffix('.pre').name
        output_timing = self.timings_dir / (dt_path.stem + '_timing.json')
        
        # Determine mode based on source file extension
        mode = "dt" if source_path.suffix == ".dt" else "mesh"
        
        # Preprocess dataset with grid generation
        cmd = [
            str(self.preprocess_exec),
            "--mode", mode,
            "--dataset", str(source_path),
            "--output-geometry", str(output_geometry),
            "--output-timing", str(output_timing),
            "--generate-grid",
            "--grid-resolution", str(self.grid_resolution)
        ]
        
        print(f"[{self.name}] Preprocessing {source_path.name} (output: {dt_path.name}) with grid (resolution={self.grid_resolution})...")
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = adapter_log_dir / f"preprocess_{dt_path.stem}_{int(time.time())}.log"
            run_command_streaming(cmd, timeout=None, log_path=str(log_path), prefix=f"[{self.name}]")
        else:
            # Stream to terminal without logging
            run_command_streaming(cmd, timeout=None, log_path=None, prefix=f"[{self.name}]")

    def run_overlap(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
        query_direction: str = "both",
        pairs_output: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the overlap join query."""
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}

        # Use preprocessed files if they exist in the preprocessed directory
        input_path1 = Path(file1)
        input_path2 = Path(file2)
        p1 = self.preprocessed_dir / input_path1.with_suffix('.pre').name
        p2 = self.preprocessed_dir / input_path2.with_suffix('.pre').name
        
        f1 = str(p1) if p1.exists() else file1
        f2 = str(p2) if p2.exists() else file2

        runtimes = []
        breakdown_accum = {} # key: phase name, value: list of durations
        num_obj1 = 0
        num_obj2 = 0
        num_intersections = 0
        universe_extents1 = [0.0, 0.0, 0.0]
        universe_extents2 = [0.0, 0.0, 0.0]
        
        print(f"[{self.name}] Running benchmark...")

        adapter_log_dir = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "exact":
            expected_prefixes = [
                "raytrace_",
                "gpu deduplication",
                "download results",
                "query",
            ]
        elif self.mode in ("estimated", "direct_estimation"):
            # For estimated mode, include selectivity estimation in query time
            expected_prefixes = [
                "selectivity estimation",
                "raytrace_hash_",
                "execute hash query",
                "download results",
                "query",
            ]
        else:
            # estimate_only
            expected_prefixes = ["selectivity estimation"]

        # Execute num_runs times, each with warmup
        for run_idx in range(num_runs):
            json_output = self.timings_dir / f"timing_{self.mode}_{int(time.time())}_{run_idx}.json"
            
            cmd = [
                str(self.executable),
                "--mesh1", f1,
                "--mesh2", f2,
                "--runs", "1",
                "--warmup-runs", str(self.warmup_runs),
                "--no-export",
                "--output", str(json_output)
            ]

            if self.mode == "direct_estimation":
                cmd.extend(["--query-direction", query_direction])
                if pairs_output and run_idx == (num_runs - 1):
                    cmd.extend(["--pairs-output", str(pairs_output)])

            if self.mode == "estimate_only":
                cmd.append("--estimate-only")
            
            try:
                log_path = None
                if adapter_log_dir is not None:
                    log_path = str(adapter_log_dir / f"run_{run_idx:03d}.log")

                stdout_text, stderr_text = run_command_streaming(
                    cmd,
                    timeout=timeout,
                    log_path=log_path,
                    prefix=f"[{self.name}]",
                )
                
                # Parse summary from stdout on the first run
                if run_idx == 0:
                    lines = stdout_text.splitlines()
                    def parse_vec3(l):
                        return [float(p.strip()) for p in l.split("[")[1].split("]")[0].split(",")]
                        
                    for i, line in enumerate(lines):
                        if "Mesh1 objects:" in line:
                            num_obj1 = int(line.split(":")[1].strip())
                        elif "Mesh2 objects:" in line:
                            num_obj2 = int(line.split(":")[1].strip())
                        elif "Unique object pairs:" in line:
                            num_intersections = int(line.split(":")[1].strip())
                        elif "Hash Table Query found" in line:
                            num_intersections = int(line.split("found")[1].split("unique")[0].strip())
                        elif "Final Estimated Pairs:" in line:
                            try:
                                num_intersections = int(line.split(":", 1)[1].strip())
                            except ValueError: pass
                        elif "Universe Extents:" in line:
                            try:
                                ext = parse_vec3(line)
                                universe_extents1 = ext
                                universe_extents2 = ext
                            except Exception: pass
                        elif "Mesh1 Universe Min:" in line and (i+1) < len(lines) and "Mesh1 Universe Max:" in lines[i+1]:
                            try:
                                v_min = parse_vec3(line)
                                v_max = parse_vec3(lines[i+1])
                                universe_extents1 = [v_max[j] - v_min[j] for j in range(3)]
                            except Exception: pass
                        elif "Mesh2 Universe Min:" in line and (i+1) < len(lines) and "Mesh2 Universe Max:" in lines[i+1]:
                            try:
                                v_min = parse_vec3(line)
                                v_max = parse_vec3(lines[i+1])
                                universe_extents2 = [v_max[j] - v_min[j] for j in range(3)]
                            except Exception: pass

                if not json_output.exists():
                    return {"error": f"Timing JSON not found at {json_output}. Output:\n{stdout_text + stderr_text}"}

                with open(json_output, 'r') as f:
                    data = json.load(f)

                phases = data.get("phases", {})
                phase_values = {}
                for key, phase_data in phases.items():
                    normalized_key = re.sub(r"_\d+$", "", key.lower())
                    phase_values[normalized_key] = phase_values.get(normalized_key, 0.0) + phase_data.get("duration_ms", 0.0)

                has_detailed_raytrace = any(k.startswith("raytrace_") for k in phase_values.keys())

                if self.mode == "exact":
                    query_time = phase_values.get("query", 0.0)
                    if query_time <= 0.0:
                        query_time = sum(v for k, v in phase_values.items() if k.startswith("raytrace_"))
                        query_time += phase_values.get("gpu deduplication", 0.0)
                    query_time += phase_values.get("download results", 0.0)
                elif self.mode in ("estimated", "direct_estimation"):
                    query_time = phase_values.get("selectivity estimation", 0.0)
                    execute_hash = phase_values.get("execute hash query", 0.0)
                    if execute_hash > 0.0:
                        query_time += execute_hash
                    else:
                        query_time += sum(v for k, v in phase_values.items() if k.startswith("raytrace_hash_"))
                    query_time += phase_values.get("download results", 0.0)
                else:
                    query_time = phase_values.get("selectivity estimation", 0.0)

                found = query_time > 0.0

                for normalized_key, duration in phase_values.items():
                    if not any(normalized_key.startswith(prefix) for prefix in expected_prefixes):
                        continue
                    if has_detailed_raytrace and normalized_key in ("query", "execute hash query"):
                        continue
                    if normalized_key not in breakdown_accum:
                        breakdown_accum[normalized_key] = []
                    breakdown_accum[normalized_key].append(duration)

                if not found:
                    return {"error": f"Expected timing phases not found in {json_output}"}

                runtimes.append(query_time)

            except subprocess.TimeoutExpired:
                print(f"[{self.name}] Timeout reached ({timeout}s)")
                return {"error": f"Timeout reached ({timeout}s)"}
            except subprocess.CalledProcessError as e:
                return {"error": f"Raytracer failed with exit code {e.returncode}: {e.stderr}"}
            except json.JSONDecodeError:
                return {"error": "Failed to parse timing JSON"}
            finally:
                if json_output.exists():
                    json_output.unlink()

        if not runtimes:
            return {"error": "No timing results collected for Raytracer"}

        # Calculate mean breakdown
        breakdown_stats = {}
        for phase, times in breakdown_accum.items():
            breakdown_stats[phase] = np.mean(times)

        return {
            "mean": np.mean(runtimes),
            "min": np.min(runtimes),
            "max": np.max(runtimes),
            "std": np.std(runtimes),
            "raw_times": runtimes,
            "breakdown": breakdown_stats,
            "num_obj1": num_obj1,
            "num_obj2": num_obj2,
            "num_intersections": num_intersections,
            "universe_extents1": universe_extents1,
            "universe_extents2": universe_extents2
        }
