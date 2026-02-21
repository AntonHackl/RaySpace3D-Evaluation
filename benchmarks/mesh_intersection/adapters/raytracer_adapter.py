import subprocess
import time
import json
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from benchmarks.common.adapters.base import IntersectionBenchmarkAdapter, run_command_streaming


class RaytracerIntersectionAdapter(IntersectionBenchmarkAdapter):
    def __init__(
        self,
        rayspace_dir: str,
        mode: str = "two_pass",
        preprocessed_dir: str = "preprocessed",
        timings_dir: str = "timings",
        grid_resolution: int = 10,
        warmup_runs: int = 2,
    ):
        super().__init__(f"Raytracer_{mode}")
        self.rayspace_dir = Path(rayspace_dir)
        self.mode = mode
        self.preprocessed_dir = Path(preprocessed_dir)
        self.timings_dir = Path(timings_dir)
        self.grid_resolution = grid_resolution
        self.warmup_runs = warmup_runs

        self.timings_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        query_bin_dir = self.rayspace_dir / "query" / "build" / "bin"
        if self.mode == "two_pass":
            self.executable = query_bin_dir / "raytracer_mesh_intersection"
        elif self.mode in ("estimated", "estimate_only"):
            self.executable = query_bin_dir / "raytracer_intersection_estimated"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.preprocess_exec = self.rayspace_dir / "preprocess" / "build" / "bin" / "preprocess_dataset"

    def check_preprocessed(self, file_path: str) -> bool:
        input_path = Path(file_path)
        pre_file = self.preprocessed_dir / input_path.with_suffix('.pre').name
        return pre_file.exists()

    def preprocess_from_source(self, source_file: str, dt_file: str, log_dir: Optional[str] = None):
        source_path = Path(source_file)
        dt_path = Path(dt_file)

        output_geometry = self.preprocessed_dir / dt_path.with_suffix('.pre').name
        output_timing = self.timings_dir / (dt_path.stem + '_timing.json')

        mode = "dt" if source_path.suffix == ".dt" else "mesh"

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
            run_command_streaming(cmd, timeout=None, log_path=None, prefix=f"[{self.name}]")

    def run_intersection(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}

        input_path1 = Path(file1)
        input_path2 = Path(file2)
        p1 = self.preprocessed_dir / input_path1.with_suffix('.pre').name
        p2 = self.preprocessed_dir / input_path2.with_suffix('.pre').name

        f1 = str(p1) if p1.exists() else file1
        f2 = str(p2) if p2.exists() else file2

        runtimes = []
        breakdown_accum = {}
        num_obj1 = 0
        num_obj2 = 0
        num_intersections = 0

        print(f"[{self.name}] Running benchmark...")

        adapter_log_dir = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "two_pass":
            expected_prefixes = ["query_", "gpu deduplication_", "download results_"]
        elif self.mode == "estimated":
            expected_prefixes = ["selectivity estimation_", "query_"]
        else:
            expected_prefixes = ["selectivity estimation_"]

        for run_idx in range(num_runs):
            json_output = self.timings_dir / f"timing_{self.mode}_{int(time.time())}_{run_idx}.json"

            cmd = [
                str(self.executable),
                "--mesh1", f1,
                "--mesh2", f2,
                "--output", str(json_output)
            ]

            if self.mode == "two_pass":
                cmd.extend([
                    "--runs", "1",
                    "--warmup-runs", str(self.warmup_runs),
                    "--no-export"
                ])
            elif self.mode == "estimate_only":
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

                if run_idx == 0:
                    for line in stdout_text.splitlines():
                        if "Mesh1 objects:" in line:
                            num_obj1 = int(line.split(":")[1].strip())
                        elif "Mesh2 objects:" in line:
                            num_obj2 = int(line.split(":")[1].strip())
                        elif "Unique intersecting object pairs:" in line:
                            num_intersections = int(line.split(":")[1].strip())
                        elif "Actual Intersection Pairs:" in line:
                            num_intersections = int(line.split(":")[1].strip())
                        elif "Hash Table Query found" in line:
                            num_intersections = int(line.split("found")[1].split("unique")[0].strip())
                        elif "Final Estimated Pairs:" in line:
                            try:
                                num_intersections = int(line.split(":", 1)[1].strip())
                            except ValueError:
                                pass

                if not json_output.exists():
                    return {"error": f"Timing JSON not found at {json_output}. Output:\n{stdout_text + stderr_text}"}

                with open(json_output, 'r') as f:
                    data = json.load(f)

                phases = data.get("phases", {})
                query_time = 0.0
                found = False

                for prefix in expected_prefixes:
                    key = f"{prefix}1"
                    if key in phases:
                        duration = phases[key].get("duration_ms", 0.0)
                        query_time += duration
                        if prefix not in breakdown_accum:
                            breakdown_accum[prefix] = []
                        breakdown_accum[prefix].append(duration)
                        found = True

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
            "num_intersections": num_intersections
        }
